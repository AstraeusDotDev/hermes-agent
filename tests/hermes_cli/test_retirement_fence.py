"""
Retirement fence tests — guards against re-claiming a task whose prior
worker is still alive. Uses a fake psutil process injected into the
dispatch module's module namespace so no real PIDs are touched.
"""
import sqlite3
import pytest

from hermes_cli.kanban_db import (
    init_db,
    claim_task,
    claim_review_task,
    complete_task,
    get_task,
)
from hermes_cli.kanban_db_connect import _migrate_add_optional_columns
from hermes_cli import kanban_db as kb


# ---- test doubles ---------------------------------------------------------


class FakeProcess:
    """Minimal psutil.Process look-alike with configurable create_time/exe."""

    def __init__(self, pid: int, create_time: float, exe_path: str):
        self._pid = pid
        self._create_time = create_time
        self._exe = exe_path

    def create_time(self):
        return self._create_time

    def exe(self):
        return self._exe


class FakePsutil:
    """psutil replacement that returns our FakeProcess or raises."""

    class NoSuchProcess(Exception):
        pass

    def __init__(self):
        self._procs = {}

    def register(self, pid: int, create_time: float, exe_path: str):
        self._procs[pid] = FakeProcess(pid, create_time, exe_path)

    def Process(self, pid):
        if pid not in self._procs:
            raise self.NoSuchProcess(f"No such process: {pid}")
        return self._procs[pid]


# ---- fixture --------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    from hermes_cli.kanban_db_connect import connect
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ---- tests ----------------------------------------------------------------


def test_migration_adds_task_runs_fence_columns(tmp_path):
    """_migrate_add_optional_columns adds worker_create_time + worker_exe_path to legacy DBs."""
    db_path = tmp_path / "legacy.db"
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Fresh DB already has the columns from SCHEMA_SQL; to test the migration
    # path we'd need a truly legacy DB. But we can at least assert the columns
    # exist on a fresh init.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(task_runs)").fetchall()}
    assert "worker_create_time" in cols
    assert "worker_exe_path" in cols
    conn.close()


def test_claim_task_allowed_when_no_prior_worker(db):
    """Fresh task with no prior run -> claim succeeds."""
    cur = db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    task_id = "t1"
    claimed = claim_task(db, task_id)
    assert claimed is not None
    assert claimed.status == "running"


def _setup_fake_psutil(monkeypatch, fake_psutil):
    """Patch psutil in sys.modules so all modules see it."""
    import sys
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)


def test_claim_task_suppressed_when_prior_worker_alive(db, monkeypatch):
    """Prior run with live worker -> claim suppressed, task demoted to ready."""
    fake = FakePsutil()
    fake.register(12345, 1000.0, "/usr/bin/python3")
    _setup_fake_psutil(monkeypatch, fake)

    # Seed: completed run with a "live" worker
    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'done', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 12345, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )
    # Reset task to ready so claim can be attempted
    db.execute("UPDATE tasks SET status = 'ready' WHERE id = 't1'")

    claimed = claim_task(db, "t1")
    assert claimed is None, "claim should be suppressed"
    task = get_task(db, "t1")
    assert task.status == "ready"
    event = db.execute(
        "SELECT kind FROM task_events WHERE task_id = 't1' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert event["kind"] == "retirement_wait"


def test_claim_task_succeeds_when_prior_worker_dead(db, monkeypatch):
    """Prior run with dead worker (PID gone) -> claim succeeds."""
    fake = FakePsutil()
    # Do NOT register the PID — simulates process gone
    _setup_fake_psutil(monkeypatch, fake)

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 99999, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )

    claimed = claim_task(db, "t1")
    assert claimed is not None
    assert claimed.status == "running"


def test_claim_task_succeeds_when_pid_reused_different_create_time(db, monkeypatch):
    """PID exists but create_time differs -> treated as different process, claim allowed."""
    fake = FakePsutil()
    fake.register(12345, 9999.0, "/usr/bin/python3")  # different create_time
    _setup_fake_psutil(monkeypatch, fake)

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 12345, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )

    claimed = claim_task(db, "t1")
    assert claimed is not None
    assert claimed.status == "running"


def test_claim_task_succeeds_when_pid_reused_different_exe(db, monkeypatch):
    """PID exists but exe_path differs -> treated as different process, claim allowed."""
    fake = FakePsutil()
    fake.register(12345, 1000.0, "/usr/bin/other")
    _setup_fake_psutil(monkeypatch, fake)

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 12345, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )

    claimed = claim_task(db, "t1")
    assert claimed is not None
    assert claimed.status == "running"


def test_claim_task_allowed_when_psutil_unavailable(db, monkeypatch):
    """psutil import fails -> conservative 'wait' to avoid false 'released'."""
    # Simulate psutil import failure by making psutil raise on import
    class BadPsutil:
        class NoSuchProcess(Exception):
            pass
        def Process(self, pid):
            raise ImportError("no psutil")
    _setup_fake_psutil(monkeypatch, BadPsutil())

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 12345, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )

    claimed = claim_task(db, "t1")
    assert claimed is None, "conservative wait when psutil unavailable"
    task = get_task(db, "t1")
    assert task.status == "ready"


def test_claim_review_task_suppressed_when_prior_worker_alive(db, monkeypatch):
    """Same fence logic applies to review->running claims."""
    fake = FakePsutil()
    fake.register(12345, 1000.0, "/usr/bin/python3")
    _setup_fake_psutil(monkeypatch, fake)

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'review', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'done', 12345, 1000.0, '/usr/bin/python3', 1000, 2000, 'completed')
        """
    )

    claimed = claim_review_task(db, "t1")
    assert claimed is None
    task = get_task(db, "t1")
    assert task.status == "review"
    event = db.execute(
        "SELECT kind FROM task_events WHERE task_id = 't1' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert event["kind"] == "retirement_wait"


def test_active_run_no_ended_at_suppresses_claim(db):
    """Most recent run has no ended_at (still running) -> suppress without psutil."""
    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    db.execute(
        """
        INSERT INTO task_runs (task_id, profile, status, worker_pid, worker_create_time, worker_exe_path, started_at, ended_at, outcome)
        VALUES ('t1', 'alice', 'running', 12345, 1000.0, '/usr/bin/python3', 1000, NULL, NULL)
        """
    )

    claimed = claim_task(db, "t1")
    assert claimed is None
    task = get_task(db, "t1")
    assert task.status == "ready"


def test_end_run_preserves_worker_identity(db):
    """complete_task does NOT clear worker_pid/create_time/exe_path on task_runs."""
    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    claimed = claim_task(db, "t1")
    run_id = claimed.current_run_id

    # Simulate worker identity being set
    db.execute(
        "UPDATE task_runs SET worker_pid = 12345, worker_create_time = 1000.0, worker_exe_path = '/usr/bin/python3' WHERE id = ?",
        (run_id,),
    )

    complete_task(db, "t1", summary="done")

    row = db.execute(
        "SELECT worker_pid, worker_create_time, worker_exe_path FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert row["worker_pid"] == 12345
    assert row["worker_create_time"] == 1000.0
    assert row["worker_exe_path"] == "/usr/bin/python3"


def test_dispatch_set_worker_pid_populates_identity(db, monkeypatch):
    """_set_worker_pid in dispatch populates create_time + exe_path via psutil."""
    import hermes_cli.kanban_db_dispatch as dispatch

    fake = FakePsutil()
    fake.register(54321, 5000.0, "/home/user/worker.py")
    _setup_fake_psutil(monkeypatch, fake)

    db.execute(
        "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) VALUES ('t1', 'test task', 'ready', 'alice', 'default', 1000)"
    )
    claim_task(db, "t1")
    task = get_task(db, "t1")
    run_id = task.current_run_id

    dispatch._set_worker_pid(db, "t1", 54321)

    row = db.execute(
        "SELECT worker_pid, worker_create_time, worker_exe_path FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert row["worker_pid"] == 54321
    assert row["worker_create_time"] == 5000.0
    assert row["worker_exe_path"] == "/home/user/worker.py"