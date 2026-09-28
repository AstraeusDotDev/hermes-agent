"""Tests: retired-worker fence in the dispatcher claim path.

A lifecycle transition (complete / block / schedule / request-review /
request-changes / reclaim) clears the task claim before the prior
mutation-capable worker has actually exited, which lets a reviewer or repair
worker overlap the same worktree.  The fence in ``_dispatch_lane_task`` checks
the most recent ended run's process identity *before* the claim attempt and
suppresses the spawn while the exact prior process is still live.

Identity model (main): ``_set_worker_pid`` persists ``worker_pid`` +
``worker_started_at`` (the ``_process_fingerprint`` string ``"<boot_id>|<start_time>"``)
on BOTH the ``tasks`` row and the active ``task_runs`` row; ``_end_run`` retains
those columns on the closed run, so the most recent ended run carries the exact
process identity.  ``check_retired_worker`` uses ``_worker_alive`` with the
string fingerprint for exact-match liveness; legacy integer / unverified rows
are treated as "not exact" and are never suppressed (fail-closed for this card).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


@pytest.fixture
def host(conn):
    return kb._claimer_id().split(":", 1)[0]

@pytest.fixture(autouse=True)
def dispatch_preconditions(monkeypatch):
    # These cases exercise the retirement seam, not installation of profile w
    # or the recent-success cooldown. Keep the real claim/spawn/fence paths.
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(kbd, "check_respawn_guard", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------
# Helpers: build a task with a closed run that carries worker identity, then
# stand up a live / dead / recycled / identity-changed process for the fence to
# inspect.
# ---------------------------------------------------------------------------


def _spawn_and_close_run(conn, host, task_id, *, pid=None):
    """Claim the task, record a worker pid + fingerprint, then close the run
    (so the fence sees an ended run with identity).

    Returns the closed run id.
    """
    claimer = f"{host}:A"
    kb.claim_task(conn, task_id, claimer=claimer)
    if pid is None:
        # Capture an exact identity while the child is alive, then reap it
        # before dispatch. /proc cannot fingerprint an already-reaped PID.
        proc = subprocess.Popen(["sleep", "30"])
        try:
            kbd._set_worker_pid(conn, task_id, proc.pid)
            run_id = kb._current_run_id(conn, task_id)
            assert run_id is not None
            assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
            return run_id
        finally:
            proc.terminate()
            proc.wait()
    kbd._set_worker_pid(conn, task_id, pid)
    run_id = kb._current_run_id(conn, task_id)
    assert run_id is not None
    kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
    return run_id


def _latest_retirement_event(conn, task_id):
    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN (?, ?) "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kbd._RETIREMENT_WAIT_KIND, kbd._RETIREMENT_RELEASED_KIND),
    ).fetchone()
    return (row["kind"], kb._json_dict(row["payload"])) if row else (None, None)


# ---------------------------------------------------------------------------
# Fence exercised through the real dispatch seam: _dispatch_lane_task.
# ---------------------------------------------------------------------------


def test_retired_worker_suppresses_ready_spawn(conn, host):
    """A ready task whose prior worker is still live is NOT re-claimed this
    tick — the fence returns "retirement_wait" and the spawn is skipped."""
    tid = kb.create_task(conn, title="retired", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=sleeper.pid)
        # Back to ready, unclaimed.
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, "w", result,
            lane="ready", dry_run=False, ttl_seconds=None, board=None,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
            spawn_fn=None, per_profile_cap=None, per_profile_running={},
        ) is False
        assert result.respawn_guarded
        assert result.respawn_guarded[-1][1] == kbd._RETIREMENT_WAIT_KIND
        assert tid not in {t for t, _, _ in result.spawned}
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_dead_prior_worker_releases_for_dispatch(conn, host):
    """When the prior worker is dead, the fence emits retirement_released and
    the task is dispatchable (spawn slot consumed)."""
    tid = kb.create_task(conn, title="dead", assignee="w")
    run_id = _spawn_and_close_run(conn, host, tid)
    ended = kbd._ended_run_with_identity(conn, tid)
    assert ended is not None and ended["id"] == run_id
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    result = kbd.DispatchResult()
    row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
    assert kbd._dispatch_lane_task(
        conn, row, "w", result,
        lane="ready", dry_run=False, ttl_seconds=None, board=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
        spawn_fn=None, per_profile_cap=None, per_profile_running={},
    ) is True
    assert not result.respawn_guarded
    kind, payload = _latest_retirement_event(conn, tid)
    assert kind == kbd._RETIREMENT_RELEASED_KIND
    assert payload is not None and payload["run_id"] == run_id
    assert len(result.spawned) == 1
    assert result.spawned[0][:2] == (tid, "w")
    assert result.spawned[0][2]


def test_pid_reuse_does_not_suppress_unrelated_reused_pid(conn, host):
    """A live PID with a different start fingerprint than the ended run must
    release. Reuse is modeled on the isolated run row; the OS need not recycle
    a PID during this test."""
    tid = kb.create_task(conn, title="reuse", assignee="w")
    previous = subprocess.Popen(["sleep", "30"])
    try:
        old_fingerprint = kbd._process_fingerprint(previous.pid)
        assert old_fingerprint is not None
    finally:
        previous.terminate()
        previous.wait()
    # Linux records process start in clock ticks; separate launches by more
    # than one tick so two valid fingerprints do not coincide.
    time.sleep(0.15)
    newer = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=newer.pid)
        new_fingerprint = kbd._process_fingerprint(newer.pid)
        assert new_fingerprint is not None and old_fingerprint != new_fingerprint
        # Model a recycled PID: the ended run retains the original process's
        # fingerprint, while the current holder of its recorded PID is newer.
        conn.execute(
            "UPDATE task_runs SET worker_started_at=? WHERE id=?",
            (old_fingerprint, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, "w", result,
            lane="ready", dry_run=False, ttl_seconds=None, board=None,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
            spawn_fn=None, per_profile_cap=None, per_profile_running={},
        ) is True
        assert not result.respawn_guarded
        assert result.spawned[0][:2] == (tid, "w")
        kind, payload = _latest_retirement_event(conn, tid)
        assert kind == kbd._RETIREMENT_RELEASED_KIND
        assert payload is not None and payload["run_id"] == run_id
    finally:
        newer.terminate()
        newer.wait()


def test_unverified_fingerprint_fails_closed(conn, host):
    """A live worker with an unverified fingerprint cannot be identified as
    the exact predecessor. The card is released rather than PID-only fenced."""
    tid = kb.create_task(conn, title="unverified", assignee="w")
    proc = subprocess.Popen(["sleep", "30"])
    try:
        kb.claim_task(conn, tid, claimer=f"{host}:A")
        run_id = kb._current_run_id(conn, tid)
        assert run_id is not None
        # Retain a real, still-live PID but deliberately remove exact identity.
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
            (proc.pid, kbd.UNVERIFIED_WORKER_FINGERPRINT, tid),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
            (proc.pid, kbd.UNVERIFIED_WORKER_FINGERPRINT, run_id),
        )
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, "w", result,
            lane="ready", dry_run=False, ttl_seconds=None, board=None,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
            spawn_fn=None, per_profile_cap=None, per_profile_running={},
        ) is True
        assert not result.respawn_guarded
        assert result.spawned[0][:2] == (tid, "w")
        kind, payload = _latest_retirement_event(conn, tid)
        assert kind == kbd._RETIREMENT_RELEASED_KIND
        assert payload is not None and payload["run_id"] == run_id
    finally:
        proc.terminate()
        proc.wait()


def test_retired_worker_in_review_lane(conn, host):
    """The fence also guards the review lane — a reviewer must not be spawned
    beside a still-live prior implementer."""
    tid = kb.create_task(conn, title="review-retired", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='review', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, "w", result,
            lane="review", dry_run=False, ttl_seconds=None, board=None,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
            spawn_fn=None, per_profile_cap=None, per_profile_running={},
        ) is False
        assert result.respawn_guarded and result.respawn_guarded[-1][1] == kbd._RETIREMENT_WAIT_KIND
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_coalesced_retirement_wait_no_duplicate_events(conn, host):
    """Concurrent dispatcher ticks do not spam retirement_wait events for the
    same ended run."""
    tid = kb.create_task(conn, title="coalesce", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        for tick in range(3):
            result = kbd.DispatchResult()
            row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
            kbd._dispatch_lane_task(
                conn, row, "w", result,
                lane="ready", dry_run=False, ttl_seconds=None, board=None,
                failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
                spawn_fn=None, per_profile_cap=None, per_profile_running={},
            )
            if tick == 0:
                # An intervening event does not turn this into a new run.
                with kb.write_txn(conn):
                    kb._append_event(conn, tid, "commented", {"body": "still waiting"})
        # Exactly one retirement_wait event for this task.
        n = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = ?",
            (tid, kbd._RETIREMENT_WAIT_KIND),
        ).fetchone()[0]
        assert n == 1
        kind, payload = _latest_retirement_event(conn, tid)
        assert kind == kbd._RETIREMENT_WAIT_KIND
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_check_retired_worker_dry_run_is_read_only(conn, host):
    """A dry-run reports the fence verdict without writing an event; a real
    dispatch records one wait receipt for the same identity-bearing run."""
    tid = kb.create_task(conn, title="dry", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]
        assert kbd.check_retired_worker(conn, tid, dry_run=True) == kbd._RETIREMENT_WAIT_KIND
        after = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (tid,)).fetchone()[0]
        assert after == before
        assert _latest_retirement_event(conn, tid) == (None, None)
        assert kbd.check_retired_worker(conn, tid, dry_run=False) == kbd._RETIREMENT_WAIT_KIND
        assert kbd.check_retired_worker(conn, tid, dry_run=False) == kbd._RETIREMENT_WAIT_KIND
        kind, payload = _latest_retirement_event(conn, tid)
        assert kind == kbd._RETIREMENT_WAIT_KIND
        assert payload is not None and payload["run_id"] == run_id
        count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind=?",
            (tid, kbd._RETIREMENT_WAIT_KIND),
        ).fetchone()[0]
        assert count == 1
    finally:
        sleeper.terminate()
        sleeper.wait()


# ---------------------------------------------------------------------------
# Reviewer counterexamples: weak identities and capacity reservation must not
# suppress unrelated ready work or write receipts during read-only preflight.
# ---------------------------------------------------------------------------

def test_start_only_fingerprint_cannot_suppress_live_pid(conn, host):
    tid = kb.create_task(conn, title="malformed", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, tid, pid=sleeper.pid)
        actual = kbd._process_fingerprint(sleeper.pid)
        assert actual is not None
        start = actual.rsplit("|", 1)[1]
        start_only = start[:1] + "_" + start[1:]  # float accepts _, SQLite keeps TEXT
        assert kbd._worker_alive(sleeper.pid, start_only) is True
        conn.execute("UPDATE task_runs SET worker_started_at=? WHERE id=?", (start_only, run_id))
        conn.commit()
        selected = kbd._ended_run_with_identity(conn, tid)
        assert selected is not None
        assert kbd._has_exact_identity(conn, selected) is False
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_fenced_review_does_not_reserve_only_ready_slot(conn, host):
    review_id = kb.create_task(conn, title="fenced-review", assignee="w")
    ready_id = kb.create_task(conn, title="ready-work", assignee="w")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        run_id = _spawn_and_close_run(conn, host, review_id, pid=sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='review', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (review_id,),
        )
        conn.commit()
        rows = kbd._lane_rows(conn, "review")
        assert [row["id"] for row in rows] == [review_id]
        before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (review_id,)).fetchone()[0]
        assert not kbd._any_spawnable_review(conn, rows)
        after = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (review_id,)).fetchone()[0]
        assert after == before, "capacity preflight must not emit fence receipts"
        assert kb.get_task(conn, ready_id).status == "ready"
        assert run_id is not None
    finally:
        sleeper.terminate()
        sleeper.wait()


# ---------------------------------------------------------------------------
# Selector regression: an identity-less run row must not hide the
# identity-bearing run the fence depends on.
#
# A later completion with no active run can synthesize an identity-less row
# after an earlier worker completion. If that worker is still alive and the
# two rows share an ``ended_at`` second, ordering alone picks the higher id
# and silently loses the retirement fence.
# ---------------------------------------------------------------------------


def test_identity_selector_ignores_synthesized_identityless_run(conn, host):
    """The most recent ended run *with identity* wins even when a later
    identity-less row shares its ``ended_at`` second."""
    tid = kb.create_task(conn, title="selector", assignee="w")
    _spawn_and_close_run(conn, host, tid)

    real = conn.execute(
        "SELECT id, worker_pid, worker_started_at FROM task_runs "
        "WHERE task_id=? AND worker_pid IS NOT NULL",
        (tid,),
    ).fetchone()
    assert real is not None, "fixture must produce an identity-bearing run"
    ended_at = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id=?", (real["id"],)
    ).fetchone()["ended_at"]

    # A later administrative completion can synthesize an identity-less row
    # when there is no current run, even though the previous worker lives.
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?", (tid,)
    )
    conn.commit()
    assert kb.complete_task(conn, tid, summary="administrative completion")
    synthetic = conn.execute(
        "SELECT id, ended_at, worker_pid FROM task_runs WHERE task_id=? "
        "ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert synthetic["id"] > real["id"] and synthetic["worker_pid"] is None
    # The selector regression needs a same-second tie. Align the first row's
    # timestamp if the two genuine completions straddled a second boundary.
    conn.execute("UPDATE task_runs SET ended_at=? WHERE id=?", (synthetic["ended_at"], real["id"]))
    conn.commit()

    top = conn.execute(
        "SELECT id, worker_pid FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert top["id"] != real["id"], "fixture must make the NULL row sort first"
    assert top["worker_pid"] is None

    # The fence must still see the real worker's identity.
    selected = kbd._ended_run_with_identity(conn, tid)
    assert selected is not None, "identity-bearing run must remain selectable"
    assert selected["id"] == real["id"]
    assert selected["worker_pid"] == real["worker_pid"]
    assert selected["worker_started_at"] == real["worker_started_at"]


def test_identity_selector_returns_none_when_no_run_carries_identity(conn, host):
    """With only identity-less rows, the fence applies nothing and returns
    None (no suppression) rather than a bogus run."""
    tid = kb.create_task(conn, title="no-identity", assignee="w")
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
        "VALUES (?, ?, 'completed', ?, ?)",
        (tid, host, now, now),
    )
    conn.commit()
    assert kbd._ended_run_with_identity(conn, tid) is None
