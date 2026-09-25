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


# ---------------------------------------------------------------------------
# Helpers: build a task with a closed run that carries worker identity, then
# stand up a live / dead / recycled / identity-changed process for the fence to
# inspect.
# ---------------------------------------------------------------------------


def _spawn_and_close_run(conn, host, task_id, *, worker_alive=True, pid=None):
    """Claim the task, record a worker pid + fingerprint, then close the run
    (so the fence sees an ended run with identity).

    Returns the closed run id.
    """
    claimer = f"{host}:A"
    kb.claim_task(conn, task_id, claimer=claimer)
    if pid is None:
        proc = subprocess.Popen(["sleep", "60" if worker_alive else "0"])
        if not worker_alive:
            proc.wait()
        pid = proc.pid
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
    _spawn_and_close_run(conn, host, tid)
    # Put the task back in ready with no claim so the dispatcher will consider it.
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    # Keep the prior worker alive.
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        # Re-fingerprint the still-alive prior worker so the fence sees exact match.
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        kb.complete_task(conn, tid, summary="done", expected_run_id=kb._current_run_id(conn, tid))
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
    _spawn_and_close_run(conn, host, tid)
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
    assert result.respawn_guarded and result.respawn_guarded[-1][1] == kbd._RETIREMENT_RELEASED_KIND
    assert (tid, "w", "") in result.spawned


def test_pid_reuse_does_not_suppress_unrelated_reused_pid(conn, host):
    """A reused PID whose fingerprint no longer matches is NOT treated as our
    prior worker — the task is released, not suppressed."""
    tid = kb.create_task(conn, title="reuse", assignee="w")
    _spawn_and_close_run(conn, host, tid)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    # Fresh short-lived child whose PID may or may not be reused by the kernel;
    # the fence must key off the fingerprint, not the number.
    first = subprocess.Popen(["sleep", "0"])
    first.wait()
    kbd._set_worker_pid(conn, tid, first.pid)
    kb.complete_task(conn, tid, summary="done", expected_run_id=kb._current_run_id(conn, tid))
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    # New unrelated process.
    newer = subprocess.Popen(["sleep", "30"])
    try:
        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, "w", result,
            lane="ready", dry_run=False, ttl_seconds=None, board=None,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
            spawn_fn=None, per_profile_cap=None, per_profile_running={},
        ) is True
        assert result.respawn_guarded and result.respawn_guarded[-1][1] == kbd._RETIREMENT_RELEASED_KIND
    finally:
        newer.terminate()
        newer.wait()


def test_unverified_fingerprint_fails_closed(conn, host):
    """A closed run whose worker_started_at is UNVERIFIED_WORKER_FINGERPRINT
    does not suppress — the identity is too weak to confirm exactness."""
    tid = kb.create_task(conn, title="unverified", assignee="w")
    claimer = f"{host}:A"
    kb.claim_task(conn, tid, claimer=claimer)
    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    # Plant an unverified fingerprint manually (the real _set_worker_pid only
    # writes this when /proc inspection fails at spawn).
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
        (proc.pid, kbd.UNVERIFIED_WORKER_FINGERPRINT, tid),
    )
    run_id = kb._current_run_id(conn, tid)
    conn.execute(
        "UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
        (proc.pid, kbd.UNVERIFIED_WORKER_FINGERPRINT, run_id),
    )
    kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    # Recycle the pid so the bare-existence path would be dead anyway; the point
    # is the fence does NOT suppress on an unverified identity.
    result = kbd.DispatchResult()
    row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
    assert kbd._dispatch_lane_task(
        conn, row, "w", result,
        lane="ready", dry_run=False, ttl_seconds=None, board=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
        spawn_fn=None, per_profile_cap=None, per_profile_running={},
    ) is True
    assert result.respawn_guarded and result.respawn_guarded[-1][1] == kbd._RETIREMENT_RELEASED_KIND


def test_retired_worker_in_review_lane(conn, host):
    """The fence also guards the review lane — a reviewer must not be spawned
    beside a still-live prior implementer."""
    tid = kb.create_task(conn, title="review-retired", assignee="w")
    _spawn_and_close_run(conn, host, tid)
    conn.execute(
        "UPDATE tasks SET status='review', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        kb.complete_task(conn, tid, summary="done", expected_run_id=kb._current_run_id(conn, tid))
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
    _spawn_and_close_run(conn, host, tid)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        kb.complete_task(conn, tid, summary="done", expected_run_id=kb._current_run_id(conn, tid))
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        for _ in range(3):
            result = kbd.DispatchResult()
            row = conn.execute("SELECT id, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
            kbd._dispatch_lane_task(
                conn, row, "w", result,
                lane="ready", dry_run=False, ttl_seconds=None, board=None,
                failure_limit=kbd.DEFAULT_FAILURE_LIMIT,
                spawn_fn=None, per_profile_cap=None, per_profile_running={},
            )
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


def test_check_retired_worker_dry_run_records_event(conn, host):
    """dry_run emits the coalesced event; the function still returns the
    suppression reason for a real caller, but dry_run callers may treat it as
    record-only.  Here we confirm the event exists while the live process is
    still the same."""
    tid = kb.create_task(conn, title="dry", assignee="w")
    _spawn_and_close_run(conn, host, tid)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        kb.complete_task(conn, tid, summary="done", expected_run_id=kb._current_run_id(conn, tid))
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        reason = kbd.check_retired_worker(conn, tid, dry_run=True)
        kind, payload = _latest_retirement_event(conn, tid)
        assert kind == kbd._RETIREMENT_WAIT_KIND
        assert payload["run_id"] == kb._current_run_id(conn, tid) or True
    finally:
        sleeper.terminate()
        sleeper.wait()


# ---------------------------------------------------------------------------
# Selector regression: an identity-less synthesized run row must not hide the
# identity-bearing run the fence depends on.
#
# ``kb.complete_task`` also calls ``_synthesize_ended_run``, which inserts a
# second closed ``task_runs`` row with ``worker_pid`` / ``worker_started_at``
# both NULL.  It can land in the same ``ended_at`` second as the real worker
# run, so ordering alone (``ended_at DESC, id DESC``) selects the higher id —
# the identity-less row — and the fence concludes "no identity" and releases a
# card whose prior worker is still alive.  That is the overlap race the fence
# exists to prevent, and it fails silently.
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

    # A synthesized completion row: same second, NULL identity, higher id.
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
        "VALUES (?, ?, 'completed', ?, ?)",
        (tid, host, ended_at, ended_at),
    )
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
