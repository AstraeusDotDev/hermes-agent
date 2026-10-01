"""Parked /goal barriers must wake from the TUI's empty-queue idle boundary."""
from __future__ import annotations

import importlib
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    server = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    emitted, submitted = [], []
    real_submit = server._run_prompt_submit
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: submitted.append((a, k)) or True)
    ready = threading.Event()
    ready.set()
    session = {
        "session_key": "parked-goal", "agent": SimpleNamespace(session_id="parked-goal"),
        "agent_ready": ready, "running": False, "history": [], "history_lock": threading.Lock(),
    }
    mgr = goals.GoalManager(session["session_key"])
    mgr.set("finish the isolated goal", max_turns=7)
    mgr.wait_for_seconds(1, reason="isolated timer")
    assert mgr.state is not None
    mgr.state.waiting_until = time.time() - 1
    mgr.state.turns_used = 2
    mgr._save()
    yield SimpleNamespace(server=server, goals=goals, session=session, manager=mgr,
                          submitted=submitted, emitted=emitted, home=home, real_submit=real_submit)
    goals._DB_CACHE.clear()
    server._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


def _run_empty_poller(env, monkeypatch, passes=2, *, real_loop=False):
    """No events or sleeps: each queue wait advances one real idle poller pass."""
    from tools import async_delegation
    from tools.process_registry import process_registry

    stop = threading.Event()
    empty = MagicMock()
    waits = []

    def idle_wait(timeout):
        waits.append(timeout)
        # A finished ordinary turn releases its claim before the next idle pass.
        env.session["running"] = False
        if len(waits) == passes:
            stop.set()
        raise queue.Empty

    empty.get.side_effect = idle_wait
    empty.qsize.return_value = 0
    monkeypatch.setattr(process_registry, "completion_queue", empty)
    monkeypatch.setattr(process_registry, "restore_completions", lambda: None)
    monkeypatch.setattr(async_delegation, "maybe_sweep_orphaned_completions", lambda q: None)
    for name in ("_poll_bot_live_delivery_guarded", "_notif_poll_kanban",
                 "_maybe_fire_tui_loop_tick", "_maybe_fire_tui_heartbeat_tick"):
        if real_loop and name == "_maybe_fire_tui_loop_tick":
            continue
        monkeypatch.setattr(env.server, name, lambda *a: None)
    ticks = iter(10.0 * (i + 1) for i in range(passes))
    monkeypatch.setattr(env.server, "time", SimpleNamespace(
        time=time.time, monotonic=lambda: next(ticks), sleep=time.sleep))
    env.server._notification_poller_loop(stop, "ui-parked", env.session)
    assert waits == [0.5] * passes
    empty.get_nowait.assert_not_called()
    empty.put.assert_not_called()


def test_empty_idle_poller_resumes_once_without_spending_judge_budget(env, monkeypatch):
    _run_empty_poller(env, monkeypatch)
    saved = env.goals.load_goal(env.manager.session_id)
    print({"executable": sys.executable, "python": sys.version.split()[0],
           "server": env.server.__file__,
           "poller_source": env.server._notification_poller_scoped_loop.__code__.co_filename,
           "submitted": len(env.submitted), "waiting_until": saved.waiting_until,
           "turns_used": saved.turns_used})
    assert len(env.submitted) == 1, "expired parked goal never wakes on an empty completion queue"
    args, kwargs = env.submitted[0]
    assert args[1:3] == ("ui-parked", env.session)
    assert args[3] == env.manager.next_continuation_prompt()
    assert not kwargs.get("terminal_callback"), "wake must remain an ordinary resumable turn"
    assert saved.waiting_until == 0 and saved.waiting_reason is None
    assert saved.status == "active" and saved.turns_used == 2 and saved.max_turns == 7
    assert any(event == "status.update" and payload["kind"] == "goal"
               for event, sid, payload in env.emitted)


def test_failed_idle_wake_retries_even_when_a_due_loop_shares_the_poller(env, monkeypatch):
    from hermes_cli.loops import LoopManager

    loop = LoopManager(env.manager.session_id)
    loop.set("unrelated loop", interval_seconds=60)
    attempts = []

    def submit(*a, **k):
        attempts.append(a[3])
        return len(attempts) > 1

    monkeypatch.setattr(env.server, "_run_prompt_submit", submit)
    _run_empty_poller(env, monkeypatch, real_loop=True)
    assert attempts == [env.manager.next_continuation_prompt()] * 2
    assert LoopManager(env.manager.session_id).state.ticks_fired == 0
    saved = env.goals.load_goal(env.manager.session_id)
    assert saved.waiting_until == 0 and saved.turns_used == 2


def test_barrier_satisfied_between_goal_and_loop_probes_still_wakes(env, monkeypatch):
    from hermes_cli.loops import LoopManager

    clock = [time.time()]
    monkeypatch.setattr(env.goals, "time", SimpleNamespace(
        time=lambda: clock[0], monotonic=time.monotonic, sleep=time.sleep))
    env.manager.wait_for_seconds(100, reason="expires between idle probes")
    deadline = env.manager.state.waiting_until
    loop = LoopManager(env.manager.session_id)
    loop.set("unrelated loop", interval_seconds=60)
    fire_loop = env.server._maybe_fire_tui_loop_tick

    def expire_before_loop(*args):
        clock[0] = deadline + 1
        return fire_loop(*args)

    monkeypatch.setattr(env.server, "_maybe_fire_tui_loop_tick", expire_before_loop)
    _run_empty_poller(env, monkeypatch, real_loop=True)
    assert len(env.submitted) == 1, "loop probe consumed the barrier without resuming the goal"
    assert env.submitted[0][0][3] == env.manager.next_continuation_prompt()
    assert LoopManager(env.manager.session_id).state.ticks_fired == 0
    saved = env.goals.load_goal(env.manager.session_id)
    assert saved.waiting_until == 0 and saved.turns_used == 2


def test_wake_helper_clears_once_and_leaves_the_judge_to_the_ordinary_turn(env, monkeypatch):
    clears = []
    original = env.goals.GoalState.clear_wait

    def clear_wait(state):
        clears.append(state.waiting_until)
        original(state)

    monkeypatch.setattr(env.goals.GoalState, "clear_wait", clear_wait)
    judge = MagicMock(side_effect=AssertionError("idle wake must not judge"))
    monkeypatch.setattr(env.goals, "judge_goal", judge)
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    env.session["running"] = False
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    assert clears == [env.manager.state.waiting_until]
    assert len(env.submitted) == 1
    assert env.submitted[0][1] == {
        "image_paths": [], "queued_prompt_generation": 0,
        "goal_claim": (env.manager.session_id, env.goals.load_goal(env.manager.session_id).to_json())}
    assert env.goals.load_goal(env.manager.session_id).turns_used == 2
    judge.assert_not_called()


@pytest.mark.parametrize("blocked", [
    "unexpired", "busy", "closing", "finalized", "interrupted", "no-agent", "not-ready",
    "paused", "done", "cleared", "no-goal", "budget", "not-parked", "queued", "queue-list",
    "auto-continue", "pid", "process-session", "delegations", "no-key",
])
def test_waits_and_ineligible_sessions_neither_wake_nor_mutate_goal(env, monkeypatch, blocked):
    from tools.process_registry import process_registry

    state, session = env.manager.state, env.session
    monkeypatch.setattr(process_registry, "is_session_waiting", lambda key: key == "watcher")
    monkeypatch.setattr(env.goals, "count_active_delegations", lambda key: 2)
    mutations = {
        "unexpired": lambda: setattr(state, "waiting_until", time.time() + 100),
        "busy": lambda: session.update(running=True),
        "closing": lambda: session.update(_closing=True),
        "finalized": lambda: session.update(_finalized=True),
        "interrupted": lambda: session.update(_turn_cancel_requested=True),
        "no-agent": lambda: session.update(agent=None),
        "not-ready": session["agent_ready"].clear,
        "paused": lambda: setattr(state, "status", "paused"),
        "done": lambda: setattr(state, "status", "done"),
        "cleared": lambda: setattr(state, "status", "cleared"),
        "no-goal": lambda: session.update(session_key="no-goal-here", agent=SimpleNamespace(session_id="no-goal-here")),
        "budget": lambda: setattr(state, "turns_used", state.max_turns),
        "not-parked": state.clear_wait,
        "queued": lambda: session.update(queued_prompt={"text": "a person is waiting"}),
        "queue-list": lambda: session.update(queued_prompts=["a person is waiting"]),
        "auto-continue": lambda: session.update(_auto_continue_scheduled=True),
        "pid": lambda: env.manager.wait_on(os.getpid(), reason="the test process is alive"),
        "process-session": lambda: env.manager.wait_on_session("watcher"),
        "delegations": lambda: env.manager.wait_for_seconds(100, on_delegations=2),
        "no-key": lambda: session.update(session_key="", agent=SimpleNamespace(session_id="")),
    }
    mutations[blocked]()
    env.manager._save()
    before = env.goals.load_goal(env.manager.session_id)
    before = before.to_json() if before is not None else None
    was_running = session["running"]
    env.server._maybe_resume_tui_parked_goal("ui-parked", session)
    after = env.goals.load_goal(env.manager.session_id)
    assert (after.to_json() if after is not None else None) == before
    assert session["running"] is was_running
    assert env.submitted == [] and env.emitted == []


@pytest.mark.parametrize("refusal", [False, RuntimeError("worker unavailable")], ids=["refused", "raised"])
def test_failed_submission_retains_the_exact_wait_for_a_successful_retry(env, monkeypatch, refusal):
    before = env.goals.load_goal(env.manager.session_id).to_json()

    def refuse(*a, **k):
        if isinstance(refusal, Exception):
            raise refusal
        return refusal

    monkeypatch.setattr(env.server, "_run_prompt_submit", refuse)
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    assert env.session["running"] is False
    assert env.goals.load_goal(env.manager.session_id).to_json() == before
    assert env.emitted == []
    monkeypatch.setattr(env.server, "_run_prompt_submit", lambda *a, **k: env.submitted.append(a) or True)
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    assert len(env.submitted) == 1
    assert env.goals.load_goal(env.manager.session_id).waiting_until == 0
    assert env.goals.load_goal(env.manager.session_id).turns_used == 2


@pytest.mark.parametrize("mutation", ["pause", "replace"])
def test_refusal_rollback_does_not_clobber_a_write_after_its_read(env, monkeypatch, mutation):
    db = env.goals._get_session_db()
    execute_write = db._execute_write
    armed = [False]
    updated = []

    def write_after_rollback_read(fn, *args, **kwargs):
        if armed[0]:
            armed[0] = False
            newer = env.goals.GoalManager(env.manager.session_id)
            if mutation == "pause":
                newer.pause(reason="user pause at the persistence boundary")
            else:
                newer.set("replacement goal must survive")
            updated.append(newer.state.to_json())
        return execute_write(fn, *args, **kwargs)

    def refuse(*args, **kwargs):
        armed[0] = True
        return False

    monkeypatch.setattr(db, "_execute_write", write_after_rollback_read)
    monkeypatch.setattr(env.server, "_run_prompt_submit", refuse)
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    assert updated, "the competing persisted goal write was not exercised"
    assert env.goals.load_goal(env.manager.session_id).to_json() == updated[0], (
        "refusal rollback overwrote the concurrent goal command")
    assert env.session["running"] is False
    assert env.submitted == []


@pytest.mark.parametrize("accepted", [True, False])
def test_dispatch_never_overwrites_a_newer_goal_state(env, monkeypatch, accepted):
    updated = []

    def submit(*a, **k):
        newer = env.goals.GoalManager(env.manager.session_id)
        if accepted:
            # The prompt worker may finish and re-park before submit returns.
            newer.wait_for_seconds(100, reason="new worker wait")
            newer.state.turns_used += 1
            newer._save()
        else:
            newer.pause(reason="a racing user paused the goal")
        updated.append(newer.state.to_json())
        return accepted

    monkeypatch.setattr(env.server, "_run_prompt_submit", submit)
    env.server._maybe_resume_tui_parked_goal("ui-parked", env.session)
    assert env.goals.load_goal(env.manager.session_id).to_json() == updated[0]
    assert env.session["running"] is accepted


@pytest.mark.parametrize("boundary", ["persistence", "admitted"])
@pytest.mark.parametrize("successor", ["none", "user", "replacement"])
def test_stop_invalidates_parked_wake_before_worker_launch(env, monkeypatch, boundary, successor):
    server, session = env.server, env.session
    before = env.goals.load_goal(env.manager.session_id).to_json()
    session["agent"].clear_interrupt = MagicMock()
    session["agent"].interrupt = MagicMock()
    workers = []
    injected = []
    newer = []
    save = env.goals.save_goal_if_unchanged
    admit = server._admit_prompt_turn
    monkeypatch.setattr(server, "_run_prompt_submit", env.real_submit)

    def start_worker(target, *, name, session, publish=None):
        thread = SimpleNamespace(is_alive=lambda: True)
        if publish is not None and not publish(thread):
            return None
        workers.append(name)
        session["_run_thread"] = thread
        return thread  # no model/network work: exercise the real final admission boundary

    monkeypatch.setattr(server, "_start_session_work", start_worker)

    def stop_then_successor():
        if injected:
            return
        injected.append(boundary)
        assert session["running"] is True
        assert not session.get("_run_thread")
        assert server._interrupt_session_turn("ui-parked", session, request_id="real-stop") is False
        assert session["_turn_cancel_requested"] is True
        assert session["_queued_prompt_generation"] == 1
        assert session["running"] is False
        if successor == "replacement":
            replacement = env.goals.GoalManager(env.manager.session_id)
            replacement.set("new goal must survive the cancelled wake")
            newer.append(replacement.state.to_json())
        if successor != "none":
            err, _ = server._lock_in_submit_turn(
                "new-user", "ui-parked", session, "explicit new user prompt", {}, False, None, None, None)
            assert err is None
            assert session["_turn_cancel_requested"] is False
            assert env.real_submit("new-user", "ui-parked", session, "explicit new user prompt") is True
            newer.append(session["inflight_turn"])

    def save_then_stop(*args, **kwargs):
        saved = save(*args, **kwargs)
        if saved and boundary == "persistence":
            stop_then_successor()
        return saved

    def admit_then_stop(*args, **kwargs):
        admitted = admit(*args, **kwargs)
        if admitted is not None and boundary == "admitted":
            stop_then_successor()
        return admitted

    monkeypatch.setattr(env.goals, "save_goal_if_unchanged", save_then_stop)
    monkeypatch.setattr(server, "_admit_prompt_turn", admit_then_stop)
    assert server._maybe_resume_tui_parked_goal("ui-parked", session) is True
    assert injected == [boundary], "real Stop injection did not fire"
    print({"boundary": boundary, "successor": successor, "stop_injected": injected,
           "workers": workers, "executable": sys.executable, "python": sys.version,
           "submit_source": env.real_submit.__code__.co_filename,
           "stop_source": server._interrupt_session_turn.__code__.co_filename})
    assert len(workers) == (0 if successor == "none" else 1), "cancelled parked wake launched a worker"
    assert session["running"] is (successor != "none"), "stale wake released a newer user turn's claim"
    if successor != "none":
        assert session["inflight_turn"] is newer[-1]
    assert env.goals.load_goal(env.manager.session_id).to_json() == (
        newer[0] if successor == "replacement" else before)
    assert not any(frame[0] == "status.update" and frame[2]["kind"] == "goal"
                   for frame in env.emitted)


def test_profile_scope_and_compression_tip_never_borrow_a_sibling_or_stale_goal(env, monkeypatch, tmp_path):
    import hermes_state
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import get_hermes_home, get_hermes_home_override

    # Use real call-time profile DB resolution, not conftest's single-store pin.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    env.goals._DB_CACHE.clear()
    set_multiplex_active(True)
    homes = [tmp_path / name for name in ("launch-a", "sibling-b")]
    key, stale = "compression-tip", "compression-parent"
    records, originals = [], []
    for home in homes:
        home.mkdir()
        record = {**env.session, "session_key": stale, "profile_home": str(home),
                  "agent": SimpleNamespace(session_id=key), "history_lock": threading.Lock(), "running": False}
        with env.server._session_profile_runtime_scope(record):
            mgr = env.goals.GoalManager(key)
            mgr.set(home.name)
            mgr.wait_for_seconds(100 if home == homes[1] else 1)
            if home == homes[0]:
                mgr.state.waiting_until = time.time() - 1
                mgr._save()
            old = env.goals.GoalManager(stale)
            old.set("stale parent must not run")
            old.wait_for_seconds(1)
            old.state.waiting_until = time.time() - 1
            old._save()
            originals.append((mgr.state.to_json(), old.state.to_json()))
        records.append(record)
    seen = []
    monkeypatch.setattr(env.server, "_run_prompt_submit",
                        lambda *a, **k: seen.append((get_hermes_home(), a[3])) or True)
    prior_scope = get_hermes_home_override()
    try:
        for record in (records[0], records[1], records[0]):
            record["running"] = False
            env.server._maybe_resume_tui_parked_goal("ui-profile", record)
            assert get_hermes_home_override() == prior_scope
        assert len(seen) == 1 and seen[0][0] == homes[0]
        assert homes[0].name in seen[0][1] and "stale parent" not in seen[0][1]
        for index, record in enumerate(records):
            with env.server._session_profile_runtime_scope(record):
                assert env.goals.load_goal(stale).to_json() == originals[index][1]
                current = env.goals.load_goal(key)
                if index == 0:
                    assert current.waiting_until == 0 and current.turns_used == 0
                else:
                    assert current.to_json() == originals[index][0]
    finally:
        set_multiplex_active(False)


def _real_worker_fixture(env, monkeypatch):
    """Keep production admission/publication/finally; replace only model/I/O work."""
    from contextlib import nullcontext
    from hermes_cli import plugins
    from tools import approval, async_delegation

    server, session = env.server, env.session
    monkeypatch.setattr(server, "_run_prompt_submit", env.real_submit)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda s: True)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_routing_provenance_db", lambda s: nullcontext(None))
    monkeypatch.setattr(server, "_record_turn_marker", lambda *a, **k: "")
    for name in ("_finish_turn", "_retire_turn_marker", "_emit_settled_session_info",
                 "_run_post_turn_followups", "_clear_pending"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **k: None)
    monkeypatch.setattr(async_delegation, "interrupt_for_session", lambda *a, **k: None)
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *a, **k: None)
    session["agent"].clear_interrupt = MagicMock()
    session["agent"].interrupt = MagicMock()
    entered, release = threading.Event(), threading.Event()

    def prepare(*a):
        entered.set()
        assert release.wait(3), "test did not release real worker"
        return None

    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    return entered, release


def test_hosted_cleanup_preserves_successor_rpc_claim_before_worker_admission(env, monkeypatch):
    """RPC claims precede worker owner tokens; old cleanup cannot erase their task."""
    from tui_gateway import hosted_room_member_activity

    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    old_task = {"task_id": "old-task", "room_id": "room"}
    new_task = {"task_id": "successor-task", "room_id": "room"}
    errors, activity = [], []
    worker = None

    def retire_marker(*a):
        cleanup_entered.set()
        if not cleanup_release.wait(3):
            errors.append("old final cleanup was not released")

    monkeypatch.setattr(server, "_retire_turn_marker", retire_marker)
    monkeypatch.setattr(hosted_room_member_activity, "emit_room_member_activity",
                        lambda task, **kw: activity.append((task, kw)) or True)
    try:
        err, _ = server._lock_in_submit_turn(
            "old", "ui-parked", session, "old hosted turn", {}, False, None, old_task, None)
        assert err is None
        assert env.real_submit("old", "ui-parked", session, "old hosted turn") is True
        worker = session["_run_thread"]
        assert entered.wait(3), "old production worker never entered"
        owner = session["_prompt_turn_owner"]
        release.set()
        assert cleanup_entered.wait(3), "old final cleanup injection did not fire"
        assert worker.is_alive() and session["running"] is False
        # Exercise the real RPC claim while B's deferred agent build has not yet
        # reached worker admission. Persistence/build/model I/O are outside this seam.
        err, _ = server._lock_in_submit_turn(
            "new", "ui-parked", session, "successor hosted turn", {}, False, None, new_task, None)
        assert err is None
        claimed_task = session["_hosted_room_task"]
        assert claimed_task == new_task
        assert session["_prompt_turn_owner"] is owner, "RPC claim unexpectedly refreshed worker ownership"
        cleanup_release.set()
        worker.join(3)
        assert not worker.is_alive() and not errors, errors
        assert session.get("_hosted_room_task") is claimed_task, "old cleanup erased successor task identity"
        assert session["running"] is True
        assert hosted_room_member_activity.project_room_member_activity({
            "method": "event", "params": {"type": "tool.start", "session_id": "ui-parked",
                                         "payload": {"name": "successor-tool"}}},
            {"ui-parked": session}) is True
        assert activity[0][0] is claimed_task
        assert activity[0][1]["kind"] == "tool.started"
    finally:
        release.set()
        cleanup_release.set()
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive(), "old production worker leaked"


def test_owned_hosted_completion_retires_task_before_idle(env, monkeypatch):
    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    worker = None
    try:
        err, _ = server._lock_in_submit_turn(
            "owned", "ui-parked", session, "owned hosted turn", {}, False, None,
            {"task_id": "owned-task"}, None)
        assert err is None
        assert env.real_submit("owned", "ui-parked", session, "owned hosted turn") is True
        worker = session["_run_thread"]
        assert entered.wait(3), "owned production worker never entered"
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert session["running"] is False
        assert "_hosted_room_task" not in session
        assert not session.get("inflight_turn")
    finally:
        release.set()
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive(), "owned production worker leaked"


def test_stop_samples_real_worker_published_before_cancellation_lock(env, monkeypatch):
    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    stop_boundary, allow_stop = threading.Event(), threading.Event()
    lock = threading.Lock()
    errors = []

    class PublicationBarrier:
        def __enter__(self):
            if threading.current_thread().name == "regression-stop":
                stop_boundary.set()
                assert allow_stop.wait(3), "worker publication was not released"
            lock.acquire()

        def __exit__(self, *a):
            lock.release()

    session["history_lock"] = PublicationBarrier()

    def stop():
        try:
            server._interrupt_session_turn("ui-parked", session)
        except BaseException as exc:
            errors.append(exc)

    stopper = threading.Thread(target=stop, name="regression-stop")
    stopper.start()
    worker = None
    try:
        assert stop_boundary.wait(3), "Stop never reached pre-lock barrier"
        assert server._maybe_resume_tui_parked_goal("ui-parked", session) is True
        worker = session["_run_thread"]
        assert entered.wait(3) and worker.is_alive(), "real publication/start did not fire"
        allow_stop.set()
        stopper.join(3)
        assert not stopper.is_alive() and not errors, errors
        assert session["_turn_cancel_requested"] is True
        assert session["_queued_prompt_generation"] == 1
        assert session["running"] is True, "Stop released a worker it sampled before publication"
        assert isinstance(session.get("inflight_turn"), dict)
    finally:
        allow_stop.set()
        release.set()
        stopper.join(3)
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive(), "real worker deadlocked against Stop"


def test_stale_worker_finally_preserves_real_explicit_successor(env, monkeypatch):
    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    assert server._maybe_resume_tui_parked_goal("ui-parked", session) is True
    old_worker = session["_run_thread"]
    assert entered.wait(3), "old production worker never entered"
    successor_entered, successor_release = threading.Event(), threading.Event()
    successor = []
    errors = []

    def successor_prepare(*a):
        successor_entered.set()
        assert successor_release.wait(3), "successor was not released"
        return None

    def finish_old(*a):
        if threading.current_thread() is not old_worker:
            return
        try:
            # Inject a later published owner at the old body's cleanup boundary.
            # Admission and worker launch remain production code, including Lock.
            with session["history_lock"]:
                session["running"] = False
                server._clear_inflight_turn(session)
            monkeypatch.setattr(server, "_prepare_turn_input", successor_prepare)
            error, _ = server._lock_in_submit_turn(
                "successor", "ui-parked", session, "explicit successor", {}, False, None, None, None)
            assert error is None
            assert env.real_submit("successor", "ui-parked", session, "explicit successor") is True
            successor.append((session["_run_thread"], session["inflight_turn"]))
            assert successor_entered.wait(3), "real successor publication did not fire"
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(server, "_finish_turn", finish_old)
    try:
        assert server._interrupt_session_turn("ui-parked", session) is False
        release.set()
        old_worker.join(3)
        assert not old_worker.is_alive() and not errors, errors
        assert len(successor) == 1, "successor injection did not fire"
        thread, inflight = successor[0]
        assert thread.is_alive()
        assert session["running"] is True, "old finally released the real explicit successor"
        assert session["inflight_turn"] is inflight
    finally:
        release.set()
        successor_release.set()
        old_worker.join(3)
        for thread, _ in successor:
            thread.join(3)
            assert not thread.is_alive(), "successor deadlocked during old cleanup"


def test_stop_preserves_published_worker_before_its_body_can_start(env, monkeypatch):
    _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    ran = []
    session["running"] = True

    def publish(worker):
        with session["history_lock"]:
            session["_run_thread"] = worker
            session["_run_thread_starting"] = worker
        assert not worker.is_alive() and not ran
        assert server._interrupt_session_turn("ui-parked", session) is False
        assert session["running"] is True, "Stop freed a published pending worker"
        assert not ran, "body ran before the publication boundary settled"
        return True

    worker = server._start_session_work(
        lambda: ran.append("body"), name="pending-publication-regression", session=session,
        publish=publish)
    assert worker is session["_run_thread"]
    worker.join(3)
    assert not worker.is_alive() and ran == ["body"]
    assert session.get("_run_thread_starting") is None


@pytest.mark.parametrize("settlement", ["retry", "rollback"])
def test_goal_publication_transaction_never_runs_a_body_before_settlement(env, monkeypatch, settlement):
    import sqlite3

    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    before = env.goals.load_goal(env.manager.session_id).to_json()
    db = env.goals._get_session_db()
    execute = db._execute_write
    transactions, attempts, handles = [], [], []

    def execute_with_settlement_fault(fn, *args, **kwargs):
        transactions.append(fn)
        if len(transactions) != 2:  # successful wait-clearing CAS precedes publication
            return execute(fn, *args, **kwargs)

        def fault(conn):
            result = fn(conn)
            attempts.append(result)
            assert not entered.is_set(), "worker body ran inside unsettled publication transaction"
            handles.append(session["_run_thread"])
            assert not handles[-1].is_alive()
            if settlement == "rollback":
                raise RuntimeError("injected publication rollback")
            if len(attempts) == 1:
                raise sqlite3.OperationalError("database is locked")
            return result

        return execute(fault, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", execute_with_settlement_fault)
    try:
        server._maybe_resume_tui_parked_goal("ui-parked", session)
        assert attempts, "publication settlement injection did not fire"
        if settlement == "rollback":
            assert attempts == [True] and not entered.is_set()
            assert not session.get("_run_thread") and not session.get("_run_thread_starting")
            assert session["running"] is False
            assert env.goals.load_goal(env.manager.session_id).to_json() == before
        else:
            assert attempts == [True, True] and handles[0] is handles[1]
            assert entered.wait(3) and handles[0].is_alive()
    finally:
        release.set()
        for worker in set(handles):
            if worker.ident is not None:
                worker.join(3)
                assert not worker.is_alive(), "settlement worker did not unwind"


@pytest.mark.parametrize("mutation", ["pause", "replace"])
@pytest.mark.parametrize("boundary", ["persistence", "admitted", "final-launch"])
def test_persisted_goal_edit_revokes_real_wake_without_stop(env, monkeypatch, mutation, boundary):
    from contextlib import contextmanager

    entered, release = _real_worker_fixture(env, monkeypatch)
    server, session = env.server, env.session
    edit_requested, edit_done = threading.Event(), threading.Event()
    edits, errors, launched, injections = [], [], [], []
    save, admit, start = env.goals.save_goal_if_unchanged, server._admit_prompt_turn, server._start_session_work

    def edit_goal():
        try:
            assert edit_requested.wait(3), "persisted edit boundary never fired"
            newer = env.goals.GoalManager(env.manager.session_id)
            assert newer.state.waiting_until == 0, "wake CAS did not actually succeed"
            if mutation == "pause":
                newer.pause(reason="user revoked the wake without Stop")
            else:
                newer.set("replacement goal owns the persisted authority")
            edits.append(env.goals.load_goal(env.manager.session_id).to_json())
        except BaseException as exc:
            errors.append(exc)
        finally:
            edit_done.set()

    editor = threading.Thread(target=edit_goal, name="persisted-goal-editor")
    editor.start()

    def inject():
        if injections:
            return
        injections.append(boundary)
        assert session["running"] is True
        assert not session.get("_run_thread"), "edit must precede actual worker publication"
        edit_requested.set()
        assert edit_done.wait(3), "persisted edit deadlocked before final launch"
        assert not errors, errors

    def save_then_edit(*args, **kwargs):
        saved = save(*args, **kwargs)
        if saved and boundary == "persistence":
            inject()
        return saved

    def admit_then_edit(*args, **kwargs):
        result = admit(*args, **kwargs)
        if result is not None and boundary == "admitted":
            inject()
        return result

    @contextmanager
    def routing_boundary(record):
        if boundary == "final-launch":
            inject()
        yield None

    def start_real(*args, **kwargs):
        worker = start(*args, **kwargs)
        if worker is not None:
            launched.append(worker)
        return worker

    monkeypatch.setattr(env.goals, "save_goal_if_unchanged", save_then_edit)
    monkeypatch.setattr(server, "_admit_prompt_turn", admit_then_edit)
    monkeypatch.setattr(server, "_routing_provenance_db", routing_boundary)
    monkeypatch.setattr(server, "_start_session_work", start_real)
    try:
        assert server._maybe_resume_tui_parked_goal("ui-parked", session) is True
        assert injections == [boundary] and len(edits) == 1, "persisted injection did not fire"
        assert not session.get("_turn_cancel_requested")
        assert int(session.get("_queued_prompt_generation", 0)) == 0, "test accidentally used Stop"
        assert env.goals.load_goal(env.manager.session_id).to_json() == edits[0]
        assert not launched, "revoked persisted goal launched a stale synthesized worker"
        assert session["running"] is False
        assert not any(frame[0] == "status.update" and frame[2]["kind"] == "goal"
                       for frame in env.emitted)
        # Pausing/replacing a goal revokes only its synthetic claim, not user authority.
        error, _ = server._lock_in_submit_turn(
            "explicit", "ui-parked", session, "legitimate explicit user", {}, False, None, None, None)
        assert error is None
        assert env.real_submit("explicit", "ui-parked", session, "legitimate explicit user") is True
        assert len(launched) == 1 and entered.wait(3) and launched[0].is_alive()
        assert env.goals.load_goal(env.manager.session_id).to_json() == edits[0]
    finally:
        edit_requested.set()
        release.set()
        editor.join(3)
        assert not editor.is_alive(), "persisted editor did not terminate"
        for worker in launched:
            worker.join(3)
            assert not worker.is_alive(), "real prompt worker did not terminate"
