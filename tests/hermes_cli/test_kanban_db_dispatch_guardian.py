"""GPU reservations are surfaced and production workers hand off to guardian-run."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_gpu_governor_adapter as adapter
from hermes_cli.kanban_guardian_launch import guardian_argv
from hermes_cli.resource_intent import canonicalize_resource_intent

GPU_UUID = "GPU-1A2B3C4D-0000-0000-0000-00000000ABCD"


@pytest.fixture
def dispatch_conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    with kbc.connect() as conn:
        yield conn


def _request() -> dict[str, Any]:
    return {
        "resource_kind": "gpu",
        "schema_version": 1,
        "selector_type": "exact",
        "selector_value": GPU_UUID,
        "exclusive": True,
        "min_free_mib": 8192,
        "headroom_mib": 1024,
        "reservation_ttl_seconds": 600,
    }


def _gpu_task(conn, *, status: str = "ready") -> str:
    task_id = kb.create_task_with_resource(
        conn,
        title="GPU task",
        assignee="worker",
        resource_request=_request(),
    )
    if status != "ready":
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    return task_id


def _receipt() -> dict[str, Any]:
    _, digest = canonicalize_resource_intent(_request())
    return {
        "lease_id": "lease-test",
        "fencing_token": "opaque-fence-test",
        "request_digest": digest,
        "gpu_selector": GPU_UUID,
        "expiry": int(time.time()) + 300,
        "canonical_receipt": json.dumps({
            "protocol_version": "1",
            "request_hash": "a" * 64,
        }),
    }


def test_guardian_argv_preserves_worker_args_and_uses_configured_paths(tmp_path: Path):
    governor = tmp_path / "gpu_governor.py"
    lock_root = tmp_path / "locks"
    argv = guardian_argv(
        _receipt(),
        ["hermes", "-p", "worker", "chat", "-q", "a value with spaces"],
        config={
            "kanban": {
                "gpu_governor": {
                    "bin": str(governor),
                    "python": sys.executable,
                    "lock_root": str(lock_root),
                }
            }
        },
    )
    assert argv[:6] == [
        str(Path(sys.executable).absolute()),
        str(governor.resolve()),
        "guardian-run",
        "lease-test",
        "opaque-fence-test",
        "--lock-root",
    ]
    assert argv[6:8] == [str(lock_root.resolve()), "--"]
    assert argv[8:] == ["hermes", "-p", "worker", "chat", "-q", "a value with spaces"]


def test_gpu_wait_events_coalesce_and_reservation_result_omits_fencing_token(
    dispatch_conn,
    monkeypatch: pytest.MonkeyPatch,
):
    task_id = _gpu_task(dispatch_conn)
    decisions = [
        SimpleNamespace(granted=False, reason="governor_busy"),
        SimpleNamespace(granted=False, reason="governor_busy"),
        SimpleNamespace(granted=False, reason="governor_timeout"),
        SimpleNamespace(granted=True, receipt=_receipt()),
    ]
    monkeypatch.setattr(
        adapter, "reserve_gpu_resource", lambda *args, **kwargs: decisions.pop(0)
    )
    spawn = lambda task, workspace, board=None: 421  # noqa: E731

    first = kbd.dispatch_once(dispatch_conn, spawn_fn=spawn, max_in_progress=10)
    second = kbd.dispatch_once(dispatch_conn, spawn_fn=spawn, max_in_progress=10)
    changed = kbd.dispatch_once(dispatch_conn, spawn_fn=spawn, max_in_progress=10)
    admitted = kbd.dispatch_once(dispatch_conn, spawn_fn=spawn, max_in_progress=10)

    kinds = [event.kind for event in kb.list_events(dispatch_conn, task_id)]
    assert kinds.count("resource_wait_started") == 1
    assert kinds.count("resource_reason_changed") == 1
    assert kinds.count("resource_reserved") == 1
    assert "heartbeat" not in kinds
    assert first.deferred_resources[0]["reason"] == "governor_busy"
    assert second.deferred_resources[0]["reason"] == "governor_busy"
    assert changed.deferred_resources[0]["reason"] == "governor_timeout"
    assert admitted.deferred_resources == []
    assert admitted.resource_reservations[0]["lease_id"] == "lease-test"
    assert "fencing_token" not in admitted.resource_reservations[0]
    assert kb.get_task(dispatch_conn, task_id).status == "running"


def test_review_gpu_card_uses_admission_path(
    monkeypatch: pytest.MonkeyPatch, dispatch_conn
):
    task_id = _gpu_task(dispatch_conn, status="review")
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    monkeypatch.setattr(
        adapter,
        "reserve_gpu_resource",
        lambda *args, **kwargs: SimpleNamespace(granted=True, receipt=_receipt()),
    )
    spawned: list[str] = []

    result = kbd.dispatch_once(
        dispatch_conn,
        spawn_fn=lambda task, workspace, board=None: spawned.append(task.id) or 422,
        max_spawn=1,
        max_in_progress=10,
    )

    assert spawned == [task_id]
    assert result.resource_reservations[0]["lease_id"] == "lease-test"
    assert kb.get_task(dispatch_conn, task_id).status == "running"
    assert any(
        event.kind == "resource_reserved"
        for event in kb.list_events(dispatch_conn, task_id)
    )


@pytest.mark.parametrize("failure_stage", ["claim", "workspace", "spawn"])
def test_pre_guardian_failures_cancel_reservation_once(
    failure_stage: str,
    dispatch_conn,
    monkeypatch: pytest.MonkeyPatch,
):
    task_id = _gpu_task(dispatch_conn)
    monkeypatch.setattr(
        adapter,
        "reserve_gpu_resource",
        lambda *args, **kwargs: SimpleNamespace(granted=True, receipt=_receipt()),
    )
    cancellations: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.kanban_guardian_launch.cancel_reservation",
        lambda reservation: cancellations.append(reservation["lease_id"]) or True,
    )
    if failure_stage == "claim":
        monkeypatch.setattr(kb, "claim_task", lambda *args, **kwargs: None)
    elif failure_stage == "workspace":
        monkeypatch.setattr(
            kbd._kbw,
            "resolve_workspace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("workspace")),
        )
        monkeypatch.setattr(
            kbd._kbw,
            "_resolve_worktree_workspace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("workspace")),
        )
    else:
        monkeypatch.setattr(
            kbd,
            "_default_spawn",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn")),
        )

    result = kbd.dispatch_once(dispatch_conn, max_in_progress=10)

    assert cancellations == ["lease-test"]
    assert result.resource_reservations[0]["status"] == "cancelled"
    if failure_stage == "claim":
        assert kb.get_task(dispatch_conn, task_id).status == "ready"


def test_cancel_reservation_uses_one_fenced_cli_call(tmp_path: Path):
    from hermes_cli.kanban_guardian_launch import cancel_reservation

    governor = tmp_path / "gpu_governor.py"
    ledger = tmp_path / "ledger.sqlite3"
    lock_root = tmp_path / "locks"
    captured: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv, **kwargs):
        captured.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "protocol_version": "1",
                "lease_id": "lease-test",
                "transition": "released",
            }),
        )

    config = {
        "kanban": {
            "gpu_governor": {
                "bin": str(governor),
                "python": sys.executable,
                "ledger": str(ledger),
                "lock_root": str(lock_root),
            }
        }
    }
    assert cancel_reservation(_receipt(), runner=runner, config=config)
    assert len(captured) == 1
    argv, kwargs = captured[0]
    assert argv == [
        str(Path(sys.executable).absolute()),
        str(governor.resolve()),
        "--db",
        str(ledger.resolve()),
        "cancel-reservation",
        "lease-test",
        "opaque-fence-test",
        "a" * 64,
        "--lock-root",
        str(lock_root.resolve()),
    ]
    assert kwargs["check"] is False
    assert "shell" not in kwargs


def test_default_spawn_wraps_only_reserved_worker_with_guardian(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_guardian_test",
        title="guardian worker",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=1,
        completed_at=None,
        workspace_kind="worktree",
        workspace_path=str(workspace),
        claim_lock="host:dispatcher",
        claim_expires=999,
        tenant=None,
        branch_name="wt/t_guardian_test",
        current_run_id=7,
    )
    child_argv = ["hermes", "-p", "worker", "chat", "-q", "original worker argv"]
    captured: dict[str, Any] = {}

    class FakeProc:
        pid = 423

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return FakeProc()

    governor = tmp_path / "gpu_governor"
    lock_root = tmp_path / "governor-locks"
    monkeypatch.setattr(
        adapter,
        "_load_config",
        lambda: {
            "kanban": {
                "gpu_governor": {
                    "bin": str(governor),
                    "ledger": str(tmp_path / "ledger.sqlite3"),
                    "lock_root": str(lock_root),
                }
            }
        },
    )
    monkeypatch.setattr(kbd, "_worker_argv", lambda *args: child_argv)
    monkeypatch.setattr(kbd, "_propagate_module_import_root", lambda *args: None)
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda task, argv: list(argv))
    monkeypatch.setattr(kbd, "_open_worker_log", lambda *args: io.BytesIO())
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    monkeypatch.setattr(
        "tools.environments.local.build_subprocess_env",
        lambda **kwargs: {"PATH": "/bin"},
    )
    monkeypatch.setattr(
        "tools.environments.local.strip_launch_profile_env", lambda *args: None
    )
    monkeypatch.setattr("tools.process_registry.systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda *args: None)

    assert (
        kbd._default_spawn(task, str(workspace), resource_reservation=_receipt()) == 423
    )
    argv = captured["argv"]
    separator = argv.index("--")
    assert argv[:separator] == [
        str(governor.resolve()),
        "guardian-run",
        "lease-test",
        "opaque-fence-test",
        "--lock-root",
        str(lock_root.resolve()),
    ]
    assert argv[separator + 1 :] == child_argv
    assert captured["kwargs"]["cwd"] == str(workspace)
    assert captured["kwargs"]["env"]["HERMES_KANBAN_TASK"] == task.id
    assert captured["kwargs"]["env"]["HERMES_KANBAN_RUN_ID"] == "7"

    captured.clear()
    assert kbd._default_spawn(task, str(workspace)) == 423
    assert "guardian-run" not in captured["argv"]
    assert captured["argv"] == child_argv
