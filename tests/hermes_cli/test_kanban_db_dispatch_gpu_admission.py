"""Dispatch-time GPU admission is fail-closed and leaves CPU dispatch alone."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_gpu_governor_adapter as adapter
from hermes_cli.resource_intent import canonicalize_resource_intent

GPU_UUID = "GPU-1A2B3C4D-0000-0000-0000-00000000ABCD"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as connection:
        yield connection


def _request(**overrides: Any) -> dict[str, Any]:
    request = {
        "resource_kind": "gpu",
        "schema_version": 1,
        "selector_type": "exact",
        "selector_value": GPU_UUID,
        "exclusive": True,
        "min_free_mib": 8192,
        "headroom_mib": 1024,
        "reservation_ttl_seconds": 600,
    }
    request.update(overrides)
    return request


def _gpu_card(conn, *, title="GPU work", priority=0, created_at=None) -> str:
    task_id = kb.create_task_with_resource(
        conn,
        title=title,
        assignee="worker",
        priority=priority,
        resource_request=_request(),
    )
    if created_at is not None:
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (created_at, task_id))
    return task_id


def _cpu_card(conn, *, title="CPU work", priority=0) -> str:
    return kb.create_task(
        conn, title=title, assignee="worker", priority=priority,
    )


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "kanban": {
            "gpu_governor": {
                "bin": str(tmp_path / "governor"),
                "ledger": str(tmp_path / "governor.sqlite3"),
                "lock_root": str(tmp_path),
                "admission_window_seconds": 2.0,
            }
        }
    }


def _success_output(
    tmp_path: Path,
    task_id: str,
    request: dict[str, Any],
    *,
    priority: int,
    owner: str = "worker",
) -> str:
    _, request_digest = canonicalize_resource_intent(request)
    request_id = adapter._request_id(task_id, 1)
    ledger = (tmp_path / "governor.sqlite3").resolve()
    lock_root = tmp_path.resolve()
    receipt = {
        "protocol_version": str(adapter.ADAPTER_PROTOCOL_VERSION),
        "request_hash": adapter._governor_request_hash(
            request_id=request_id,
            gpu_uuid=GPU_UUID,
            min_free_mib=request["min_free_mib"],
            headroom_mib=request["headroom_mib"],
            priority=priority,
            owner=owner,
        ),
        "request_id": request_id,
        "gpu_uuid": GPU_UUID,
        "gpu_index": 0,
        "lease_id": "lease-dispatch-test",
        "fencing_token": "fence-dispatch-test",
        "expires_at": time.time() + 1.0,
        "ledger_digest": hashlib.sha256(str(ledger).encode()).hexdigest(),
        "lock_root_digest": hashlib.sha256(str(lock_root).encode()).hexdigest(),
    }
    assert request_digest
    return json.dumps(receipt)


def _install_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter, "_load_config", lambda: _config(tmp_path))


def _dispatch(conn, spawn_fn, **kwargs):
    return kbd.dispatch_once(conn, spawn_fn=spawn_fn, **kwargs)


def _no_spawn(*args, **kwargs):
    raise AssertionError("a deferred task must not be spawned")


def test_valid_gpu_reservation_is_validated_before_claim(conn, monkeypatch, tmp_path):
    task_id = _gpu_card(conn, priority=6)
    _install_config(monkeypatch, tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=_success_output(tmp_path, task_id, _request(), priority=6),
            stderr="",
        )

    monkeypatch.setattr(adapter.subprocess, "run", runner)
    spawned = []
    result = _dispatch(conn, lambda task, workspace, board=None: spawned.append(task.id) or 31415)

    assert len(calls) == 1
    assert spawned == [task_id]
    assert [item[0] for item in result.spawned] == [task_id]
    assert result.deferred_resources == []
    claimed = kb.get_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"
    assert claimed.current_run_id is not None


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("busy", "governor_busy"),
        ("unknown_exit", "governor_exit_nonzero"),
        ("timeout", "governor_timeout"),
        ("malformed", "malformed_governor_output"),
    ],
)
def test_gpu_reservation_failures_defer_without_claim_or_failure(
    conn, monkeypatch, tmp_path, failure, expected_reason,
):
    task_id = _gpu_card(conn)
    _install_config(monkeypatch, tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "malformed":
            return SimpleNamespace(returncode=0, stdout="{not-json", stderr="")
        return SimpleNamespace(
            returncode=75 if failure == "busy" else 42,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(adapter.subprocess, "run", runner)
    result = _dispatch(conn, _no_spawn)

    assert len(calls) == 1
    assert result.spawned == []
    assert result.deferred_resources == [
        {"task_id": task_id, "status": "deferred_resources", "reason": expected_reason}
    ]
    deferred = kb.get_task(conn, task_id)
    assert deferred is not None
    assert deferred.status == "ready"
    assert deferred.claim_lock is None
    assert deferred.current_run_id is None
    assert deferred.consecutive_failures == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'claimed'", (task_id,)
    ).fetchone()[0] == 0


def test_only_oldest_highest_priority_gpu_card_attempts_once(conn, monkeypatch, tmp_path):
    older = _gpu_card(conn, title="older same priority", priority=10, created_at=100)
    newer = _gpu_card(conn, title="newer same priority", priority=10, created_at=200)
    lower_priority = _gpu_card(conn, title="lower priority but older", priority=3, created_at=50)
    _install_config(monkeypatch, tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=75, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", runner)
    result = _dispatch(conn, _no_spawn)

    assert len(calls) == 1
    assert adapter._request_id(older, 1) in calls[0]
    assert adapter._request_id(newer, 1) not in calls[0]
    assert adapter._request_id(lower_priority, 1) not in calls[0]
    assert result.deferred_resources[0] == {
        "task_id": older, "status": "deferred_resources", "reason": "governor_busy"
    }
    assert {item["task_id"] for item in result.deferred_resources[1:]} == {newer, lower_priority}
    assert all(
        item["reason"] == "reservation_attempt_limit_reached"
        for item in result.deferred_resources[1:]
    )
    remaining = [kb.get_task(conn, task_id) for task_id in (older, newer, lower_priority)]
    assert all(task is not None and task.status == "ready" for task in remaining)


def test_invalidated_oldest_gpu_request_allows_next_candidate_to_attempt(conn, monkeypatch, tmp_path):
    cpu_id = _cpu_card(conn, title="CPU first", priority=20)
    invalidated = _gpu_card(conn, title="request removed before admission", priority=10, created_at=100)
    next_gpu = _gpu_card(conn, title="next valid candidate", priority=9, created_at=200)
    _install_config(monkeypatch, tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=75, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", runner)

    def spawn(task, workspace, board=None):
        if task.id == cpu_id:
            conn.execute("DELETE FROM task_resource_requests WHERE card_id = ?", (invalidated,))
        return 27182

    result = _dispatch(conn, spawn)

    assert len(calls) == 1
    assert adapter._request_id(next_gpu, 1) in calls[0]
    assert result.spawned[0][0] == cpu_id
    assert {item["task_id"] for item in result.deferred_resources} == {invalidated, next_gpu}
    assert next(item for item in result.deferred_resources if item["task_id"] == invalidated)["reason"] == "resource_request_changed"
    assert next(item for item in result.deferred_resources if item["task_id"] == next_gpu)["reason"] == "governor_busy"
    invalidated_task = kb.get_task(conn, invalidated)
    next_task = kb.get_task(conn, next_gpu)
    assert invalidated_task is not None and invalidated_task.status == "ready"
    assert next_task is not None and next_task.status == "ready"


def test_gpu_reservation_is_deferred_when_dispatch_lock_cannot_be_resolved(conn, monkeypatch):
    task_id = _gpu_card(conn)

    def no_lock_path(**kwargs):
        raise RuntimeError("test lock-path failure")

    monkeypatch.setattr(kb, "kanban_db_path", no_lock_path)
    monkeypatch.setattr(
        adapter.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("reserve must stay under the dispatch lock"),
    )

    result = _dispatch(conn, _no_spawn)

    assert result.deferred_resources == [
        {"task_id": task_id, "status": "deferred_resources", "reason": "dispatch_lock_unavailable"}
    ]
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert task.claim_lock is None
    assert task.current_run_id is None


@pytest.mark.parametrize("governor_state", ["absent", "unreachable", "malformed"])
def test_cpu_card_dispatch_does_not_consult_gpu_governor(
    conn, monkeypatch, tmp_path, governor_state,
):
    task_id = _cpu_card(conn)
    if governor_state == "absent":
        monkeypatch.setattr(adapter, "_load_config", lambda: {})
    elif governor_state == "unreachable":
        monkeypatch.setattr(adapter, "_load_config", lambda: _config(tmp_path))
    else:
        monkeypatch.setattr(adapter, "_load_config", lambda: {"kanban": {"gpu_governor": []}})

    def unexpected_governor_call(*args, **kwargs):
        raise AssertionError("CPU dispatch must not consult the GPU governor")

    monkeypatch.setattr(adapter.subprocess, "run", unexpected_governor_call)
    spawned = []
    result = _dispatch(conn, lambda task, workspace, board=None: spawned.append(task.id) or 42424)

    assert spawned == [task_id]
    assert [item[0] for item in result.spawned] == [task_id]
    assert result.deferred_resources == []
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "running"
    assert task.current_run_id is not None
