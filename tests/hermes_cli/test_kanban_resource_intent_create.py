"""Typed GPU resource intent writes through the internal, CLI, and tool paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.resource_intent import canonicalize_resource_intent


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as connection:
        yield connection


def _intent(**overrides):
    return {
        "resource_kind": "gpu",
        "schema_version": 1,
        "selector_type": "exact",
        "selector_value": "GPU-abc12345-1234-1234-1234-123456789abc",
        "exclusive": True,
        "min_free_mib": 0,
        "headroom_mib": 0,
        "reservation_ttl_seconds": 600,
        **overrides,
    }


def _resource_row(conn, task_id):
    return conn.execute(
        "SELECT request_json, request_sha256 FROM task_resource_requests WHERE card_id = ?",
        (task_id,),
    ).fetchone()


def _cli_args(argv):
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    return root.parse_args(["kanban", "create", *argv])


def _invoke_cli_create(title, intent, key="resource-key"):
    args = _cli_args([
        title,
        "--assignee",
        "worker",
        "--idempotency-key",
        key,
        "--resources-json",
        json.dumps(intent),
        "--json",
    ])
    return kc.kanban_command(args)


def test_create_task_validates_before_writing_and_persists_canonical_intent(conn):
    request = _intent(selector_value="GPU-abc12345-1234-1234-1234-123456789abc")
    task_id = kb.create_task(
        conn,
        title="GPU task",
        assignee="worker",
        resource_intent=request,
        idempotency_key="internal-resource",
    )
    canonical, digest = canonicalize_resource_intent(request)
    row = _resource_row(conn, task_id)
    assert row["request_json"] == canonical
    assert row["request_sha256"] == digest

    with pytest.raises(ValueError, match="missing resource-intent keys"):
        kb.create_task(
            conn,
            title="bad GPU task",
            resource_intent={"resource_kind": "gpu"},
        )
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert (
        conn.execute("SELECT COUNT(*) FROM task_resource_requests").fetchone()[0] == 1
    )


def test_create_task_idempotency_matches_bytes_or_fails_closed(conn):
    task_id = kb.create_task(
        conn,
        title="first",
        assignee="worker",
        resource_intent=_intent(),
        idempotency_key="same-card",
    )
    before_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    same_id = kb.create_task(
        conn,
        title="retry title is ignored",
        assignee="worker",
        resource_intent=_intent(
            selector_value="GPU-abc12345-1234-1234-1234-123456789abc"
        ),
        idempotency_key="same-card",
    )
    assert same_id == task_id
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        == before_events
    )

    before = dict(_resource_row(conn, task_id))
    with pytest.raises(ValueError, match="different resource intent"):
        kb.create_task(
            conn,
            title="must not replace",
            assignee="other",
            resource_intent=_intent(min_free_mib=1),
            idempotency_key="same-card",
        )
    assert dict(_resource_row(conn, task_id)) == before
    task = kb.get_task(conn, task_id)
    assert task is not None and task.title == "first"
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    with pytest.raises(ValueError, match="different resource intent"):
        kb.create_task(conn, title="missing intent", idempotency_key="same-card")
    unchanged = kb.get_task(conn, task_id)
    assert unchanged is not None and unchanged.title == "first"


def test_existing_card_without_intent_rejects_intent_retry(conn):
    task_id = kb.create_task(conn, title="plain", idempotency_key="plain-card")
    with pytest.raises(ValueError, match="different resource intent"):
        kb.create_task(
            conn,
            title="GPU",
            resource_intent=_intent(),
            idempotency_key="plain-card",
        )
    task = kb.get_task(conn, task_id)
    assert task is not None and task.title == "plain"
    assert (
        conn.execute("SELECT COUNT(*) FROM task_resource_requests").fetchone()[0] == 0
    )


def test_cli_resources_json_writes_and_idempotently_reuses_card(kanban_home, capsys):
    request = _intent()
    assert _invoke_cli_create("CLI GPU task", request) == 0
    first = json.loads(capsys.readouterr().out)
    task_id = first["id"]
    with kbc.connect() as connection:
        before = dict(_resource_row(connection, task_id))
        event_count = connection.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

    # The parser consumes one argv value and the engine normalizes the UUID.
    retry = _intent(selector_value="GPU-" + request["selector_value"][4:].lower())
    assert _invoke_cli_create("different title", retry) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["id"] == task_id

    with kbc.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert dict(_resource_row(connection, task_id)) == before
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
            ).fetchone()[0]
            == event_count
        )

    assert _invoke_cli_create("divergent", _intent(min_free_mib=8)) == 1
    capsys.readouterr()
    with kbc.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert dict(_resource_row(connection, task_id)) == before
        task = kb.get_task(connection, task_id)
        assert task is not None and task.title == "CLI GPU task"


def test_cli_rejects_bad_or_multiple_json_values_before_database_creation(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for raw in (
        '{"resource_kind":"gpu"}',
        json.dumps(_intent()) + " {}",
        json.dumps(_intent()).replace(
            '"resource_kind": "gpu"', '"resource_kind": "gpu", "resource_kind": "gpu"'
        ),
    ):
        with pytest.raises(SystemExit):
            _cli_args(["invalid", "--resources-json", raw])
    assert not list(home.rglob("*.db"))


def test_kanban_create_tool_uses_typed_intent_and_rejects_before_connect(
    kanban_home,
    monkeypatch,
):
    from tools import kanban_tools  # register the production handler
    from tools.registry import registry
    from hermes_cli import kanban_db_notify as kbn

    original_connect = kbc.connect

    with kbc.connect() as connection:
        # Ensure the database begins empty after fixture initialization.
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "first-chat")

    def should_not_connect(*args, **kwargs):
        raise AssertionError("invalid resource intent reached the DB connection")

    monkeypatch.setattr(kbc, "connect", should_not_connect)
    invalid = registry.dispatch(
        "kanban_create",
        {
            "title": "invalid",
            "assignee": "worker",
            "resource_intent": {"resource_kind": "gpu"},
        },
    )
    assert "missing resource-intent keys" in json.loads(cast(str, invalid))["error"]

    monkeypatch.setattr(kbc, "connect", original_connect)
    request = _intent()
    result = json.loads(
        cast(
            str,
            registry.dispatch(
                "kanban_create",
                {
                    "title": "tool GPU task",
                    "assignee": "worker",
                    "resource_intent": request,
                    "idempotency_key": "tool-resource",
                },
            ),
        )
    )
    assert result["ok"] is True
    task_id = result["task_id"]
    assert result["subscribed"] is True
    with kbc.connect() as connection:
        canonical, digest = canonicalize_resource_intent(request)
        row = _resource_row(connection, task_id)
        assert (row["request_json"], row["request_sha256"]) == (canonical, digest)
        before_events = connection.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "retry-chat")
    retry = json.loads(
        cast(
            str,
            registry.dispatch(
                "kanban_create",
                {
                    "title": "duplicate",
                    "assignee": "other",
                    "resource_intent": _intent(
                        selector_value="GPU-" + request["selector_value"][4:].lower()
                    ),
                    "idempotency_key": "tool-resource",
                },
            ),
        )
    )
    assert retry["ok"] is True and retry["task_id"] == task_id
    assert retry["subscribed"] is False
    divergent = json.loads(
        cast(
            str,
            registry.dispatch(
                "kanban_create",
                {
                    "title": "must not mutate",
                    "assignee": "other",
                    "resource_intent": _intent(min_free_mib=16),
                    "idempotency_key": "tool-resource",
                },
            ),
        )
    )
    assert "different resource intent" in divergent["error"]
    with kbc.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        task = kb.get_task(connection, task_id)
        assert task is not None and task.title == "tool GPU task"
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
            ).fetchone()[0]
            == before_events
        )
        assert len(kbn.list_notify_subs(connection, task_id)) == 1


def test_kanban_create_schema_exposes_the_validated_intent_shape():
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA

    intent_schema = KANBAN_CREATE_SCHEMA["parameters"]["properties"]["resource_intent"]
    assert set(intent_schema["required"]) == set(_intent())
    assert set(intent_schema["properties"]) == set(_intent())
    assert intent_schema["additionalProperties"] is False
    assert intent_schema["properties"]["resource_kind"]["enum"] == ["gpu"]
    assert intent_schema["properties"]["selector_type"]["enum"] == ["auto", "exact"]
    assert intent_schema["properties"]["reservation_ttl_seconds"]["maximum"] == 86400
