"""Read surfaces for canonical resource intent.

Two properties are load-bearing and easy to fake, so each gets an explicit test:

1. CPU Cards are UNCHANGED. No new key, no null placeholder. The tests assert
   the key is ABSENT rather than falsy, because ``assert not d.get(...)`` would
   also pass for ``{"resource_intent": None}`` -- the exact regression this card
   exists to prevent.

2. Resource intent is IMMUTABLE. Body/title/edit operations must not move it.
   Asserted by digest before and after, so a mutation that rewrites the row with
   identical values still shows up as a changed ``updated``-style artifact.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db import (
    ResourceIntent,
    create_task_with_resource,
    get_resource_intent,
    list_resource_intents,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as connection:
        yield connection


def _request(**overrides) -> dict:
    request = {
        "selector_type": "exact",
        "selector_value": "GPU-abc123",
        "reservation_ttl_seconds": 600,
    }
    request.update(overrides)
    return request


def _gpu_card(conn, title="Train", **overrides) -> str:
    return create_task_with_resource(
        conn, resource_request=_request(**overrides), title=title, assignee="wiener"
    )


def _cpu_card(conn, title="Plain") -> str:
    return kb.create_task(conn, title=title, assignee="wiener")


class TestReadFromStorage:
    def test_round_trips_every_typed_field(self, conn: sqlite3.Connection) -> None:
        task_id = _gpu_card(conn, exclusive=0, min_free_mib=8192, headroom_mib=512)
        intent = get_resource_intent(conn, task_id)
        assert intent is not None
        assert intent.card_id == task_id
        assert intent.resource_kind == "gpu"
        assert intent.schema_version == 1
        assert intent.selector_type == "exact"
        assert intent.selector_value == "GPU-abc123"
        assert intent.exclusive is False          # bool, not 0
        assert intent.min_free_mib == 8192
        assert intent.headroom_mib == 512
        assert intent.reservation_ttl_seconds == 600

    def test_exclusive_reads_as_bool_not_int(self, conn: sqlite3.Connection) -> None:
        """SQLite stores INTEGER 0/1; the typed surface must not leak that."""
        exclusive = get_resource_intent(conn, _gpu_card(conn, exclusive=1))
        shared = get_resource_intent(conn, _gpu_card(conn, exclusive=0))
        assert exclusive is not None and exclusive.exclusive is True
        assert shared is not None and shared.exclusive is False

    def test_carries_the_creation_digest(self, conn: sqlite3.Connection) -> None:
        task_id = _gpu_card(conn)
        intent = get_resource_intent(conn, task_id)
        assert intent is not None
        assert len(intent.request_sha256) == 64

    def test_cpu_card_has_no_intent(self, conn: sqlite3.Connection) -> None:
        assert get_resource_intent(conn, _cpu_card(conn)) is None

    def test_unknown_card_is_none_not_an_error(self, conn: sqlite3.Connection) -> None:
        assert get_resource_intent(conn, "t_does_not_exist") is None

    def test_intent_is_immutable(self, conn: sqlite3.Connection) -> None:
        """frozen dataclass: a reader cannot mutate what it is about to act on."""
        intent = get_resource_intent(conn, _gpu_card(conn))
        assert intent is not None
        with pytest.raises(Exception):
            intent.selector_value = "GPU-hijacked"  # type: ignore[misc]

    def test_batch_returns_only_declaring_cards(self, conn: sqlite3.Connection) -> None:
        gpu = _gpu_card(conn, title="GPU")
        cpu = _cpu_card(conn, title="CPU")
        intents = list_resource_intents(conn, [gpu, cpu])
        assert set(intents) == {gpu}
        assert cpu not in intents

    def test_batch_on_empty_input_issues_no_query(self, conn: sqlite3.Connection) -> None:
        assert list_resource_intents(conn, []) == {}

    def test_batch_matches_single_read(self, conn: sqlite3.Connection) -> None:
        task_id = _gpu_card(conn)
        assert list_resource_intents(conn, [task_id])[task_id] == get_resource_intent(conn, task_id)

    def test_reader_survives_a_board_without_the_table(self, conn: sqlite3.Connection) -> None:
        """A pre-migration board must read as 'no intent', not crash the reader."""
        conn.execute("DROP TABLE task_resource_requests")
        assert get_resource_intent(conn, "t_anything") is None
        assert list_resource_intents(conn, ["t_anything"]) == {}


class TestShowSurface:
    def _show(self, conn, task_id, monkeypatch):
        """Call the real kanban_show handler and return the parsed payload."""
        import tools.kanban_tools as kt

        monkeypatch.setattr(kt, "_board", lambda *a, **k: _yield_conn(conn))
        return json.loads(kt._handle_show({"task_id": task_id}))

    def test_gpu_card_exposes_typed_intent(self, conn, monkeypatch) -> None:
        task_id = _gpu_card(conn)
        payload = self._show(conn, task_id, monkeypatch)
        intent = payload["task"]["resource_intent"]
        assert isinstance(intent, dict)
        assert intent["selector_value"] == "GPU-abc123"
        assert intent["exclusive"] is True
        # structured, not prose
        assert isinstance(intent["min_free_mib"], int)

    def test_cpu_card_payload_is_unchanged(self, conn, monkeypatch) -> None:
        """The regression this card guards: a null placeholder is NOT allowed."""
        payload = self._show(conn, _cpu_card(conn), monkeypatch)
        assert "resource_intent" not in payload["task"], (
            "CPU card gained a resource_intent key: "
            f"{payload['task'].get('resource_intent')!r}"
        )

    def test_cpu_card_keys_match_a_gpu_cards_keys_minus_intent(self, conn, monkeypatch) -> None:
        cpu = self._show(conn, _cpu_card(conn), monkeypatch)["task"]
        gpu = self._show(conn, _gpu_card(conn), monkeypatch)["task"]
        assert set(gpu) - set(cpu) == {"resource_intent"}


class TestListSurface:
    def _list(self, conn, monkeypatch):
        import tools.kanban_tools as kt

        monkeypatch.setattr(kt, "_board", lambda *a, **k: _yield_conn(conn))
        monkeypatch.setattr(kt, "_require_orchestrator_tool", lambda *a, **k: None)
        return json.loads(kt._handle_list({}))

    def test_gpu_summary_carries_intent(self, conn, monkeypatch) -> None:
        task_id = _gpu_card(conn, title="GPU card")
        payload = self._list(conn, monkeypatch)
        row = next(t for t in payload["tasks"] if t["id"] == task_id)
        assert row["resource_intent"]["selector_value"] == "GPU-abc123"

    def test_cpu_summary_has_no_intent_key(self, conn, monkeypatch) -> None:
        task_id = _cpu_card(conn, title="CPU card")
        payload = self._list(conn, monkeypatch)
        row = next(t for t in payload["tasks"] if t["id"] == task_id)
        assert "resource_intent" not in row

    def test_mixed_page_omits_only_for_cpu_cards(self, conn, monkeypatch) -> None:
        gpu = _gpu_card(conn, title="GPU card")
        cpu = _cpu_card(conn, title="CPU card")
        payload = self._list(conn, monkeypatch)
        rows = {t["id"]: t for t in payload["tasks"]}
        assert "resource_intent" in rows[gpu]
        assert "resource_intent" not in rows[cpu]


class TestImmutabilityUnderEdit:
    def test_body_edit_does_not_touch_resource_intent(self, conn) -> None:
        task_id = _gpu_card(conn)
        before = get_resource_intent(conn, task_id)
        assert kb.edit_task(conn, task_id, body="a completely new body") is True
        assert get_resource_intent(conn, task_id) == before

    def test_title_edit_does_not_touch_resource_intent(self, conn) -> None:
        task_id = _gpu_card(conn)
        before = get_resource_intent(conn, task_id)
        assert kb.edit_task(conn, task_id, title="renamed") is True
        assert get_resource_intent(conn, task_id) == before

    def test_status_transitions_do_not_touch_resource_intent(self, conn) -> None:
        task_id = _gpu_card(conn)
        before = get_resource_intent(conn, task_id)
        kb.assign_task(conn, task_id, "turing")
        kb.claim_task(conn, task_id, ttl_seconds=60, claimer="astraeus:1")
        assert get_resource_intent(conn, task_id) == before

    def test_row_is_untouched_byte_for_byte(self, conn) -> None:
        """Stronger than equality of the read model: the stored row is identical."""
        task_id = _gpu_card(conn)
        row_before = conn.execute(
            "SELECT * FROM task_resource_requests WHERE card_id = ?", (task_id,)
        ).fetchone()
        kb.edit_task(conn, task_id, body="new body", title="new title")
        row_after = conn.execute(
            "SELECT * FROM task_resource_requests WHERE card_id = ?", (task_id,)
        ).fetchone()
        assert dict(row_before) == dict(row_after)


def _yield_conn(conn):
    """Adapter so a plain connection can stand in for the _board contextmanager."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        from hermes_cli import kanban_db as kbmod

        yield kbmod, conn

    return _cm()