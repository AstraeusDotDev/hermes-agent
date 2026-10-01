"""Tests for atomic Card + resource-request persistence.

The load-bearing property here is ATOMICITY: a Card that demanded a GPU
reservation must never exist without the reservation a dispatcher would rely on
to admit it, and a reservation must never exist for a Card the dispatcher cannot
see. Every atomicity test therefore asserts on BOTH tables after a forced
failure — checking only one would pass while the other leaked.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db import (
    canonical_resource_request,
    create_task_with_resource,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors test_kanban_db)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    """A REAL board connection.

    Deliberately not a raw ``SCHEMA_SQL`` in-memory DB: that schema predates
    additive columns such as ``completion_contract``, which ``create_task``
    writes, so a raw fixture fails for reasons unrelated to this feature. Using
    ``kbc.connect()`` exercises the production schema plus migrations.
    """
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


def _counts(conn: sqlite3.Connection) -> tuple[int, int]:
    tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    res = conn.execute("SELECT COUNT(*) FROM task_resource_requests").fetchone()[0]
    return tasks, res


class TestCanonicalisation:
    def test_sorted_keys_and_compact_separators(self) -> None:
        payload, _ = canonical_resource_request({"b": 1, "a": 2, "c": [3, 4]})
        assert payload == '{"a":2,"b":1,"c":[3,4]}'
        assert " " not in payload

    def test_key_order_does_not_change_the_digest(self) -> None:
        _, d1 = canonical_resource_request({"a": 1, "b": 2})
        _, d2 = canonical_resource_request({"b": 2, "a": 1})
        assert d1 == d2

    def test_digest_is_sha256_of_the_canonical_text(self) -> None:
        payload, digest = canonical_resource_request({"a": 1, "b": 2})
        assert digest == hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert len(digest) == 64

    def test_value_change_changes_the_digest(self) -> None:
        _, d1 = canonical_resource_request({"selector_value": "GPU-a"})
        _, d2 = canonical_resource_request({"selector_value": "GPU-b"})
        assert d1 != d2


class TestPersistence:
    def test_creates_card_and_resource_row(self, conn: sqlite3.Connection) -> None:
        task_id = create_task_with_resource(
            conn, resource_request=_request(), title="Train", assignee="wiener"
        )
        row = conn.execute(
            "SELECT * FROM task_resource_requests WHERE card_id = ?", (task_id,)
        ).fetchone()
        assert row is not None
        assert row["card_id"] == task_id
        assert row["selector_type"] == "exact"
        assert row["selector_value"] == "GPU-abc123"
        assert row["reservation_ttl_seconds"] == 600

    def test_stored_json_is_canonical_and_hash_matches_it(self, conn: sqlite3.Connection) -> None:
        request = _request()
        task_id = create_task_with_resource(
            conn, resource_request=request, title="Train", assignee="wiener"
        )
        row = conn.execute(
            "SELECT request_json, request_sha256 FROM task_resource_requests WHERE card_id = ?",
            (task_id,),
        ).fetchone()
        # canonical: sorted, compact, and re-parsing yields the caller's dict
        assert row["request_json"] == json.dumps(
            request, sort_keys=True, separators=(",", ":")
        )
        assert row["request_sha256"] == hashlib.sha256(
            row["request_json"].encode("utf-8")
        ).hexdigest()

    def test_defaults_are_applied(self, conn: sqlite3.Connection) -> None:
        task_id = create_task_with_resource(
            conn, resource_request=_request(), title="Train", assignee="wiener"
        )
        row = conn.execute(
            "SELECT * FROM task_resource_requests WHERE card_id = ?", (task_id,)
        ).fetchone()
        assert row["resource_kind"] == "gpu"
        assert row["schema_version"] == 1
        assert row["exclusive"] == 1
        assert row["min_free_mib"] == 0
        assert row["headroom_mib"] == 0

    def test_optional_fields_are_honoured(self, conn: sqlite3.Connection) -> None:
        task_id = create_task_with_resource(
            conn,
            resource_request=_request(
                resource_kind="npu", exclusive=0, min_free_mib=8192, headroom_mib=512
            ),
            title="Train",
            assignee="wiener",
        )
        row = conn.execute(
            "SELECT * FROM task_resource_requests WHERE card_id = ?", (task_id,)
        ).fetchone()
        assert (row["resource_kind"], row["exclusive"]) == ("npu", 0)
        assert (row["min_free_mib"], row["headroom_mib"]) == (8192, 512)

    def test_no_resource_request_writes_no_resource_row(self, conn: sqlite3.Connection) -> None:
        """Intent is never inferred. A plain card stays plain."""
        create_task_with_resource(conn, title="Plain", assignee="wiener")
        assert _counts(conn) == (1, 0)

    def test_matches_create_task_when_no_request(self, conn: sqlite3.Connection) -> None:
        task_id = create_task_with_resource(conn, title="Plain", assignee="wiener")
        assert conn.execute(
            "SELECT title FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()["title"] == "Plain"

    def test_forwards_task_kwargs(self, conn: sqlite3.Connection) -> None:
        task_id = create_task_with_resource(
            conn,
            resource_request=_request(),
            title="Train",
            body="body text",
            assignee="wiener",
            priority=42,
        )
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "Train"
        assert row["body"] == "body text"
        assert row["assignee"] == "wiener"
        assert row["priority"] == 42


class TestValidation:
    @pytest.mark.parametrize("missing", ["selector_type", "selector_value", "reservation_ttl_seconds"])
    def test_missing_required_key_raises_before_any_write(
        self, conn: sqlite3.Connection, missing: str
    ) -> None:
        request = _request()
        del request[missing]
        with pytest.raises(ValueError, match=missing):
            create_task_with_resource(
                conn, resource_request=request, title="Train", assignee="wiener"
            )
        assert _counts(conn) == (0, 0), "validation must precede any write"

    @pytest.mark.parametrize("ttl", [0, -1, "600", 1.5])
    def test_bad_ttl_raises_before_any_write(
        self, conn: sqlite3.Connection, ttl: object
    ) -> None:
        with pytest.raises(ValueError, match="reservation_ttl_seconds"):
            create_task_with_resource(
                conn,
                resource_request=_request(reservation_ttl_seconds=ttl),
                title="Train",
                assignee="wiener",
            )
        assert _counts(conn) == (0, 0)


class TestAtomicity:
    def test_resource_failure_rolls_back_the_card_too(
        self, conn: sqlite3.Connection
    ) -> None:
        """The core guarantee: no orphaned Card when the resource write fails.

        The failure is made reachable by dropping the table's own FK parent
        reference for this connection: with ``task_resource_requests`` renamed out
        from under the INSERT, the resource write raises while the Card insert has
        already succeeded inside the same transaction. That is exactly the
        interleaving that would orphan a Card if the two writes were not atomic.
        """
        before = _counts(conn)
        conn.execute("ALTER TABLE task_resource_requests RENAME TO _rr_hidden")
        try:
            with pytest.raises(sqlite3.Error):
                create_task_with_resource(
                    conn, resource_request=_request(), title="Orphan", assignee="wiener"
                )
        finally:
            conn.execute("ALTER TABLE _rr_hidden RENAME TO task_resource_requests")
        after = _counts(conn)
        assert after == before, f"orphaned Card survived rollback: {before} -> {after}"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'Orphan'"
        ).fetchone()[0] == 0

    def test_card_creation_failure_writes_no_resource_row(
        self, conn: sqlite3.Connection
    ) -> None:
        """The mirror case: a bad Card must not leave a reservation behind."""
        with pytest.raises(ValueError, match="title is required"):
            create_task_with_resource(
                conn, resource_request=_request(), title="  ", assignee="wiener"
            )
        assert _counts(conn) == (0, 0)

    def test_failure_inside_transaction_leaves_both_tables_empty(
        self, conn: sqlite3.Connection
    ) -> None:
        """A non-IntegrityError escaping the resource write also rolls back."""
        original = conn.execute

        class Boom(Exception):
            pass

        def exploding(sql: str, *args):
            if "task_resource_requests" in sql:
                raise Boom("injected")
            return original(sql, *args)

        conn.execute = exploding  # type: ignore[method-assign]
        with pytest.raises(Boom):
            create_task_with_resource(
                conn, resource_request=_request(), title="Train", assignee="wiener"
            )
        conn.execute = original  # type: ignore[method-assign]
        assert _counts(conn) == (0, 0)

    def test_successful_write_commits_both(self, conn: sqlite3.Connection) -> None:
        create_task_with_resource(
            conn, resource_request=_request(), title="Train", assignee="wiener"
        )
        assert _counts(conn) == (1, 1)

    def test_composes_under_an_outer_transaction(self, conn: sqlite3.Connection) -> None:
        """write_txn nesting: composition must not raise 'already in a transaction'."""
        from hermes_cli.kanban_db_connect import write_txn

        with write_txn(conn, allow_nested=True):
            create_task_with_resource(
                conn, resource_request=_request(), title="Nested", assignee="wiener"
            )
        assert _counts(conn) == (1, 1)