"""Schema tests for ``task_resource_requests``.

Covers the migration contract: it must run on a fresh board, be safe to re-run,
carry the specified columns with their defaults, and actually ENFORCE its
constraints. The enforcement cases are behavioural -- a schema that declares a
constraint it does not enforce looks identical to one that does, so every
constraint here is proved by an attempted violation.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli.kanban_db import SCHEMA_SQL


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """A fresh in-memory board with the real schema and FK enforcement on.

    ``kanban_db_connect`` sets ``PRAGMA foreign_keys=ON`` for real boards, so a
    test that leaves it off would silently stop proving the FK constraint.
    """
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA_SQL)
    for card_id in ("t_card_a", "t_card_b"):
        connection.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, 'todo', 1)",
            (card_id, card_id),
        )
    connection.commit()
    return connection


def _insert(conn: sqlite3.Connection, card_id: str, kind: str = "gpu", sha: str = "deadbeef") -> None:
    conn.execute(
        """
        INSERT INTO task_resource_requests
            (card_id, resource_kind, schema_version, selector_type, selector_value,
             request_json, request_sha256, reservation_ttl_seconds, exclusive,
             min_free_mib, headroom_mib, created_at)
        VALUES (?, ?, 1, 'exact', ?, '{}', ?, 600, 1, 0, 0, 1)
        """,
        (card_id, kind, f"GPU-{card_id}", sha),
    )


class TestMigration:
    def test_runs_on_a_fresh_database(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)  # must not raise

    @pytest.mark.parametrize("repeat", [2, 3])
    def test_is_idempotent_when_re_run(self, repeat: int) -> None:
        """Re-running the migration must be a no-op, not an error.

        Every statement is ``IF NOT EXISTS``; if one ever stops being, a board
        that reconnects on every dispatch would fail to open.
        """
        connection = sqlite3.connect(":memory:")
        for _ in range(repeat):
            connection.executescript(SCHEMA_SQL)
            connection.commit()

    def test_does_not_disturb_existing_rows(self) -> None:
        """Additive only: re-running must not drop or recreate data."""
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES ('t_keep', 'Keep', 'todo', 1)"
        )
        connection.commit()
        connection.executescript(SCHEMA_SQL)
        connection.commit()
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


class TestColumns:
    #: column -> (declared type, notnull, default)
    EXPECTED = {
        "id": ("INTEGER", 0, None),
        "card_id": ("TEXT", 1, None),
        "resource_kind": ("TEXT", 1, "'gpu'"),
        "schema_version": ("INTEGER", 1, "1"),
        "selector_type": ("TEXT", 1, None),
        "selector_value": ("TEXT", 1, None),
        "exclusive": ("INTEGER", 1, "1"),
        "min_free_mib": ("INTEGER", 1, "0"),
        "headroom_mib": ("INTEGER", 1, "0"),
        "reservation_ttl_seconds": ("INTEGER", 1, None),
        "request_json": ("TEXT", 1, None),
        "request_sha256": ("TEXT", 1, None),
        "created_at": ("INTEGER", 1, None),
    }

    def test_every_specified_column_is_present(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)
        names = {
            row[1] for row in connection.execute("PRAGMA table_info(task_resource_requests)")
        }
        assert set(self.EXPECTED) <= names

    def test_types_and_notnull(self) -> None:
        """Each column's declared type and NOT NULL flag match the contract.

        ``pragma_table_info`` returns ``(type, notnull)``; ``notnull`` is an int
        flag, ``type`` is a string. Asserting ``(type, notnull) == (notnull,
        notnull)`` would compare a string to an int and fail for the wrong
        reason, so build the expectation per-field rather than reusing one value.
        """
        expected_types = {name: kind for name, (kind, _n, _d) in self.EXPECTED.items()}
        expected_notnull = {name: bool(n) for name, (_t, n, _d) in self.EXPECTED.items()}

        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)
        actual = {
            row[1]: (row[2], bool(row[3]))
            for row in connection.execute("PRAGMA table_info(task_resource_requests)")
        }
        for name in expected_types:
            assert actual[name] == (expected_types[name], expected_notnull[name]), (
                f"{name}: type/notnull mismatch"
            )

    def test_declared_defaults(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)
        actual = {
            row[1]: row[4]
            for row in connection.execute("PRAGMA table_info(task_resource_requests)")
        }
        assert actual["resource_kind"] == "'gpu'"
        assert actual["schema_version"] == "1"
        assert actual["exclusive"] == "1"
        assert actual["min_free_mib"] == "0"
        assert actual["headroom_mib"] == "0"

    def test_card_id_is_text_to_match_tasks_id(self) -> None:
        """``tasks.id`` is ``TEXT PRIMARY KEY``.

        An INTEGER ``card_id`` would still *create* fine but could never satisfy
        the foreign key against a real card id like ``t_b435e9bb`` -- a migration
        that looks correct and admits nothing.
        """
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA_SQL)
        types = {
            name: kind
            for name, kind in connection.execute(
                "SELECT name, type FROM pragma_table_info('tasks') WHERE name='id'"
            )
        }
        rr = {
            name: kind
            for name, kind in connection.execute(
                "SELECT name, type FROM pragma_table_info('task_resource_requests') WHERE name='card_id'"
            )
        }
        assert types["id"] == "TEXT"
        assert rr["card_id"] == "TEXT"


class TestConstraints:
    def test_unique_on_card_id_and_resource_kind_is_enforced(self, conn: sqlite3.Connection) -> None:
        _insert(conn, "t_card_a")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            _insert(conn, "t_card_a")

    def test_same_kind_allowed_for_a_different_card(self, conn: sqlite3.Connection) -> None:
        """Uniqueness is per-card, not global -- two cards may both want a GPU."""
        _insert(conn, "t_card_a")
        _insert(conn, "t_card_b")
        assert conn.execute("SELECT COUNT(*) FROM task_resource_requests").fetchone()[0] == 2

    def test_different_kind_allowed_for_the_same_card(self, conn: sqlite3.Connection) -> None:
        """A card may request more than one resource kind."""
        _insert(conn, "t_card_a", kind="gpu")
        _insert(conn, "t_card_a", kind="npu")
        assert conn.execute("SELECT COUNT(*) FROM task_resource_requests").fetchone()[0] == 2

    def test_foreign_key_rejects_unknown_card(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert(conn, "t_does_not_exist")

    def test_defaults_apply_on_insert(self, conn: sqlite3.Connection) -> None:
        """Omitted optional columns must take their declared defaults."""
        conn.execute(
            """
            INSERT INTO task_resource_requests
                (card_id, selector_type, selector_value,
                 request_json, request_sha256, reservation_ttl_seconds, created_at)
            VALUES ('t_card_a', 'exact', 'GPU-x', '{}', 'abc', 600, 1)
            """
        )
        row = conn.execute(
            """
            SELECT resource_kind, schema_version, exclusive, min_free_mib, headroom_mib
            FROM task_resource_requests WHERE card_id = 't_card_a'
            """
        ).fetchone()
        assert row == ("gpu", 1, 1, 0, 0)