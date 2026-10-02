"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc

# These probe the kanban guard against the real root on purpose.
pytestmark = pytest.mark.allow_real_home_io


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kbc.connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    import tests.conftest as _conftest

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: _conftest._REAL_KANBAN_ROOT
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: _conftest._REAL_KANBAN_ROOT / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    import tests.conftest as _conftest

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(_conftest._REAL_KANBAN_ROOT / "kanban.db")


def test_module_purging_fixtures_do_not_disable_write_guard_in_suite_order():
    """The two module-purging fixtures must not strand this file on stale imports."""
    project_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "tests/hermes_cli/test_kanban_cli_dispatch_passthrough.py::test_cli_dispatch_passes_max_in_progress_from_config",
        "tests/hermes_cli/test_kanban_per_profile_cap.py::test_cap_2_balances_two_profiles",
        "tests/hermes_cli/test_kanban_write_guard.py::test_connect_raises_when_kanban_home_is_real_root",
        "tests/hermes_cli/test_kanban_write_guard.py::test_connect_raises_for_explicit_db_path_under_real_root",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
