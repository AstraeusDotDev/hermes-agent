"""Conditional metadata writes must never overwrite a newer persisted value."""
from hermes_state import SessionDB


def test_compare_and_set_requires_the_exact_existing_value(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        key = "goal:conditional-retry"
        db.set_meta(key, "cleared barrier")
        assert db.compare_and_set_meta(key, "cleared barrier", "parked barrier") is True
        assert db.get_meta(key) == "parked barrier"
        db.set_meta(key, "user paused")
        assert db.compare_and_set_meta(key, "cleared barrier", "parked barrier") is False
        assert db.get_meta(key) == "user paused"
        assert db.compare_and_set_meta("missing-goal", "cleared barrier", "parked barrier") is False
        assert db.get_meta("missing-goal") is None
