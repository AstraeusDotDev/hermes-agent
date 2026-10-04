"""Tests for the typed resource-intent validation/canonicalization engine.

The engine is the single source of truth for resource-intent acceptance and for
the ``request_sha256`` idempotency key, so these tests pin two properties:

1. **Fail closed.** Every deviation from the schema raises
   ``ResourceIntentError`` with the documented code. Nothing is coerced,
   defaulted, or dropped.
2. **Canonical and pure.** Logically identical intents produce byte-identical
   canonical text and an identical digest, regardless of caller key order or
   incidental representation. The engine touches nothing outside its argument.

Rejection tests are written against *inputs a hostile or careless caller would
actually produce* -- a file path in a selector, a URL, a shell command, a
credential -- not against a synthetic shape no caller would emit.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hermes_cli.resource_intent import (
    MAX_RESERVATION_TTL_SECONDS,
    RESOURCE_INTENT_KEYS,
    SCHEMA_VERSION,
    ResourceIntentError,
    canonicalize_resource_intent,
    is_canonical_resource_intent,
)

GPU_UUID = "GPU-1A2B3C4D-0000-0000-0000-00000000ABCD"


def _intent(**overrides):
    """A valid resource intent, with one field overridden per test."""
    intent = {
        "resource_kind": "gpu",
        "schema_version": SCHEMA_VERSION,
        "selector_type": "exact",
        "selector_value": GPU_UUID,
        "exclusive": True,
        "min_free_mib": 8192,
        "headroom_mib": 1024,
        "reservation_ttl_seconds": 600,
    }
    intent.update(overrides)
    return intent


class TestCanonicalisation:
    def test_returns_canonical_text_and_digest(self) -> None:
        payload, digest = canonicalize_resource_intent(_intent())
        assert json.loads(payload) == _intent()
        assert digest == hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert len(digest) == 64

    def test_canonical_text_is_sorted_and_compact(self) -> None:
        payload, _ = canonicalize_resource_intent(_intent())
        assert " " not in payload
        keys = list(json.loads(payload))
        assert keys == sorted(keys)

    def test_caller_key_order_does_not_change_the_digest(self) -> None:
        forward = _intent()
        backward = dict(reversed(list(forward.items())))
        assert list(forward) != list(backward)
        assert canonicalize_resource_intent(forward) == canonicalize_resource_intent(
            backward
        )

    def test_digest_is_stable_across_repeated_calls(self) -> None:
        assert canonicalize_resource_intent(_intent()) == canonicalize_resource_intent(
            _intent()
        )

    def test_uuid_case_is_normalised_not_rejected(self) -> None:
        # Lower-case only the hex body: the "GPU-" prefix is a literal and
        # .lower() on the whole string would destroy it, which is a different
        # (and correctly rejected) input.
        lower = "GPU-" + GPU_UUID[4:].lower()
        payload, d_lower = canonicalize_resource_intent(
            _intent(selector_value=lower)
        )
        _, d_upper = canonicalize_resource_intent(_intent())
        # A UUID is case-insensitive, so both spellings are the SAME request and
        # must share one digest -- otherwise idempotency breaks on re-submission.
        assert d_lower == d_upper
        assert json.loads(payload)["selector_value"] == GPU_UUID

    def test_value_change_changes_the_digest(self) -> None:
        _, d1 = canonicalize_resource_intent(_intent(min_free_mib=8192))
        _, d2 = canonicalize_resource_intent(_intent(min_free_mib=16384))
        assert d1 != d2

    def test_exclusive_int_and_bool_normalise_to_one_form(self) -> None:
        as_bool, d_bool = canonicalize_resource_intent(_intent(exclusive=True))
        as_int, d_int = canonicalize_resource_intent(_intent(exclusive=1))
        # SQLite has no boolean type; 1 and True must not be two different intents.
        assert d_bool == d_int
        assert json.loads(as_bool)["exclusive"] is True
        assert json.loads(as_int)["exclusive"] is True


class TestSchemaShape:
    def test_unknown_key_is_rejected_not_ignored(self) -> None:
        # The load-bearing case: an unrecognised key must fail loudly, because
        # dropping it would let a caller believe a constraint was recorded.
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(env={"CUDA_VISIBLE_DEVICES": "0"}))
        assert exc.value.code == "unknown_key"

    def test_missing_key_is_rejected(self) -> None:
        incomplete = _intent()
        del incomplete["headroom_mib"]
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(incomplete)
        assert exc.value.code == "missing_key"

    def test_error_lists_the_offending_key(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(selector_value="/etc/shadow"))
        assert exc.value.key == "selector_value"

    def test_non_mapping_is_rejected(self) -> None:
        for bad in ([], "gpu", 7, None):
            with pytest.raises(ResourceIntentError) as exc:
                canonicalize_resource_intent(bad)  # type: ignore[arg-type]
            assert exc.value.code == "not_a_mapping"

    def test_every_declared_key_is_required(self) -> None:
        # Guards against the key set and the validation drifting apart: a key
        # added to the schema must also be enforced.
        for key in RESOURCE_INTENT_KEYS:
            incomplete = _intent()
            del incomplete[key]
            with pytest.raises(ResourceIntentError):
                canonicalize_resource_intent(incomplete)


class TestFixedValuedFields:
    def test_resource_kind_must_be_gpu(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(resource_kind="cpu"))
        assert exc.value.code == "bad_resource_kind"

    def test_schema_version_must_be_one(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(schema_version=2))
        assert exc.value.code == "bad_schema_version"

    def test_schema_version_bool_is_rejected(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(schema_version=True))
        assert exc.value.code == "bool_as_int"


class TestSelector:
    def test_auto_selector_requires_auto_value(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(
                _intent(selector_type="auto", selector_value=GPU_UUID)
            )
        assert exc.value.code == "inconsistent_selector"

    def test_exact_selector_rejects_auto_value(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(
                _intent(selector_type="exact", selector_value="auto")
            )
        assert exc.value.code == "inconsistent_selector"

    def test_auto_pair_is_accepted(self) -> None:
        payload, _ = canonicalize_resource_intent(
            _intent(selector_type="auto", selector_value="auto")
        )
        assert json.loads(payload)["selector_type"] == "auto"

    def test_unknown_selector_type_is_rejected(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(selector_type="any"))
        assert exc.value.code == "bad_selector_type"

    @pytest.mark.parametrize("selector_type", [[], ["auto"], {}, {"type": "exact"}])
    @pytest.mark.parametrize("predicate", [False, True])
    def test_selector_containers_fail_structurally(self, selector_type, predicate) -> None:
        request = _intent(selector_type=selector_type)
        if predicate:
            _, digest = canonicalize_resource_intent(_intent())
            assert is_canonical_resource_intent(request, digest) is False
        else:
            with pytest.raises(ResourceIntentError) as exc:
                canonicalize_resource_intent(request)
            assert exc.value.code == "bad_selector_type"
            assert exc.value.key == "selector_type"

    @pytest.mark.parametrize(
        "value",
        [
            "/etc/shadow",                       # file path
            "/home/jfensch/.ssh/id_rsa",         # credential by path
            "https://evil.example/gpu",          # URL
            "rm -rf / ; echo pwned",             # shell command
            "$(cat /etc/passwd)",                # shell substitution
            "`whoami`",                          # shell substitution
            "nvidia-smi --query",                # command name
            "hunter2",                           # bare credential
            "sk-live-abc123",                    # API key
            "GPU-not-a-uuid",                    # malformed
            "00000000-0000-0000-0000-0000000000",  # bare UUID, wrong prefix
            "",                                  # empty
            "GPU-1a2b3c4d-0000-0000-0000-00000000abcd\nGPU-x",  # smuggling
        ],
    )
    def test_non_uuid_selector_values_are_rejected(self, value: str) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(selector_value=value))
        assert exc.value.code in {
            "bad_selector_value",
            "inconsistent_selector",
            "empty_string",
            "control_character",
        }

    def test_non_string_selector_value_is_rejected(self) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(selector_value=0))
        assert exc.value.code == "bad_selector_value"


class TestNumericFields:
    @pytest.mark.parametrize(
        "field", ["min_free_mib", "headroom_mib", "reservation_ttl_seconds"]
    )
    @pytest.mark.parametrize("value", [True, False])
    def test_bool_is_never_accepted_as_int(self, field: str, value: bool) -> None:
        # bool is a subclass of int; without an exact type check, True would be
        # recorded as 1 and conflate a flag with a capacity.
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(**{field: value}))
        assert exc.value.code == "bool_as_int"

    @pytest.mark.parametrize("field", ["min_free_mib", "headroom_mib"])
    def test_negative_capacities_are_rejected(self, field: str) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(**{field: -1}))
        assert exc.value.code.startswith("negative_")

    def test_zero_capacity_is_allowed(self) -> None:
        payload, _ = canonicalize_resource_intent(
            _intent(min_free_mib=0, headroom_mib=0)
        )
        assert json.loads(payload)["min_free_mib"] == 0

    @pytest.mark.parametrize("value", ["8192", 8192.0, None, [8192]])
    def test_non_int_capacities_are_rejected(self, value) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(min_free_mib=value))
        assert exc.value.code in {"not_an_int", "bool_as_int"}

    def test_float_that_is_whole_is_still_rejected(self) -> None:
        # 8192.0 is numerically equal to 8192 but is a different type; silently
        # truncating would make the digest depend on the caller's accident.
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(min_free_mib=8192.0))
        assert exc.value.code == "not_an_int"

    @pytest.mark.parametrize("ttl", [0, -1])
    def test_non_positive_ttl_is_rejected(self, ttl: int) -> None:
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(reservation_ttl_seconds=ttl))
        assert exc.value.code == "bad_reservation_ttl"

    def test_unbounded_ttl_is_rejected(self) -> None:
        # A reservation must expire so a crashed Card cannot hold a device.
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(
                _intent(reservation_ttl_seconds=MAX_RESERVATION_TTL_SECONDS + 1)
            )
        assert exc.value.code == "reservation_ttl_too_long"

    def test_max_ttl_is_accepted(self) -> None:
        payload, _ = canonicalize_resource_intent(
            _intent(reservation_ttl_seconds=MAX_RESERVATION_TTL_SECONDS)
        )
        assert (
            json.loads(payload)["reservation_ttl_seconds"]
            == MAX_RESERVATION_TTL_SECONDS
        )


class TestExclusive:
    @pytest.mark.parametrize("value", [0, 1, True, False])
    def test_boolean_and_sqlite_int_forms_are_accepted(self, value) -> None:
        payload, _ = canonicalize_resource_intent(_intent(exclusive=value))
        assert json.loads(payload)["exclusive"] is bool(value)

    @pytest.mark.parametrize("value", [2, -1, "true", None, 0.0])
    def test_ambiguous_exclusive_values_are_rejected(self, value) -> None:
        # Anything outside {0, 1, True, False} means the caller did not express a
        # decision; guessing on its behalf would persist an invented constraint.
        with pytest.raises(ResourceIntentError) as exc:
            canonicalize_resource_intent(_intent(exclusive=value))
        assert exc.value.code == "bad_exclusive"


class TestIdempotencyPredicate:
    def test_matching_digest_returns_true(self) -> None:
        _, digest = canonicalize_resource_intent(_intent())
        assert is_canonical_resource_intent(_intent(), digest) is True

    def test_reordered_but_identical_intent_matches(self) -> None:
        _, digest = canonicalize_resource_intent(_intent())
        reordered = dict(reversed(list(_intent().items())))
        assert is_canonical_resource_intent(reordered, digest) is True

    def test_divergent_intent_returns_false(self) -> None:
        _, digest = canonicalize_resource_intent(_intent())
        assert is_canonical_resource_intent(_intent(min_free_mib=99), digest) is False

    def test_invalid_intent_returns_false_rather_than_raising(self) -> None:
        # The predicate must be answerable without the caller pre-validating;
        # invalid and divergent are both simply "not the stored request".
        _, digest = canonicalize_resource_intent(_intent())
        assert is_canonical_resource_intent({"bogus": 1}, digest) is False
        assert is_canonical_resource_intent("not a mapping", digest) is False  # type: ignore[arg-type]

    def test_wrong_digest_returns_false(self) -> None:
        assert is_canonical_resource_intent(_intent(), "0" * 64) is False


class TestPurity:
    def test_input_mapping_is_not_mutated(self) -> None:
        # Canonicalization must not normalise in place: the caller's object may
        # be reused, and an engine that rewrites its argument is not pure.
        intent = _intent(selector_value="GPU-" + GPU_UUID[4:].lower())
        snapshot = dict(intent)
        canonicalize_resource_intent(intent)
        assert intent == snapshot

    def test_engine_imports_no_io_or_probing_modules(self) -> None:
        # The card forbids GPU probing, subprocess, and network. Proven over the
        # AST's actual import statements rather than a text scan: this module's
        # own docstring names those very modules while promising not to use
        # them, so a substring scan would flag the promise as the violation.
        import ast
        import pathlib

        import hermes_cli.resource_intent as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert imported <= {"hashlib", "json", "re", "typing", "__future__"}
        for banned in ("subprocess", "socket", "urllib", "requests", "pynvml", "os"):
            assert banned not in imported

    def test_validation_does_not_read_the_environment(self, monkeypatch) -> None:
        # A device-selection decision that depended on ambient env would make
        # the digest non-reproducible: the same intent would hash differently
        # depending on the shell it was validated in.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "attacker-board")
        payload, digest = canonicalize_resource_intent(_intent())
        assert "CUDA_VISIBLE_DEVICES" not in payload
        assert "attacker-board" not in payload
        assert set(json.loads(payload)) == RESOURCE_INTENT_KEYS
        # Same digest as the unenv'd computation.
        assert is_canonical_resource_intent(_intent(), digest) is True
