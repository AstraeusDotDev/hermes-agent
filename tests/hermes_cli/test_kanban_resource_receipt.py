"""CPU-only tests for GPU reservation receipt validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from hermes_cli.resource_intent import canonicalize_resource_intent
from hermes_cli.kanban_resource_receipt import (
    ReceiptValidationResult,
    validate_resource_receipt,
)

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
GPU_UUID = "GPU-1A2B3C4D-0000-0000-0000-00000000ABCD"


def _intent(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
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


def _receipt(**overrides: Any) -> dict[str, Any]:
    _, digest = canonicalize_resource_intent(_intent())
    receipt: dict[str, Any] = {
        "lease_id": "lease-123",
        "fencing_token": "fence-9",
        "request_digest": digest,
        "gpu_selector": GPU_UUID,
        "expiry": int(NOW.timestamp()) + 60,
        "canonical_receipt": "signed-by-the-broker",
    }
    receipt.update(overrides)
    return receipt


def _validate(receipt: dict[str, Any], **kwargs: Any) -> ReceiptValidationResult:
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("admitted_request", _intent())
    return validate_resource_receipt(receipt, **kwargs)


def test_valid_receipt_passes_and_digest_uses_intent_engine() -> None:
    result = _validate(_receipt())
    assert result == ReceiptValidationResult(valid=True, reasons=())


def test_mapping_canonical_receipt_does_not_prove_request_was_admitted() -> None:
    receipt = _receipt(canonical_receipt=_intent())
    result = validate_resource_receipt(receipt, now=NOW)
    assert not result.valid
    assert "request_digest:admitted_request_required" in result.reasons


@pytest.mark.parametrize(
    "field",
    [
        "lease_id",
        "fencing_token",
        "request_digest",
        "gpu_selector",
        "expiry",
        "canonical_receipt",
    ],
)
def test_each_field_is_required(field: str) -> None:
    receipt = _receipt()
    del receipt[field]
    result = _validate(receipt)
    assert not result.valid
    assert f"missing:{field}" in result.reasons


@pytest.mark.parametrize(
    "field",
    [
        "lease_id",
        "fencing_token",
        "request_digest",
        "gpu_selector",
        "expiry",
        "canonical_receipt",
    ],
)
def test_each_field_rejects_null(field: str) -> None:
    result = _validate(_receipt(**{field: None}))
    assert not result.valid
    assert any(reason.startswith(f"{field}:") for reason in result.reasons)


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [
        ("lease_id", ""),
        ("fencing_token", ""),
        ("request_digest", ""),
        ("gpu_selector", ""),
        ("expiry", ""),
        ("canonical_receipt", ""),
    ],
)
def test_each_field_rejects_empty_value(field: str, empty_value: str) -> None:
    result = _validate(_receipt(**{field: empty_value}))
    assert not result.valid
    assert any(reason.startswith(f"{field}:") for reason in result.reasons)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("lease_id", 17),
        ("fencing_token", 17),
        ("request_digest", 17),
        ("gpu_selector", []),
        ("expiry", []),
        ("canonical_receipt", []),
    ],
)
def test_each_field_rejects_wrong_type(field: str, wrong_value: Any) -> None:
    result = _validate(_receipt(**{field: wrong_value}))
    assert not result.valid
    assert any(reason.startswith(f"{field}:") for reason in result.reasons)


@pytest.mark.parametrize("selector", [0, 12, GPU_UUID, "1a2b3c4d-0000-0000-0000-00000000abcd"])
def test_gpu_selector_accepts_index_or_uuid(selector: Any) -> None:
    assert _validate(_receipt(gpu_selector=selector)).valid


@pytest.mark.parametrize("selector", [True, False, -1, "not-a-uuid"])
def test_gpu_selector_rejects_bool_negative_index_and_bad_uuid(selector: Any) -> None:
    result = _validate(_receipt(gpu_selector=selector))
    assert not result.valid
    assert "gpu_selector:wrong_type_or_format" in result.reasons


@pytest.mark.parametrize("expiry", [True, False])
def test_expiry_rejects_bool_as_int(expiry: bool) -> None:
    result = _validate(_receipt(expiry=expiry))
    assert "expiry:wrong_type_or_format" in result.reasons


def test_future_epoch_expiry_passes_and_expired_epoch_fails() -> None:
    assert _validate(_receipt(expiry=int(NOW.timestamp()) + 1)).valid
    expired = _validate(_receipt(expiry=int(NOW.timestamp()) - 1))
    assert "expiry:not_in_future" in expired.reasons


def test_future_and_expired_iso_expiry() -> None:
    assert _validate(_receipt(expiry=(NOW + timedelta(seconds=1)).isoformat())).valid
    expired = _validate(_receipt(expiry=(NOW - timedelta(seconds=1)).isoformat()))
    assert "expiry:not_in_future" in expired.reasons


def test_expiry_equal_to_now_is_rejected_for_epoch_and_iso() -> None:
    for value in (int(NOW.timestamp()), NOW.isoformat()):
        result = _validate(_receipt(expiry=value))
        assert "expiry:not_in_future" in result.reasons


def test_expiry_requires_caller_supplied_now() -> None:
    _, digest = canonicalize_resource_intent(_intent())
    receipt = _receipt()
    result = validate_resource_receipt(receipt, admitted_request=_intent())
    assert not result.valid
    assert "expiry:now_required" in result.reasons
    assert receipt["request_digest"] == digest


def test_request_digest_requires_lowercase_sha256_hex() -> None:
    _, digest = canonicalize_resource_intent(_intent())
    result = _validate(_receipt(request_digest=digest.upper()))
    assert "request_digest:invalid_sha256_hex" in result.reasons


def test_unknown_key_is_rejected() -> None:
    result = _validate(_receipt(unrecognized=True))
    assert not result.valid
    assert "unknown_key:'unrecognized'" in result.reasons


def test_digest_mismatch_is_rejected_against_admitted_intent() -> None:
    different_intent = _intent(min_free_mib=16384)
    result = validate_resource_receipt(
        _receipt(), now=NOW, admitted_request=different_intent
    )
    assert "request_digest:mismatch" in result.reasons


def test_invalid_admitted_request_fails_closed() -> None:
    result = validate_resource_receipt(
        _receipt(), now=NOW, admitted_request={"unexpected": "intent"}
    )
    assert any(
        reason.startswith("request_digest:admitted_request_invalid:")
        for reason in result.reasons
    )


def test_missing_admitted_request_cannot_be_inferred_from_string_receipt() -> None:
    result = validate_resource_receipt(_receipt(), now=NOW)
    assert "request_digest:admitted_request_required" in result.reasons


def test_all_simultaneous_failures_are_reported() -> None:
    result = _validate(
        _receipt(
            lease_id=" lease ",
            fencing_token="   ",
            request_digest="A" * 64,
            gpu_selector=True,
            expiry=int(NOW.timestamp()),
            canonical_receipt=[],
            unexpected="field",
        )
    )
    assert not result.valid
    assert {
        "lease_id:surrounding_whitespace",
        "fencing_token:empty",
        "request_digest:invalid_sha256_hex",
        "request_digest:mismatch",
        "gpu_selector:wrong_type_or_format",
        "expiry:not_in_future",
        "canonical_receipt:wrong_type",
        "unknown_key:'unexpected'",
    }.issubset(set(result.reasons))


def test_result_is_immutable() -> None:
    result = _validate(_receipt())
    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]


def test_non_mapping_input_fails_closed() -> None:
    result = validate_resource_receipt(None)  # type: ignore[arg-type]
    assert result.reasons == ("receipt:not_a_mapping",)
