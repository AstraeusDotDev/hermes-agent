"""Pure validation for GPU reservation grant receipts.

A resource intent describes a request; a receipt attests that a grant was made.
This module validates the latter and binds its request digest to an admitted
intent supplied separately by the caller. It performs no I/O and never reads
the clock.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from hermes_cli.resource_intent import (
    ResourceIntentError,
    canonicalize_resource_intent,
)

__all__ = ["ReceiptValidationResult", "RESOURCE_RECEIPT_KEYS", "validate_resource_receipt"]

RESOURCE_RECEIPT_KEYS = frozenset(
    {
        "lease_id",
        "fencing_token",
        "request_digest",
        "gpu_selector",
        "expiry",
        "canonical_receipt",
    }
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")


@dataclass(frozen=True)
class ReceiptValidationResult:
    """Validation outcome; ``reasons`` contains every detected failure."""

    valid: bool
    reasons: tuple[str, ...]


def _valid_gpu_selector(value: Any) -> bool:
    """Return whether a selector is a non-negative index or a UUID string."""
    if type(value) is int:
        return value >= 0
    if not isinstance(value, str) or not value:
        return False
    if _GPU_UUID_RE.fullmatch(value):
        return True
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return value.lower() == str(parsed)


def _as_utc(value: datetime) -> datetime:
    """Treat naive caller-provided datetimes as UTC; normalize aware values."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_expiry(value: Any) -> datetime | int | None:
    """Parse supported expiry forms without consulting a clock."""
    if type(value) is int:
        return value
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    iso_value = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        return datetime.fromisoformat(iso_value)
    except ValueError:
        return None


def validate_resource_receipt(
    receipt: Mapping[str, Any],
    *,
    now: datetime | None = None,
    admitted_request: Mapping[str, Any] | None = None,
) -> ReceiptValidationResult:
    """Validate a receipt and bind its digest to an admitted resource intent.

    ``now`` is supplied by the caller. If omitted, the expiry cannot be proven
    future-dated and validation fails closed. ``admitted_request`` must be the
    request already accepted by the caller's trust boundary; receipt contents
    are never treated as proof of admission.

    No clock, filesystem, network, or subprocess operation is performed.
    Reasons are stable field-oriented strings, and validation does not stop at
    the first invalid field.
    """
    reasons: list[str] = []
    if not isinstance(receipt, Mapping):
        return ReceiptValidationResult(False, ("receipt:not_a_mapping",))

    keys = set(receipt)
    for key in sorted(keys - RESOURCE_RECEIPT_KEYS, key=repr):
        reasons.append(f"unknown_key:{key!r}")
    for key in sorted(RESOURCE_RECEIPT_KEYS - keys):
        reasons.append(f"missing:{key}")

    # lease_id: stable, non-empty identifier without accidental edge spaces.
    if "lease_id" in receipt:
        value = receipt["lease_id"]
        if not isinstance(value, str):
            reasons.append("lease_id:wrong_type")
        elif not value:
            reasons.append("lease_id:empty")
        elif value != value.strip():
            reasons.append("lease_id:surrounding_whitespace")

    # fencing_token is opaque, but it must contain at least one non-space char.
    if "fencing_token" in receipt:
        value = receipt["fencing_token"]
        if not isinstance(value, str):
            reasons.append("fencing_token:wrong_type")
        elif not value.strip():
            reasons.append("fencing_token:empty")

    digest: Any = None
    if "request_digest" in receipt:
        digest = receipt["request_digest"]
        if not isinstance(digest, str):
            reasons.append("request_digest:wrong_type")
        elif not digest:
            reasons.append("request_digest:empty")
        elif _SHA256_HEX_RE.fullmatch(digest) is None:
            reasons.append("request_digest:invalid_sha256_hex")

    if "gpu_selector" in receipt:
        value = receipt["gpu_selector"]
        if value is None:
            reasons.append("gpu_selector:null")
        elif isinstance(value, str) and not value:
            reasons.append("gpu_selector:empty")
        elif not _valid_gpu_selector(value):
            reasons.append("gpu_selector:wrong_type_or_format")

    if "expiry" in receipt:
        parsed_expiry = _parse_expiry(receipt["expiry"])
        if parsed_expiry is None:
            value = receipt["expiry"]
            if value is None:
                reasons.append("expiry:null")
            elif isinstance(value, str) and not value:
                reasons.append("expiry:empty")
            else:
                reasons.append("expiry:wrong_type_or_format")
        elif now is None:
            reasons.append("expiry:now_required")
        elif not isinstance(now, datetime):
            reasons.append("expiry:invalid_now")
        else:
            current = _as_utc(now)
            if isinstance(parsed_expiry, int):
                is_future = parsed_expiry > current.timestamp()
            else:
                is_future = _as_utc(parsed_expiry) > current
            if not is_future:
                reasons.append("expiry:not_in_future")

    if "canonical_receipt" in receipt:
        value = receipt["canonical_receipt"]
        if isinstance(value, str):
            if not value.strip():
                reasons.append("canonical_receipt:empty")
        elif isinstance(value, Mapping):
            if not value:
                reasons.append("canonical_receipt:empty")
        else:
            reasons.append("canonical_receipt:wrong_type")

    # A well-formed digest alone proves nothing: it must equal the canonical
    # digest generated by the request-intent engine for the admitted request.
    request = admitted_request
    if request is None:
        reasons.append("request_digest:admitted_request_required")
    else:
        try:
            _, expected_digest = canonicalize_resource_intent(request)
        except ResourceIntentError as exc:
            reasons.append(f"request_digest:admitted_request_invalid:{exc.code}")
        else:
            if isinstance(digest, str) and digest != expected_digest:
                reasons.append("request_digest:mismatch")

    return ReceiptValidationResult(not reasons, tuple(reasons))
