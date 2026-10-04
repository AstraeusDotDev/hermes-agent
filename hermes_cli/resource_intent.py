"""Typed validation and canonicalization for GPU resource intent.

This module is the single source of truth for resource-intent validation and
idempotency. Every write surface (CLI, tool API, internal ``create_task``) must
funnel through :func:`canonicalize_resource_intent` rather than re-deriving
canonical bytes, because the stored ``request_sha256`` is only a meaningful
idempotency key if exactly one function decides what "the same request" means.

Design constraints, all load-bearing:

* **Pure.** No GPU probing, no subprocess, no network, no clock, no filesystem.
  Validation is a decision procedure over its argument alone, so it is safe to
  call before a transaction, inside one, or from a test.
* **Fail closed.** Every rejection raises :class:`ResourceIntentError` with a
  stable ``code``. Nothing is coerced, defaulted, or silently dropped. A caller
  that cannot be understood must not reach the database.
* **Canonical output.** Success returns a *text* canonical JSON string whose
  SHA-256 is the digest. The digest is taken over the same bytes that get
  stored, so a stored digest always corresponds byte-for-byte to a stored
  ``request_json``.

The threat model is that a resource intent is attacker-influenced text arriving
over a CLI or tool boundary. This is a *typed* field, not a payload: there is no
key whose value is a file path, URL, shell command, or credential, so those are
rejected structurally rather than pattern-matched after the fact.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

__all__ = [
    "ResourceIntentError",
    "RESOURCE_INTENT_KEYS",
    "RESOURCE_KIND",
    "SCHEMA_VERSION",
    "MAX_RESERVATION_TTL_SECONDS",
    "canonicalize_resource_intent",
    "is_canonical_resource_intent",
]

#: The only resource kind this engine understands.
RESOURCE_KIND = "gpu"

#: The only schema version this engine understands.
SCHEMA_VERSION = 1

#: Reservation TTL is bounded: a reservation must expire so a crashed Card
#: cannot hold a device forever. 24h is long enough for any real training run
#: and short enough that a forgotten reservation self-heals.
MAX_RESERVATION_TTL_SECONDS = 86_400

#: Exactly these keys, no more and no fewer. A key outside this set is rejected
#: rather than ignored, because silently dropping an unrecognised key would let
#: a caller believe a constraint was recorded when it was not.
RESOURCE_INTENT_KEYS = frozenset(
    {
        "resource_kind",
        "schema_version",
        "selector_type",
        "selector_value",
        "exclusive",
        "min_free_mib",
        "headroom_mib",
        "reservation_ttl_seconds",
    }
)

#: Accepted ``selector_type`` values.
#:
#: ``auto`` defers device choice to the dispatcher; ``exact`` names one device.
_SELECTOR_TYPES = frozenset({"auto", "exact"})

# A GPU UUID as NVIDIA reports it: ``GPU-`` plus 8-4-4-4-12 lowercase hex. The
# form is matched case-insensitively but the *canonical* form is uppercase, so a
# lowercase UUID normalises rather than being rejected -- a UUID is not a
# secret and case is not meaningful in one.
_GPU_UUID_RE = re.compile(
    r"^GPU-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


class ResourceIntentError(ValueError):
    """A resource intent was rejected.

    ``code`` is a stable, machine-readable reason. Callers that must fail closed
    on a specific class of bad input (e.g. a bool passed where an int is
    required) should branch on ``code``, not on message text.
    """

    def __init__(self, code: str, message: str, *, key: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.key = key

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ResourceIntentError(code={self.code!r}, key={self.key!r})"


def _require_int(value: Any, key: str) -> int:
    """Return ``value`` as an int, rejecting bool.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(x, int)`` is
    true for ``True``. Persisting ``True`` into an INTEGER column would record
    ``1`` and silently conflate "exclusive" with "min_free_mib = 1". The type
    check is therefore on ``type(x) is int``.
    """
    if type(value) is bool:
        raise ResourceIntentError(
            "bool_as_int", f"{key} must be an int, not a bool", key=key
        )
    if not isinstance(value, int):
        raise ResourceIntentError(
            "not_an_int", f"{key} must be an int, got {type(value).__name__}", key=key
        )
    return value


def _reject_unsafe_str(value: str, key: str) -> None:
    """Reject string values that are not plain identifiers.

    ``selector_value`` is the only free-form string in the schema, and it is
    constrained to either the literal ``auto`` or a GPU UUID. That closed
    grammar is the real defence: a path (``/etc/shadow``), a URL
    (``https://evil/x``), a shell command (``rm -rf /;``) or a credential
    cannot match it, so there is nothing to sanitise after the fact.
    """
    if not value:
        raise ResourceIntentError("empty_string", f"{key} must be non-empty", key=key)
    if any(ch in value for ch in "\n\r\t\x00"):
        raise ResourceIntentError(
            "control_character", f"{key} must not contain control characters", key=key
        )


def canonicalize_resource_intent(request: Mapping[str, Any]) -> tuple[str, str]:
    """Validate a resource intent and return ``(canonical_json, sha256_hex)``.

    The input is validated strictly; the canonical form is
    ``json.dumps(..., sort_keys=True, separators=(',', ':'))`` over the
    *normalised* mapping, so two requests that mean the same thing produce
    byte-identical output and therefore an identical digest.

    Normalisation is limited to values whose representation is genuinely
    non-canonical for their type: a GPU UUID is upper-cased, and a real ``bool``
    is used for ``exclusive``. Nothing is defaulted -- a missing key is an
    error, not an invitation to guess.

    Raises:
        ResourceIntentError: on any deviation from the schema.
    """
    if not isinstance(request, Mapping):
        raise ResourceIntentError(
            "not_a_mapping",
            f"resource intent must be a mapping, got {type(request).__name__}",
        )

    keys = set(request)

    # Unknown keys are rejected, never ignored. A caller that passes
    # ``{"min_free_mib": 0, "env": {...}}`` must not be told the intent was
    # accepted: the ``env`` would not be recorded, and believing otherwise is
    # exactly the failure this engine exists to prevent.
    unknown = sorted(keys - RESOURCE_INTENT_KEYS)
    if unknown:
        raise ResourceIntentError(
            "unknown_key", f"unknown resource-intent keys: {unknown}"
        )

    missing = sorted(RESOURCE_INTENT_KEYS - keys)
    if missing:
        raise ResourceIntentError(
            "missing_key", f"missing resource-intent keys: {missing}"
        )

    # --- fixed-valued fields -------------------------------------------------
    kind = request["resource_kind"]
    if kind != RESOURCE_KIND:
        raise ResourceIntentError(
            "bad_resource_kind",
            f"resource_kind must be {RESOURCE_KIND!r}, got {kind!r}",
            key="resource_kind",
        )

    version = _require_int(request["schema_version"], "schema_version")
    if version != SCHEMA_VERSION:
        raise ResourceIntentError(
            "bad_schema_version",
            f"schema_version must be {SCHEMA_VERSION}, got {version}",
            key="schema_version",
        )

    # --- device selection ----------------------------------------------------
    selector_type = request["selector_type"]
    if not isinstance(selector_type, str) or selector_type not in _SELECTOR_TYPES:
        raise ResourceIntentError(
            "bad_selector_type",
            f"selector_type must be one of {sorted(_SELECTOR_TYPES)}, "
            f"got {selector_type!r}",
            key="selector_type",
        )

    selector_value = request["selector_value"]
    if not isinstance(selector_value, str):
        raise ResourceIntentError(
            "bad_selector_value",
            f"selector_value must be a str, got {type(selector_value).__name__}",
            key="selector_value",
        )
    _reject_unsafe_str(selector_value, "selector_value")

    if selector_type == "auto":
        # Exactly one device, chosen by the dispatcher. Anything else here would
        # let a caller believe the reservation is device-agnostic while
        # recording a pin.
        if selector_value != "auto":
            raise ResourceIntentError(
                "inconsistent_selector",
                "selector_type 'auto' requires selector_value 'auto'",
                key="selector_value",
            )
        normalised_selector_value = "auto"
    else:  # "exact"
        # Exactly one device, named. A GPU UUID or nothing.
        if selector_value == "auto":
            raise ResourceIntentError(
                "inconsistent_selector",
                "selector_type 'exact' requires a GPU UUID, not 'auto'",
                key="selector_value",
            )
        match = _GPU_UUID_RE.match(selector_value)
        if match is None:
            raise ResourceIntentError(
                "bad_selector_value",
                "selector_value must be 'auto' or a GPU UUID of the form "
                "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                key="selector_value",
            )
        normalised_selector_value = "GPU-" + match.group(1).upper()

    # --- exclusivity ---------------------------------------------------------
    # A real bool, not an int. ``1`` is accepted and normalised to ``True``
    # because SQLite has no boolean type and existing rows store 0/1; ``0``
    # normalises to ``False``. What is rejected is the *ambiguous* case, where
    # the caller cannot be read as having made a decision at all.
    exclusive_raw = request["exclusive"]
    if isinstance(exclusive_raw, bool):
        exclusive = exclusive_raw
    elif type(exclusive_raw) is int and exclusive_raw in (0, 1):
        exclusive = bool(exclusive_raw)
    else:
        raise ResourceIntentError(
            "bad_exclusive",
            "exclusive must be a bool (or the int 0/1 SQLite stores)",
            key="exclusive",
        )

    # --- capacities ----------------------------------------------------------
    min_free = _require_int(request["min_free_mib"], "min_free_mib")
    if min_free < 0:
        raise ResourceIntentError(
            "negative_min_free_mib",
            f"min_free_mib must be non-negative, got {min_free}",
            key="min_free_mib",
        )

    headroom = _require_int(request["headroom_mib"], "headroom_mib")
    if headroom < 0:
        raise ResourceIntentError(
            "negative_headroom_mib",
            f"headroom_mib must be non-negative, got {headroom}",
            key="headroom_mib",
        )

    # --- reservation lifetime ------------------------------------------------
    ttl = _require_int(request["reservation_ttl_seconds"], "reservation_ttl_seconds")
    if ttl <= 0:
        raise ResourceIntentError(
            "bad_reservation_ttl",
            f"reservation_ttl_seconds must be positive, got {ttl}",
            key="reservation_ttl_seconds",
        )
    if ttl > MAX_RESERVATION_TTL_SECONDS:
        raise ResourceIntentError(
            "reservation_ttl_too_long",
            f"reservation_ttl_seconds must be <= {MAX_RESERVATION_TTL_SECONDS}, "
            f"got {ttl}",
            key="reservation_ttl_seconds",
        )

    canonical_obj = {
        "resource_kind": RESOURCE_KIND,
        "schema_version": SCHEMA_VERSION,
        "selector_type": selector_type,
        "selector_value": normalised_selector_value,
        "exclusive": exclusive,
        "min_free_mib": min_free,
        "headroom_mib": headroom,
        "reservation_ttl_seconds": ttl,
    }

    # ``allow_nan=False`` so a NaN/Infinity can never reach the digest: such a
    # value is not valid JSON and would produce a digest nothing can reproduce.
    payload = json.dumps(
        canonical_obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_canonical_resource_intent(request: Mapping[str, Any], digest: str) -> bool:
    """Return True when ``request`` canonicalises to exactly ``digest``.

    This is the idempotency predicate. It is a *total* function: any rejection
    returns ``False`` rather than propagating, because "does this incoming
    intent match the stored one?" must be answerable without a caller having to
    pre-validate. Divergence and invalidity are both simply "not equal".
    """
    try:
        _, computed = canonicalize_resource_intent(request)
    except ResourceIntentError:
        return False
    return computed == digest
