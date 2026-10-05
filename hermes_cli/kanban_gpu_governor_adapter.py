"""Versioned Hermes-to-Astraeus GPU reservation adapter.

The governor owns a separate request-hash contract. Protocol v1 re-derives its
hash from the exact reserve arguments (request id, GPU UUID, capacity floors,
priority, owner, empty payload, ``exclusive`` compatible key, and non-CPU flag)
and never confuses it with Hermes' resource-intent digest. The path digests
mirror Astraeus ``_ledger_digest`` / ``_lock_root_digest``: SHA-256 of
``str(Path(path).resolve())``. Tests cross-check those formulas against the
real governor, whose reserve rejects incorrect digests.

The configured ``bin`` is either the governor Python script or an executable;
Python scripts run with the current interpreter. The CLI's zero-wait contract
is an absolute deadline equal to the current epoch time. Calls are bounded by
an independent eight-second subprocess timeout. This adapter never mutates
Kanban card state; any governor-ledger mutation belongs to the governor CLI.
It returns a verified receipt or a deferral decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hermes_cli.kanban_resource_receipt import validate_resource_receipt
from hermes_cli.resource_intent import canonicalize_resource_intent

ADAPTER_PROTOCOL_VERSION = 1
SUBPROCESS_TIMEOUT_SECONDS = 8

_CONFIG_PATHS = {
    "bin": ("ASTRAEUS_GPU_GOVERNOR_BIN", "bin"),
    "ledger": ("ASTRAEUS_GPU_GOVERNOR_LEDGER", "ledger"),
    "lock_root": ("ASTRAEUS_GPU_GOVERNOR_LOCK_ROOT", "lock_root"),
}


@dataclass(frozen=True)
class GPUAdapterResult:
    """The adapter's non-mutating admission decision."""

    status: str
    reason: str
    receipt: dict[str, Any] | None = None

    @property
    def granted(self) -> bool:
        return self.status == "granted"


def _defer(reason: str) -> GPUAdapterResult:
    return GPUAdapterResult("deferred_resources", reason)


def _config_value(config: Mapping[str, Any], leaf: str) -> tuple[bool, Any]:
    current: Any = config
    for key in ("kanban", "gpu_governor", leaf):
        if not isinstance(current, Mapping):
            return False, None
        if key not in current:
            return False, None
        current = current[key]
    return True, current


def _resolve_path(config: Mapping[str, Any], name: str) -> Path | None:
    env_name, leaf = _CONFIG_PATHS[name]
    present, configured = _config_value(config, leaf)
    if present and configured not in (None, ""):
        value = configured
    else:
        value = os.environ.get(env_name)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _load_config() -> Mapping[str, Any]:
    from hermes_cli.config_effective import load_user_config_effective

    loaded = load_user_config_effective(fail_closed=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("effective config is not a mapping")
    return loaded


def _canonical_path_digest(path: Path) -> str:
    """Match the governor's canonical absolute-path digest functions."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _strict_json_loads(value: str) -> Any:
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(
        value,
        object_pairs_hook=object_without_duplicates,
        parse_constant=reject_constant,
    )


def _resource_row(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        row = to_dict()
        if isinstance(row, Mapping):
            return row
    raise ValueError("resource request must be a persisted mapping")


def _hermes_intent(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    intent = {
        "resource_kind": row["resource_kind"],
        "schema_version": row["schema_version"],
        "selector_type": row["selector_type"],
        "selector_value": row["selector_value"],
        "exclusive": row["exclusive"],
        "min_free_mib": row["min_free_mib"],
        "headroom_mib": row["headroom_mib"],
        "reservation_ttl_seconds": row["reservation_ttl_seconds"],
    }
    _, digest = canonicalize_resource_intent(intent)
    if row.get("request_sha256") != digest:
        raise ValueError("persisted resource intent digest mismatch")
    return intent, digest


def _request_id(card_id: str, attempt: int) -> str:
    stable = hashlib.sha256(f"{card_id}\0{attempt}".encode("utf-8")).hexdigest()
    return f"hermes-{stable}"


def _governor_request_hash(
    *, request_id: str, gpu_uuid: str, min_free_mib: int,
    headroom_mib: int, priority: int, owner: str,
) -> str:
    """Version-1 mirror of Astraeus ``_canonical_request_hash``."""
    canonical = json.dumps(
        {
            "request_id": request_id,
            "gpu_uuid": gpu_uuid,
            "min_free_mib": min_free_mib,
            "headroom_mib": headroom_mib,
            "priority": priority,
            "owner": owner,
            "payload": "{}",
            "compatible_key": "exclusive",
            "cpu_only": 0,
        },
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_argv(
    *, bin_path: Path, ledger: Path, lock_root: Path,
    request_id: str, gpu_uuid: str, min_free_mib: int,
    headroom_mib: int, priority: int, owner: str, deadline: float,
) -> list[str]:
    argv = [str(bin_path)] if bin_path.suffix.lower() != ".py" else [sys.executable, str(bin_path)]
    argv.extend(
        [
            "--db", str(ledger), "reserve", request_id,
            "--gpu", gpu_uuid,
            "--min-free-mib", str(min_free_mib),
            "--headroom-mib", str(headroom_mib),
            "--priority", str(priority),
            "--owner", owner,
            "--deadline", repr(deadline),
            "--ledger-digest", _canonical_path_digest(ledger),
            "--lock-root-digest", _canonical_path_digest(lock_root),
            "--lock-root", str(lock_root),
        ]
    )
    return argv


def _verified_receipt(
    raw: Any,
    *,
    expected_request_hash: str,
    request_id: str,
    gpu_uuid: str,
    ledger_digest: str,
    lock_root_digest: str,
    intent: Mapping[str, Any],
    intent_digest: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    expected_protocol = str(ADAPTER_PROTOCOL_VERSION)
    if raw.get("protocol_version") != expected_protocol:
        return None
    if raw.get("request_hash") != expected_request_hash:
        return None
    if raw.get("request_id") != request_id or raw.get("gpu_uuid") != gpu_uuid:
        return None
    if raw.get("ledger_digest") != ledger_digest:
        return None
    if raw.get("lock_root_digest") != lock_root_digest:
        return None
    lease_id, token, expiry = raw.get("lease_id"), raw.get("fencing_token"), raw.get("expires_at")
    if not isinstance(lease_id, str) or not lease_id.strip():
        return None
    if not isinstance(token, str) or not token.strip():
        return None
    if type(expiry) is int:
        hermes_expiry: int | str = expiry
    elif type(expiry) is float:
        try:
            hermes_expiry = datetime.fromtimestamp(expiry, timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    else:
        return None
    try:
        canonical_governor_receipt = json.dumps(
            raw, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        # Hermes receipts accept integer epochs or ISO timestamps; the
        # governor emits a fractional epoch, so preserve its instant as UTC ISO.
        hermes_receipt = {
            "lease_id": lease_id,
            "fencing_token": token,
            "request_digest": intent_digest,
            "gpu_selector": gpu_uuid,
            "expiry": hermes_expiry,
            "canonical_receipt": canonical_governor_receipt,
        }
        validation = validate_resource_receipt(
            hermes_receipt,
            now=datetime.now(timezone.utc),
            admitted_request=intent,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return hermes_receipt if validation.valid else None


def reserve_gpu_resource(
    resource_request: Any,
    *,
    attempt: int,
    owner: str,
    priority: int,
    config: Mapping[str, Any] | None = None,
    runner: Callable[..., Any] | None = None,
) -> GPUAdapterResult:
    """Request a zero-wait GPU reservation for a persisted Kanban resource row.

    ``priority`` is supplied from the persisted card record because
    ``task_resource_requests`` has no priority column. The request row itself is
    verified against its stored Hermes digest before any subprocess is started.
    All expected operational failures become ``deferred_resources`` results.
    """
    try:
        if config is None:
            try:
                config = _load_config()
            except Exception:
                return _defer("config_unavailable")
        if not isinstance(config, Mapping):
            return _defer("invalid_config")
        paths = {name: _resolve_path(config, name) for name in _CONFIG_PATHS}
        if any(path is None for path in paths.values()):
            missing = [name for name, path in paths.items() if path is None]
            return _defer("unresolved_config:" + ",".join(missing))
        bin_path = paths["bin"]
        ledger = paths["ledger"]
        lock_root = paths["lock_root"]
        assert bin_path is not None and ledger is not None and lock_root is not None
        if ledger.parent.resolve() != lock_root:
            return _defer("governor_cli_lock_root_must_match_ledger_parent")

        row = _resource_row(resource_request)
        intent, intent_digest = _hermes_intent(row)
        card_id = row.get("card_id")
        if not isinstance(card_id, str) or not card_id.strip():
            return _defer("invalid_card_id")
        if type(attempt) is not int or attempt < 1:
            return _defer("invalid_attempt")
        if type(priority) is not int:
            return _defer("invalid_priority")
        if not isinstance(owner, str) or not owner.strip() or any(c in owner for c in "\x00\r\n"):
            return _defer("invalid_owner")
        if intent["selector_type"] != "exact":
            return _defer("gpu_selector_requires_exact_uuid")
        gpu_uuid = intent["selector_value"]
        request_id = _request_id(card_id, attempt)
        request_hash = _governor_request_hash(
            request_id=request_id,
            gpu_uuid=gpu_uuid,
            min_free_mib=intent["min_free_mib"],
            headroom_mib=intent["headroom_mib"],
            priority=priority,
            owner=owner,
        )
        ledger_digest = _canonical_path_digest(ledger)
        lock_root_digest = _canonical_path_digest(lock_root)
        deadline = time.time()
        argv = _build_argv(
            bin_path=bin_path,
            ledger=ledger,
            lock_root=lock_root,
            request_id=request_id,
            gpu_uuid=gpu_uuid,
            min_free_mib=intent["min_free_mib"],
            headroom_mib=intent["headroom_mib"],
            priority=priority,
            owner=owner,
            deadline=deadline,
        )
        result = (runner or subprocess.run)(
            argv,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
        returncode = getattr(result, "returncode", None)
        if type(returncode) is not int:
            return _defer("invalid_subprocess_result")
        if returncode != 0:
            return _defer("governor_busy" if returncode == 75 else "governor_exit_nonzero")
        stdout = getattr(result, "stdout", None)
        if not isinstance(stdout, str):
            return _defer("malformed_governor_output")
        try:
            governor_receipt = _strict_json_loads(stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return _defer("malformed_governor_output")
        verified = _verified_receipt(
            governor_receipt,
            expected_request_hash=request_hash,
            request_id=request_id,
            gpu_uuid=gpu_uuid,
            ledger_digest=ledger_digest,
            lock_root_digest=lock_root_digest,
            intent=intent,
            intent_digest=intent_digest,
        )
        if verified is None:
            if isinstance(governor_receipt, dict) and governor_receipt.get("protocol_version") != str(ADAPTER_PROTOCOL_VERSION):
                return _defer("unsupported_governor_protocol")
            if isinstance(governor_receipt, dict) and governor_receipt.get("request_hash") != request_hash:
                return _defer("governor_request_hash_mismatch")
            return _defer("governor_receipt_verification_failed")
        return GPUAdapterResult("granted", "verified", verified)
    except subprocess.TimeoutExpired:
        return _defer("governor_timeout")
    except Exception:
        return _defer("adapter_error")


__all__ = [
    "ADAPTER_PROTOCOL_VERSION",
    "GPUAdapterResult",
    "SUBPROCESS_TIMEOUT_SECONDS",
    "reserve_gpu_resource",
]
