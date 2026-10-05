"""Versioned Hermes-to-Astraeus GPU reservation adapter.

The governor owns a separate request-hash contract. Protocol v1 re-derives its
hash from the exact reserve arguments (request id, GPU UUID, capacity floors,
priority, owner, empty payload, ``exclusive`` compatible key, and non-CPU flag)
and never confuses it with Hermes' resource-intent digest. The path digests
mirror Astraeus ``_ledger_digest`` / ``_lock_root_digest``: SHA-256 of
``str(Path(path).resolve())``. Tests cross-check those formulas against the
real governor, whose reserve rejects incorrect digests.

"Zero-wait" means one bounded, non-queuing reservation attempt, not an expired
deadline: the default admission window is two seconds, configurable with
``kanban.gpu_governor.admission_window_seconds`` or
``ASTRAEUS_GPU_GOVERNOR_ADMISSION_WINDOW``. The adapter's subprocess timeout is
capped by both the remaining window and eight seconds (and remains below the
adapter's ten-second maximum). The caller's dispatch loop
still owns the at-most-one-attempt-per-tick invariant. Configure ``bin`` as the
governor Python script or a native executable. A ``.py`` script requires the
explicit ``kanban.gpu_governor.python`` / ``ASTRAEUS_GPU_GOVERNOR_PYTHON``
interpreter so it cannot silently import Astraeus from Hermes' environment.
This adapter never mutates Kanban card state; any governor-ledger mutation
belongs to the governor CLI. It returns a verified receipt or a deferral.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hermes_cli.kanban_resource_receipt import validate_resource_receipt
from hermes_cli.resource_intent import canonicalize_resource_intent

ADAPTER_PROTOCOL_VERSION = 1
SUBPROCESS_TIMEOUT_SECONDS = 8
DEFAULT_ADMISSION_WINDOW_SECONDS = 2.0
MAX_ADMISSION_WINDOW_SECONDS = 10.0

_CONFIG_PATHS = {
    "bin": ("ASTRAEUS_GPU_GOVERNOR_BIN", "bin"),
    "ledger": ("ASTRAEUS_GPU_GOVERNOR_LEDGER", "ledger"),
    "lock_root": ("ASTRAEUS_GPU_GOVERNOR_LOCK_ROOT", "lock_root"),
    "python": ("ASTRAEUS_GPU_GOVERNOR_PYTHON", "python"),
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
    path = Path(value).expanduser()
    if name == "python":
        # Venv launchers are often symlinks; resolving one bypasses pyvenv.cfg.
        return Path(os.path.abspath(path))
    return path.resolve()


def _load_config() -> Mapping[str, Any]:
    from hermes_cli.config_effective import load_user_config_effective

    loaded = load_user_config_effective(fail_closed=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("effective config is not a mapping")
    return loaded


def _canonical_path_digest(path: Path) -> str:
    """Match the governor's canonical absolute-path digest functions."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _admission_window(config: Mapping[str, Any]) -> float | None:
    present, configured = _config_value(config, "admission_window_seconds")
    if present and configured not in (None, ""):
        value = configured
    else:
        value = os.environ.get("ASTRAEUS_GPU_GOVERNOR_ADMISSION_WINDOW")
        if value in (None, ""):
            return DEFAULT_ADMISSION_WINDOW_SECONDS
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    window = float(value)
    if window <= 0 or window > MAX_ADMISSION_WINDOW_SECONDS:
        return None
    return window


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
    *, bin_path: Path, python_path: Path | None, ledger: Path, lock_root: Path,
    request_id: str, gpu_uuid: str, min_free_mib: int,
    headroom_mib: int, priority: int, owner: str, deadline: float,
) -> list[str]:
    if bin_path.suffix.lower() == ".py":
        if python_path is None:
            return []
        argv = [str(python_path), str(bin_path)]
    else:
        argv = [str(bin_path)]
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
    deadline: float,
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
        if expiry > deadline:
            return None
        hermes_expiry: int | str = expiry
    elif type(expiry) is float:
        if not math.isfinite(expiry) or expiry > deadline:
            return None
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
    """Request one bounded, non-queuing reservation attempt for a persisted row.

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
        paths = {name: _resolve_path(config, name) for name in ("bin", "ledger", "lock_root")}
        if any(path is None for path in paths.values()):
            missing = [name for name, path in paths.items() if path is None]
            return _defer("unresolved_config:" + ",".join(missing))
        bin_path = paths["bin"]
        ledger = paths["ledger"]
        lock_root = paths["lock_root"]
        assert bin_path is not None and ledger is not None and lock_root is not None
        python_path = _resolve_path(config, "python") if bin_path.suffix.lower() == ".py" else None
        if bin_path.suffix.lower() == ".py" and python_path is None:
            return _defer("governor_python_unresolved")
        admission_window = _admission_window(config)
        if admission_window is None:
            return _defer("invalid_admission_window")
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
        started = time.monotonic()
        deadline = time.time() + admission_window
        argv = _build_argv(
            bin_path=bin_path,
            python_path=python_path,
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
        remaining_window = admission_window - max(0.0, time.monotonic() - started)
        timeout = min(SUBPROCESS_TIMEOUT_SECONDS, remaining_window)
        if not argv or timeout <= 0:
            return _defer("admission_window_elapsed")
        result = (runner or subprocess.run)(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
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
            deadline=deadline,
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
