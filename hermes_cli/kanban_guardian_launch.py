"""Launch a reserved Kanban GPU worker through the governor guardian CLI.

Hermes trusts this module only with a reservation already validated by the
versioned GPU adapter. It forwards the opaque lease identity and the original
worker argv to ``guardian-run``; Astraeus is an external executable and is
never imported here. The guardian owns child launch, lease heartbeats, and
release/quarantine after handoff.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, Callable


def guardian_argv(
    reservation: Mapping[str, Any],
    child_argv: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Translate a validated receipt into ``guardian-run ... -- <child>`` argv."""
    lease_id = reservation.get("lease_id")
    token = reservation.get("fencing_token")
    if not isinstance(lease_id, str) or not lease_id or lease_id != lease_id.strip():
        raise ValueError("reservation lease_id is invalid")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("reservation fencing_token is invalid")
    if not child_argv or any(
        not isinstance(arg, str) or "\x00" in arg for arg in child_argv
    ):
        raise ValueError("worker argv must be a non-empty sequence of strings")

    from hermes_cli import kanban_gpu_governor_adapter as adapter

    effective_config = config if config is not None else adapter._load_config()
    if not isinstance(effective_config, Mapping):
        raise ValueError("GPU governor config is invalid")
    bin_path = adapter._resolve_path(effective_config, "bin")
    lock_root = adapter._resolve_path(effective_config, "lock_root")
    if bin_path is None or lock_root is None:
        raise ValueError("GPU guardian executable or lock root is unresolved")
    if bin_path.suffix.lower() == ".py":
        python_path = adapter._resolve_path(effective_config, "python")
        if python_path is None:
            raise ValueError("GPU guardian Python interpreter is unresolved")
        argv = [str(python_path), str(bin_path)]
    else:
        argv = [str(bin_path)]
    argv.extend([
        "guardian-run",
        lease_id,
        token,
        "--lock-root",
        str(lock_root),
        "--",
        *child_argv,
    ])
    return argv


def launch_guardian(
    reservation: Mapping[str, Any],
    child_argv: Sequence[str],
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    config: Mapping[str, Any] | None = None,
    argv_transform: Callable[[list[str]], list[str]] | None = None,
    **popen_kwargs: Any,
) -> Any:
    """Start the guardian process with the worker's existing execution context."""
    argv = guardian_argv(reservation, child_argv, config=config)
    if argv_transform is not None:
        argv = argv_transform(argv)
    return popen(argv, **popen_kwargs)


def cancel_reservation(
    reservation: Mapping[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Issue one fenced cancellation when dispatch cannot reach guardian-run.

    The governor's cancel command is used only for pre-launch failures in the
    Hermes claim/workspace/spawn path. A nonzero, malformed, or unavailable
    receipt is uncertainty; callers must not retry with a weaker identity.
    """
    lease_id = reservation.get("lease_id")
    token = reservation.get("fencing_token")
    canonical = reservation.get("canonical_receipt")
    try:
        raw = json.loads(canonical) if isinstance(canonical, str) else canonical
    except (TypeError, ValueError):
        return False
    request_hash = raw.get("request_hash") if isinstance(raw, Mapping) else None
    if (
        not isinstance(lease_id, str)
        or not lease_id
        or not isinstance(token, str)
        or not token.strip()
        or not isinstance(request_hash, str)
        or not request_hash
    ):
        return False

    from hermes_cli import kanban_gpu_governor_adapter as adapter

    try:
        effective_config = config if config is not None else adapter._load_config()
        bin_path = adapter._resolve_path(effective_config, "bin")
        ledger = adapter._resolve_path(effective_config, "ledger")
        lock_root = adapter._resolve_path(effective_config, "lock_root")
    except Exception:
        return False
    if bin_path is None or ledger is None or lock_root is None:
        return False
    if bin_path.suffix.lower() == ".py":
        python_path = adapter._resolve_path(effective_config, "python")
        if python_path is None:
            return False
        argv = [str(python_path), str(bin_path)]
    else:
        argv = [str(bin_path)]
    argv.extend([
        "--db",
        str(ledger),
        "cancel-reservation",
        lease_id,
        token,
        request_hash,
        "--lock-root",
        str(lock_root),
    ])
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=adapter.SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if getattr(result, "returncode", None) != 0:
        return False
    try:
        response = json.loads(getattr(result, "stdout", ""))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(response, Mapping)
        and response.get("protocol_version") == str(adapter.ADAPTER_PROTOCOL_VERSION)
        and response.get("lease_id") == lease_id
        and response.get("transition") == "released"
    )


__all__ = ["cancel_reservation", "guardian_argv", "launch_guardian"]
