"""Tests for the versioned Hermes-to-Astraeus GPU governor adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli.resource_intent import canonicalize_resource_intent
from hermes_cli.kanban_resource_receipt import validate_resource_receipt
from hermes_cli.kanban_gpu_governor_adapter import (
    ADAPTER_PROTOCOL_VERSION,
    SUBPROCESS_TIMEOUT_SECONDS,
    reserve_gpu_resource,
)

GPU_UUID = "GPU-1A2B3C4D-0000-0000-0000-00000000ABCD"
ASTRAEUS_ROOT = Path.home() / "astraeus"
ASTRAEUS_GOVERNOR_SCRIPT = ASTRAEUS_ROOT / "scripts" / "gpu_governor.py"
ASTRAEUS_PYTHON = ASTRAEUS_ROOT / ".venv" / "bin" / "python"


def _row(**overrides: Any) -> dict[str, Any]:
    intent: dict[str, Any] = {
        "resource_kind": "gpu",
        "schema_version": 1,
        "selector_type": "exact",
        "selector_value": GPU_UUID,
        "exclusive": True,
        "min_free_mib": 8192,
        "headroom_mib": 1024,
        "reservation_ttl_seconds": 600,
    }
    intent.update(overrides)
    _, digest = canonicalize_resource_intent(intent)
    return {"card_id": "t_0123abcd", **intent, "request_sha256": digest}


def _config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    values = {
        "bin": str(ASTRAEUS_GOVERNOR_SCRIPT),
        "ledger": str(tmp_path / "governor.sqlite3"),
        "lock_root": str(tmp_path),
    }
    values.update(overrides)
    return {"kanban": {"gpu_governor": values}}


def _request_id(card_id: str = "t_0123abcd", attempt: int = 3) -> str:
    stable = hashlib.sha256(f"{card_id}\0{attempt}".encode()).hexdigest()
    return f"hermes-{stable}"


def _governor_hash(
    *, request_id: str | None = None, gpu_uuid: str = GPU_UUID,
    min_free_mib: int = 8192, headroom_mib: int = 1024,
    priority: int = 7, owner: str = "colony-coder",
) -> str:
    canonical = json.dumps(
        {
            "request_id": request_id or _request_id(),
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


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _success_json(tmp_path: Path, **overrides: Any) -> str:
    _, request_digest = canonicalize_resource_intent(
        {key: value for key, value in _row().items() if key not in {"card_id", "request_sha256"}}
    )
    receipt: dict[str, Any] = {
        "protocol_version": str(ADAPTER_PROTOCOL_VERSION),
        "request_hash": _governor_hash(),
        "lease_id": "lease-0123",
        "request_id": _request_id(),
        "fencing_token": "fence-44",
        "gpu_uuid": GPU_UUID,
        "gpu_index": 0,
        "snapshot": {"uuid": GPU_UUID, "index": 0},
        "created_at": time.time(),
        "expires_at": time.time() + 600,
        "ledger_digest": _path_digest(tmp_path / "governor.sqlite3"),
        "lock_root_digest": _path_digest(tmp_path),
    }
    receipt.update(overrides)
    assert request_digest
    return json.dumps(receipt)


def _invoke(
    tmp_path: Path,
    runner,
    *,
    config: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
):
    return reserve_gpu_resource(
        row or _row(),
        attempt=3,
        owner="colony-coder",
        priority=7,
        config=config or _config(tmp_path),
        runner=runner,
    )


def test_success_builds_zero_wait_argv_and_valid_hermes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hermes_cli.kanban_gpu_governor_adapter.time.time", lambda: 1_900_000_000.0)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=_success_json(tmp_path), stderr="")

    result = _invoke(tmp_path, runner)

    assert result.status == "granted"
    assert result.reason == "verified"
    assert result.receipt is not None
    validation = validate_resource_receipt(
        result.receipt,
        now=datetime.now(timezone.utc),
        admitted_request={key: value for key, value in _row().items() if key not in {"card_id", "request_sha256"}},
    )
    assert validation.valid, validation.reasons
    governor_receipt = json.loads(result.receipt["canonical_receipt"])
    assert result.receipt["request_digest"] != governor_receipt["request_hash"]
    assert governor_receipt["lease_id"] == result.receipt["lease_id"]
    argv, kwargs = calls[0]
    assert argv[:3] == [sys.executable, str(ASTRAEUS_GOVERNOR_SCRIPT.resolve()), "--db"]
    assert str((tmp_path / "governor.sqlite3").resolve()) == argv[3]
    assert argv[4:6] == ["reserve", _request_id()]
    assert argv[argv.index("--gpu") + 1] == GPU_UUID
    assert argv[argv.index("--min-free-mib") + 1] == "8192"
    assert argv[argv.index("--headroom-mib") + 1] == "1024"
    assert argv[argv.index("--priority") + 1] == "7"
    assert argv[argv.index("--owner") + 1] == "colony-coder"
    assert float(argv[argv.index("--deadline") + 1]) == 1_900_000_000.0
    assert argv[argv.index("--ledger-digest") + 1] == _path_digest(tmp_path / "governor.sqlite3")
    assert argv[argv.index("--lock-root-digest") + 1] == _path_digest(tmp_path)
    assert argv[argv.index("--lock-root") + 1] == str(tmp_path.resolve())
    assert kwargs["timeout"] == SUBPROCESS_TIMEOUT_SECONDS <= 10
    assert kwargs["capture_output"] and kwargs["text"] and not kwargs["check"]


@pytest.mark.parametrize(
    ("returncode", "stdout", "reason"),
    [
        (75, '{"protocol_version":"1","ok":false,"deferred":true}', "governor_busy"),
        (17, "{}", "governor_exit_nonzero"),
        (0, "not-json", "malformed_governor_output"),
        (0, "{\"protocol_version\":\"1\",\"protocol_version\":\"1\"}", "malformed_governor_output"),
    ],
)
def test_subprocess_failure_matrix_defers(
    tmp_path: Path, returncode: int, stdout: str, reason: str
) -> None:
    def runner(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="ignored")

    result = _invoke(tmp_path, runner)
    assert result.status == "deferred_resources"
    assert result.reason == reason
    assert result.receipt is None


def test_timeout_defers_and_never_escapes(tmp_path: Path) -> None:
    def runner(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    result = _invoke(tmp_path, runner)
    assert result.status == "deferred_resources"
    assert result.reason == "governor_timeout"


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"request_hash": "0" * 64}, "governor_request_hash_mismatch"),
        ({"protocol_version": "999"}, "unsupported_governor_protocol"),
    ],
)
def test_success_exit_with_invalid_governor_receipt_defers(
    tmp_path: Path, override: dict[str, Any], reason: str
) -> None:
    def runner(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=_success_json(tmp_path, **override), stderr="")

    result = _invoke(tmp_path, runner)
    assert result.status == "deferred_resources"
    assert result.reason == reason
    assert result.receipt is None


def test_unresolved_config_defers_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env_name in (
        "ASTRAEUS_GPU_GOVERNOR_BIN",
        "ASTRAEUS_GPU_GOVERNOR_LEDGER",
        "ASTRAEUS_GPU_GOVERNOR_LOCK_ROOT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    calls = []
    result = _invoke(
        tmp_path,
        lambda *args, **kwargs: calls.append(args),
        config={"kanban": {"gpu_governor": {}}},
    )
    assert result.status == "deferred_resources"
    assert result.reason == "unresolved_config:bin,ledger,lock_root"
    assert calls == []


def test_config_precedes_environment_and_environment_fills_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "from-env"
    env_dir.mkdir()
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_BIN", str(tmp_path / "env-governor.py"))
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_LEDGER", str(env_dir / "env.db"))
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_LOCK_ROOT", str(env_dir))
    config = _config(tmp_path, bin=str(tmp_path / "config-governor.py"))
    config["kanban"]["gpu_governor"].pop("ledger")
    config["kanban"]["gpu_governor"].pop("lock_root")
    seen: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(argv)
        return SimpleNamespace(returncode=75, stdout="{}", stderr="")

    result = _invoke(tmp_path, runner, config=config)
    assert result.reason == "governor_busy"
    assert seen[0][0:2] == [sys.executable, str((tmp_path / "config-governor.py").resolve())]
    assert seen[0][3] == str((env_dir / "env.db").resolve())


def test_persisted_intent_digest_mismatch_defers_without_subprocess(tmp_path: Path) -> None:
    row = _row()
    row["min_free_mib"] = 16384
    calls: list[Any] = []
    result = _invoke(tmp_path, lambda *args, **kwargs: calls.append(args), row=row)
    assert result.status == "deferred_resources"
    assert result.reason == "adapter_error"
    assert calls == []


def test_auto_selector_and_bad_card_fields_defer(tmp_path: Path) -> None:
    auto_row = _row(selector_type="auto", selector_value="auto")
    result = _invoke(tmp_path, lambda *args, **kwargs: None, row=auto_row)
    assert result.reason == "gpu_selector_requires_exact_uuid"

    result = reserve_gpu_resource(
        _row(), attempt=0, owner="colony-coder", priority=7,
        config=_config(tmp_path), runner=lambda *args, **kwargs: None,
    )
    assert result.reason == "invalid_attempt"


@pytest.mark.skipif(
    not (ASTRAEUS_PYTHON.is_file() and ASTRAEUS_GOVERNOR_SCRIPT.is_file()),
    reason="NOT_RUN: Astraeus checkout or its Python environment is unavailable",
)
def test_real_governor_zero_wait_smoke_returns_75_promptly(tmp_path: Path) -> None:
    ledger = (tmp_path / "smoke.sqlite3").resolve()
    lock_root = tmp_path.resolve()
    args = [
        str(ASTRAEUS_PYTHON), str(ASTRAEUS_GOVERNOR_SCRIPT), "--db", str(ledger),
        "reserve", "hermes-zero-wait-smoke", "--gpu",
        "GPU-00000000-0000-0000-0000-000000000000", "--min-free-mib", "0",
        "--deadline", repr(time.time() - 0.1),
        "--ledger-digest", _path_digest(ledger),
        "--lock-root-digest", _path_digest(lock_root),
        "--lock-root", str(lock_root),
    ]
    started = time.monotonic()
    completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    wall_seconds = time.monotonic() - started
    assert completed.returncode == 75, completed.stdout
    assert wall_seconds < 10
    envelope = json.loads(completed.stdout)
    assert envelope["exit_code"] == 75
    assert envelope["protocol_version"] == "1"


@pytest.mark.skipif(
    not (ASTRAEUS_PYTHON.is_file() and ASTRAEUS_ROOT.joinpath("src/astraeus/colony/gpu_governor.py").is_file()),
    reason="NOT_RUN: Astraeus checkout or its Python environment is unavailable",
)
def test_real_governor_receipt_is_accepted_by_adapter(tmp_path: Path) -> None:
    """Exercise real GovernorLedger/GPUGovernor with a deterministic probe.

    The public adapter sends an already-expired deadline to enforce zero-wait,
    so the real reserve API necessarily rejects it. For this cross-check only,
    the runner invokes the same real API with a short future deadline and a
    temporary ledger; this verifies the v1 request/path digests and receipt
    serialization against Astraeus without touching its checkout.
    """
    config = _config(tmp_path)
    runner_errors: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        db = Path(argv[argv.index("--db") + 1])
        request_id = argv[argv.index("reserve") + 1]
        def arg(name: str) -> str:
            return argv[argv.index(name) + 1]

        script = r'''import json, sys, time
from pathlib import Path
from astraeus.colony.gpu_governor import GPUGovernor, GPUSnapshot, GovernorLedger, ReservationReceipt, serialize_model, _ledger_digest, _lock_root_digest
ledger_path = Path(sys.argv[1])
gpu_uuid = sys.argv[2]
request_id = sys.argv[3]
min_free_mib, headroom_mib, priority = map(int, sys.argv[4:7])
owner = sys.argv[7]
lock_root = Path(sys.argv[8])
clock = time.time
ledger = GovernorLedger(ledger_path, clock=clock)
def probe():
    return (GPUSnapshot(gpu_uuid, 0, 65536, 0, 65536, (), time.time()),)
governor = GPUGovernor(ledger, probe, lock_directory=lock_root, clock=clock)
receipt = governor.reserve(
    request_id=request_id, gpu_uuid=gpu_uuid, min_free_mib=min_free_mib,
    headroom_mib=headroom_mib, priority=priority, owner=owner,
    created_at=time.time(), deadline=time.time() + 120,
    ledger_path=ledger_path, lock_root=lock_root,
    ledger_digest=_ledger_digest(ledger_path),
    lock_root_digest=_lock_root_digest(lock_root),
)
if not isinstance(receipt, ReservationReceipt):
    print(json.dumps({"error": "real governor did not grant", "result": receipt}))
    raise SystemExit(75 if receipt == 75 else 2)
print(json.dumps({"protocol_version": receipt.protocol_version, **serialize_model(receipt)}, sort_keys=True))
'''
        completed = subprocess.run(
            [
                str(ASTRAEUS_PYTHON), "-c", script,
                str(db), arg("--gpu"), request_id,
                arg("--min-free-mib"), arg("--headroom-mib"), arg("--priority"),
                arg("--owner"), str(db.parent),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=ASTRAEUS_ROOT,
        )
        runner_errors.append(completed.stderr)
        return SimpleNamespace(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    result = _invoke(tmp_path, runner, config=config)
    assert result.status == "granted", runner_errors
    assert result.receipt is not None
    assert result.receipt["canonical_receipt"]
    _, admitted_intent_digest = canonicalize_resource_intent(
        {key: value for key, value in _row().items() if key not in {"card_id", "request_sha256"}}
    )
    assert result.receipt["request_digest"] == admitted_intent_digest
    assert validate_resource_receipt(
        result.receipt,
        now=datetime.now(timezone.utc),
        admitted_request={key: value for key, value in _row().items() if key not in {"card_id", "request_sha256"}},
    ).valid
