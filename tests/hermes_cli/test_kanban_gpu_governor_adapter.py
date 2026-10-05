"""Tests for the versioned Hermes-to-Astraeus GPU governor adapter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
        "python": str(ASTRAEUS_PYTHON),
        "ledger": str(tmp_path / "governor.sqlite3"),
        "lock_root": str(tmp_path),
        "admission_window_seconds": 2.0,
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


def _install_nvidia_smi_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu_uuid: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  --query-gpu=*) printf "%s\\n" "' + gpu_uuid + ',0,65536,0,65536" ;;\n'
        '  --query-compute-apps=*) exit 0 ;;\n'
        '  *) exit 2 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))


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
        "expires_at": time.time() + 1,
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
    assert argv[:3] == [str(ASTRAEUS_PYTHON), str(ASTRAEUS_GOVERNOR_SCRIPT.resolve()), "--db"]
    assert str((tmp_path / "governor.sqlite3").resolve()) == argv[3]
    assert argv[4:6] == ["reserve", _request_id()]
    assert argv[argv.index("--gpu") + 1] == GPU_UUID
    assert argv[argv.index("--min-free-mib") + 1] == "8192"
    assert argv[argv.index("--headroom-mib") + 1] == "1024"
    assert argv[argv.index("--priority") + 1] == "7"
    assert argv[argv.index("--owner") + 1] == "colony-coder"
    assert float(argv[argv.index("--deadline") + 1]) == 1_900_000_002.0
    assert argv[argv.index("--ledger-digest") + 1] == _path_digest(tmp_path / "governor.sqlite3")
    assert argv[argv.index("--lock-root-digest") + 1] == _path_digest(tmp_path)
    assert argv[argv.index("--lock-root") + 1] == str(tmp_path.resolve())
    assert 0 < kwargs["timeout"] <= 2.0
    assert kwargs["timeout"] <= SUBPROCESS_TIMEOUT_SECONDS <= 10
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
        "ASTRAEUS_GPU_GOVERNOR_PYTHON",
        "ASTRAEUS_GPU_GOVERNOR_ADMISSION_WINDOW",
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


def test_python_script_requires_explicit_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ASTRAEUS_GPU_GOVERNOR_PYTHON", raising=False)
    config = _config(tmp_path)
    config["kanban"]["gpu_governor"].pop("python")
    result = _invoke(tmp_path, lambda *args, **kwargs: pytest.fail("unexpected subprocess"), config=config)
    assert result.status == "deferred_resources"
    assert result.reason == "governor_python_unresolved"


def test_config_precedes_environment_and_environment_fills_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "from-env"
    env_dir.mkdir()
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_BIN", str(tmp_path / "env-governor.py"))
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_PYTHON", str(tmp_path / "env-python"))
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_ADMISSION_WINDOW", "4.5")
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_LEDGER", str(env_dir / "env.db"))
    monkeypatch.setenv("ASTRAEUS_GPU_GOVERNOR_LOCK_ROOT", str(env_dir))
    config = _config(tmp_path, bin=str(tmp_path / "config-governor.py"))
    config["kanban"]["gpu_governor"].pop("ledger")
    config["kanban"]["gpu_governor"].pop("lock_root")
    config["kanban"]["gpu_governor"]["python"] = str(tmp_path / "config-python")
    seen: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append((argv, kwargs))
        return SimpleNamespace(returncode=75, stdout="{}", stderr="")

    started = time.time()
    result = _invoke(tmp_path, runner, config=config)
    assert result.reason == "governor_busy"
    argv, kwargs = seen[0]
    assert argv[0:2] == [str((tmp_path / "config-python").resolve()), str((tmp_path / "config-governor.py").resolve())]
    assert argv[3] == str((env_dir / "env.db").resolve())
    assert 2.0 <= float(argv[argv.index("--deadline") + 1]) - started < 2.1
    assert kwargs["timeout"] <= 2.0

    config["kanban"]["gpu_governor"].pop("admission_window_seconds")
    started = time.time()
    _invoke(tmp_path, runner, config=config)
    argv, kwargs = seen[1]
    assert 4.5 <= float(argv[argv.index("--deadline") + 1]) - started < 4.6
    assert kwargs["timeout"] <= 4.5


def test_invalid_admission_window_defers_before_subprocess(tmp_path: Path) -> None:
    result = _invoke(
        tmp_path,
        lambda *args, **kwargs: pytest.fail("unexpected subprocess"),
        config=_config(tmp_path, admission_window_seconds=float("nan")),
    )
    assert result.status == "deferred_resources"
    assert result.reason == "invalid_admission_window"


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
    not (ASTRAEUS_PYTHON.is_file() and ASTRAEUS_ROOT.joinpath("src/astraeus/colony/gpu_governor.py").is_file()),
    reason="NOT_RUN: Astraeus checkout or its Python environment is unavailable",
)
def test_real_governor_grant_is_accepted_by_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the configured real CLI with deterministic telemetry and a temp ledger."""
    gpu_uuid = "GPU-ABCDEF01-2345-6789-ABCD-EF0123456789"
    _install_nvidia_smi_stub(tmp_path, monkeypatch, gpu_uuid)
    row = _row(selector_value=gpu_uuid, min_free_mib=0, headroom_mib=0)
    config = _config(tmp_path, admission_window_seconds=8.0)
    started_monotonic = time.monotonic()
    started = time.time()
    result = reserve_gpu_resource(
        row,
        attempt=3,
        owner="colony-coder",
        priority=7,
        config=config,
    )
    assert result.status == "granted", result.reason
    assert time.monotonic() - started_monotonic < 8.0
    assert result.receipt is not None
    assert result.receipt["canonical_receipt"]
    governor_receipt = json.loads(result.receipt["canonical_receipt"])
    _, admitted_intent_digest = canonicalize_resource_intent({
        key: value for key, value in row.items() if key not in {"card_id", "request_sha256"}
    })
    assert result.receipt["request_digest"] == admitted_intent_digest
    assert governor_receipt["request_hash"] == _governor_hash(
        gpu_uuid=gpu_uuid,
        min_free_mib=0,
        headroom_mib=0,
    )
    assert float(governor_receipt["expires_at"]) <= started + 8.1
    assert validate_resource_receipt(
        result.receipt,
        now=datetime.now(timezone.utc),
        admitted_request={key: value for key, value in row.items() if key not in {"card_id", "request_sha256"}},
    ).valid


@pytest.mark.skipif(
    not (ASTRAEUS_PYTHON.is_file() and ASTRAEUS_ROOT.joinpath("src/astraeus/colony/gpu_governor.py").is_file()),
    reason="NOT_RUN: Astraeus checkout or its Python environment is unavailable",
)
def test_injected_runner_cross_checks_real_governor_api(tmp_path: Path) -> None:
    """Keep the direct GPUGovernor digest/receipt cross-check alongside CLI smoke."""
    config = _config(tmp_path)
    script = """\
import json, sys, time
from pathlib import Path
from astraeus.colony.gpu_governor import GPUGovernor, GPUSnapshot, GovernorLedger, ReservationReceipt, serialize_model

ledger_path = Path(sys.argv[1])
gpu_uuid, request_id = sys.argv[2], sys.argv[3]
min_free_mib, headroom_mib, priority = map(int, sys.argv[4:7])
owner, deadline = sys.argv[7], float(sys.argv[8])
lock_root = Path(sys.argv[9])
ledger_digest, lock_root_digest = sys.argv[10:12]
ledger = GovernorLedger(ledger_path)
def probe():
    return (GPUSnapshot(gpu_uuid, 0, 65536, 0, 65536, (), time.time()),)
governor = GPUGovernor(ledger, probe, lock_directory=lock_root)
receipt = governor.reserve(
    request_id=request_id, gpu_uuid=gpu_uuid, min_free_mib=min_free_mib,
    headroom_mib=headroom_mib, priority=priority, owner=owner,
    created_at=time.time(), deadline=deadline, ledger_path=ledger_path,
    lock_root=lock_root, ledger_digest=ledger_digest,
    lock_root_digest=lock_root_digest,
)
if not isinstance(receipt, ReservationReceipt):
    print(json.dumps({"protocol_version": "1", "exit_code": receipt, "deferred": True}))
    raise SystemExit(receipt if type(receipt) is int else 2)
print(json.dumps({"protocol_version": receipt.protocol_version, **serialize_model(receipt)}, sort_keys=True))
"""

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        def arg(name: str) -> str:
            return argv[argv.index(name) + 1]

        db = Path(arg("--db"))
        completed = subprocess.run(
            [
                str(ASTRAEUS_PYTHON), "-c", script,
                str(db), arg("--gpu"), argv[argv.index("reserve") + 1],
                arg("--min-free-mib"), arg("--headroom-mib"), arg("--priority"),
                arg("--owner"), arg("--deadline"), arg("--lock-root"),
                arg("--ledger-digest"), arg("--lock-root-digest"),
            ],
            capture_output=True,
            text=True,
            timeout=kwargs["timeout"],
            check=False,
            cwd=ASTRAEUS_ROOT,
        )
        return SimpleNamespace(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    result = _invoke(tmp_path, runner, config=config)
    assert result.status == "granted", result.reason
    assert result.receipt is not None
    assert result.receipt["request_digest"] == _row()["request_sha256"]


@pytest.mark.skipif(
    not (ASTRAEUS_PYTHON.is_file() and ASTRAEUS_GOVERNOR_SCRIPT.is_file()),
    reason="NOT_RUN: Astraeus checkout or its Python environment is unavailable",
)
def test_real_governor_ungrantable_request_defers_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_nvidia_smi_stub(tmp_path, monkeypatch, GPU_UUID)
    row = _row(selector_value="GPU-00000000-0000-0000-0000-000000000000", min_free_mib=0)
    started = time.monotonic()
    result = reserve_gpu_resource(
        row,
        attempt=3,
        owner="colony-coder",
        priority=7,
        config=_config(tmp_path),
    )
    assert result.status == "deferred_resources"
    assert result.reason == "governor_busy"
    assert time.monotonic() - started < 2.0
