"""Tests: retirement recognition in the agent conversation loop.

Dispatcher-owned workers retire only after a SUCCESSFUL terminal lifecycle tool
result (kanban_complete / kanban_block / kanban_request_review /
kanban_request_changes / reclaim / schedule, plus legacy tool-name aliases)
for the worker's own task. A failed (error-carrying) result does NOT retire the
worker — the model may retry.  The loop stops further model sampling on a
successful retirement and enters ordinary bounded finalization; it never signals,
kills, pauses or relaunches the prior process.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

from agent.conversation_loop import (
    _LIFECYCLE_RETIREMENT_KEY,
    _TERMINAL_LIFECYCLE_TOOLS,
    _successful_terminal_lifecycle_tool_result,
    _tool_batch_retired,
)
from agent.delegation_context import non_dispatcher_owned_context
from agent.turn_tool_round import run_tool_round
from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_tool_call
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG


# ── _successful_terminal_lifecycle_tool_result ─────────────────────────────


def test_accepts_kanban_complete_success():
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_complete", {"ok": True})
        is True
    )
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_complete", {"ok": True, "summary": "done"}
        )
        is True
    )


def test_accepts_kanban_block_success():
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_block", {"ok": True, "reason": "blocked"}
        )
        is True
    )


def test_accepts_kanban_request_review_success():
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_request_review", {"ok": True, "summary": "review"}
        )
        is True
    )


def test_accepts_kanban_request_changes_success():
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_request_changes", {"ok": True, "reason": "changes"}
        )
        is True
    )


def test_accepts_request_review_alias_success():
    # Legacy / emitted tool-name aliases use hyphenated forms in some transports.
    assert (
        _successful_terminal_lifecycle_tool_result("request-review", {"ok": True})
        is True
    )
    assert (
        _successful_terminal_lifecycle_tool_result("request-changes", {"ok": True})
        is True
    )


def test_accepts_schedule_success_via_success_field():
    # Scheduling tools report {"success": true} rather than {"ok": true}.
    assert (
        _successful_terminal_lifecycle_tool_result("schedule", {"success": True})
        is True
    )


def test_accepts_reclaim_success():
    assert _successful_terminal_lifecycle_tool_result("reclaim", {"ok": True}) is True


def test_accepts_cronjob_manage_success():
    # Cron maintenance never closes the worker's card, even on success.
    assert (
        _successful_terminal_lifecycle_tool_result("cronjob_manage", {"success": True})
        is False
    )
    assert (
        _successful_terminal_lifecycle_tool_result(
            "cronjob_manage", {"success": True, "count": 0, "jobs": []}
        )
        is False
    )


def test_cron_list_tool_batch_does_not_retire_worker():
    call = {
        "id": "cron-list",
        "function": {"name": "cronjob_manage", "arguments": '{"action":"list"}'},
    }
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": "cron-list",
            "content": '{"success": true, "count": 0, "jobs": []}',
        },
    ]
    fake = type("M", (), {"tool_calls": [call]})()
    assert _tool_batch_retired(messages, fake) is False


def test_worker_only_retires_on_its_own_task_handoff():
    call = {"id": "complete", "function": {"name": "kanban_complete"}}
    fake = type("M", (), {"tool_calls": [call]})()
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": "complete",
            "content": '{"ok": true, "task_id": "another"}',
        },
    ]
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is False
    messages[1]["content"] = '{"ok": true, "task_id": "mine"}'
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is True
    messages[1]["content"] = '{"ok": true}'
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is False


def test_rejects_tool_error():
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_complete", {"error": "something"}
        )
        is False
    )
    assert (
        _successful_terminal_lifecycle_tool_result(
            "kanban_block", {"error": "blocked", "ok": True}
        )
        is False
    )


def test_rejects_non_terminal_tool():
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_comment", {"ok": True})
        is False
    )
    assert (
        _successful_terminal_lifecycle_tool_result("file_write", {"ok": True}) is False
    )
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_heartbeat", {"ok": True})
        is False
    )


def test_rejects_unknown_tool():
    assert (
        _successful_terminal_lifecycle_tool_result("some_unknown_tool", {"ok": True})
        is False
    )


def test_string_content_json_parsed():
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_complete", '{"ok": true}')
        is True
    )
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_complete", '{"ok": false}')
        is False
    )


def test_malformed_string_content_not_retired():
    assert (
        _successful_terminal_lifecycle_tool_result("kanban_complete", "not json")
        is False
    )


def test_non_dict_content_after_json_not_retired():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", "[]") is False
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", "42") is False


def test_none_content_not_retired():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", None) is False


# ── _tool_batch_retired ────────────────────────────────────────────────────


def test_empty_messages_not_retired():
    fake = type("M", (), {"tool_calls": []})()
    assert _tool_batch_retired([], fake) is False


def test_no_assistant_tool_call_message_not_retired():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    fake = type("M", (), {"tool_calls": []})()
    assert _tool_batch_retired(messages, fake) is False


def test_retires_single_tool_success():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "kanban_complete",
            "content": '{"ok": true}',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_retires_scheduled_success_via_success_field():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "schedule", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "schedule",
            "content": '{"success": true}',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_fails_on_tool_error_not_retired():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "kanban_complete",
            "content": '{"error": "fatal"',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_non_terminal_tool_result_not_retired():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "kanban_comment",
            "content": "ok",
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_batched_mixed_tools_retires_on_terminal_success():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "kanban_comment",
            "content": "ok",
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "kanban_complete",
            "content": '{"ok": true}',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_tool_call_id_resolves_name():
    # name lookup prefers tool_call_id match over positional fallback.
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "name": "kanban_complete",
            "content": '{"ok": true}',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_single_tool_falls_back_to_names_when_no_id():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "name": "kanban_complete", "content": '{"ok": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_multiple_tools_without_ids_uses_last_assistant_name():
    # When there is no tool_call_id linkage and multiple tool calls were made,
    # the function refuses to guess the matching name (no retirement).
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                },
                {
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "name": "kanban_complete", "content": '{"ok": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_unlinked_result_id_cannot_retire_a_multi_tool_batch():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "comment", "function": {"name": "kanban_comment"}},
                {"id": "complete", "function": {"name": "kanban_complete"}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "different",
            "name": "kanban_complete",
            "content": '{"ok": true}',
        },
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_tool_result_after_assistant_message_respected():
    # Only results after the newest assistant-with-tool-calls message count; an
    # older tool result from a prior turn does not retire.
    messages = [
        {"role": "assistant", "content": "step 1", "tool_calls": []},
        {"role": "tool", "name": "kanban_complete", "content": '{"ok": true}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "kanban_comment",
            "content": "ok",
        },
    ]
    fake = type("M", (), {"tool_calls": messages[-2]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


# ── _TERMINAL_LIFECYCLE_TOOLS coverage ────────────────────────────────────


def test_terminal_tools_set_contains_core_kanban_tools():
    assert "kanban_complete" in _TERMINAL_LIFECYCLE_TOOLS
    assert "kanban_block" in _TERMINAL_LIFECYCLE_TOOLS
    assert "kanban_request_review" in _TERMINAL_LIFECYCLE_TOOLS
    assert "kanban_request_changes" in _TERMINAL_LIFECYCLE_TOOLS
    assert "kanban_reclaim" in _TERMINAL_LIFECYCLE_TOOLS
    assert "schedule" in _TERMINAL_LIFECYCLE_TOOLS
    assert "reclaim" in _TERMINAL_LIFECYCLE_TOOLS


def test_terminal_tools_set_contains_aliases():
    assert "complete" in _TERMINAL_LIFECYCLE_TOOLS
    assert "block" in _TERMINAL_LIFECYCLE_TOOLS
    assert "request-review" in _TERMINAL_LIFECYCLE_TOOLS
    assert "request-changes" in _TERMINAL_LIFECYCLE_TOOLS


def test_terminal_tools_set_contains_cronjob_manage():
    assert "cronjob_manage" not in _TERMINAL_LIFECYCLE_TOOLS


@pytest.mark.parametrize("max_iterations", [1, 10])
@pytest.mark.parametrize("micro_enabled", [False, True])
def test_terminal_handoff_skips_pressure_compaction_and_next_model_call(
    monkeypatch, max_iterations, micro_enabled
):
    """A real conversation tool round must stop before post-tool LLM compaction."""
    tool_def = {
        "type": "function",
        "function": {
            "name": "kanban_complete",
            "description": "Complete a task",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_def]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch(
            "agent.context_compressor.get_model_context_length", return_value=256_000
        ),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=max_iterations,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.max_compression_attempts = 1
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names.add("skill_manage")
    agent._spawn_background_review = MagicMock()
    agent._deferred_title_upgrade = MagicMock(ident=None)
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = (
        90_000  # preflight 33 tokens is safe; post-tool 100k is not
    )
    compressor.context_length = 256_000
    compressor.last_prompt_tokens = 0
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.side_effect = lambda tokens: tokens >= 90_000
    compressor.should_compress_info.side_effect = lambda tokens: (
        tokens >= 90_000,
        "threshold" if tokens >= 90_000 else None,
    )
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""
    compressor._micro_compact_enabled = micro_enabled
    compressor._micro_compact.side_effect = AssertionError(
        "micro-compaction sampled after handoff"
    )
    agent.compression_enabled = True
    agent.context_compressor = compressor

    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="kanban_complete", arguments="{}"),
    )
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[call],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=100_000,
            completion_tokens=1,
            total_tokens=100_001,
        ),
    )
    agent.client.chat.completions.create.side_effect = [
        response,
        AssertionError("model sampled after handoff"),
    ]
    agent._compress_context = MagicMock(
        side_effect=AssertionError("compression sampled after handoff")
    )

    def fake_execute(assistant_message, messages, *_args):
        messages.append({
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": assistant_message.tool_calls[0].id,
            "content": '{"ok": true, "task_id": "work-card"}',
        })

    agent._execute_tool_calls = fake_execute
    monkeypatch.setenv("HERMES_KANBAN_TASK", "work-card")
    with (
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.title_generator.start_title_upgrade") as start_title_upgrade,
    ):
        result = agent.run_conversation("finish worker task")
    assert agent.client.chat.completions.create.call_count == 1
    assert agent._compress_context.call_count == 0
    compressor._micro_compact.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    start_title_upgrade.assert_not_called()
    assert result["completed"] is True
    assert result["final_response"] == "Worker task handoff recorded."
    assert result.get("turn_exit_reason") == "terminal_lifecycle_tool"
    assert result["messages"][-1]["role"] == "assistant"


# ── Oversized lifecycle results: spillover must not swallow the handoff ────
#
# An oversized tool result is replaced by a ``<persisted-output>`` preview BEFORE
# ``_tool_batch_retired`` reads it (``maybe_persist_tool_result`` per result, and
# ``enforce_turn_budget`` for the whole turn). ``kanban_complete`` returns the entire
# attachment set, so a real handoff with deliverables is exactly the shape that spills —
# and the worker then kept sampling the model instead of retiring.


def _lifecycle_result(task_id: str, *, ok: bool = True, pad_chars: int = 0) -> str:
    """A kanban-shaped success payload, optionally padded past the 100K spill threshold."""
    payload = {
        "ok": ok,
        "task_id": task_id,
        "summary": "done",
        "attachments": [{"path": f"/tmp/artifact-{i}.bin", "size": 4096} for i in range(3)],
    }
    if pad_chars:
        payload["metadata"] = "m" * pad_chars
    return json.dumps(payload)


def _run_sequential(agent, result: str, *, name: str = "kanban_complete", call_id: str = "call_1"):
    """Dispatch ONE real tool call through the production sequential path.

    Mirrors the real turn: the assistant(tool_calls) row is appended BEFORE dispatch,
    which is the anchor ``_tool_batch_retired`` scans back to.
    """
    tool_calls = [_mock_tool_call(name=name, call_id=call_id)]
    messages: list = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    with patch("model_tools.handle_function_call", return_value=result):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    fake = type("M", (), {"tool_calls": tool_calls})()
    return messages, fake


@pytest.fixture
def spill_home(tmp_path, monkeypatch):
    """Isolated result directory: real persistence, never the live board/home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path, raising=False)
    return tmp_path


@pytest.mark.parametrize("tool_name", ["kanban_complete", "kanban_block", "kanban_request_review"])
def test_oversized_current_task_handoff_still_retires(spill_home, tool_name):
    """RED->GREEN: a spilled success for the worker's OWN task must still retire."""
    messages, fake = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", pad_chars=120_000), name=tool_name
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    # The canonical spill really happened — that is the precondition for this regression.
    assert PERSISTED_OUTPUT_TAG in tool_msg["content"]
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is True


def test_oversized_handoff_for_another_task_does_not_retire(spill_home):
    """Spillover must not become an ownership bypass."""
    messages, fake = _run_sequential(
        _make_agent(), _lifecycle_result("other-card", pad_chars=120_000)
    )
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_oversized_failed_handoff_does_not_retire(spill_home):
    """A spilled FAILURE must stay a failure — the model still gets to retry."""
    messages, fake = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", ok=False, pad_chars=120_000)
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    assert PERSISTED_OUTPUT_TAG in tool_msg["content"]
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_turn_budget_spillover_also_preserves_retirement(spill_home, monkeypatch):
    """The AGGREGATE spill site is independent of the per-result one and must be covered.

    Every result stays UNDER the 100K per-result threshold, so only
    ``enforce_turn_budget`` (``_finalize_tool_batch``) can stub it. Three 90K results
    against a 150K turn budget keep the aggregate over budget after the first spill, so
    the handoff's own row is stubbed too.
    """
    agent = _make_agent()
    monkeypatch.setattr(
        "agent.tool_executor._budget_for_agent",
        lambda _agent: BudgetConfig(default_result_size=100_000, turn_budget=150_000),
    )
    tool_calls = [
        _mock_tool_call(name="kanban_heartbeat", call_id="c1"),
        _mock_tool_call(name="kanban_complete", call_id="c2"),
        _mock_tool_call(name="kanban_heartbeat", call_id="c3"),
    ]
    messages: list = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    results = {
        "kanban_heartbeat": _lifecycle_result("work-card", pad_chars=90_000),
        "kanban_complete": _lifecycle_result("work-card", pad_chars=90_000),
    }
    with patch("model_tools.handle_function_call", side_effect=lambda n, a, *rest, **kw: results[n]):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    complete_msg = next(m for m in messages if m.get("tool_call_id") == "c2")
    assert PERSISTED_OUTPUT_TAG in complete_msg["content"]  # aggregate spill fired
    fake = type("M", (), {"tool_calls": tool_calls})()
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is True


def test_non_lifecycle_tool_with_oversized_success_payload_does_not_retire(spill_home):
    """Tool NAME still gates retirement: a big ``ok: true`` from another tool is not a handoff."""
    messages, fake = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", pad_chars=120_000), name="kanban_comment"
    )
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_uncommitted_result_does_not_retire_when_flush_fails(spill_home, monkeypatch):
    """Durability fence: a result that never reached the session DB must not retire.

    The fence is the ORDER in ``run_tool_round``: a failed canonical append sets
    ``_incremental_persistence_failed`` and breaks the turn BEFORE the retirement check
    runs. An uncommitted result therefore must not end the turn as a handoff, however
    readable its in-memory content still is.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "work-card")
    agent = MagicMock()
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.valid_tool_names = {"kanban_complete"}
    agent.stream_delta_callback = None
    agent._tool_guardrail_halt_decision = None
    agent._incremental_persistence_failed = False
    agent._flush_messages_to_session_db.return_value = True
    agent._cap_delegate_task_calls.side_effect = lambda calls: calls
    agent._deduplicate_tool_calls.side_effect = lambda calls: calls

    call = _mock_tool_call(name="kanban_complete", call_id="handoff")
    assistant = SimpleNamespace(tool_calls=[call], content=None)
    messages: list = []

    def execute(_message, rows, *_args):
        # A committed-looking result that never made it to the session DB.
        agent._incremental_persistence_failed = True
        rows.append({
            "role": "assistant", "tool_calls": [call],
        })
        rows.append({
            "role": "tool", "tool_call_id": "handoff",
            "content": '{"ok": true, "task_id": "work-card"}',
        })

    agent._execute_tool_calls.side_effect = execute
    post = SimpleNamespace(
        messages=messages, active_system_prompt="system", conversation_history=None,
        compression_attempts=0, final_response=None, turn_exit_reason="unknown",
        current_turn_user_idx=0, end_turn=False,
    )
    kwargs = dict(
        assistant_message=assistant, finish_reason="tool_calls", messages=messages,
        conversation_history=None, api_call_count=1, effective_task_id=None,
        user_message="task", system_message="system", active_system_prompt="system",
        compression_attempts=0, max_compression_attempts=1, final_response=None,
        failed=False, _turn_exit_reason="unknown", truncated_tool_call_retries=0,
        current_turn_user_idx=0,
    )
    with (
        patch("agent.turn_tool_round.validate_tool_calls", return_value=SimpleNamespace(
            action="proceed", mixed_invalid_batch=False
        )),
        patch("agent.turn_tool_round.stage_tool_call_message", return_value=(
            {"role": "assistant", "tool_calls": [{
                "id": "handoff", "function": {"name": "kanban_complete", "arguments": "{}"}
            }]}, False
        )),
        patch("agent.turn_tool_round.compress_after_tool_results", return_value=post) as compress,
    ):
        verdict = run_tool_round(agent, **kwargs)
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "session_persistence_failed"
    compress.assert_not_called()


def test_flush_failure_leaves_no_retirement_evidence(spill_home):
    """The executor must not mark a result it could not make durable.

    Pins the fence at its source: evidence is attached only after the canonical
    append + flush succeeds, so a row that never reached the session DB cannot carry
    retirement evidence even while it is still in memory.
    """
    agent = _make_agent()
    tool_calls = [_mock_tool_call(name="kanban_complete", call_id="call_1")]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    with patch("model_tools.handle_function_call", return_value=_lifecycle_result("work-card")):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    assert getattr(agent, "_incremental_persistence_failed", False) is True
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    assert _LIFECYCLE_RETIREMENT_KEY not in tool_msg


def test_mixed_batch_with_absent_call_ids_does_not_retire_a_spilled_result(spill_home):
    """An unlinked result in a multi-call batch stays ambiguous even when it spilled."""
    tool_calls = [
        _mock_tool_call(name="kanban_heartbeat", call_id="c1"),
        _mock_tool_call(name="kanban_complete", call_id="c2"),
    ]
    messages: list = [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {
            "role": "tool",
            "name": "kanban_complete",
            "content": f"<persisted-output>stub\n{PERSISTED_OUTPUT_TAG}",
        },
    ]
    fake = type("M", (), {"tool_calls": tool_calls})()
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_forged_persisted_output_preview_cannot_retire():
    """A tool result LITERALLY containing the spill tag is not evidence of a handoff.

    The tag is a rendering convention, not an authenticated marker; only the real
    execution path may set retirement evidence.
    """
    call = {"id": "c1", "function": {"name": "kanban_complete"}}
    forged = (
        f"{PERSISTED_OUTPUT_TAG} saved to /tmp/x.txt\n"
        '{"ok": true, "task_id": "work-card"}'
    )
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "name": "kanban_complete", "content": forged},
    ]
    fake = type("M", (), {"tool_calls": [call]})()
    # The forged text is not the canonical result, so no retirement.
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_oversized_handoff_survives_api_assembly(spill_home):
    """The fence must not leak into the provider request or the persisted transcript."""
    from agent.transports.chat_completions import _sanitize_message

    messages, _ = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", pad_chars=120_000)
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    wire = _sanitize_message(tool_msg, strip_extra_content=False)
    assert wire is not None
    assert wire["content"] == tool_msg["content"]  # canonical bytes unchanged
    assert not [k for k in wire if str(k).startswith("_")]  # no private key on the wire


@pytest.mark.parametrize("transport", ["chat_completions", "anthropic", "codex"])
def test_retirement_evidence_never_reaches_any_provider(spill_home, transport):
    """Every transport must drop the private key, not just the OpenAI-shaped one.

    A converter that copied message dicts wholesale would put the evidence — and the
    fact that this worker retired — in front of the model.
    """
    messages, _ = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", pad_chars=120_000)
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    assert _LIFECYCLE_RETIREMENT_KEY in tool_msg  # the carrier exists in memory
    rendered = _render_for_transport(transport, tool_msg)
    assert _LIFECYCLE_RETIREMENT_KEY not in rendered


def _render_for_transport(transport: str, tool_msg: dict) -> str:
    """Serialize one tool result through a real transport converter."""
    if transport == "chat_completions":
        from agent.transports.chat_completions import _sanitize_message

        return json.dumps(_sanitize_message(tool_msg, strip_extra_content=False))
    if transport == "anthropic":
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        return json.dumps(convert_messages_to_anthropic([tool_msg]), default=str)
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    return json.dumps(_chat_messages_to_responses_input([tool_msg]), default=str)


# ── Independent review counterexamples (exact tip ffa9498) ───────────────────
#
# An external read-only review of 64b8595..ffa9498 returned REQUEST_CHANGES with
# four executable probes. Each is pinned here as a contract, not a snapshot.

# A real kanban id shape (``t_`` + 8 hex, hermes_cli/kanban_db.py::_new_task_id).
CARD = "t_731b564c"


def test_contradictory_ok_true_success_false_does_not_retire():
    """A stale ``ok: true`` beside an explicit ``success: false`` is a REJECTION.

    Both fields are real result contracts (Kanban uses ``ok``, scheduling uses
    ``success``), so the pair is contradictory rather than merely absent. A single
    true field must not outvote the other reporting failure.
    """
    assert _successful_terminal_lifecycle_tool_result(
        "kanban_complete",
        {"ok": True, "success": False, "message": "rejected", "task_id": "work-card"},
    ) is False
    # The mirror image is equally a contradiction.
    assert _successful_terminal_lifecycle_tool_result(
        "kanban_complete", {"ok": False, "success": True},
    ) is False
    # An uncontradicted single-true field is still the documented contract.
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", {"ok": True}) is True
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", {"success": True}) is True


def test_contradictory_fields_do_not_retire_through_executor_evidence(spill_home):
    """The evidence path is a second reader of the same contradiction.

    ``_tool_batch_retired`` re-runs the predicate against the executor's evidence when
    the content is a spill preview, so the veto must hold on BOTH paths.
    """
    agent = _make_agent()
    result = json.dumps({
        "ok": True, "success": False, "message": "rejected", "task_id": "work-card",
    })
    messages, fake = _run_sequential(agent, result, call_id="call_1")
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    # The raw (unspilled) result is contradictory, so no retirement.
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False
    assert _successful_terminal_lifecycle_tool_result(
        "kanban_complete", tool_msg.get(_LIFECYCLE_RETIREMENT_KEY) or {},
    ) is False


def test_error_bearing_payload_withholds_evidence(spill_home):
    """A REAL refusal must not stamp evidence, whatever the executor's verdict.

    ``ok: true`` plus a real ``error``/``user_summary`` is exactly the shape a
    partially-applied or denied handoff produces. The authoritative signal is the
    parsed payload, so both must veto on the evidence path.
    """
    from agent.tool_executor import _lifecycle_handoff_evidence

    for payload in (
        {"ok": True, "error": "board write failed", "task_id": "work-card"},
        {"ok": True, "user_summary": "approval denied", "task_id": "work-card"},
    ):
        raw = json.dumps(payload)
        evidence = _lifecycle_handoff_evidence("kanban_complete", raw)
        assert evidence is None or evidence.get("error") is True
        assert _successful_terminal_lifecycle_tool_result("kanban_complete", evidence or {}) is False
        assert _successful_terminal_lifecycle_tool_result("kanban_complete", raw) is False

    # Through the real commit path: a refusal never retires. The evidence may still be
    # present — it carries the bounded ``error`` veto marker — but that marker can only
    # ever BLOCK retirement, so the contract is the verdict, not the key's presence.
    agent = _make_agent()
    messages, fake = _run_sequential(
        agent,
        json.dumps({"ok": True, "user_summary": "approval denied", "task_id": "work-card"}),
        call_id="call_1",
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    evidence = tool_msg.get(_LIFECYCLE_RETIREMENT_KEY)
    if evidence is not None:
        assert evidence.get("error") is True
        assert _successful_terminal_lifecycle_tool_result("kanban_complete", evidence) is False
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_attachment_named_error_does_not_strand_a_completed_worker(spill_home):
    """REGRESSION GUARD: the substring heuristic must not veto a genuine handoff.

    ``_detect_tool_failure`` classifies any result whose first 500 chars contain the
    token ``"error"``/``"failed"`` as a failure. A real ``kanban_complete`` returns its
    whole attachment set, so an ordinary file named ``error`` lands inside that window.
    Gating retirement on that verdict stranded a genuinely completed worker — the exact
    failure the fence exists to prevent. Retirement is decided on the parsed payload.
    """
    def att(i, name):
        return {
            "id": i, "filename": name, "content_type": "application/octet-stream",
            "size": 4096, "uploaded_by": "agent",
            # A realistic stored_path is long, and 500 of them push the payload past
            # the 100K per-result spill threshold — the shape this fence exists for.
            "stored_path": f"/home/u/.hermes/kanban/attachments/{CARD}/blob-{i:04d}-deliverable",
        }

    result = json.dumps({
        "ok": True, "task_id": CARD, "run_id": 7,
        "attachments": [att(1, "error")] + [att(i, f"p{i}.bin") for i in range(2, 500)],
    })
    assert len(result) > 100_000, f"payload must cross the spill threshold, got {len(result)}"
    # Precondition: the payload really does trip the substring heuristic...
    from agent.display import _detect_tool_failure

    is_error, _ = _detect_tool_failure("kanban_complete", result)
    assert is_error is True
    # ...and really does spill, so the content path is dead and only evidence remains.
    messages, fake = _run_sequential(_make_agent(), result)
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    assert PERSISTED_OUTPUT_TAG in tool_msg["content"]
    # ...yet the worker must still retire.
    assert _tool_batch_retired(messages, fake, worker_task_id=CARD) is True


def test_composite_bridge_id_pairs_with_canonical_result_id():
    """A Responses bridge publishes ``call_1|fc_1``; the result carries ``call_1``.

    Pairing on the raw id string alone missed this legitimate handoff, so a worker
    that had genuinely completed kept sampling the model. The canonical composite
    half is the pairing key (same convention as ``coalesce_tool_call_id``).
    """
    call = SimpleNamespace(
        id="call_1|fc_1", type="function",
        function=SimpleNamespace(name="kanban_complete", arguments="{}"),
    )
    assistant = SimpleNamespace(tool_calls=[call], content=None)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool", "tool_call_id": "call_1",
            "content": json.dumps({"ok": True, "task_id": "work-card"}),
        },
    ]
    assert _tool_batch_retired(messages, assistant, worker_task_id="work-card") is True


def test_duplicate_call_ids_in_one_batch_are_ambiguous_and_do_not_retire():
    """Two calls claiming one id cannot be attributed by name.

    ``uniquify_tool_call_ids`` repairs this upstream, but a batch that reaches the
    loop still ambiguous must not retire on a guess.
    """
    calls = [
        {"id": "dup", "function": {"name": "kanban_complete", "arguments": "{}"}},
        {"id": "dup", "function": {"name": "kanban_heartbeat", "arguments": "{}"}},
    ]
    fake = type("M", (), {"tool_calls": calls})()
    messages = [
        {"role": "assistant", "content": "", "tool_calls": calls},
        {
            "role": "tool", "tool_call_id": "dup",
            "content": json.dumps({"ok": True, "task_id": "work-card"}),
        },
    ]
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is False


def test_flush_returning_none_is_not_durable():
    """``None`` from the flush means NOTHING WAS WRITTEN.

    ``_flush_messages_to_session_db_unlocked`` returns ``None`` on the
    no-session-DB / persistence-disabled paths. Reading that as durable stamped
    retirement evidence with no DB row behind it. The two verdicts are deliberately
    SEPARATE: ``None`` must not stop a batch (an agent with no DB still has to work),
    but it must never be read as a durable write.
    """
    from agent.tool_executor import _flush_session_db_after_tool_progress

    agent = SimpleNamespace(
        _flush_messages_to_session_db=lambda _messages: None,
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
    )
    # Not a batch-stopping failure: non-lifecycle dispatch must continue.
    assert _flush_session_db_after_tool_progress(agent, [], stage="probe") is True
    assert agent._incremental_persistence_failed is False
    # ...but nothing was written, so it is not durable evidence.
    assert agent._last_flush_wrote_row is False

    # An explicit True is both non-failing and durable.
    ok_agent = SimpleNamespace(
        _flush_messages_to_session_db=lambda _messages: True,
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
    )
    assert _flush_session_db_after_tool_progress(ok_agent, [], stage="probe") is True
    assert ok_agent._incremental_persistence_failed is False
    assert ok_agent._last_flush_wrote_row is True

    # An explicit False is a real write failure: batch stops, nothing durable.
    fail_agent = SimpleNamespace(
        _flush_messages_to_session_db=lambda _messages: False,
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
    )
    assert _flush_session_db_after_tool_progress(fail_agent, [], stage="probe") is False
    assert fail_agent._incremental_persistence_failed is True
    assert fail_agent._last_flush_wrote_row is False


def test_flush_exception_is_not_durable():
    """A raising flush is a failure, and never a durable write."""
    from agent.tool_executor import _flush_session_db_after_tool_progress

    def boom(_messages):
        raise RuntimeError("disk gone")

    agent = SimpleNamespace(
        _flush_messages_to_session_db=boom,
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
    )
    assert _flush_session_db_after_tool_progress(agent, [], stage="probe") is False
    assert agent._incremental_persistence_failed is True
    assert agent._last_flush_wrote_row is False


def test_no_session_db_agent_does_not_retire(tmp_path, monkeypatch):
    """Real agent, real persistence-disabled path: no DB row, so no retirement.

    Deliberately NOT a mock return value — this drives the real
    ``_flush_messages_to_session_db`` (which returns ``None`` for a disabled or absent
    session DB) through the production commit path.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _make_agent()
    agent._session_db = None
    agent._persist_disabled = True
    tool_calls = [_mock_tool_call(name="kanban_complete", call_id="call_1")]
    messages: list = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    with patch("model_tools.handle_function_call", return_value=_lifecycle_result("work-card")):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    # Nothing durable was written, so the executor must not CLAIM durability...
    assert _LIFECYCLE_RETIREMENT_KEY not in tool_msg
    assert getattr(agent, "_last_flush_wrote_row", False) is not True
    # ...and no batch-stopping failure is invented: an agent with nowhere to persist
    # must still be able to run its turn.
    assert getattr(agent, "_incremental_persistence_failed", False) is False

    # The unspilled content is still readable, and the loop's CONTENT path retires from
    # it. That is pre-existing behavior (verified identical at the base commit) and is
    # deliberately NOT gated on the session DB: the kanban_complete already mutated the
    # board, so the handoff really did happen and the worker really should stop. Gating
    # this on a DB row would strand a genuinely completed worker in a sampling loop —
    # the exact failure the retirement fence exists to prevent. The fix is therefore
    # scoped to the durability CLAIM (the sidecar), not to the content verdict.
    fake = type("M", (), {"tool_calls": tool_calls})()
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is True


def test_real_session_db_row_carries_no_evidence_key(tmp_path, monkeypatch, spill_home):
    """A real SQLite session DB must not receive the private key.

    Uses a genuine ``SessionDB`` (not a mock) so the assertion covers the real
    row-building path rather than a stubbed return value.
    """
    from tests.agent.test_tool_call_incremental_persistence import _attach_real_session_db

    agent = _make_agent()
    _attach_real_session_db(agent, tmp_path / "state.db", "retirement-evidence-probe")
    # Dict-shaped tool calls: the real flush JSON-serializes the assistant row, so an
    # SDK-object stub would fail serialization and exercise the wrong branch.
    call = {"id": "call_1", "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"}}
    messages: list = [{"role": "assistant", "content": "", "tool_calls": [call]}]
    assistant_message = SimpleNamespace(
        content="", tool_calls=[SimpleNamespace(
            id="call_1", type="function",
            function=SimpleNamespace(name="kanban_complete", arguments="{}"),
        )]
    )
    with patch("model_tools.handle_function_call", return_value=_lifecycle_result("work-card")):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    # Durable now, so evidence IS present in memory...
    assert getattr(agent, "_incremental_persistence_failed", False) is False
    assert _LIFECYCLE_RETIREMENT_KEY in tool_msg
    # ...and the real row builder drops the private key while keeping canonical content.
    from agent.session_persistence import _db_flush_row

    row = _db_flush_row(agent, tool_msg, False)
    assert _LIFECYCLE_RETIREMENT_KEY not in row
    assert row["content"] == tool_msg["content"]  # canonical content unchanged
    # The row really landed in SQLite.
    from tests.agent.test_tool_call_incremental_persistence import _durable_roles

    assert "tool" in _durable_roles(tmp_path / "state.db", "retirement-evidence-probe")


def test_oversized_task_id_is_not_carried_as_evidence():
    """A 120K-char ``task_id`` is a body, not an id.

    The sidecar exists to survive spillover, so an unbounded field would defeat the
    bound it was introduced to provide.
    """
    from agent.tool_executor import _lifecycle_handoff_evidence

    huge = json.dumps({"ok": True, "task_id": "t" * 120_000})
    evidence = _lifecycle_handoff_evidence("kanban_complete", huge)
    assert evidence is None or "task_id" not in evidence
    assert len(json.dumps(evidence or {})) < 1000
    # A non-boolean verdict is malformed, not an acceptance, and must not ride along.
    assert _lifecycle_handoff_evidence(
        "kanban_complete", json.dumps({"ok": "yes" * 50_000})
    ) is None
    # The documented short id still survives.
    assert _lifecycle_handoff_evidence(
        "kanban_complete", json.dumps({"ok": True, "task_id": "t_731b564c"})
    ) == {"ok": True, "task_id": "t_731b564c"}


def test_oversized_result_still_retires_valid_own_card_handoff(spill_home):
    """The bound must not cost the original fix: a truly oversized result still retires.

    Guards against over-correcting into "reject anything big" — the spillover bug
    this fence exists for was precisely a large genuine handoff failing to retire.
    """
    messages, fake = _run_sequential(
        _make_agent(), _lifecycle_result("work-card", pad_chars=120_000)
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    assert PERSISTED_OUTPUT_TAG in tool_msg["content"]
    assert _tool_batch_retired(messages, fake, worker_task_id="work-card") is True


def test_bounded_evidence_never_reaches_session_db_or_provider(spill_home):
    """Bounded AND private: the sidecar reaches neither the DB row nor the wire."""
    from agent.session_persistence import _db_flush_row
    from agent.transports.chat_completions import _sanitize_message

    agent = _make_agent()
    messages, _ = _run_sequential(
        agent, _lifecycle_result("work-card", pad_chars=120_000)
    )
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    evidence = tool_msg[_LIFECYCLE_RETIREMENT_KEY]
    assert len(json.dumps(evidence)) < 1000
    assert _LIFECYCLE_RETIREMENT_KEY not in _sanitize_message(tool_msg, strip_extra_content=False)
    assert _LIFECYCLE_RETIREMENT_KEY not in _db_flush_row(agent, tool_msg, False)
