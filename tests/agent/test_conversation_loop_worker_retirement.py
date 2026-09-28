"""Tests: retirement recognition in the agent conversation loop.

Dispatcher-owned workers retire only after a SUCCESSFUL terminal lifecycle tool
result (kanban_complete / kanban_block / kanban_request_review /
kanban_request_changes / reclaim / schedule, plus the legacy tool-name aliases
and cronjob_manage).  A failed (error-carrying) result does NOT retire the
worker — the model may retry.  The loop stops further model sampling on a
successful retirement and enters ordinary bounded finalization; it never signals,
kills, pauses or relaunches the prior process.
"""

from __future__ import annotations

from agent.conversation_loop import (
    _TERMINAL_LIFECYCLE_TOOLS,
    _successful_terminal_lifecycle_tool_result,
    _tool_batch_retired,
)


# ── _successful_terminal_lifecycle_tool_result ─────────────────────────────


def test_accepts_kanban_complete_success():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", {"ok": True}) is True
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", {"ok": True, "summary": "done"}) is True


def test_accepts_kanban_block_success():
    assert _successful_terminal_lifecycle_tool_result("kanban_block", {"ok": True, "reason": "blocked"}) is True


def test_accepts_kanban_request_review_success():
    assert _successful_terminal_lifecycle_tool_result("kanban_request_review", {"ok": True, "summary": "review"}) is True


def test_accepts_kanban_request_changes_success():
    assert _successful_terminal_lifecycle_tool_result("kanban_request_changes", {"ok": True, "reason": "changes"}) is True


def test_accepts_request_review_alias_success():
    # Legacy / emitted tool-name aliases use hyphenated forms in some transports.
    assert _successful_terminal_lifecycle_tool_result("request-review", {"ok": True}) is True
    assert _successful_terminal_lifecycle_tool_result("request-changes", {"ok": True}) is True


def test_accepts_schedule_success_via_success_field():
    # Scheduling tools report {"success": true} rather than {"ok": true}.
    assert _successful_terminal_lifecycle_tool_result("schedule", {"success": True}) is True


def test_accepts_reclaim_success():
    assert _successful_terminal_lifecycle_tool_result("reclaim", {"ok": True}) is True


def test_accepts_cronjob_manage_success():
    # Cron maintenance never closes the worker's card, even on success.
    assert _successful_terminal_lifecycle_tool_result("cronjob_manage", {"success": True}) is False
    assert _successful_terminal_lifecycle_tool_result("cronjob_manage", {"success": True, "count": 0, "jobs": []}) is False


def test_cron_list_tool_batch_does_not_retire_worker():
    call = {"id": "cron-list", "function": {"name": "cronjob_manage", "arguments": '{"action":"list"}'}}
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "cron-list", "content": '{"success": true, "count": 0, "jobs": []}'},
    ]
    fake = type("M", (), {"tool_calls": [call]})()
    assert _tool_batch_retired(messages, fake) is False


def test_worker_only_retires_on_its_own_task_handoff():
    call = {"id": "complete", "function": {"name": "kanban_complete"}}
    fake = type("M", (), {"tool_calls": [call]})()
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "complete", "content": '{"ok": true, "task_id": "another"}'},
    ]
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is False
    messages[1]["content"] = '{"ok": true, "task_id": "mine"}'
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is True
    messages[1]["content"] = '{"ok": true}'
    assert _tool_batch_retired(messages, fake, worker_task_id="mine") is False


def test_rejects_tool_error():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", {"error": "something"}) is False
    assert _successful_terminal_lifecycle_tool_result("kanban_block", {"error": "blocked", "ok": True}) is False


def test_rejects_non_terminal_tool():
    assert _successful_terminal_lifecycle_tool_result("kanban_comment", {"ok": True}) is False
    assert _successful_terminal_lifecycle_tool_result("file_write", {"ok": True}) is False
    assert _successful_terminal_lifecycle_tool_result("kanban_heartbeat", {"ok": True}) is False


def test_rejects_unknown_tool():
    assert _successful_terminal_lifecycle_tool_result("some_unknown_tool", {"ok": True}) is False


def test_string_content_json_parsed():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", '{"ok": true}') is True
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", '{"ok": false}') is False


def test_malformed_string_content_not_retired():
    assert _successful_terminal_lifecycle_tool_result("kanban_complete", "not json") is False


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
                {"id": "call_1", "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "kanban_complete", "content": '{"ok": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_retires_scheduled_success_via_success_field():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "schedule", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "schedule", "content": '{"success": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_fails_on_tool_error_not_retired():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "kanban_complete", "content": '{"error": "fatal"'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_non_terminal_tool_result_not_retired():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "kanban_comment", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "kanban_comment", "content": "ok"},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_batched_mixed_tools_retires_on_terminal_success():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "kanban_comment", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "kanban_comment", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_2", "name": "kanban_complete", "content": '{"ok": true}'},
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
                {"id": "call_a", "type": "function", "function": {"name": "kanban_comment", "arguments": "{}"}},
                {"id": "call_b", "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_b", "name": "kanban_complete", "content": '{"ok": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is True


def test_single_tool_falls_back_to_names_when_no_id():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
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
                {"type": "function", "function": {"name": "kanban_comment", "arguments": "{}"}},
                {"type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "name": "kanban_complete", "content": '{"ok": true}'},
    ]
    fake = type("M", (), {"tool_calls": messages[0]["tool_calls"]})()
    assert _tool_batch_retired(messages, fake) is False


def test_unlinked_result_id_cannot_retire_a_multi_tool_batch():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "comment", "function": {"name": "kanban_comment"}},
            {"id": "complete", "function": {"name": "kanban_complete"}},
        ]},
        {"role": "tool", "tool_call_id": "different", "name": "kanban_complete", "content": '{"ok": true}'},
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
                {"id": "c1", "type": "function", "function": {"name": "kanban_comment", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "kanban_comment", "content": "ok"},
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
