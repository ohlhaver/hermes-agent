"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)


def test_failure_only_turn_halts_identical_errors_and_next_request_can_use_tools():
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    limit = agent._tool_guardrails.config.exact_failure_block_after
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call("web_search", "{}")])
        for _ in range(limit + 1)
    ]
    with patch("run_agent.handle_function_call", return_value='{"error":"missing query"}') as dispatch:
        result = agent.run_conversation("synthetic repeated invalid call")
    assert dispatch.call_count == limit
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["api_calls"] == limit + 1

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call("web_search", '{"query":"corrected"}')]),
        _mock_response(content="New request completed."),
    ]
    with patch("run_agent.handle_function_call", return_value='{"ok":true}') as dispatch:
        followup = agent.run_conversation("synthetic new request")
    dispatch.assert_called_once()
    assert followup["final_response"] == "New request completed."
    assert "guardrail" not in followup


def test_failure_only_turn_allows_a_corrected_call_and_successful_repeated_reads():
    agent = _make_agent("web_search", "read_file", max_iterations=15,
                        config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    calls = [
        _mock_tool_call("web_search", "{}"),
        _mock_tool_call("web_search", '{"query":"corrected"}'),
    ] + [_mock_tool_call("read_file", '{"path":"synthetic"}') for _ in range(7)]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[call]) for call in calls
    ] + [_mock_response(content="Completed after correction.")]
    successful_queries = []
    def handle(name, args, *a, **kw):
        if name == "web_search" and not args:
            return '{"error":"missing query"}'
        if name == "web_search":
            successful_queries.append(args)
            return '{"ok":true}'
        return "same read result"
    with patch("run_agent.handle_function_call", side_effect=handle) as dispatch:
        result = agent.run_conversation("synthetic correction and repeated reads")
    assert successful_queries == [{"query": "corrected"}]
    assert dispatch.call_count == len(calls)
    assert result["final_response"] == "Completed after correction."
    assert "guardrail" not in result


@pytest.mark.parametrize("arguments", ['"not an object"', '[]', '{broken}'])
def test_failure_only_also_bounds_invalid_arguments_before_dispatch(arguments):
    agent = _make_agent("web_search", max_iterations=12,
                        config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    limit = agent._tool_guardrails.config.exact_failure_block_after
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call("web_search", arguments)])
        for _ in range(12)
    ]
    with patch("run_agent.handle_function_call") as dispatch:
        result = agent.run_conversation("synthetic malformed arguments")
    dispatch.assert_not_called()
    assert result["api_calls"] == limit + 1
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"


def test_config_enabled_hard_stop_run_conversation_returns_controlled_guardrail_halt_without_top_level_error():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 2
    assert result["api_calls"] == 3
    assert result["api_calls"] < agent.max_iterations
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "error" not in result
    assert result["completed"] is True
    assert "stopped retrying" in result["final_response"]
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["guardrail"]["tool_name"] == "web_search"

    assistant_tool_calls = [m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    for assistant_msg in assistant_tool_calls:
        call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        following_results = [m for m in result["messages"] if m.get("role") == "tool" and m.get("tool_call_id") in call_ids]
        assert len(following_results) == len(call_ids)


def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


def test_failure_only_invalid_json_does_not_count_valid_same_tool_sibling():
    agent = _make_agent("web_search", max_iterations=15,
                        config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    limit = agent._tool_guardrails.config.same_tool_failure_halt_after
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[
            _mock_tool_call("web_search", '{broken' + str(i) + '}', f"bad-{i}"),
            _mock_tool_call("web_search", '{"query":"valid"}', f"good-{i}"),
        ]) for i in range(limit)
    ] + [_mock_response(content="Stopped safely.")]
    with patch("run_agent.handle_function_call") as dispatch:
        result = agent.run_conversation("synthetic mixed same-tool batch")
    dispatch.assert_not_called()
    assert result["api_calls"] == limit
    assert result["guardrail"]["code"] == "same_tool_failure_halt"
    good = [m for m in result["messages"] if m.get("tool_call_id") == f"good-{limit-1}"]
    assert len(good) == 1 and good[0]["content"].startswith("Skipped:")


@pytest.mark.parametrize("answer", [
    "Ich konnte diesen Schritt nicht abschließen. Die Datei bleibt unverändert.",
    "Der gewünschte Code ist `retry_count = 3`. Der weitere Schritt ist fehlgeschlagen.",
])
def test_failure_only_uses_one_tool_free_localized_completion_and_streams_it(answer):
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    _seed_exact_failures(agent, "web_search", {}, 5)
    decision = agent._tool_guardrails.before_call("web_search", {})
    agent._set_tool_guardrail_halt(decision)
    agent.client.chat.completions.create.return_value = _mock_response(content=answer)
    messages = [{"role":"user", "content":"Bitte antworte auf Deutsch und zeige den gewünschten Code."}]
    deltas = []
    agent.stream_delta_callback = deltas.append
    with patch.object(agent, "_emit_status") as status:
        result = agent._finish_tool_guardrail_halt(messages)
    status.assert_not_called()
    assert result == answer
    assert deltas == [answer, None]
    assert messages.count({"role":"assistant", "content":answer}) == 1
    assert len(messages) == 2  # No synthetic English user instruction in persisted history.
    request = agent.client.chat.completions.create.call_args.kwargs
    agent.client.chat.completions.create.assert_called_once()
    assert "tools" not in request
    assert "conversation's language" in request["messages"][-1]["content"]
    assert "repeated_exact_failure_block" not in result


@pytest.mark.parametrize("summary", ["", RuntimeError("internal-provider-code")])
def test_failure_only_summary_failure_has_no_retry_or_internal_diagnostics(summary):
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    _seed_exact_failures(agent, "web_search", {}, 5)
    agent._set_tool_guardrail_halt(agent._tool_guardrails.before_call("web_search", {}))
    if isinstance(summary, Exception):
        agent.client.chat.completions.create.side_effect = summary
    else:
        agent.client.chat.completions.create.return_value = _mock_response(content=summary)
    result = agent._finish_tool_guardrail_halt([{"role":"user", "content":"Please do the task."}])
    agent.client.chat.completions.create.assert_called_once()
    assert "couldn't complete" in result
    assert "internal-provider-code" not in result
    assert "repeated_exact_failure_block" not in result


def test_failure_only_invalid_json_halt_removes_thinking_prefill():
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {
        "failure_hard_stop_enabled": True, "hard_stop_after": {"same_tool_failure": 1},
    }})
    thinking = _mock_response(content="")
    thinking.choices[0].message.reasoning_content = "Need to search."
    agent.client.chat.completions.create.side_effect = [
        thinking,
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[
            _mock_tool_call("web_search", "{broken}", "broken-call")]),
        _mock_response(content="I could not complete this step."),
    ]
    with patch("run_agent.handle_function_call") as dispatch:
        result = agent.run_conversation("synthetic thinking then malformed tool call")
    dispatch.assert_not_called()
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert not any(m.get("_thinking_prefill") for m in result["messages"])
    pairs = [(a["role"], b["role"]) for a,b in zip(result["messages"], result["messages"][1:])]
    assert ("assistant", "assistant") not in pairs
    assert sum(m.get("tool_call_id") == "broken-call" for m in result["messages"]) == 1


def test_failure_only_config_loads_from_real_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("tool_loop_guardrails:\n  failure_hard_stop_enabled: true\n")
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(api_key="synthetic-test", base_url="https://example.test/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    config = agent._tool_guardrails.config
    assert config.failure_hard_stop_enabled is True
    assert config.hard_stop_enabled is False
    assert config.exact_failure_block_after == 5
    assert config.same_tool_failure_halt_after == 8
    assert agent.max_iterations == 90


def test_failure_only_codex_completion_has_one_request_and_one_visible_answer():
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    agent.api_mode = "codex_responses"
    agent.model = "gpt-5-codex"
    _seed_exact_failures(agent, "web_search", {}, 5)
    agent._set_tool_guardrail_halt(agent._tool_guardrails.before_call("web_search", {}))
    answer = "Ich konnte diesen Schritt nicht abschließen."
    commentary = SimpleNamespace(type="message", phase="commentary", id="commentary",
                                 content=[SimpleNamespace(type="output_text", text="internal commentary")])
    agent.client.responses.create.return_value = iter([
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.added", item=commentary),
        SimpleNamespace(type="response.output_text.delta", delta="internal commentary"),
        SimpleNamespace(type="response.output_item.done", item=commentary),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="internal reasoning"),
        SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="message", phase="final_answer")),
        SimpleNamespace(type="response.output_text.delta", delta=answer),
        SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(type="message", phase="final_answer", content=[SimpleNamespace(type="output_text", text=answer)])),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[
            SimpleNamespace(type="message", phase="final_answer", content=[SimpleNamespace(type="output_text", text=answer)])])),
    ])
    deltas, reasoning, interim = [], [], []
    agent.stream_delta_callback = deltas.append
    agent.reasoning_delta_callback = reasoning.append
    agent.interim_assistant_callback = lambda *a, **k: interim.append(a)
    messages = [{"role":"user", "content":"Bitte hilf mir."}]
    with patch.object(agent, "_emit_status") as status:
        result = agent._finish_tool_guardrail_halt(messages)
    status.assert_not_called()
    assert result == answer
    assert deltas == [answer, None]
    assert not reasoning and not interim
    assert len(messages) == 2
    agent.client.responses.create.assert_called_once()
    assert "tools" not in agent.client.responses.create.call_args.kwargs


def test_failure_only_codex_transport_error_is_not_retried():
    import httpx
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    agent.api_mode = "codex_responses"
    agent.model = "gpt-5-codex"
    _seed_exact_failures(agent, "web_search", {}, 5)
    agent._set_tool_guardrail_halt(agent._tool_guardrails.before_call("web_search", {}))
    agent.client.responses.create.side_effect = httpx.ConnectError("synthetic internal connection detail")
    result = agent._finish_tool_guardrail_halt([{"role":"user", "content":"Please help."}])
    agent.client.responses.create.assert_called_once()
    assert "couldn't complete" in result
    assert "internal connection" not in result


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
@pytest.mark.parametrize("empty", [True, False])
def test_failure_completion_uses_explicit_platform_localized_fallback(mode, empty):
    agent = _make_agent("web_search", config={"tool_loop_guardrails": {"failure_hard_stop_enabled": True}})
    agent.api_mode = mode
    agent.model = "gpt-5-codex" if mode == "codex_responses" else "test/model"
    agent._tool_failure_fallback = "Ich konnte diesen Schritt nicht abschließen. Du kannst eine neue Anfrage senden."
    _seed_exact_failures(agent, "web_search", {}, 5)
    agent._set_tool_guardrail_halt(agent._tool_guardrails.before_call("web_search", {}))
    create = agent.client.responses.create if mode == "codex_responses" else agent.client.chat.completions.create
    if empty:
        create.return_value = SimpleNamespace(output=[], status="completed") if mode == "codex_responses" else _mock_response(content="")
    else:
        create.side_effect = RuntimeError("synthetic failure")
    messages = [{"role":"user", "content":"Bitte hilf mir."}]
    result = agent._finish_tool_guardrail_halt(messages)
    assert result == agent._tool_failure_fallback
    assert len(messages) == 2
    create.assert_called_once()
