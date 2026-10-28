"""HPD-1808: a direct call of a deferred (Tool Search) tool runs through tool_call.

Measured on fc-justus-3 on 2026-10-28: the model called
``mcp__basic_memory__search_notes`` by its exact name, which Tool Search had
deferred behind the bridge, and was told "Tool ... does not exist".
"""
import json
from types import SimpleNamespace

import pytest

DEFERRED = "mcp__basic_memory__search_notes"


def _tool_call(name: str, arguments):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeChatCompletions:
    def __init__(self, name, arguments):
        self.calls = 0
        self.requests = []
        self.name, self.arguments = name, arguments

    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        if self.calls == 1:
            assistant = SimpleNamespace(
                content=None, reasoning=None,
                tool_calls=[_tool_call(self.name, self.arguments)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=assistant, finish_reason="tool_calls")], usage=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", reasoning=None, tool_calls=[]),
                finish_reason="stop")],
            usage=None,
        )


def _run(monkeypatch, name, arguments, scoped=frozenset({DEFERRED})):
    from run_agent import AIAgent

    completions = _FakeChatCompletions(name, arguments)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [
            {"function": {"name": "tool_search"}},
            {"function": {"name": "tool_call"}},
        ],
    )
    monkeypatch.setattr("agent.tool_executor._tool_search_scoped_names", lambda agent: scoped)
    dispatched = []

    def _handle(name, args, task_id=None, **kwargs):
        # The sequential path hands the bridge call to handle_function_call,
        # which unwraps it (model_tools); the concurrent path unwraps first.
        # (no MCP server is registered here, so the bridge is read directly).
        if name == "tool_call":
            name, args = args["name"], args["arguments"]
        dispatched.append((name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr("run_agent.handle_function_call", _handle)
    agent = AIAgent(
        model="test-model", api_key="test-key", base_url="http://localhost:8080/v1",
        platform="cli", max_iterations=3, quiet_mode=True, skip_memory=True,
    )
    agent._disable_streaming = True
    result = agent.run_conversation("search my notes")
    tool_messages = [m for m in result["messages"] if m.get("role") == "tool"]
    return dispatched, tool_messages, result


def test_direct_call_of_deferred_tool_runs(monkeypatch):
    dispatched, tool_messages, _ = _run(monkeypatch, DEFERRED, json.dumps({"query": "Aman"}))
    assert dispatched == [(DEFERRED, {"query": "Aman"})]
    assert tool_messages and "does not exist" not in tool_messages[0]["content"]


def test_direct_call_with_dict_arguments(monkeypatch):
    dispatched, _, _ = _run(monkeypatch, DEFERRED, {"query": "Aman"})
    assert dispatched == [(DEFERRED, {"query": "Aman"})]


def test_out_of_scope_name_still_does_not_exist(monkeypatch):
    dispatched, tool_messages, _ = _run(
        monkeypatch, "mcp__other__secret_tool", json.dumps({}), scoped=frozenset({DEFERRED}))
    assert dispatched == []
    assert "does not exist" in tool_messages[0]["content"]


@pytest.mark.parametrize("raw,expected", [
    ('{"query": "x"}', {"query": "x"}),
    ("", {}),
    (None, {}),
    ({"query": "x"}, {"query": "x"}),
    ("{broken", "{broken"),
])
def test_route_helper_wraps_arguments(monkeypatch, raw, expected):
    from agent.conversation_loop import _route_direct_deferred_tool_call

    monkeypatch.setattr("agent.tool_executor._tool_search_scoped_names", lambda agent: frozenset({DEFERRED}))
    agent = SimpleNamespace(valid_tool_names={"tool_search", "tool_call"})
    tc = _tool_call(DEFERRED, raw)
    assert _route_direct_deferred_tool_call(agent, tc) is True
    assert tc.function.name == "tool_call"
    assert json.loads(tc.function.arguments) == {"name": DEFERRED, "arguments": expected}


def test_route_helper_needs_the_bridge(monkeypatch):
    from agent.conversation_loop import _route_direct_deferred_tool_call

    monkeypatch.setattr("agent.tool_executor._tool_search_scoped_names", lambda agent: frozenset({DEFERRED}))
    agent = SimpleNamespace(valid_tool_names={"read_file"})
    tc = _tool_call(DEFERRED, "{}")
    assert _route_direct_deferred_tool_call(agent, tc) is False
    assert tc.function.name == DEFERRED
