"""HPD-1785: a hallucinated tool name must never poison a conversation.

Strict Responses backends (openai-codex) reject a replayed
``function_call.name`` outside ``^[a-zA-Z0-9_.-]+\\Z`` with a non-retryable
HTTP 400 ("Invalid 'input[2].name': string does not match pattern").  The
agent used to store the model's raw invalid name in the history, so every
later request failed and the chat was dead for good.  Names are now made
provider-safe when stored AND when a request is built, so an already
poisoned history recovers on its own.
"""

import re
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)
from agent.message_sanitization import (
    INVALID_TOOL_NAME_SENTINEL,
    is_valid_tool_call_name,
    sanitize_tool_call_name,
)

# The exact pattern from the provider's 400 (``\Z``: no trailing newline).
PROVIDER_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+\Z")

BAD_NAMES = [
    "browser click",          # space
    "browser_klick_ä",        # umlaut
    "browser_click\n",        # trailing newline ($ would accept it, \Z does not)
    "browser\nnavigate",      # embedded newline
    "<|tool_call|>browser_snapshot",  # chat-template token
    "functions:browser_click",
    "äöü",                    # nothing usable left
    "   ",                    # blank
    "",
    "x y" * 100,              # overlong and invalid -> capped
]


@pytest.mark.parametrize("name", ["browser_click", "mcp.server.tool", "web-search", "a"])
def test_valid_names_are_unchanged(name):
    assert is_valid_tool_call_name(name)
    assert sanitize_tool_call_name(name) is name


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_names_become_provider_safe(name):
    assert not is_valid_tool_call_name(name)
    out = sanitize_tool_call_name(name)
    assert PROVIDER_PATTERN.match(out), out
    assert len(out) <= 64


def test_specific_normalisations():
    assert sanitize_tool_call_name("browser click") == "browser_click"
    assert sanitize_tool_call_name("browser_click\n") == "browser_click"
    assert sanitize_tool_call_name("äöü") == INVALID_TOOL_NAME_SENTINEL
    assert sanitize_tool_call_name(None) == INVALID_TOOL_NAME_SENTINEL


def _poisoned_history(bad_name):
    """Shape of the dead fc-e2e-2 home chat: compaction summary, then an
    assistant turn whose stored tool call carries an invalid name."""
    return [
        {"role": "user", "content": "[Context summary] earlier Uber Eats run hit a bot block"},
        {"role": "assistant", "content": "Ich versuche es noch einmal."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "call_id": "call_bad",
                    "type": "function",
                    "function": {"name": bad_name, "arguments": "{}"},
                },
                {
                    "id": "call_good",
                    "call_id": "call_good",
                    "type": "function",
                    "function": {"name": "browser_snapshot", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "name": bad_name, "tool_call_id": "call_bad",
         "content": f"Tool '{bad_name}' does not exist."},
        {"role": "tool", "name": "browser_snapshot", "tool_call_id": "call_good", "content": "ok"},
        {"role": "user", "content": "Hallo?"},
    ]


@pytest.mark.parametrize("bad_name", ["browser click", "brówser_click", "browser_click\n", "<|x|>"])
def test_responses_input_recovers_poisoned_history(bad_name):
    items = _chat_messages_to_responses_input(_poisoned_history(bad_name))
    calls = [i for i in items if i.get("type") == "function_call"]
    outputs = [i for i in items if i.get("type") == "function_call_output"]
    assert len(calls) == 2
    for item in calls:
        assert PROVIDER_PATTERN.match(item["name"]), item["name"]
    # The valid sibling is untouched and every output keeps its call.
    assert calls[1]["name"] == "browser_snapshot"
    assert {c["call_id"] for c in calls} == {o["call_id"] for o in outputs}

    # The preflight that runs right before the HTTP request agrees.
    normalized = _preflight_codex_input_items(items)
    for item in normalized:
        if item.get("type") == "function_call":
            assert PROVIDER_PATTERN.match(item["name"]), item["name"]


def test_preflight_normalises_raw_function_call_name():
    out = _preflight_codex_input_items([
        {"type": "function_call", "call_id": "c1", "name": "browser click\n", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "x"},
    ])
    assert out[0]["name"] == "browser_click"


def test_pre_call_sanitizer_repairs_invalid_names_and_keeps_pairing():
    from run_agent import AIAgent

    sanitized = AIAgent._sanitize_api_messages(_poisoned_history("browser klick ä"))
    assistant = next(m for m in sanitized if m.get("tool_calls"))
    names = [tc["function"]["name"] for tc in assistant["tool_calls"]]
    assert names[1] == "browser_snapshot"
    assert PROVIDER_PATTERN.match(names[0])
    tools = [m for m in sanitized if m.get("role") == "tool"]
    assert [t["tool_call_id"] for t in tools] == ["call_bad", "call_good"]
    assert all(PROVIDER_PATTERN.match(t["name"]) for t in tools)


def test_stored_assistant_message_has_provider_safe_name():
    """Store-time end: the assistant turn written to the history (and to
    state.db) never carries the raw invalid name."""
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()

    tc = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="browser click\nä", arguments="{}"),
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    stored = agent._build_assistant_message(msg, "tool_calls")
    stored_name = stored["tool_calls"][0]["function"]["name"]
    assert PROVIDER_PATTERN.match(stored_name), stored_name
    # The live call object is not mutated: dispatch still sees the original
    # name and answers it with "unknown tool" instead of running a tool.
    assert tc.function.name == "browser click\nä"
