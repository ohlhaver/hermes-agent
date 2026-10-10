"""HPD-1151 step 2: only the current user message carries its api_content.

``compression.current_turn_sidecar_models`` (default empty) names the models
for which older user rows stop replaying and counting their ``api_content``
sidecar. Only the current turn's user message keeps it, for every call of the
turn. Default: stock Hermes behaviour, byte for byte.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import current_turn_sidecar_models_from_config
from tests.agent.test_turn_context import _FakeAgent, _build

HAIKU = "anthropic/claude-haiku-5.5"
SIDECAR = "older question\n\n" + ("recalled memory context " * 400)


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


class _FakeMemory:
    def on_turn_start(self, *_a, **_k):
        pass

    def prefetch_all(self, _query):
        return "fresh memory for this turn"


def _history():
    return [
        {"role": "user", "content": "older question", "api_content": SIDECAR},
        {"role": "assistant", "content": "older answer", "api_content": "older answer (sent bytes)"},
        {"role": "user", "content": "second question", "api_content": "second question + memory"},
        {"role": "assistant", "content": "second answer"},
    ]


def _agent(model: str, models=()):
    agent = _FakeAgent()
    agent.model = model
    agent._current_turn_sidecar_models = tuple(models)
    agent._memory_manager = _FakeMemory()
    return agent


def test_default_keeps_every_sidecar():
    agent = _agent(HAIKU)
    ctx = _build(agent, conversation_history=_history())
    assert ctx.messages[0]["api_content"] == SIDECAR
    assert ctx.messages[2]["api_content"] == "second question + memory"


def test_listed_model_drops_older_user_sidecars_only():
    agent = _agent(HAIKU, (HAIKU,))
    ctx = _build(agent, conversation_history=_history(), user_message="current question")
    assert "api_content" not in ctx.messages[0]
    assert "api_content" not in ctx.messages[2]
    # Assistant sanitize-divergence sidecars are untouched.
    assert ctx.messages[1]["api_content"] == "older answer (sent bytes)"
    # The current turn's user message carries its injection for the turn.
    current = ctx.messages[ctx.current_turn_user_idx]
    assert current["content"] == "current question"
    assert "fresh memory for this turn" in current["api_content"]


def test_unlisted_model_is_unchanged():
    agent = _agent("deepseek/deepseek-v4", (HAIKU,))
    ctx = _build(agent, conversation_history=_history())
    assert ctx.messages[0]["api_content"] == SIDECAR


def test_size_estimate_no_longer_counts_older_sidecars():
    before = estimate_messages_tokens_rough(_history())
    agent = _agent(HAIKU, (HAIKU,))
    ctx = _build(agent, conversation_history=_history(), user_message="q")
    older = ctx.messages[: ctx.current_turn_user_idx]
    assert estimate_messages_tokens_rough(older) < before - 2000


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        ("", ()),
        ([], ()),
        ([HAIKU], (HAIKU,)),
        (["  Anthropic/Claude-Haiku-5.5 ", "", 7], (HAIKU,)),
        ("anthropic/claude-haiku-5.5", (HAIKU,)),
        ({"x": 1}, ()),
    ],
)
def test_config_parsing(value, expected):
    assert current_turn_sidecar_models_from_config({"current_turn_sidecar_models": value}) == expected
    assert current_turn_sidecar_models_from_config({}) == ()
    assert current_turn_sidecar_models_from_config(None) == ()
