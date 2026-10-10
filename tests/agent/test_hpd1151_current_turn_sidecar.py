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


# --- the trigger reads the provider's native count (HPD-1151, step 2b) ---
#
# Hermes' preflight and pre-API checks use estimate_request_tokens_rough
# (chars/4). For Claude Haiku 5.5 the provider-reported (native) prompt was
# about 2x that estimate on fc-e2e-2, so an 85k trigger on the estimate fired
# near 170k native, far above the 100k long-price line. For the listed models
# the last provider-reported prompt counts as a floor of the pressure.

from agent.turn_context import trigger_pressure_tokens  # noqa: E402


class _Compressor:
    def __init__(self, last):
        self.last_prompt_tokens = last


def _pressure_agent(model, models, last):
    agent = _agent(model, models)
    agent.context_compressor = _Compressor(last)
    return agent


def test_pressure_uses_native_usage_for_listed_model():
    agent = _pressure_agent(HAIKU, (HAIKU,), 90_000)
    assert trigger_pressure_tokens(agent, 40_000) == 90_000
    assert trigger_pressure_tokens(agent, 120_000) == 120_000


def test_pressure_unchanged_for_unlisted_model_or_default():
    assert trigger_pressure_tokens(_pressure_agent("deepseek/deepseek-v4", (HAIKU,), 90_000), 40_000) == 40_000
    assert trigger_pressure_tokens(_pressure_agent(HAIKU, (), 90_000), 40_000) == 40_000


def test_pressure_ignores_post_compaction_sentinel_and_missing_usage():
    # -1 = compaction just ran, no provider reading yet (#36718); 0 = no usage.
    assert trigger_pressure_tokens(_pressure_agent(HAIKU, (HAIKU,), -1), 40_000) == 40_000
    assert trigger_pressure_tokens(_pressure_agent(HAIKU, (HAIKU,), 0), 40_000) == 40_000


def test_preflight_compacts_on_native_usage_for_listed_model():
    from unittest.mock import MagicMock

    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(model=HAIKU, threshold_percent=0.08535, protect_first_n=0, protect_last_n=3, quiet_mode=True)
    compressor.last_prompt_tokens = 90_000
    compressor.last_real_prompt_tokens = 90_000
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"short question {i}"})
        history.append({"role": "assistant", "content": f"short answer {i}"})

    def run(models):
        agent = _agent(HAIKU, models)
        agent.compression_enabled = True
        agent.context_compressor = compressor
        agent._compress_context = MagicMock(side_effect=lambda messages, *_a, **_k: (messages[-3:], "SYSTEM"))
        _build(agent, conversation_history=list(history), user_message="next")
        return agent._compress_context.call_count

    assert run(()) == 0          # stock: the tiny estimate never reaches 85k
    assert run((HAIKU,)) == 1    # listed: the 90k native reading does
