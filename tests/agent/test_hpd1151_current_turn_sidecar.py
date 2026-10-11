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
def _isolated_ratio_store(tmp_path, monkeypatch):
    from agent import turn_context as _tc

    monkeypatch.setattr(_tc, "_native_ratio_store_path", lambda: tmp_path / "native_token_ratios.json")


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


# --- compaction in provider units (HPD-1151) ---
#
# Hermes' preflight and pre-API checks and its tail/summary budgets work in
# chars/4 units. Claude Haiku 5.5's provider count, which the price and the
# 100k long-price line use, was 1.76x that on fc-e2e-2 (estimate 84,184,
# provider 148,151). For the listed models the estimate is scaled by a ratio
# calibrated from the last answered request (seed: the last ratio stored for
# the model, else 1.0), and the tail and
# summary budgets are divided by it.

from agent.turn_context import (  # noqa: E402
    calibrate_native_token_ratio,
    native_token_ratio,
    trigger_pressure_tokens,
)


class _Compressor:
    def __init__(self, ratio=0.0, last=0):
        self.native_token_ratio = ratio
        self.last_prompt_tokens = last


def _ratio_agent(model, models, ratio=0.0, last=0):
    agent = _agent(model, models)
    agent.context_compressor = _Compressor(ratio, last)
    return agent


def test_unlisted_or_default_is_the_bare_estimate():
    assert trigger_pressure_tokens(_ratio_agent("deepseek/deepseek-v4", (HAIKU,), 1.76), 50_000) == 50_000
    assert trigger_pressure_tokens(_ratio_agent(HAIKU, (), 1.76), 50_000) == 50_000
    assert native_token_ratio(_ratio_agent(HAIKU, ())) == 0.0


def test_listed_model_uses_seed_then_calibrated_ratio():
    agent = _ratio_agent(HAIKU, (HAIKU,))
    assert trigger_pressure_tokens(agent, 50_000) == 50_000          # unmeasured model: seed 1.0
    calibrate_native_token_ratio(agent, 84_184, 148_151)            # fc-e2e-2 measurement
    assert round(agent.context_compressor.native_token_ratio, 2) == 1.76
    assert trigger_pressure_tokens(agent, 50_000) == int(50_000 * 148_151 / 84_184)


def test_calibration_is_clamped_and_ignores_bad_readings():
    agent = _ratio_agent(HAIKU, (HAIKU,), 1.5)
    calibrate_native_token_ratio(agent, 10_000, 90_000)
    assert agent.context_compressor.native_token_ratio == 3.0
    calibrate_native_token_ratio(agent, 10_000, 5_000)
    assert agent.context_compressor.native_token_ratio == 1.0
    for rough, prompt in ((0, 90_000), (10_000, 0), (10_000, -1)):
        calibrate_native_token_ratio(agent, rough, prompt)
        assert agent.context_compressor.native_token_ratio == 1.0
    other = _ratio_agent("deepseek/deepseek-v4", (HAIKU,))
    calibrate_native_token_ratio(other, 10_000, 20_000)
    assert other.context_compressor.native_token_ratio == 0.0


def test_dropped_sidecar_is_not_counted_again_by_the_trigger():
    # A stale provider reading that still contained the previous turn's
    # sidecar must not force a compaction: the pressure follows the current
    # message list, not last_prompt_tokens.
    agent = _ratio_agent(HAIKU, (HAIKU,), 1.76, last=150_000)
    assert trigger_pressure_tokens(agent, 40_000) == int(40_000 * 1.76)


def _haiku_compressor(ratio):
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        from agent.context_compressor import ContextCompressor

        c = ContextCompressor(model=HAIKU, threshold_percent=0.08535, protect_first_n=0,
                              protect_last_n=3, summary_target_ratio=0.10, quiet_mode=True)
    c.native_token_ratio = ratio
    return c


def test_tail_and_summary_budgets_shrink_by_the_ratio_only_when_set():
    stock = _haiku_compressor(0.0)
    scaled = _haiku_compressor(1.76)
    # 85,000 with a 4,096 output reservation, 85,350 without one.
    assert 85_000 <= stock.threshold_tokens <= 85_350
    tail = int(stock.threshold_tokens * 0.10)
    assert stock._native_scaled(stock.tail_token_budget) == stock.tail_token_budget == tail
    assert scaled._native_scaled(scaled.tail_token_budget) == int(tail / 1.76)
    turns = [{"role": "user", "content": "x" * 400_000}]
    assert stock._compute_summary_budget(turns) == stock.max_summary_tokens == 10_000
    assert scaled._compute_summary_budget(turns) == int(10_000 / 1.76)


def test_tail_cut_keeps_less_when_scaled():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(40):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": "y" * 4_000})
    stock = _haiku_compressor(0.0)._find_tail_cut_by_tokens(msgs, 1)
    scaled = _haiku_compressor(1.76)._find_tail_cut_by_tokens(msgs, 1)
    assert scaled > stock  # the tail starts later, so it is shorter


def test_preflight_compacts_in_provider_units_for_listed_model():
    from unittest.mock import MagicMock

    compressor = _haiku_compressor(1.76)
    history = []
    # ~55k estimate: below 85k as chars/4, above it (~97k) in provider units.
    for i in range(20):
        history.append({"role": "user", "content": f"q{i} " + "z" * 5_500})
        history.append({"role": "assistant", "content": f"a{i} " + "z" * 5_500})

    def run(models):
        agent = _agent(HAIKU, models)
        agent.compression_enabled = True
        agent.context_compressor = compressor
        compressor.last_prompt_tokens = 0
        agent._compress_context = MagicMock(side_effect=lambda messages, *_a, **_k: (messages[-3:], "SYSTEM"))
        _build(agent, conversation_history=list(history), user_message="next")
        return agent._compress_context.call_count

    assert run(()) == 0
    assert run((HAIKU,)) == 1


def test_turn_start_seeds_the_compressor_so_depth_matches_the_trigger():
    # A fresh agent (restart, resumed session) compacts before any response:
    # the compressor must already shrink its budgets by the same seed.
    compressor = _haiku_compressor(0.0)
    agent = _agent(HAIKU, (HAIKU,))
    agent.context_compressor = compressor
    _build(agent, conversation_history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    # Unmeasured model: seed 1.0 (no scaling); a stored ratio would be used instead.
    assert compressor.native_token_ratio == 1.0
    assert compressor._native_scaled(compressor.tail_token_budget) == compressor.tail_token_budget


def test_turn_start_does_not_seed_unlisted_models():
    compressor = _haiku_compressor(0.0)
    agent = _agent(HAIKU, ())
    agent.context_compressor = compressor
    _build(agent, conversation_history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert compressor.native_token_ratio == 0.0
