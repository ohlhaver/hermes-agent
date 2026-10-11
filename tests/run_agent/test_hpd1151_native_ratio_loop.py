"""HPD-1151: the turn loop calibrates the provider/estimate ratio for listed models."""

from unittest.mock import patch

import pytest

from tests.run_agent.test_413_compression import _mock_response, agent  # noqa: F401  (fixture)

HAIKU = "anthropic/claude-haiku-5.5"


@pytest.fixture(autouse=True)
def _isolated_ratio_store(tmp_path, monkeypatch):
    from agent import turn_context as _tc

    monkeypatch.setattr(_tc, "_native_ratio_store_path", lambda: tmp_path / "native_token_ratios.json")


def _run(agent, models, prompt_tokens):  # noqa: F811
    agent.model = HAIKU
    agent._current_turn_sidecar_models = tuple(models)
    agent.context_compressor.native_token_ratio = 0.0
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="ok", usage={"prompt_tokens": prompt_tokens, "completion_tokens": 3, "total_tokens": prompt_tokens + 3}),
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")
    assert result["completed"] is True
    return agent


def test_listed_model_calibrates_from_its_response(agent):  # noqa: F811
    _run(agent, (HAIKU,), 0)  # a response without usage leaves the turn-start seed
    assert agent.context_compressor.native_token_ratio == 1.0
    _run(agent, (HAIKU,), 1)
    rough = agent._hpd1151_rough_request_tokens
    assert rough > 0
    _run(agent, (HAIKU,), int(rough * 1.76))
    assert abs(agent.context_compressor.native_token_ratio - 1.76) < 0.05


def test_unlisted_model_never_calibrates(agent):  # noqa: F811
    _run(agent, (), 50_000)
    assert agent.context_compressor.native_token_ratio == 0.0


def test_pre_api_check_uses_provider_units_for_listed_model(agent):  # noqa: F811
    # Estimate 60k: below an 85k threshold as chars/4, above it x1.76.
    from unittest.mock import MagicMock

    agent.model = HAIKU
    agent.context_compressor.threshold_tokens = 85_000

    def run(models):
        agent._current_turn_sidecar_models = tuple(models)
        agent.context_compressor.native_token_ratio = 1.76
        agent.context_compressor.last_prompt_tokens = 0
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="ok", usage={"prompt_tokens": 50_000, "completion_tokens": 3, "total_tokens": 50_003}),
        ]
        compress = MagicMock(side_effect=lambda messages, system, **_k: (messages, system))
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 3
        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=60_000),
            patch.object(agent, "_compress_context", compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("hello", conversation_history=history)
        return compress.call_count

    assert run(()) == 0
    assert run((HAIKU,)) >= 1
