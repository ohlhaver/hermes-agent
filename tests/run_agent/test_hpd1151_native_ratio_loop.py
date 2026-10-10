"""HPD-1151: the turn loop calibrates the provider/estimate ratio for listed models."""

from unittest.mock import patch

from tests.run_agent.test_413_compression import _mock_response, agent  # noqa: F401  (fixture)

HAIKU = "anthropic/claude-haiku-5.5"


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
    _run(agent, (HAIKU,), 0)  # a response without usage leaves it uncalibrated
    assert agent.context_compressor.native_token_ratio == 0.0
    _run(agent, (HAIKU,), 1)
    rough = agent._hpd1151_rough_request_tokens
    assert rough > 0
    _run(agent, (HAIKU,), int(rough * 1.76))
    assert abs(agent.context_compressor.native_token_ratio - 1.76) < 0.05


def test_unlisted_model_never_calibrates(agent):  # noqa: F811
    _run(agent, (), 50_000)
    assert agent.context_compressor.native_token_ratio == 0.0
