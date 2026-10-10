"""HPD-1151: the two compression settings reach the agent from config.yaml."""

from unittest.mock import patch

import pytest


def _agent(compression):
    cfg = {"compression": {"enabled": True, **compression}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=1_000_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-haiku-5.5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_defaults_are_off():
    agent = _agent({})
    assert agent._current_turn_sidecar_models == ()
    assert agent.context_compressor.restart_summary_probe is False


@pytest.mark.parametrize("flag", [True, "true"])
def test_settings_are_read(flag):
    agent = _agent({
        "current_turn_sidecar_models": ["anthropic/claude-haiku-5.5"],
        "restart_summary_probe": flag,
    })
    assert agent._current_turn_sidecar_models == ("anthropic/claude-haiku-5.5",)
    assert agent.context_compressor.restart_summary_probe is True
