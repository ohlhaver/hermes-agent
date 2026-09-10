"""Exercise the real cron agent request boundary with synthetic provider replies."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("manual_override", [False, True])
def test_execution_output_contract_is_fixed_and_stable(monkeypatch, tmp_path, manual_override):
    import cron.scheduler as scheduler
    import run_agent
    import hermes_cli.config
    import hermes_cli.runtime_provider
    import hermes_cli.env_loader
    import tools.mcp_tool
    import tools.skills_tool

    requirement = "Synthetic request: German technical status only; no summaries."
    task = "Perform the synthetic work and persist its structured result."
    job = {"id": "synthetic-output", "name": "synthetic", "prompt": task, "skills": ["synthetic-report"]}
    if manual_override:
        job.update(_execution_output_requirements=True, prompt=task + "\n\nOutput requirements for this execution only:\n" + requirement)
    original = copy.deepcopy(job)
    config = {"model": {"default": "synthetic", "context_length": 256000}}
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(scheduler, "_resolve_origin", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_cli.env_loader, "load_hermes_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_cli.env_loader, "reset_secret_source_cache", lambda: None)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda **kw: config)
    monkeypatch.setattr(hermes_cli.runtime_provider, "resolve_runtime_provider", lambda **kw: {"api_key": "synthetic", "base_url": "https://example.invalid/v1", "provider": "openrouter", "api_mode": "chat_completions"})
    monkeypatch.setattr(tools.mcp_tool, "discover_mcp_tools", lambda: [])
    monkeypatch.setattr(tools.skills_tool, "skill_view", lambda *a, **kw: json.dumps({"success": True, "content": "Report the synthetic findings after writing the structured result."}))
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [{"type": "function", "function": {"name": "terminal", "description": "synthetic result writer", "parameters": {"type": "object", "properties": {}}}}])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda **kw: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    writes = []
    def dispatch(name, args, *a, **kw):
        writes.append(name)
        return '{"persisted":true}'
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    tool = SimpleNamespace(id="synthetic-write", type="function", function=SimpleNamespace(name="terminal", arguments="{}"))
    replies = iter([SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool]), finish_reason="tool_calls")], model="synthetic", usage=None), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Technisch abgeschlossen.", tool_calls=None), finish_reason="stop")], model="synthetic", usage=None)])
    requests = []
    real_agent = run_agent.AIAgent
    class CapturedAgent(real_agent):
        def __init__(self, **kwargs):
            kwargs.update(skip_context_files=True, skip_memory=True, quiet_mode=True)
            super().__init__(**kwargs)
            self.client = MagicMock()
            self._cached_system_prompt = "Synthetic platform instructions."
            self._use_prompt_caching = False
            self.tool_delay = 0
            self.compression_enabled = False
            self.save_trajectories = False
            def complete(**request):
                requests.append(copy.deepcopy(request))
                return next(replies)
            self.client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(run_agent, "AIAgent", CapturedAgent)
    success, output, final, error = scheduler.run_job(job)
    assert success is True, error
    assert final == "Technisch abgeschlossen."
    assert writes == ["terminal"]
    assert len(requests) == 2  # No extra model call or repair retry.
    systems = []
    for request in requests:
        system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        systems.append(system)
        assert ("Execution-only response contract" in system) is manual_override
        assert requirement not in system  # Never promote free tool arguments.
        users = "\n".join(m["content"] for m in request["messages"] if m["role"] == "user")
        assert task in users
        assert (requirement in users) is manual_override
        if manual_override:
            assert "Output requirements for this execution only:" in users
            assert users.rstrip().endswith(requirement)
    assert systems[0] == systems[1]
    assert job == original
