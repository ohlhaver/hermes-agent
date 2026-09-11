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
    import tools.cronjob_tools as cron_tools

    requirement = "Synthetic request: German technical status only; no summaries."
    task = "Perform the synthetic work and persist its structured result."
    job = {"id": "synthetic-output", "name": "synthetic", "prompt": task, "skills": ["synthetic-report"]}
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
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [
        {"type": "function", "function": {"name": "terminal", "description": "synthetic result writer", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": cron_tools.CRONJOB_SCHEMA},
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda **kw: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    writes = []
    def dispatch(name, args, *a, **kw):
        if name == "cronjob":
            return cron_tools.cronjob(**args)
        writes.append(name)
        return '{"persisted":true}'
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    tool = SimpleNamespace(id="synthetic-write", type="function", function=SimpleNamespace(name="terminal", arguments="{}"))
    def response(content, call=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[call] if call else None), finish_reason="tool_calls" if call else "stop")], model="synthetic", usage=None)
    replies_list = [response("", tool), response("Technisch abgeschlossen.")]
    if manual_override:
        run_call = SimpleNamespace(id="synthetic-run", type="function", function=SimpleNamespace(name="cronjob", arguments=json.dumps({"action": "run", "job_id": job["id"], "prompt": requirement})))
        replies_list = [response("", run_call), *replies_list, response("Technisch abgeschlossen.")]
    replies = iter(replies_list)
    requests = []
    child_results = []
    dispatched_jobs = []
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
            is_child = kwargs.get("platform") == "cron"
            def complete(**request):
                requests.append((is_child, copy.deepcopy(request)))
                return next(replies)
            self.client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(run_agent, "AIAgent", CapturedAgent)
    if manual_override:
        from cron.executions import finish_execution
        monkeypatch.setattr(cron_tools, "resolve_job_ref", lambda *_: job)
        monkeypatch.setattr(cron_tools, "get_job", lambda *_: job)
        monkeypatch.setattr(cron_tools, "claim_job_for_fire", lambda *_: True)
        def run_one(execution_job, **kwargs):
            dispatched_jobs.append(copy.deepcopy(execution_job))
            result = scheduler.run_job(execution_job)
            child_results.append(result)
            finish_execution(execution_job["execution_id"], success=result[0], error=result[3])
            return True
        monkeypatch.setattr(scheduler, "run_one_job", run_one)
        parent = CapturedAgent(model="synthetic", api_key="synthetic", base_url="https://example.invalid/v1", provider="openrouter")
        try:
            parent.run_conversation("Run the saved synthetic job once. " + requirement)
        finally:
            parent.close()
        success, output, final, error = child_results[0]
        assert len(dispatched_jobs) == 1
        assert dispatched_jobs[0]["prompt"].endswith(requirement)
        for is_child, request in requests:
            if not is_child:
                schema = next(t["function"] for t in request["tools"] if t["function"]["name"] == "cronjob")
                assert "verbatim" in schema["description"]
                assert "do not read or process" in schema["description"]
                assert "without inventing restrictions" in schema["parameters"]["properties"]["prompt"]["description"]
    else:
        success, output, final, error = scheduler.run_job(job)
    assert success is True, error
    assert final == "Technisch abgeschlossen."
    assert writes == ["terminal"]
    assert len(requests) == (4 if manual_override else 2)  # Two per agent; no repair retry.
    systems = []
    for is_child, request in requests:
        if not is_child:
            continue
        system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        systems.append(system)
        assert ("Execution-only response contract" in system) is manual_override
        assert requirement not in system  # Never promote free tool arguments.
        users = "\n".join(m["content"] for m in request["messages"] if m["role"] == "user")
        assert task in users
        assert (requirement in users) is manual_override
        if manual_override:
            assert "Output requirements for this execution only" in users
            assert users.rstrip().endswith(requirement)
    assert systems[0] == systems[1]
    assert job == original
