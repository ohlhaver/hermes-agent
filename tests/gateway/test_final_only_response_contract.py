"""Final-only visibility is disclosed in the real model request, without retries."""
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import _make_runner, MetadataEditProgressCaptureAdapter


def response(content, tool=None):
    call = SimpleNamespace(id=f"call-{tool}", type="function", function=SimpleNamespace(name=tool, arguments="{}")) if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[call] if call else None), finish_reason="tool_calls" if tool else "stop")], model="synthetic", usage=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("interim", [False, True])
@pytest.mark.parametrize("complete_final", [False, True])
async def test_final_only_contract_reaches_requests_after_housekeeping(monkeypatch, tmp_path, interim, complete_final):
    import run_agent
    import gateway.run as gateway_run
    import hermes_cli.config
    import tools.memory_tool

    # Same output shape as the native incident, with synthetic text only:
    # terminal -> 1325-character report + memory -> 233-character reference.
    report = ("Statusbericht: Scanner und Ranker sind abgeschlossen. " + "x" * 1325)[:1325]
    reference = ("Die vorherige Antwort enthält den Statusbericht. Memory-Zeichenlimit. " + "x" * 233)[:233]
    final = report + '\n```python\nprint(56)\n```\n"The answer is 56."' if complete_final else reference
    replies = iter([response("", "terminal"), response(report, "memory"), response(final)])
    requests = []
    dispatches = []
    config = {"model": {"default": "synthetic", "context_length": 256000}, "display": {"tool_progress": "off", "interim_assistant_messages": interim}}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "synthetic", "base_url": "https://example.invalid/v1"})
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [
        {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}
        for name in ["terminal", "memory"]
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda **kwargs: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    def dispatch(name, args, *a, **kw):
        dispatches.append(name)
        return '{"error":"synthetic memory character limit"}' if name == "memory" else '{"status":"complete"}'
    monkeypatch.setattr(run_agent, "handle_function_call", dispatch)
    monkeypatch.setattr(tools.memory_tool, "memory_tool", lambda **kwargs: dispatch("memory", {}))
    real_agent = run_agent.AIAgent
    class CapturedAgent(real_agent):
        def __init__(self, **kwargs):
            kwargs.update(skip_context_files=True, skip_memory=True, quiet_mode=True)
            super().__init__(**kwargs)
            agent = self
            agent.client = MagicMock()
            agent._cached_system_prompt = "Synthetic platform instructions."
            agent._use_prompt_caching = False
            agent.tool_delay = 0
            agent.compression_enabled = False
            agent.save_trajectories = False
            def complete(**request):
                requests.append(copy.deepcopy(request))
                return next(replies)
            agent.client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(run_agent, "AIAgent", CapturedAgent)
    adapter = MetadataEditProgressCaptureAdapter()
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig.from_dict({"enabled": False})
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = await runner._run_agent(message="Bitte liefere den vollständigen Statusbericht.", context_prompt="Existing channel context.", history=[], source=SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic", chat_type="dm"), session_id="synthetic-session", session_key="synthetic-key")
    assert dispatches == ["terminal", "memory"]
    assert len(requests) == 3  # No additional model call or repair retry.
    for request in requests:
        system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        assert "Existing channel context." in system
        assert ("Only your final response is delivered" in system) is (not interim)
        if not interim:
            assert "complete requested answer" in system
            assert "code and quotations" in system
            assert "housekeeping" in system
    # Transport cannot infer whether a model complied: never concatenate hidden
    # text or turn a reference into a claimed complete answer.
    assert result["final_response"] == final
    if not interim:
        assert adapter.sent == []
        assert adapter.edits == []
