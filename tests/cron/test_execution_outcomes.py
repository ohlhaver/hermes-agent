"""Real middleware -> conversation -> cron storage, without provider execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cron import executions, jobs, scheduler
from hermes_cli.middleware import LLMExecutionOutcome


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    return executions


@pytest.fixture
def native(monkeypatch):
    import run_agent
    from hermes_cli.plugins import get_plugin_manager

    provider = Mock(side_effect=AssertionError("unexpected provider dispatch"))
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent.AIAgent, "_interruptible_api_call", provider)
    monkeypatch.setattr(run_agent.AIAgent, "_interruptible_streaming_api_call", provider)
    monkeypatch.setattr(run_agent.AIAgent, "_has_stream_consumers", lambda self: False)
    # No optional model calls for compression/memory/context discovery.
    class OfflineAgent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.update(skip_context_files=True, skip_memory=True,
                          max_iterations=3, quiet_mode=True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(run_agent, "AIAgent", OfflineAgent)
    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_middleware", {})
    return OfflineAgent, manager, provider


def _install(manager, status, *, wrong_session=False):
    calls = []
    def handoff(**kw):
        calls.append(kw["session_id"])
        return LLMExecutionOutcome(
            status, "handoff " + status,
            "unrelated-session" if wrong_session else kw["session_id"],
        )
    manager._middleware["llm_execution"] = [handoff]
    return calls


@pytest.mark.parametrize("api_mode,provider", [("chat_completions", "openrouter"),
                                               ("codex_responses", "openai-codex")])
@pytest.mark.parametrize("status", ["completed", "failed", "unknown"])
def test_actual_conversation_outcomes(native, api_mode, provider, status):
    Agent, manager, dispatch = native
    calls = _install(manager, status)
    agent = Agent(model="test-model", provider=provider, api_mode=api_mode,
                  api_key="offline-test-key", base_url="http://127.0.0.1:1/v1",
                  session_id="cron-test-attempt", platform="cron")
    try:
        result = agent.run_conversation("offline task")
    finally:
        agent.close()
    assert result["execution_outcome"] == status
    assert result["completed"] is (status == "completed")
    assert result["failed"] is (status == "failed")
    assert result["final_response"] == "handoff " + status
    assert calls == ["cron-test-attempt"]
    dispatch.assert_not_called()


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
@pytest.mark.parametrize("status", ["completed", "failed", "unknown"])
def test_real_scheduler_and_direct_tool_preserve_outcomes(native, ledger, monkeypatch, status, api_mode):
    _, manager, dispatch = native
    calls = _install(manager, status)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openai-codex" if api_mode == "codex_responses" else "openrouter", "api_mode": api_mode,
        "api_key": "offline-test-key", "base_url": "http://127.0.0.1:1/v1",
    })
    delivered = []
    monkeypatch.setattr(scheduler, "_deliver_result", lambda job, content, **kw: delivered.append(content))
    job = jobs.create_job("offline task", "every 1h", name="offline", model="test-model")
    from tools.cronjob_tools import cronjob
    result = json.loads(cronjob(action="run", job_id=job["id"]))["job"]
    assert result["execution_outcome"] == status
    assert result["execution_success"] is (None if status == "unknown" else status == "completed")
    persisted = jobs.get_job(job["id"])
    assert persisted["last_status"] == {"completed": "ok", "failed": "error", "unknown": "unknown"}[status]
    assert persisted["fire_claim"] is None
    assert persisted["next_run_at"]
    assert len(ledger.list_executions(job_id=job["id"])) == 1
    assert ledger.latest_execution(job["id"])["status"] == status
    assert len(calls) == 1
    assert len(delivered) == 1
    if status == "unknown":
        assert delivered == ["handoff unknown"]
    dispatch.assert_not_called()


def test_misbound_outcome_never_leaks_other_session_or_dispatches(native):
    Agent, manager, dispatch = native
    _install(manager, "completed", wrong_session=True)
    agent = Agent(model="test-model", provider="openrouter", api_mode="chat_completions",
                  api_key="offline-test-key", base_url="http://127.0.0.1:1/v1",
                  session_id="current-attempt", platform="cron")
    try:
        result = agent.run_conversation("offline task")
    finally:
        agent.close()
    assert result["execution_outcome"] == "unknown"
    assert result["completed"] is False
    assert result["failed"] is False
    assert "handoff" not in result["final_response"]
    dispatch.assert_not_called()


@pytest.mark.parametrize("success,outcome,expected", [
    (True, None, "completed"), (False, None, "failed"),
    (None, "unknown", "unknown"), (True, "completed", "completed"),
    (False, "failed", "failed"),
])
def test_storage_compatibility_and_terminal_immutability(ledger, success, outcome, expected):
    row = ledger.create_execution("offline", source="direct")
    final = ledger.finish_execution(row["id"], success=success, outcome=outcome, error="detail")
    assert final["status"] == expected
    assert ledger.finish_execution(row["id"], success=True) is None
    assert ledger.get_execution(row["id"]) == final


@pytest.mark.parametrize("success,outcome", [(True, "unknown"), (False, "unknown"),
                                             (True, "failed"), (False, "completed"),
                                             (None, "pending"), (None, None)])
def test_contradictory_inputs_do_not_mutate_storage(ledger, success, outcome):
    row = ledger.create_execution("offline", source="direct")
    job = jobs.create_job("offline", "every 1h")
    before = jobs.get_job(job["id"])
    with pytest.raises(ValueError):
        ledger.finish_execution(row["id"], success=success, outcome=outcome)
    with pytest.raises(ValueError):
        jobs.mark_job_run(job["id"], success=success, outcome=outcome)
    assert ledger.get_execution(row["id"]) == row
    assert jobs.get_job(job["id"]) == before


def test_unknown_preserves_legacy_claim_schedule_repeat_bookkeeping(monkeypatch):
    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    results = []
    for success, outcome in [(True, None), (False, None), (None, "unknown")]:
        job = jobs.create_job("offline", "every 1h", repeat=3)
        assert jobs.claim_job_for_fire(job["id"])
        jobs.mark_job_run(job["id"], success, "detail", "delivery-only", outcome=outcome)
        state = jobs.get_job(job["id"])
        results.append({key: state.get(key) for key in (
            "last_run_at", "next_run_at", "repeat", "fire_claim", "run_claim", "last_delivery_error")})
    assert results[0] == results[1] == results[2]


def test_unknown_direct_tool_after_finite_job_removed(ledger, monkeypatch):
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (None, "unconfirmed", "unconfirmed", "unconfirmed"))
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **kw: None)
    job = jobs.create_job("offline", "1h", repeat=1)
    from tools.cronjob_tools import cronjob
    result = json.loads(cronjob(action="run", job_id=job["id"]))["job"]
    assert jobs.get_job(job["id"]) is None
    assert result["execution_outcome"] == "unknown"
    assert result["execution_success"] is None


def test_parallel_conversations_keep_exact_session_outcomes(native):
    Agent, manager, dispatch = native
    from threading import Barrier
    barrier = Barrier(3)
    expected = {"attempt-completed": "completed", "attempt-failed": "failed", "attempt-unknown": "unknown"}
    def handoff(**kw):
        session_id = kw["session_id"]
        barrier.wait(timeout=10)
        return LLMExecutionOutcome(expected[session_id], session_id, session_id)
    manager._middleware["llm_execution"] = [handoff]
    agents = [Agent(model="test-model", provider="openrouter", api_mode="chat_completions",
                    api_key="offline-test-key", base_url="http://127.0.0.1:1/v1",
                    session_id=session, platform="cron") for session in expected]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda agent: agent.run_conversation("offline task"), agents))
    finally:
        for agent in agents:
            agent.close()
    for session, result in zip(expected, results):
        assert result["execution_outcome"] == expected[session]
        assert result["final_response"] == session
    dispatch.assert_not_called()


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_ordinary_provider_dispatch_unchanged(native, api_mode):
    Agent, manager, dispatch = native
    manager._middleware["llm_execution"] = [lambda next_call, **kw: next_call()]
    dispatch.side_effect = None
    if api_mode == "chat_completions":
        dispatch.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="normal reply", role="assistant", tool_calls=None), finish_reason="stop")],
            usage=None,
        )
    else:
        dispatch.return_value = SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="normal reply")])],
            usage=None, status="completed", model="test-model",
        )
    agent = Agent(model="test-model", provider="openai-codex" if api_mode == "codex_responses" else "openrouter",
                  api_mode=api_mode, api_key="offline-test-key", base_url="http://127.0.0.1:1/v1",
                  session_id="normal-attempt", platform="cron")
    try:
        result = agent.run_conversation("offline task")
    finally:
        agent.close()
    assert result["completed"] is True
    assert result["final_response"] == "normal reply"
    assert "execution_outcome" not in result
    dispatch.assert_called_once()


@pytest.mark.parametrize("status,content,session", [
    ("pending", "pending", "same"), ("completed", "", "same"),
    ("completed", "reply", ""), ("completed", None, "same"),
])
def test_malformed_outcomes_fail_closed(status, content, session):
    result = LLMExecutionOutcome(status, content, session).for_session("same")
    assert result.status == "unknown"
    assert result.content == "Execution outcome could not be confirmed."


def test_unknown_delivery_error_is_separate(ledger, monkeypatch):
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (None, "pending", "pending", "unconfirmed"))
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **kw: "delivery unavailable")
    job = jobs.create_job("offline", "every 1h")
    assert scheduler.run_one_job(job)
    assert ledger.latest_execution(job["id"])["status"] == "unknown"
    state = jobs.get_job(job["id"])
    assert state["last_status"] == "unknown"
    assert state["last_error"] == "unconfirmed"
    assert state["last_delivery_error"] == "delivery unavailable"


def test_output_save_failure_does_not_relabel_unknown_as_execution_failure(ledger, monkeypatch):
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (None, "pending", "pending", "unconfirmed"))
    def cannot_save(*args):
        raise OSError("output unavailable")
    monkeypatch.setattr(scheduler, "save_job_output", cannot_save)
    job = jobs.create_job("offline", "every 1h")
    assert scheduler.run_one_job(job) is False  # processing failed, execution unconfirmed
    assert ledger.latest_execution(job["id"])["status"] == "unknown"
    assert jobs.get_job(job["id"])["last_status"] == "unknown"


@pytest.mark.parametrize("recorded_state", ["claimed", "running", None])
def test_manual_attempt_never_inherits_previous_job_success(ledger, monkeypatch, recorded_state):
    from tools.cronjob_tools import _execute_job_now
    job = jobs.create_job("offline", "every 1h")
    jobs.mark_job_run(job["id"], True)
    def unfinished(attempt, **kwargs):
        if recorded_state == "running":
            ledger.mark_execution_running(attempt["execution_id"])
        return True  # dispatch acknowledgement is not an execution result
    monkeypatch.setattr(scheduler, "run_one_job", unfinished)
    if recorded_state is None:
        monkeypatch.setattr(ledger, "get_execution", lambda _: None)
    result = _execute_job_now(job)
    assert result["claimed"] is True
    assert result["success"] is None
    assert result["outcome"] == (recorded_state or "unknown")
    assert result["execution_id"]
    assert jobs.get_job(job["id"])["last_status"] == "ok"


def test_same_second_runs_use_their_ledger_identity(native, ledger, monkeypatch):
    _, manager, dispatch = native
    calls = _install(manager, "completed")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openrouter", "api_mode": "chat_completions",
        "api_key": "offline-test-key", "base_url": "http://127.0.0.1:1/v1",
    })
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: datetime(2026, 9, 7, 12, tzinfo=timezone.utc))
    deliveries = []
    monkeypatch.setattr(scheduler, "_deliver_result", lambda job, content, **kw: deliveries.append(dict(job)))
    job = jobs.create_job("offline", "every 1h", model="test-model")
    assert scheduler.run_one_job(job)
    assert scheduler.run_one_job(job)
    assert "execution_id" not in job  # no attempt state in reusable job input
    assert len(set(calls)) == 2
    for call, delivered in zip(calls, deliveries):
        attempt = ledger.get_execution(delivered["execution_id"])
        assert attempt["job_id"] == job["id"]
        assert attempt["status"] == delivered["execution_outcome"] == "completed"
        assert call == ledger.execution_session_id(job["id"], attempt["id"])
    dispatch.assert_not_called()


@pytest.mark.parametrize("live_success", [True, False, None])
def test_real_delivery_paths_keep_exact_attempt_metadata(ledger, monkeypatch, live_success):
    import asyncio
    import threading
    from gateway.config import Platform, PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    import gateway.run

    captured = []
    async def standalone(config, chat_id, message, *, metadata=None, **kwargs):
        captured.append(("standalone", dict(metadata or {})))
        return {"success": True, "message_id": "standalone-receipt"}
    platform_registry.register(PlatformEntry(name="cron_identity_test", label="Synthetic", adapter_factory=lambda cfg: adapter, check_fn=lambda: True, standalone_sender_fn=standalone))
    platform = Platform("cron_identity_test")
    config = PlatformConfig(enabled=True)
    class Adapter(BasePlatformAdapter):
        async def connect(self):
            return True
        async def disconnect(self):
            pass
        async def get_chat_info(self, chat_id):
            return {"type": "dm", "name": "Synthetic"}
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            captured.append(("live", dict(metadata or {})))
            return SendResult(success=bool(live_success), message_id="live-receipt", error=None if live_success else "synthetic refusal")
    adapter = Adapter(config, platform)
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: SimpleNamespace(platforms={platform: config}))
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    loop = asyncio.new_event_loop()
    started = threading.Event()
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()
    worker = threading.Thread(target=run_loop)
    worker.start()
    assert started.wait(5)
    try:
        job = jobs.create_job("offline", "every 1h")
        first = ledger.create_execution(job["id"], source="direct")
        ledger.create_execution(job["id"], source="direct")  # newer is unrelated
        delivering = {**job, "execution_id": first["id"], "execution_outcome": "unknown", "deliver": "origin", "origin": {"platform": platform.value, "chat_id": "home"}}
        assert scheduler._deliver_result(delivering, "Unconfirmed result", adapters={platform: adapter} if live_success is not None else None, loop=loop) is None
        expected = {"job_id": job["id"], "execution_id": first["id"], "session_id": ledger.execution_session_id(job["id"], first["id"]), "execution_outcome": "unknown"}
        assert [path for path, _ in captured] == (["standalone"] if live_success is None else ["live"] if live_success else ["live", "standalone"])
        assert all(metadata == expected for _, metadata in captured)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        worker.join(5)
        loop.close()
        platform_registry.unregister(platform.value)
