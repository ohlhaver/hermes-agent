"""Registered send guards see every auxiliary request before it leaves.

The main loop's middleware never sees an auxiliary request (compression,
title generation, session search, vision, ...) or the iteration-limit summary.
``agent/auxiliary_send_guard.py`` is the seam for them; these tests pin that
every send path consults it, that a refusal sends nothing and walks no
fallback, and that a refused compression keeps the conversation unchanged.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_send_guard as guard_mod
from agent.auxiliary_send_guard import (
    AuxiliarySendRefused,
    auxiliary_session_scope,
    current_auxiliary_session_id,
    guard_auxiliary_send,
    register_auxiliary_send_guard,
    unregister_auxiliary_send_guard,
)


class _Response:
    def __init__(self, content="ok"):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason="stop")]
        self.model = "m"
        self.usage = None


@pytest.fixture
def seen():
    """Register a recording guard that refuses any request to the Anthropic host."""
    calls = []

    def guard(*, task, provider, base_url, model, request, session_id):
        calls.append({"task": task, "provider": provider, "base_url": base_url, "model": model,
                      "session_id": session_id, "messages": request.get("messages")})
        if "anthropic.com" in base_url or provider == "anthropic":
            raise AuxiliarySendRefused("refused", user_message="Nothing was sent.", reason="test")
        amended = dict(request)
        amended["extra_body"] = dict(request.get("extra_body") or {}, guarded=True)
        return amended

    register_auxiliary_send_guard(guard)
    try:
        yield calls
    finally:
        unregister_auxiliary_send_guard(guard)


def _client(base_url, response=None, side_effect=None):
    client = MagicMock()
    client.base_url = base_url
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = response or _Response()
    return client


def test_no_guard_leaves_request_untouched():
    request = {"model": "m", "messages": []}
    assert guard_auxiliary_send(task="t", provider="p", model="m", request=request) is request


def test_guard_error_fails_closed():
    def broken(**_kwargs):
        raise ValueError("cannot read store")

    register_auxiliary_send_guard(broken)
    try:
        with pytest.raises(AuxiliarySendRefused) as info:
            guard_auxiliary_send(task="compression", provider="openrouter", model="m", request={"messages": []})
        assert info.value.reason == "guard_error"
        assert info.value.status_code == 403
    finally:
        unregister_auxiliary_send_guard(broken)


def test_session_scope_wins_over_accounting_context():
    from agent.aux_accounting import reset_accounting_context, set_accounting_context

    token = set_accounting_context(object(), "turn-session")
    try:
        assert current_auxiliary_session_id() == "turn-session"
        with auxiliary_session_scope("compress-session"):
            assert current_auxiliary_session_id() == "compress-session"
        assert current_auxiliary_session_id() == "turn-session"
    finally:
        reset_accounting_context(token)
    assert current_auxiliary_session_id() is None


def test_call_llm_refused_before_sending_and_no_fallback(seen):
    from agent.auxiliary_client import call_llm

    primary = _client("https://api.anthropic.com")
    fallback = _client("https://openrouter.ai/api/v1")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "claude-haiku")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("anthropic", "claude-haiku", None, None, None)), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "fb", "openrouter")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "openrouter")), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(fallback, "fb", "openrouter")):
        with auxiliary_session_scope("s-1"):
            with pytest.raises(AuxiliarySendRefused):
                call_llm(task="compression", messages=[{"role": "user", "content": "history"}])
    assert not primary.chat.completions.create.called
    assert not fallback.chat.completions.create.called
    assert seen[0]["task"] == "compression"
    assert seen[0]["session_id"] == "s-1"
    assert seen[0]["base_url"] == "https://api.anthropic.com"


def test_call_llm_stream_path_is_guarded(seen):
    from agent.auxiliary_client import call_llm

    primary = _client("https://api.anthropic.com")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "claude-haiku")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("anthropic", "claude-haiku", None, None, None)):
        with pytest.raises(AuxiliarySendRefused):
            call_llm(task="moa_aggregator", messages=[{"role": "user", "content": "x"}], stream=True)
    assert not primary.chat.completions.create.called


def test_call_llm_amended_request_is_what_is_sent(seen):
    from agent.auxiliary_client import call_llm

    primary = _client("https://openrouter.ai/api/v1")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "deepseek/x")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("openrouter", "deepseek/x", None, None, None)):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "x"}])
    sent = primary.chat.completions.create.call_args.kwargs
    assert sent["extra_body"]["guarded"] is True


def test_fallback_candidate_is_guarded_on_its_own_route(seen):
    """Primary on OpenRouter fails with a payment error; the fallback is Anthropic and is refused."""
    from agent.auxiliary_client import call_llm

    payment = Exception("Payment Required: insufficient credits")
    payment.status_code = 402
    primary = _client("https://openrouter.ai/api/v1", side_effect=payment)
    fallback = _client("https://api.anthropic.com")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "deepseek/x")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "deepseek/x", None, None, None)), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "claude-haiku", "anthropic")):
        with pytest.raises(AuxiliarySendRefused):
            call_llm(task="session_search", messages=[{"role": "user", "content": "x"}])
    assert primary.chat.completions.create.called
    assert not fallback.chat.completions.create.called
    assert [c["provider"] for c in seen] == ["auto", "anthropic"]


def test_async_call_llm_refused_before_sending(seen):
    from agent.auxiliary_client import async_call_llm

    primary = MagicMock()
    primary.base_url = "https://api.anthropic.com"

    async def create(**_kwargs):
        raise AssertionError("must not be sent")

    primary.chat.completions.create = create
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "claude-haiku")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("anthropic", "claude-haiku", None, None, None)):
        with pytest.raises(AuxiliarySendRefused):
            asyncio.run(async_call_llm(task="vision_followup", messages=[{"role": "user", "content": "x"}]))


def test_same_provider_rebuild_is_guarded(seen):
    from agent import auxiliary_client as ac

    rebuilt = _client("https://api.anthropic.com")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(rebuilt, "claude-haiku")):
        with pytest.raises(AuxiliarySendRefused):
            ac._retry_same_provider_sync(
                task="compression", resolved_provider="anthropic", resolved_model="claude-haiku",
                resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
                main_runtime=None, final_model="claude-haiku",
                messages=[{"role": "user", "content": "x"}], temperature=None, max_tokens=None,
                tools=None, effective_timeout=30.0, effective_extra_body={}, reasoning_config=None,
            )
    assert not rebuilt.chat.completions.create.called


def test_refused_compression_keeps_the_conversation(seen):
    """A refusal is a terminal access failure: compress() returns the messages unchanged."""
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor(
        model="claude-haiku", threshold_percent=0.5, protect_first_n=1, protect_last_n=2,
        quiet_mode=False, config_context_length=64000, provider="anthropic",
    )
    messages = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 400})
    primary = _client("https://api.anthropic.com")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "claude-haiku")), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("anthropic", "claude-haiku", None, None, None)):
        with auxiliary_session_scope("s-google"):
            result = cc.compress(messages, current_tokens=60000, force=True)
    assert not primary.chat.completions.create.called
    assert [c["task"] for c in seen] and all(c["task"] == "compression" for c in seen)
    assert seen[0]["session_id"] == "s-google"
    assert result == messages
    assert cc._last_compress_aborted is True
    assert cc._last_summary_fallback_used is False


def test_compress_context_forwarder_names_the_session():
    from run_agent import AIAgent

    seen_ids = []
    fake = types.SimpleNamespace(session_id="hygiene-session")

    def compress_context(agent, messages, system_message, **_kwargs):
        seen_ids.append(current_auxiliary_session_id())
        return messages, system_message

    with patch("agent.conversation_compression.compress_context", compress_context):
        AIAgent._compress_context(fake, [], "")
    assert seen_ids == ["hygiene-session"]


def _max_iter_agent(base_url):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=base_url, quiet_mode=True,
        skip_context_files=True, skip_memory=True,
    )
    agent._cached_system_prompt = "SYS"
    agent.session_id = "s-max"
    return agent


def test_iteration_limit_summary_is_guarded_and_amended(seen):
    from agent.chat_completion_helpers import handle_max_iterations

    agent = _max_iter_agent("https://openrouter.ai/api/v1")
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "RAW"

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    transport = types.SimpleNamespace(normalize_response=lambda _r: types.SimpleNamespace(content="SUMMARY"))
    with patch.object(agent, "_ensure_primary_openai_client", return_value=client), \
         patch.object(agent, "_get_transport", return_value=transport):
        out = handle_max_iterations(agent, [{"role": "user", "content": "q"}], 5)
    assert out == "SUMMARY"
    assert captured["extra_body"]["guarded"] is True
    assert seen[-1]["task"] == "iteration_limit_summary"
    assert seen[-1]["session_id"] == "s-max"


def test_iteration_limit_summary_refusal_sends_nothing(seen):
    from agent.chat_completion_helpers import handle_max_iterations

    agent = _max_iter_agent("https://openrouter.ai/api/v1")
    agent.provider = "anthropic"
    client = MagicMock()
    with patch.object(agent, "_ensure_primary_openai_client", return_value=client), \
         patch.object(agent, "_anthropic_messages_create") as anthropic_send, \
         patch.object(agent, "_run_codex_stream") as codex_send:
        out = handle_max_iterations(agent, [{"role": "user", "content": "q"}], 5)
    assert out == "Nothing was sent."
    assert not client.chat.completions.create.called
    assert not anthropic_send.called
    assert not codex_send.called
