"""HPD-1289: an answer that used OpenRouter's server-side web search does not
calibrate the HPD-1151 native ratio, through the real turn loop.

OpenRouter sums both passes of its server-side search loop into one prompt
count. Measured on fc-justus-3 on 2026-10-17 (Claude Haiku 5.5): 132,817
reported (64,735 cached) for a request of about 38.5k rough, while the real
prompts after compression were 48,797 and 52,412 (ratio about 1.7). The usage
below is the producer's shape: OpenRouter's final stream chunk, parsed by the
OpenAI client as Hermes' stream loop keeps it (chunk.usage).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk

from tests.run_agent.test_413_compression import agent  # noqa: F401  (fixture)

HAIKU = "anthropic/claude-haiku-5.5"


@pytest.fixture(autouse=True)
def _isolated_ratio_store(tmp_path, monkeypatch):
    from agent import turn_context as _tc

    monkeypatch.setattr(_tc, "_native_ratio_store_path", lambda: tmp_path / "native_token_ratios.json")


def _final_chunk_usage(prompt, cached, web_search_requests):
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": 412,
        "total_tokens": prompt + 412,
        "cost": 0.01792,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": cached, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    if web_search_requests is not None:
        usage["server_tool_use_details"] = {"web_search_requests": web_search_requests}
    chunk = ChatCompletionChunk.construct(**{
        "id": "gen-1760689527-test",
        "object": "chat.completion.chunk",
        "created": 1760689527,
        "model": HAIKU,
        "provider": "Anthropic",
        "choices": [],
        "usage": usage,
    })
    return chunk.usage


def _response(usage):
    msg = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model=HAIKU, usage=usage)


def _turn(agent, usage, *, rough=None, history=None):  # noqa: F811
    agent.client.chat.completions.create.side_effect = [_response(usage)]
    compress = MagicMock(side_effect=lambda messages, system, **_k: (messages, system))
    patches = [
        patch.object(agent, "_compress_context", compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ]
    if rough is not None:
        patches += [
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=rough),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=rough),
        ]
    for p in patches:
        p.start()
    try:
        result = agent.run_conversation("hello", conversation_history=history)
    finally:
        for p in reversed(patches):
            p.stop()
    assert result["completed"] is True
    return compress.call_count


def _setup(agent):  # noqa: F811
    agent.model = HAIKU
    agent._current_turn_sidecar_models = (HAIKU,)
    agent.context_compressor.native_token_ratio = 0.0
    agent.context_compressor.threshold_tokens = 85_000


def _calibrate_clean(agent):  # noqa: F811
    _turn(agent, _final_chunk_usage(1, 0, None))
    rough = agent._hpd1151_rough_request_tokens
    assert rough > 0
    _turn(agent, _final_chunk_usage(int(rough * 1.76), 0, None))
    good = agent.context_compressor.native_token_ratio
    assert good == pytest.approx(1.76, abs=0.05)
    return good


HISTORY = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 3


def test_search_answer_keeps_ratio_and_next_turn_does_not_compress(agent):  # noqa: F811
    _setup(agent)
    good = _calibrate_clean(agent)

    # The 10:05:27 call: web search, in=132,817, cached 64,735.
    _turn(agent, _final_chunk_usage(132_817, 64_735, 1))
    assert agent.context_compressor.native_token_ratio == good

    # The next turn at the incident's size: rough 38,536 x 1.76 ~ 67.8k < 85k.
    calls = _turn(agent, _final_chunk_usage(52_412, 50_000, None), rough=38_536, history=HISTORY)
    assert calls == 0


def test_same_sizes_without_the_fix_signal_would_compress(agent):  # noqa: F811
    # Sensitivity check: the same count without server_tool_use_details
    # calibrates (clamped to 3.0), and the same next turn then compresses.
    _setup(agent)
    _calibrate_clean(agent)
    _turn(agent, _final_chunk_usage(132_817, 64_735, None))
    assert agent.context_compressor.native_token_ratio == 3.0
    calls = _turn(agent, _final_chunk_usage(52_412, 50_000, None), rough=38_536, history=HISTORY)
    assert calls >= 1


def test_answer_without_search_still_calibrates(agent):  # noqa: F811
    _setup(agent)
    _calibrate_clean(agent)
    rough = agent._hpd1151_rough_request_tokens
    # A clean answer, also one that reports zero searches, moves the ratio.
    _turn(agent, _final_chunk_usage(int(rough * 1.40), 0, 0))
    assert agent.context_compressor.native_token_ratio == pytest.approx(1.40, abs=0.05)


def _search_tool_call_turn(agent, web_search_requests):  # noqa: F811
    """A search answer (132,817 summed) that also calls a tool, then a clean
    answer. The post-tool check must not read the summed figure as the size."""
    tc = SimpleNamespace(id="tc1", type="function",
                         function=SimpleNamespace(name="web_search", arguments='{"query":"SNB"}'))
    msg = SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None, reasoning=None)
    tool_resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                                model=HAIKU, usage=_final_chunk_usage(132_817, 64_735, web_search_requests))
    agent.client.chat.completions.create.side_effect = [tool_resp, _response(_final_chunk_usage(48_000, 47_000, None))]
    compress = MagicMock(side_effect=lambda messages, system, **_k: (messages, system))
    with (
        patch("run_agent.handle_function_call", return_value="SNB policy rate 0%"),
        patch.object(agent, "_compress_context", compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")
    assert result["completed"] is True
    return compress.call_count


def test_search_answer_with_tool_calls_does_not_compact_mid_turn(agent):  # noqa: F811
    _setup(agent)
    agent.compression_enabled = True
    assert _search_tool_call_turn(agent, 1) == 0
    # The summed figure is no prompt size: not kept as the last prompt.
    assert agent.context_compressor.last_prompt_tokens != 132_817


def test_same_tool_call_turn_without_search_signal_compacts(agent):  # noqa: F811
    # Sensitivity check: the same 132,817 reported without server_tool_use_details
    # is a real prompt size and compacts after the tool call.
    _setup(agent)
    agent.compression_enabled = True
    assert _search_tool_call_turn(agent, None) >= 1
