"""The tail-protection budget charges the ``api_content`` sidecar (HPD-1131).

Backport of upstream NousResearch/hermes-agent 820106d4 ("fix(compaction):
count api_content in tail budget", contained in main dce1e9b3). Before it,
``_estimate_msg_budget_tokens`` measured only the clean ``content`` while the
preflight trigger and the wire both carry the sidecar. A Home conversation
whose every user turn carries ~1.5k tokens of memory and plugin context was
therefore protected as "small": the tail walk kept almost all of it and the
post-compaction prompt stayed near the threshold.

The shape here is the real one: the sidecar is built by
``compose_user_api_content`` (``<memory-context>`` fence plus the
``pre_llm_call`` user-message context) and the wire is measured after
``substitute_api_content``, exactly as ``conversation_loop`` sends it. The
compressor runs with the guest's settings: threshold 200k, target ratio 0.10,
protect_last_n 3.
"""

import copy
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough
from agent.turn_context import compose_user_api_content, substitute_api_content

CONTEXT_LENGTH = 1_000_000
THRESHOLD_PERCENT = 0.20  # 200k on a 1M window, as on the guests

MEMORY = "\n".join(
    f"- The user prefers concise answers about topic {i}; noted on 2026-10-0{i % 9 + 1}." for i in range(60)
)  # ~5k chars, about the 1.26k-token Honcho recall measured on fc-e2e-2
AUTOMATION_RUNS = "Recent automation runs:\n" + "\n".join(
    f"- job_{i:04d} daily-digest completed 2026-10-09T0{i % 10}:00Z status=ok" for i in range(14)
)  # ~1k chars of pre_llm_call context


def _wire(messages):
    out = []
    for msg in messages:
        api_msg = copy.deepcopy(msg)
        substitute_api_content(api_msg)
        out.append(api_msg)
    return out


def _home_conversation(turns: int):
    """Home turns as the guest stores them: a short user message carrying the
    sidecar, one tool call with its result, then the answer. The clean text of
    a turn is ~0.5k tokens; on the wire it is ~2k."""
    messages = [{"role": "system", "content": "You are Hermes. " * 200}]
    for i in range(turns):
        content = f"Question {i}: what changed in my portfolio since yesterday?"
        sidecar = compose_user_api_content(content, MEMORY, AUTOMATION_RUNS)
        call_id = f"call_{i:04d}"
        messages.append({"role": "user", "content": content, "api_content": sidecar})
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "finance_portfolio", "arguments": '{"range":"1d"}'}}],
        })
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": f"Portfolio {i}: " + "position unchanged; " * 60})
        messages.append({"role": "assistant", "content": f"Answer {i}: " + "Nothing material changed today. " * 18})
    return messages


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=CONTEXT_LENGTH):
        return ContextCompressor(
            model="test/model",
            threshold_percent=THRESHOLD_PERCENT,
            protect_first_n=3,
            protect_last_n=3,
            summary_target_ratio=0.10,
            quiet_mode=True,
        )


def test_sidecar_is_the_charged_content():
    content = "short question"
    sidecar = compose_user_api_content(content, MEMORY, AUTOMATION_RUNS)
    msg = {"role": "user", "content": content, "api_content": sidecar}
    assert estimate_tokens_rough(sidecar) > 1_300
    assert _estimate_msg_budget_tokens(msg) >= estimate_tokens_rough(sidecar)
    assert msg["api_content"] == sidecar  # accounting only, never mutates


def test_post_compaction_wire_prompt_is_far_below_threshold(compressor):
    messages = _home_conversation(110)
    wire_before = estimate_messages_tokens_rough(_wire(messages))
    assert wire_before >= compressor.threshold_tokens  # the trigger would fire here

    with patch.object(ContextCompressor, "_generate_summary", return_value="Earlier turns: portfolio questions answered."):
        result = compressor.compress(messages, current_tokens=wire_before)

    wire_after = estimate_messages_tokens_rough(_wire(result))
    assert wire_after < compressor.threshold_tokens * 0.4, (wire_before, wire_after, len(result))
    # The newest turn is kept verbatim, sidecar included.
    assert result[-4:] == messages[-4:]
