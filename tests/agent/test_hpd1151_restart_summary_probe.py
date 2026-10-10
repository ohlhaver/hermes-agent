"""HPD-1151 step 3: a gateway restart must not leave a second summary.

Backport of the core of upstream 08ea88f4f7 ("decay protected summaries after
restart"): ``_effective_protect_first_n`` probes the resumed head for a
persisted handoff summary before ``compress_start`` is computed. In this fork
the probe is behind ``compression.restart_summary_probe`` (default off), so a
guest without the setting behaves exactly as before.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES,
)


def _compressor(*, probe: bool, protect_first_n: int = 3) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
            restart_summary_probe=probe,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _resumed(summary_body: str):
    # What a restarted gateway loads from the session DB after one in-place
    # compaction: system prompt, the persisted handoff, then new turns.
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "first turn after resume"},
        {"role": "assistant", "content": "first answer after resume"},
        {"role": "user", "content": "second turn after resume"},
        {"role": "assistant", "content": "second answer after resume"},
        {"role": "user", "content": "third turn after resume"},
        {"role": "assistant", "content": "third answer after resume"},
        {"role": "user", "content": "fourth turn after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def _summaries(messages):
    return [
        m for m in messages
        if ContextCompressor._has_compressed_summary_metadata(m)
        or ContextCompressor._is_context_summary_content(m.get("content"))
    ]


def test_probe_off_by_default_keeps_stock_head_protection():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="test/model", protect_first_n=3, quiet_mode=True)
    assert c.restart_summary_probe is False
    assert c._protect_head_size(_resumed("old")) == 1 + 3


def test_probe_on_decays_head_when_resumed_head_holds_a_summary():
    c = _compressor(probe=True)
    assert c._protect_head_size(_resumed("old")) == 1


def test_restart_then_compaction_leaves_exactly_one_summary():
    c = _compressor(probe=True)
    old = "RESTART-FOSSIL durable facts from before restart"
    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as call:
        result = c.compress(_resumed(old))
    assert len(_summaries(result)) == 1
    assert all(old not in str(m.get("content", "")) for m in result)
    # The old summary is carried into the new one, not lost.
    prompt = call.call_args.kwargs["messages"][0]["content"]
    assert old in prompt


def test_restart_without_probe_still_stacks_two_summaries():
    # The defect this backport fixes, kept as the default until Justus decides.
    c = _compressor(probe=False)
    old = "RESTART-FOSSIL durable facts from before restart"
    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = c.compress(_resumed(old))
    assert len(_summaries(result)) == 2


def test_summary_in_live_tail_does_not_decay_first_compaction_head():
    c = _compressor(probe=True)
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(1, 11):
        msgs.append({"role": "user" if i % 2 else "assistant", "content": f"turn {i}"})
    msgs.append({"role": "assistant", "content": f"{SUMMARY_PREFIX}\nlooks like a summary"})
    msgs.append({"role": "user", "content": "final request"})
    assert c._protect_head_size(msgs) == 1 + 3


def test_probe_window_is_bounded():
    c = _compressor(probe=True)
    end = 1 + c.protect_first_n + _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES

    def at(index):
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(1, index):
            msgs.append({"role": "user" if i % 2 else "assistant", "content": f"filler {i}"})
        msgs.append({"role": "user" if index % 2 else "assistant", "content": f"{SUMMARY_PREFIX}\nboundary"})
        msgs.extend([{"role": "assistant", "content": "a"}, {"role": "user", "content": "tail"}])
        return msgs

    assert c._protect_head_size(at(end - 1)) == 1
    assert c._protect_head_size(at(end)) == 1 + 3


def test_metadata_flag_counts_as_summary():
    c = _compressor(probe=True)
    msgs = _resumed("old")
    from agent import context_compressor as cc
    msgs[1] = {"role": "user", "content": "no prefix", cc.COMPRESSED_SUMMARY_METADATA_KEY: True}
    assert c._protect_head_size(msgs) == 1
