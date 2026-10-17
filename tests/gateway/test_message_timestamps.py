from datetime import datetime
from zoneinfo import ZoneInfo

from gateway.message_timestamps import (
    coerce_message_timestamp,
    render_user_content_with_timestamp,
    strip_leading_message_timestamps,
)
from run_agent import AIAgent


BERLIN = ZoneInfo("Europe/Berlin")


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


def test_render_user_content_adds_single_context_timestamp():
    ts = _epoch(2026, 4, 28, 13, 40, 53)

    rendered = render_user_content_with_timestamp(
        "[Example User] Timestamp should be in context",
        ts,
        tz=BERLIN,
    )

    assert rendered == (
        "[Tue 2026-04-28 13:40:53 CEST] "
        "[Example User] Timestamp should be in context"
    )


def test_render_user_content_deduplicates_existing_timestamp_and_preserves_embedded_time():
    db_processing_ts = _epoch(2026, 4, 27, 15, 55, 36)
    stored_content = (
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    rendered = render_user_content_with_timestamp(
        stored_content,
        db_processing_ts,
        tz=BERLIN,
    )

    assert rendered == stored_content
    assert rendered.count("2026-04-27") == 1


def test_strip_leading_message_timestamps_removes_multiple_prefixes_and_prefers_inner_time():
    content = (
        "[Mon 2026-04-27 15:55:36 CEST] "
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    stripped, embedded_ts = strip_leading_message_timestamps(content, tz=BERLIN)

    assert stripped == "[Example User] This should go on our todo list"
    assert embedded_ts == _epoch(2026, 4, 27, 15, 54, 44)


def test_coerce_message_timestamp_accepts_datetime_and_epoch():
    dt = datetime(2026, 4, 28, 13, 40, 53, tzinfo=BERLIN)

    assert coerce_message_timestamp(dt, tz=BERLIN) == dt.timestamp()
    assert coerce_message_timestamp(dt.timestamp(), tz=BERLIN) == dt.timestamp()


def test_persist_user_message_override_keeps_clean_content_and_timestamp_metadata():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "[Example User] Clean content"
    agent._persist_user_message_timestamp = _epoch(2026, 4, 28, 13, 40, 53)
    messages = [
        {
            "role": "user",
            "content": "[Tue 2026-04-28 13:40:53 CEST] [Example User] Clean content",
        }
    ]

    agent._apply_persist_user_message_override(messages)

    assert messages == [
        {
            "role": "user",
            "content": "[Example User] Clean content",
            "timestamp": _epoch(2026, 4, 28, 13, 40, 53),
        }
    ]


# ---------------------------------------------------------------------------
# Opt-in gate: gateway.message_timestamps.enabled (default OFF)
# ---------------------------------------------------------------------------


def test_message_timestamps_enabled_defaults_off():
    from gateway.run import _message_timestamps_enabled

    assert _message_timestamps_enabled(None) is False
    assert _message_timestamps_enabled({}) is False
    assert _message_timestamps_enabled({"gateway": {}}) is False
    assert (
        _message_timestamps_enabled({"gateway": {"message_timestamps": {}}}) is False
    )


def test_message_timestamps_enabled_when_opted_in():
    from gateway.run import _message_timestamps_enabled

    assert _message_timestamps_enabled(
        {"gateway": {"message_timestamps": {"enabled": True}}}
    ) is True
    # Bare shorthand also accepted.
    assert _message_timestamps_enabled({"gateway": {"message_timestamps": True}}) is True


def test_build_history_injects_only_when_enabled():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "hello", "timestamp": _epoch(2026, 4, 28, 13, 40, 53)},
        {"role": "assistant", "content": "hi"},
    ]

    # Default (off): user content stays clean, no timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history)
    assert agent_history[0]["content"] == "hello"

    # Enabled: user content gets exactly one timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    assert agent_history[0]["content"].startswith("[")
    assert agent_history[0]["content"].endswith("hello")
    # Assistant message is never timestamped.
    assert agent_history[1]["content"] == "hi"


def test_build_history_never_stamps_a_stored_compaction_summary():
    """HPD-1282: a stored compaction summary is a user-role row. Stamping it
    hides SUMMARY_PREFIX behind ``[Sat ...]`` and every startswith-based
    summary check (anchor, auto-focus, carry-forward) then counts it as a real
    user turn. Upstream skips compaction scaffolding the same way."""
    from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor
    from gateway.run import _build_gateway_agent_history

    summary = f"{SUMMARY_PREFIX}\n## Active Task\nweather question"
    history = [
        {"role": "user", "content": summary, "timestamp": _epoch(2026, 10, 17, 10, 37, 36)},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Wie ist das Wetter heute?", "timestamp": _epoch(2026, 10, 17, 10, 58, 25)},
    ]

    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)

    summaries = [
        m for m in agent_history
        if ContextCompressor._is_context_summary_message(m)
    ]
    assert len(summaries) == 1
    assert agent_history[0]["content"] == summary
    # Ordinary user rows keep their single prefix.
    stamped = agent_history[2]["content"]
    assert stamped.startswith("[Sat 2026-10-17 ")
    assert strip_leading_message_timestamps(stamped)[0] == "Wie ist das Wetter heute?"


def test_build_history_never_stamps_a_merged_or_flagged_summary():
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        SUMMARY_PREFIX,
        _MERGED_SUMMARY_DELIMITER,
    )
    from gateway.run import _build_gateway_agent_history

    merged = f"earlier tail{_MERGED_SUMMARY_DELIMITER}{SUMMARY_PREFIX}\nbody"
    history = [
        {"role": "user", "content": merged, "timestamp": _epoch(2026, 10, 17, 10, 37, 36)},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "flagged body", COMPRESSED_SUMMARY_METADATA_KEY: True,
         "timestamp": _epoch(2026, 10, 17, 10, 38, 0)},
    ]

    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)

    assert agent_history[0]["content"] == merged
    assert agent_history[2]["content"] == "flagged body"
