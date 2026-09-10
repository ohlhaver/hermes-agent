"""Tests for ``agent.conversation_loop._restore_or_build_system_prompt``.

Validates the gateway DB-roundtrip path that keeps the system prompt
byte-stable across turns (fresh AIAgent → must restore from session DB
instead of rebuilding).  Covers:

  * Successful restore from a stored prompt (present row).
  * Legitimate first-turn build (no history).
  * Silent-failure recovery paths:
      - DB read raises → WARNING + fresh build
      - Row has system_prompt=NULL → WARNING + fresh build
      - Row has system_prompt="" → WARNING + fresh build
      - DB write fails → WARNING (subsequent turns will miss cache)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import stamp_system_prompt_contract


_CONTRACT_A = "a" * 64
_CONTRACT_B = "b" * 64


def _make_agent(
    session_db=None,
    prebuilt_prompt: str = "BUILT_PROMPT",
    contract_fingerprint: str = _CONTRACT_A,
):
    """Construct the minimal agent fake the helper needs."""
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = "cli"
    agent._session_db = session_db
    agent._build_system_prompt = MagicMock(
        return_value=stamp_system_prompt_contract(
            prebuilt_prompt, contract_fingerprint
        )
    )
    agent._prompt_contract_fingerprint = contract_fingerprint
    return agent


@pytest.fixture(autouse=True)
def _stub_prompt_contract_fingerprint(monkeypatch):
    """Keep restore tests focused on persistence decisions, not prompt assembly."""

    def build_fingerprint(agent, _system_message):
        return agent._prompt_contract_fingerprint, []

    monkeypatch.setattr(
        "agent.conversation_loop.build_system_prompt_contract_fingerprint",
        build_fingerprint,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestStoredPromptReuse:
    def test_present_row_is_reused_verbatim(self, caplog):
        """Continuing session with a stored prompt → reuse byte-for-byte."""
        stored = stamp_system_prompt_contract(
            "Stored prompt from turn 1 — byte-identical reuse", _CONTRACT_A
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        db.update_system_prompt.assert_not_called()
        # No warnings on the happy path
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_present_row_with_unicode_preserved(self):
        """Non-ASCII bytes in the stored prompt are not mangled."""
        stored = stamp_system_prompt_contract(
            "Stored prompt with unicode: ☤ ⚗ ◆ — and emoji 🦊", _CONTRACT_A
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
        assert agent._cached_system_prompt == stored

    def test_present_row_with_stale_runtime_identity_rebuilds(self, caplog):
        """Stored prompts are cache gold unless their runtime identity is stale.

        A live /model switch updates the agent and DB model_config immediately.
        If the old system_prompt snapshot still says the previous model,
        blindly restoring it makes the next turn call the new model while the
        model reads old `Model:` metadata ("what model are you?" lies).
        """
        stored = (
            "You are Hermes Agent.\n\n"
            "Conversation started: Tuesday, June 16, 2026\n"
            "Session ID: test-session-id\n"
            "Model: anthropic/claude-opus-4.8-fast\n"
            "Provider: openrouter"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(
            session_db=db,
            prebuilt_prompt=(
                "You are Hermes Agent.\n\n"
                "Conversation started: Tuesday, June 16, 2026\n"
                "Session ID: test-session-id\n"
                "Model: openai/gpt-5.5\n"
                "Provider: openrouter"
            ),
        )
        agent.model = "openai/gpt-5.5"

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        assert "Model: openai/gpt-5.5\nProvider: openrouter" in (
            agent._cached_system_prompt
        )
        agent._build_system_prompt.assert_called_once_with(None)
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, agent._cached_system_prompt
        )
        assert any("stale runtime identity" in r.getMessage() for r in caplog.records)

    def test_changed_platform_contract_rebuilds_with_same_model_and_provider(self):
        """A platform instruction change invalidates an otherwise matching row."""
        stored = stamp_system_prompt_contract(
            "Old platform prompt\nModel: test-model\nProvider: openrouter",
            _CONTRACT_A,
        )
        rebuilt = (
            "New platform prompt\nModel: test-model\nProvider: openrouter"
        )
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(
            session_db=db,
            prebuilt_prompt=rebuilt,
            contract_fingerprint=_CONTRACT_B,
        )

        history = [
            {"role": "user", "content": "Keep this question"},
            {"role": "assistant", "content": "Keep this answer"},
        ]
        before = deepcopy(history)
        with (
            patch("hermes_cli.plugins.invoke_hook") as invoke_hook,
            patch(
                "agent.credits_tracker.seed_credits_at_session_start"
            ) as seed_credits,
        ):
            _restore_or_build_system_prompt(agent, None, history)

        assert agent._cached_system_prompt == stamp_system_prompt_contract(
            rebuilt, _CONTRACT_B
        )
        assert history == before
        invoke_hook.assert_not_called()
        seed_credits.assert_not_called()
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, agent._cached_system_prompt
        )

    def test_changed_soul_or_context_input_rebuilds_without_rewriting_history(self):
        """SOUL/context changes replace only the prompt snapshot, never messages."""
        stored = stamp_system_prompt_contract(
            "Old SOUL and context\nModel: test-model\nProvider: openrouter",
            _CONTRACT_A,
        )
        rebuilt = "New SOUL and context\nModel: test-model\nProvider: openrouter"
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(
            session_db=db,
            prebuilt_prompt=rebuilt,
            contract_fingerprint=_CONTRACT_B,
        )
        history = [
            {"role": "user", "content": "Keep this question"},
            {"role": "assistant", "content": "Keep this answer"},
        ]
        before = deepcopy(history)

        _restore_or_build_system_prompt(agent, "new system input", history)

        assert history == before
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, agent._cached_system_prompt
        )
        assert "New SOUL and context" in agent._cached_system_prompt

    def test_legacy_prompt_without_contract_fingerprint_rebuilds_once(self):
        """Rows created before contract fingerprints cannot be trusted as current."""
        db = MagicMock()
        db.get_session.return_value = {
            "system_prompt": "Legacy prompt\nModel: test-model\nProvider: openrouter"
        }
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(
            agent, None, [{"role": "user", "content": "history"}]
        )

        assert agent._cached_system_prompt == agent._build_system_prompt.return_value
        db.update_system_prompt.assert_called_once_with(
            agent.session_id, agent._cached_system_prompt
        )


# ---------------------------------------------------------------------------
# Legitimate fresh-build paths (no history, no DB)
# ---------------------------------------------------------------------------


class TestLegitimateFreshBuild:
    def test_no_history_skips_db_and_builds_fresh(self, caplog):
        """First turn with empty history → build fresh, don't touch the DB."""
        db = MagicMock()
        agent = _make_agent(session_db=db)

        with (
            caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
            patch("hermes_cli.plugins.invoke_hook") as invoke_hook,
            patch(
                "agent.credits_tracker.seed_credits_at_session_start"
            ) as seed_credits,
        ):
            _restore_or_build_system_prompt(agent, None, [])

        # No history → DB read skipped entirely
        db.get_session.assert_not_called()
        agent._build_system_prompt.assert_called_once_with(None)
        assert agent._cached_system_prompt == stamp_system_prompt_contract(
            "BUILT_PROMPT", _CONTRACT_A
        )
        # Persisted to DB
        db.update_system_prompt.assert_called_once_with(
            agent.session_id,
            stamp_system_prompt_contract("BUILT_PROMPT", _CONTRACT_A),
        )
        invoke_hook.assert_called_once_with(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=agent.platform,
        )
        seed_credits.assert_called_once_with(agent)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_no_db_skips_persistence(self):
        """When session DB is None, build and skip persistence silently."""
        agent = _make_agent(session_db=None)
        _restore_or_build_system_prompt(agent, None, [])
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == stamp_system_prompt_contract(
            "BUILT_PROMPT", _CONTRACT_A
        )


# ---------------------------------------------------------------------------
# Silent-failure recovery — these are the new A/B logging paths
# ---------------------------------------------------------------------------


class TestSilentFailureWarnings:
    def test_db_read_exception_warns_and_rebuilds(self, caplog):
        """DB read raising → WARNING + fall through to fresh build."""
        db = MagicMock()
        db.get_session.side_effect = RuntimeError("disk full")
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Built fresh
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == stamp_system_prompt_contract(
            "BUILT_PROMPT", _CONTRACT_A
        )
        # Loud warning about the read failure
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("get_session failed" in r.getMessage() for r in warnings), \
            f"Expected a get_session warning, got: {[r.getMessage() for r in warnings]}"
        assert any("disk full" in r.getMessage() for r in warnings)

    def test_null_system_prompt_warns_about_unusable_stored_state(self, caplog):
        """Row exists but system_prompt is NULL → WARNING + fresh build."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": None}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("is null" in m and "rebuilding" in m for m in warnings), \
            f"Expected null-stored-prompt warning, got: {warnings}"

    def test_empty_system_prompt_warns_about_silent_persistence_bug(self, caplog):
        """Row exists but system_prompt is '' → WARNING about silent write bug."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": ""}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("is empty" in m and "rebuilding" in m for m in warnings), \
            f"Expected empty-stored-prompt warning, got: {warnings}"

    def test_db_write_failure_warns_loudly(self, caplog):
        """update_system_prompt raising → WARNING (was DEBUG before)."""
        db = MagicMock()
        # No prior row (first turn)
        db.get_session.return_value = None
        db.update_system_prompt.side_effect = RuntimeError("database is locked")
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(agent, None, [])

        # Built and assigned the cache anyway
        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == stamp_system_prompt_contract(
            "BUILT_PROMPT", _CONTRACT_A
        )
        # Warning surfaced
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "update_system_prompt failed" in m and "database is locked" in m
            for m in warnings
        ), f"Expected write-failure warning, got: {warnings}"

    def test_no_history_with_null_row_does_not_warn(self, caplog):
        """First turn (no history) hitting a null row is not surprising — no warn."""
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": None}
        agent = _make_agent(session_db=db)

        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            # Empty history → DB read is skipped entirely
            _restore_or_build_system_prompt(agent, None, [])

        db.get_session.assert_not_called()
        # No "rebuilding from scratch" warning because history is empty
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("rebuilding" in m for m in warnings)


# ---------------------------------------------------------------------------
# Byte-stability invariant
# ---------------------------------------------------------------------------


class TestPromptStabilityInvariant:
    def test_restored_prompt_is_byte_identical_to_stored(self):
        """The restored prompt must equal the stored bytes exactly — no
        normalization, trimming, or concat that could shift the prefix.

        This is the core invariant: any byte-level change at this point
        invalidates KV cache on every prefix-cache backend.
        """
        stored = (
            "You are Hermes Agent.\n"
            "\n"
            "Conversation started: Sunday, May 17, 2026\n"
            "Session ID: 20260517_153500_abc123\n"
        )
        stored = stamp_system_prompt_contract(stored, _CONTRACT_A)
        db = MagicMock()
        db.get_session.return_value = {"system_prompt": stored}
        agent = _make_agent(session_db=db)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Identity check — must be the same object reference for maximum
        # confidence we're not slicing/copying/normalizing.
        assert agent._cached_system_prompt == stored
        # Byte-level check
        assert agent._cached_system_prompt.encode("utf-8") == stored.encode("utf-8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
