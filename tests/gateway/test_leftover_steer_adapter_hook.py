"""A leftover /steer can be handed back to the adapter instead of becoming a turn.

When a steer lands after the agent's last tool batch (for example during the
final API call, or on an interrupted or empty-answer turn), the finalizer
returns it in ``result["pending_steer"]`` and the gateway runs it as the next
user turn. That is the right default for chat platforms that hand every
message straight to the agent.

An adapter that runs its own per-turn admission (one event at a time, each
with its own identity) must not have a turn started behind its back. It sets
``leftover_steer_as_next_turn = False`` and implements
``on_leftover_steer(session_key, text)``; the gateway then calls that hook
and starts no turn. Every other adapter keeps the default.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.run import GatewayRunner  # noqa: E402


def _runner():
    return object.__new__(GatewayRunner)


class _DefaultAdapter:
    pass


class _OwnAdmissionAdapter:
    leftover_steer_as_next_turn = False

    def __init__(self):
        self.calls = []

    def on_leftover_steer(self, session_key, text):
        self.calls.append((session_key, text))


class _AsyncOwnAdmissionAdapter(_OwnAdmissionAdapter):
    async def on_leftover_steer(self, session_key, text):
        self.calls.append((session_key, text))


class _RaisingAdapter(_OwnAdmissionAdapter):
    def on_leftover_steer(self, session_key, text):
        raise RuntimeError("boom")


class _FlagWithoutHandler:
    leftover_steer_as_next_turn = False


@pytest.mark.asyncio
async def test_default_adapter_runs_leftover_as_next_turn():
    pending = await _runner()._resolve_leftover_steer(
        _DefaultAdapter(), "s1", {"pending_steer": "also check B"}, None, None
    )
    assert pending == "also check B"


@pytest.mark.asyncio
async def test_default_adapter_keeps_existing_pending_and_ignores_leftover():
    pending = await _runner()._resolve_leftover_steer(
        _DefaultAdapter(), "s1", {"pending_steer": "also check B"}, "queued text", None
    )
    assert pending == "queued text"


@pytest.mark.asyncio
async def test_no_adapter_keeps_default():
    pending = await _runner()._resolve_leftover_steer(
        None, "s1", {"pending_steer": "also check B"}, None, None
    )
    assert pending == "also check B"


@pytest.mark.asyncio
async def test_no_leftover_is_noop():
    adapter = _OwnAdmissionAdapter()
    assert await _runner()._resolve_leftover_steer(adapter, "s1", {}, None, None) is None
    assert await _runner()._resolve_leftover_steer(adapter, "s1", None, None, None) is None
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_own_admission_adapter_gets_leftover_and_no_turn_starts():
    adapter = _OwnAdmissionAdapter()
    pending = await _runner()._resolve_leftover_steer(
        adapter, "s1", {"pending_steer": "also check B"}, None, None
    )
    assert pending is None
    assert adapter.calls == [("s1", "also check B")]


@pytest.mark.asyncio
async def test_own_admission_adapter_gets_leftover_even_with_other_pending():
    # Default Hermes drops the leftover when another pending text exists; an
    # adapter that asked for the leftover receives it so nothing is lost.
    adapter = _OwnAdmissionAdapter()
    pending = await _runner()._resolve_leftover_steer(
        adapter, "s1", {"pending_steer": "also check B"}, "other", None
    )
    assert pending == "other"
    assert adapter.calls == [("s1", "also check B")]


@pytest.mark.asyncio
async def test_async_hook_is_awaited():
    adapter = _AsyncOwnAdmissionAdapter()
    pending = await _runner()._resolve_leftover_steer(
        adapter, "s1", {"pending_steer": "x"}, None, None
    )
    assert pending is None
    assert adapter.calls == [("s1", "x")]


@pytest.mark.asyncio
async def test_raising_hook_starts_no_turn():
    pending = await _runner()._resolve_leftover_steer(
        _RaisingAdapter(), "s1", {"pending_steer": "x"}, None, None
    )
    assert pending is None


@pytest.mark.asyncio
async def test_flag_without_handler_falls_back_to_default():
    pending = await _runner()._resolve_leftover_steer(
        _FlagWithoutHandler(), "s1", {"pending_steer": "x"}, None, None
    )
    assert pending == "x"


def test_run_agent_inner_uses_the_helper():
    import inspect

    source = inspect.getsource(GatewayRunner._run_agent_inner)
    assert "self._resolve_leftover_steer(" in source
    assert 'pending = _leftover_steer\n' not in source
