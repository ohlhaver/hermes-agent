"""The real plugin transport keeps cron identity without changing legacy senders."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import _send_via_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_kind", ["legacy", "metadata", "kwargs", "live"])
async def test_delivery_metadata_is_opt_in_and_exact(monkeypatch, sender_kind):
    received = []
    async def legacy(config, chat, text, *, thread_id=None, media_files=None, force_document=False):
        received.append({"thread_id": thread_id, "media_files": media_files, "force_document": force_document})
        return {"success": True, "message_id": "sent"}
    async def with_metadata(config, chat, text, *, metadata=None, **kwargs):
        received.append(metadata)
        return {"success": True, "message_id": "sent"}
    async def with_kwargs(config, chat, text, **kwargs):
        received.append(kwargs.get("metadata"))
        return {"success": True, "message_id": "sent"}
    async def live(*, chat_id, content, metadata=None):
        received.append(metadata)
        return SimpleNamespace(success=True, message_id="sent")
    sender = {"legacy": legacy, "metadata": with_metadata, "kwargs": with_kwargs, "live": legacy}[sender_kind]
    name = "metadata_transport_test"
    platform_registry.register(PlatformEntry(name=name, label="Synthetic", adapter_factory=lambda _: None, check_fn=lambda: True, standalone_sender_fn=sender))
    from gateway.config import Platform
    platform = Platform(name)
    gateway_run = ModuleType("gateway.run")
    gateway_run._gateway_runner_ref = lambda: SimpleNamespace(adapters={platform: SimpleNamespace(send=live)}) if sender_kind == "live" else None
    monkeypatch.setitem(sys.modules, "gateway.run", gateway_run)
    metadata = {"job_id": "schedule", "execution_id": "a" * 32, "session_id": "cron_schedule_" + "a" * 32}
    original = dict(metadata)
    try:
        result = await _send_via_adapter(platform, SimpleNamespace(extra={}), "home", "Synthetic result", thread_id="thread", metadata=metadata)
    finally:
        platform_registry.unregister(name)
    assert result == {"success": True, "message_id": "sent"}
    if sender_kind == "legacy":
        assert received == [{"thread_id": "thread", "media_files": None, "force_document": False}]
    else:
        assert received == [{**original, **({"thread_id": "thread"} if sender_kind == "live" else {})}]
    assert metadata == original
