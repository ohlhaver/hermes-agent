import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tools.delegation_diagnostic import summarize_delegation_request
from tools import async_delegation, delegate_tool
from gateway.session_context import _SESSION_ASYNC_DELIVERY


@pytest.mark.parametrize("mode,entry", [
    ("chat_completions", {"type": "function", "function": {"name": "delegate_task", "parameters": {"type": "object"}}}),
    ("codex_responses", {"type": "function", "name": "delegate_task", "parameters": {"type": "object"}}),
    ("anthropic_messages", {"name": "delegate_task", "input_schema": {"type": "object"}}),
    ("bedrock_converse", {"toolSpec": {"name": "delegate_task", "inputSchema": {"json": {"type": "object"}}}}),
])
def test_final_schema_digest_and_no_mutation(mode, entry):
    request = {"tools": [entry], "messages": ["PRIVATE_CONTENT"]}
    if mode == "bedrock_converse":
        request = {"toolConfig": {"tools": [entry]}, "messages": ["PRIVATE_CONTENT"]}
    before = copy.deepcopy(request)
    result = summarize_delegation_request(request, mode, SimpleNamespace())
    assert result["present"] is True
    assert result["schemaSha256"] == hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert request == before
    assert "PRIVATE" not in json.dumps(result)


def test_absent_unknown_and_duplicate_schema():
    assert summarize_delegation_request({"tools": []}, "chat_completions", None)["present"] is False
    for request, mode in [({"tools": [None]}, "chat_completions"), ({}, "unknown")]:
        result = summarize_delegation_request(request, mode, None)
        assert result["present"] is None and result["schemaSha256"] is None
    entry = {"type": "function", "function": {"name": "delegate_task"}}
    result = summarize_delegation_request({"tools": [entry, entry]}, "chat_completions", None)
    assert result["present"] is True and result["schemaSha256"] is None


def test_live_guard_snapshot_without_dispatch_or_durable_io(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_spawn_paused", True)
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 1)
    monkeypatch.setattr(async_delegation, "_records", {"private-id": {"status": "running", "goal": "PRIVATE"}})
    token = _SESSION_ASYNC_DELIVERY.set(False)
    try:
        with patch.object(async_delegation, "_connect", side_effect=AssertionError("must not read/write DB")), patch.object(delegate_tool, "delegate_task", side_effect=AssertionError("must not dispatch")):
            result = summarize_delegation_request({}, "chat_completions", SimpleNamespace(_delegate_depth=1))
            assert result == dict(present=False, schemaSha256=None, parentPresent=True,
                depthAllowed=False, spawnPaused=True, asyncDeliverySupported=False, asyncCapacityAvailable=False)
            async_delegation._records["private-id"]["status"] = "finalizing"
            assert summarize_delegation_request({}, "chat_completions", None)["asyncCapacityAvailable"] is True
    finally:
        _SESSION_ASYNC_DELIVERY.reset(token)
