"""Process-bound MCP startup evidence without launching MCP or providers."""
import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import _discover_gateway_mcp
from tools import mcp_tool


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    monkeypatch.setattr(mcp_tool, "_server_connect_errors", {})
    # Any accidental attempt to launch an event loop/child is a test failure.
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: pytest.fail("MCP launch forbidden"))
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  basic_memory:\n    command: basic-memory\n    enabled: true\n"
        "    env:\n      BASIC_MEMORY_CONFIG_DIR: /synthetic/PRIVATE_PATH\n"
    )
    return tmp_path


def test_bound_runner_records_real_config_load_and_registration(isolated_config, monkeypatch):
    calls = []
    def register(servers, **kwargs):
        calls.append(list(servers))
        kwargs["_diagnostic"]["failedServerCount"] = 0
        mcp_tool._servers["basic_memory"] = SimpleNamespace(_registered_tool_names=["mcp__basic_memory__search_notes"])
        return ["mcp__basic_memory__search_notes"]
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", register)
    first, second = SimpleNamespace(), SimpleNamespace()
    asyncio.run(_discover_gateway_mcp(first))
    snapshot = dict(first._mcp_startup_diagnostic)
    asyncio.run(_discover_gateway_mcp(second))
    assert calls == [["basic_memory"], ["basic_memory"]]
    assert first._mcp_startup_diagnostic == snapshot
    assert re.fullmatch(r"[0-9a-f]{32}", snapshot.pop("startupId"))
    assert first._mcp_startup_diagnostic["startupId"] != second._mcp_startup_diagnostic["startupId"]
    assert snapshot == dict(phase="complete", errorClass="none", sdkAvailable=True,
        configLoaded=True, basicMemoryConfigured=True, discoveryAttempted=True,
        configuredServerCount=1, enabledServerCount=1, registeredToolCount=1,
        basicMemoryToolCount=1, failedServerCount=0)
    assert "PRIVATE" not in json.dumps(snapshot)


@pytest.mark.parametrize("mode,phase,config_loaded,error", [
    ("sdk", "sdk_unavailable", None, "import"),
    ("safe", "safe_mode", None, "none"),
    ("empty", "no_servers", True, "none"),
    ("config", "config_failed", False, "configuration"),
])
def test_early_exits_remain_distinct(isolated_config, monkeypatch, mode, phase, config_loaded, error):
    if mode == "sdk":
        monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", False)
    elif mode == "safe":
        monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    elif mode == "empty":
        (isolated_config / "config.yaml").write_text("{}\n")
    else:
        def broken():
            raise ValueError("PRIVATE_CONFIG_CONTENT")
        monkeypatch.setattr("hermes_cli.config.load_config", broken)
    runner = SimpleNamespace()
    asyncio.run(_discover_gateway_mcp(runner))
    data = runner._mcp_startup_diagnostic
    assert (data["phase"], data["configLoaded"], data["errorClass"]) == (phase, config_loaded, error)
    assert data["discoveryAttempted"] is False
    assert data["registeredToolCount"] is None
    assert "PRIVATE" not in json.dumps(data)


@pytest.mark.parametrize("exc,category", [(ImportError("PRIVATE"), "import"), (TimeoutError("PRIVATE"), "timeout"), (RuntimeError("PRIVATE"), "other")])
def test_startup_exception_is_bounded_and_does_not_change_fail_open(isolated_config, monkeypatch, exc, category):
    def fail(**kwargs):
        raise exc
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", fail)
    runner = SimpleNamespace()
    asyncio.run(_discover_gateway_mcp(runner))
    assert runner._mcp_startup_diagnostic["phase"] == "failed"
    assert runner._mcp_startup_diagnostic["errorClass"] == category
    assert "PRIVATE" not in json.dumps(runner._mcp_startup_diagnostic)


def test_server_failure_retains_only_category(isolated_config, monkeypatch):
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", lambda factory, **kwargs: asyncio.run(factory()))
    monkeypatch.setattr(mcp_tool, "_discover_and_register_server", AsyncMock(side_effect=TimeoutError("PRIVATE_EXCEPTION")))
    runner = SimpleNamespace()
    asyncio.run(_discover_gateway_mcp(runner))
    data = runner._mcp_startup_diagnostic
    assert data["phase"] == "complete"
    assert data["discoveryAttempted"] is True
    assert data["registeredToolCount"] == 0
    assert data["failedServerCount"] == 1
    assert data["errorClass"] == "timeout"
    assert "PRIVATE" not in json.dumps(data)


def test_registration_failure_after_connection_is_counted(isolated_config, monkeypatch):
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", lambda factory, **kwargs: asyncio.run(factory()))
    monkeypatch.setattr(mcp_tool, "_connect_server", AsyncMock(return_value=SimpleNamespace(_tools=[], name="basic_memory")))
    def broken_registration(*args):
        raise ValueError("PRIVATE_SCHEMA_CONTENT")
    monkeypatch.setattr(mcp_tool, "_register_server_tools", broken_registration)
    runner = SimpleNamespace()
    asyncio.run(_discover_gateway_mcp(runner))
    assert "basic_memory" in mcp_tool._servers
    data = runner._mcp_startup_diagnostic
    assert data["failedServerCount"] == 1
    assert data["errorClass"] == "discovery"
    assert "PRIVATE" not in json.dumps(data)
