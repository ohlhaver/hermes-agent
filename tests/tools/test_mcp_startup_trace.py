"""Synthetic entrypoint/transport tests; no MCP server or provider is started."""
import asyncio
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools.mcp_startup_trace import PREFIX, emit_phase, wrap_entrypoint

ID = "a" * 32


def records(stderr):
    return [json.loads(line[len(PREFIX):]) for line in stderr.splitlines() if line.startswith(PREFIX)]


@pytest.fixture
def entrypoint(tmp_path):
    package = tmp_path / "basic_memory"
    for relative in ("", "cli", "mcp"):
        folder = package / relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    (package / "cli/main.py").write_text("from basic_memory.mcp import tools\napp = 1\n")
    (package / "mcp/tools.py").write_text("from basic_memory.mcp.server import ready\n")
    (package / "mcp/server.py").write_text("ready = True\n")
    script = tmp_path / "basic-memory"
    script.write_text(f"#!{sys.executable}\nfrom basic_memory.cli.main import app\n"
        "import sys, json, os\nprint(json.dumps({'argv':sys.argv,'input':sys.stdin.read(),'cwd':os.getcwd()}))\nraise SystemExit(7)\n")
    script.chmod(0o700)
    return script


def test_trace_preserves_original_entrypoint_argv_stdio_and_exit(entrypoint):
    args = ["mcp", "--project", "PRIVATE_PROJECT"]
    env = {"HOME": str(entrypoint.parent), "PATH": os.defpath}
    base = subprocess.run([sys.executable, str(entrypoint), *args], input="PRIVATE_STDIN", capture_output=True, text=True, env=env, timeout=10)
    command, traced_args = wrap_entrypoint(str(entrypoint), args, ID, time.monotonic_ns())
    traced = subprocess.run([command, *traced_args], input="PRIVATE_STDIN", capture_output=True, text=True, env=env, timeout=10)
    assert base.returncode == traced.returncode == 7
    assert base.stdout == traced.stdout
    evidence = records(traced.stderr)
    phases = [item["phase"] for item in evidence]
    assert phases == ["child_entry", "cli_import_start", "tools_import_start", "server_import_start", "server_import_complete", "tools_import_complete", "cli_import_complete", "entrypoint_exit"]
    assert all(set(item) == {"startupId", "phase", "elapsedMs"} and item["startupId"] == ID for item in evidence)
    assert [item["elapsedMs"] for item in evidence] == sorted(item["elapsedMs"] for item in evidence)
    assert "PRIVATE" not in json.dumps(evidence)
    assert str(entrypoint.parent) not in json.dumps(evidence)


@pytest.mark.parametrize("mode", ["foreign_command", "foreign_interpreter", "non_mcp", "invalid_id"])
def test_unknown_entrypoints_remain_unmodified(entrypoint, mode):
    command, args, identity = str(entrypoint), ["mcp"], ID
    if mode == "foreign_command":
        command = str(entrypoint.with_name("other"))
    elif mode == "foreign_interpreter":
        entrypoint.write_text("#!/usr/bin/env python\n")
    elif mode == "non_mcp":
        args = ["--version"]
    else:
        identity = "PRIVATE"
    assert wrap_entrypoint(command, args, identity, time.monotonic_ns()) == (command, args)


def test_bounded_event_and_sink_failure_do_not_escape():
    output = io.StringIO()
    emit_phase(ID, time.monotonic_ns(), "child_entry", output)
    emit_phase("PRIVATE", time.monotonic_ns(), "child_entry", output)
    emit_phase(ID, time.monotonic_ns(), "PRIVATE", output)
    assert len(records(output.getvalue())) == 1
    class Broken:
        def write(self, *_): raise OSError("PRIVATE")
    emit_phase(ID, time.monotonic_ns(), "child_entry", Broken())


def test_serving_discovery_id_reaches_transport_and_child(entrypoint, monkeypatch):
    from tools import mcp_tool
    sink = io.StringIO()
    params_seen = []
    @asynccontextmanager
    async def stdio(params, **kwargs):
        params_seen.append(params)
        yield (object(), object())
    @asynccontextmanager
    async def session(*args, **kwargs):
        yield SimpleNamespace(initialize=AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "stdio_client", stdio)
    monkeypatch.setattr(mcp_tool, "ClientSession", session)
    monkeypatch.setattr(mcp_tool, "_get_mcp_stderr_log", lambda: sink)
    monkeypatch.setattr(mcp_tool, "_kill_orphaned_mcp_children", lambda: None)
    monkeypatch.setattr(mcp_tool, "_snapshot_child_pids", lambda: set())
    monkeypatch.setattr("tools.osv_check.check_package_for_malware", lambda *args: None)
    discovery = AsyncMock()
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_discover_tools", discovery)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event", AsyncMock(return_value="shutdown"))
    async def exercise():
        server = mcp_tool.MCPServerTask("basic_memory")
        server._startup_trace = (ID, time.monotonic_ns())
        await server._run_stdio({"command": str(entrypoint), "args": ["mcp"], "connect_timeout": 30})
        server._discover_tools.assert_awaited_once()
    asyncio.run(exercise())
    assert len(params_seen) == 1
    assert ID in params_seen[0].args
    assert str(entrypoint) in params_seen[0].args
    assert [item["phase"] for item in records(sink.getvalue())] == ["stdio_prepare", "child_spawn_start", "initialize_start", "initialize_complete", "tools_list_start", "tools_list_complete"]


def test_sibling_interpreter_alias_is_not_same_environment(entrypoint, tmp_path):
    sibling = tmp_path / "other-venv" / "bin" / "python"
    sibling.parent.mkdir(parents=True)
    sibling.symlink_to(os.path.realpath(sys.executable))
    entrypoint.write_text(f"#!{sibling}\n")
    assert os.path.realpath(sibling) == os.path.realpath(sys.executable)
    assert wrap_entrypoint(str(entrypoint), ["mcp"], ID, time.monotonic_ns()) == (str(entrypoint), ["mcp"])


def test_non_executable_entrypoint_is_not_wrapped(entrypoint):
    entrypoint.chmod(0o600)
    assert wrap_entrypoint(str(entrypoint), ["mcp"], ID, time.monotonic_ns()) == (str(entrypoint), ["mcp"])


def test_connect_binds_diagnostic_on_real_server_object(monkeypatch):
    from tools import mcp_tool
    start = AsyncMock()
    sink = io.StringIO()
    monkeypatch.setattr(mcp_tool.MCPServerTask, "start", start)
    monkeypatch.setattr(mcp_tool, "_get_mcp_stderr_log", lambda: sink)
    server = asyncio.run(mcp_tool._connect_server("basic_memory", {}, _diagnostic={"startupId": ID}))
    start.assert_awaited_once_with({})
    assert server._startup_trace[0] == ID
    assert [item["phase"] for item in records(sink.getvalue())] == ["connect_enter"]


@pytest.mark.parametrize("import_statement", ["import basic_memory.mcp.tools", "from basic_memory.mcp import tools"])
def test_failed_import_has_no_completed_marker(entrypoint, import_statement):
    (entrypoint.parent / "basic_memory/cli/main.py").write_text(import_statement + "\n")
    (entrypoint.parent / "basic_memory/mcp/tools.py").write_text("raise RuntimeError('synthetic failure')\n")
    command, args = wrap_entrypoint(str(entrypoint), ["mcp"], ID, time.monotonic_ns())
    result = subprocess.run([command, *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert result.stdout == ""
    assert [item["phase"] for item in records(result.stderr)] == ["child_entry", "cli_import_start", "tools_import_start", "entrypoint_exit"]
