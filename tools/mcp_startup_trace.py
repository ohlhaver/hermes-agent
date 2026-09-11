"""Bounded, content-free phase markers for the managed Basic Memory startup.

Runs the original same-interpreter entrypoint, preserving its argv and streams.
Markers use the already-open MCP stderr sink; stdout remains the MCP transport.
"""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time

PREFIX = "HERMES_MCP_STARTUP "
PHASES = frozenset({
    "connect_enter", "stdio_prepare", "child_spawn_start", "initialize_start",
    "initialize_complete", "tools_list_start", "tools_list_complete",
    "child_entry", "cli_import_start", "cli_import_complete",
    "tools_import_start", "tools_import_complete", "server_import_start",
    "server_import_complete", "entrypoint_exit",
})
IMPORT_PHASES = {
    "basic_memory.cli.main": "cli_import",
    "basic_memory.mcp.tools": "tools_import",
    "basic_memory.mcp.server": "server_import",
}


def emit_phase(startup_id: str, started_ns: int, phase: str, stream=None) -> None:
    """Never propagate a diagnostic write failure into MCP startup."""
    if (not isinstance(startup_id, str) or re.fullmatch(r"[0-9a-f]{32}", startup_id) is None
            or type(started_ns) is not int or started_ns <= 0 or phase not in PHASES):
        return
    try:
        elapsed_ms = max(0, min(600000, (time.monotonic_ns() - started_ns) // 1000000))
        target = stream if stream is not None else sys.stderr
        target.write(PREFIX + json.dumps({"startupId": startup_id, "phase": phase, "elapsedMs": elapsed_ms}) + "\n")
        target.flush()
    except Exception:
        pass


def wrap_entrypoint(command: str, args: list, startup_id: str, started_ns: int) -> tuple[str, list]:
    """Trace only basic-memory mcp's original same-interpreter Python script."""
    if (not isinstance(startup_id, str) or re.fullmatch(r"[0-9a-f]{32}", startup_id) is None
            or type(started_ns) is not int or started_ns <= 0
            or os.name != "posix" or not os.path.isabs(command)
            or not os.access(command, os.X_OK)
            or os.path.basename(command) != "basic-memory" or not args or args[0] != "mcp"):
        return command, args
    try:
        with open(command, "rb") as script:
            shebang = script.readline(256).decode("utf-8").strip()
        if not shebang.startswith("#!") or not os.path.isabs(shebang[2:]) or os.path.normpath(shebang[2:]) != os.path.normpath(sys.executable):
            return command, args
    except (OSError, UnicodeError):
        return command, args
    return sys.executable, [str(Path(__file__).resolve()), startup_id, str(started_ns), command, *args]


def run_entrypoint(command: str, args: list[str], startup_id: str, started_ns: int) -> None:
    original_import = builtins.__import__
    original_argv, original_path = sys.argv, list(sys.path)
    seen = set()

    def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
        targets = [name] + [name + "." + item for item in (fromlist or ()) if isinstance(item, str)]
        phases = [(target, IMPORT_PHASES[target]) for target in targets
                  if level == 0 and target in IMPORT_PHASES and target not in seen]
        if not phases:
            return original_import(name, globals, locals, fromlist, level)
        for target, phase in phases:
            seen.add(target)
            emit_phase(startup_id, started_ns, phase + "_start")
        result = original_import(name, globals, locals, fromlist, level)
        for target, phase in reversed(phases):
            emit_phase(startup_id, started_ns, phase + "_complete")
        return result

    try:
        sys.argv = [command, *args]
        sys.path[0] = os.path.dirname(os.path.abspath(command))
        builtins.__import__ = traced_import
        emit_phase(startup_id, started_ns, "child_entry")
        runpy.run_path(command, run_name="__main__")
    finally:
        builtins.__import__ = original_import
        sys.argv = original_argv
        sys.path[:] = original_path
        emit_phase(startup_id, started_ns, "entrypoint_exit")


if __name__ == "__main__":
    run_entrypoint(sys.argv[3], sys.argv[4:], sys.argv[1], int(sys.argv[2]))
