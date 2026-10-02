"""Service-owned URL checks when only the protected egress can resolve DNS."""
import socket
import os
import time
import asyncio
from types import SimpleNamespace

import pytest

from tools import browser_tool as bt
from tools import isolated_browser_service as service
from tools.isolated_browser import BrowserBoundaryError
from tools.isolated_browser import IsolatedBrowserBoundary
from tools.bot_desktop import lease


def test_epoch_free_cancel_exits_service_and_closes_owned_resources(monkeypatch, tmp_path):
    """Exercise the service loop/finally with bounded process/socket fixtures."""
    import subprocess
    import sys
    from contextlib import asynccontextmanager

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    servers, stopped = [], []
    class Server:
        def __init__(self, path, boundary, **kwargs):
            self.server_address = path
            self.boundary = boundary
            self.closed = False
            self.shutdown_called = False
            from pathlib import Path
            Path(path).touch()
            servers.append(self)
        def serve_forever(self):
            pass
        def shutdown(self):
            self.shutdown_called = True
        def server_close(self):
            self.closed = True
    @asynccontextmanager
    async def viewer_server(*args, **kwargs):
        boundary = servers[1].boundary
        boundary.dispatch(0, {"session_id": "synthetic", "method": "display.lease.acquire",
                              "epoch": lease.get().epoch, "viewer_id": "fixture-viewer"})
        assert boundary.dispatch(0, {"session_id": "synthetic", "method": "display.cancel"}) == {"revoked": True}
        yield
    executor = SimpleNamespace(process=process, execute=lambda *_: {"success": True}, safe_return=lambda: True)
    monkeypatch.setattr(service, "NativeBrowserExecutor", lambda *_a, **_k: executor)
    monkeypatch.setattr(service.os, "geteuid", lambda: 10002)
    monkeypatch.setattr(service, "BrowserRpcServer", Server)
    monkeypatch.setattr(service, "serve", viewer_server)
    monkeypatch.setattr(service.runtime, "rfb_socket_path", lambda: tmp_path / "rfb.sock")
    monkeypatch.setattr(service.runtime, "stop", lambda: stopped.append(True))
    manifest = {"session_id": "synthetic", "expires_at": time.time()+300, "agent_uid": 10001,
        "management_uid": 0, "agent_socket": str(tmp_path / "agent.sock"),
        "management_socket": str(tmp_path / "management.sock"), "agent_group": 10001, "viewer_port": 0}
    try:
        asyncio.run(service.run(manifest))
        assert process.wait(timeout=3) is not None
        assert stopped == [True] and all(s.closed and s.shutdown_called for s in servers)
        assert not (tmp_path / "agent.sock").exists() and not (tmp_path / "management.sock").exists()
        assert lease.human_holds() and not servers[0].boundary.viewer_may_input("fixture-viewer")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


@pytest.fixture
def executor_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(service.runtime, "start", lambda: None)
    monkeypatch.setattr(service.runtime, "desktop_env", lambda: {})
    monkeypatch.setattr(service.browser, "dock_launch", lambda: ("chromium", tmp_path))
    monkeypatch.setattr(service.browser, "dock_argv", lambda *_: ["chromium"])
    monkeypatch.setattr(service.browser, "running_instance_cdp_port", lambda _: 9222)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(terminate=lambda: None))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: (_ for _ in ()).throw(socket.gaierror()))
    monkeypatch.setattr(bt, "check_website_access", lambda _: None)
    calls = []
    monkeypatch.setattr(bt, "_run_browser_command", lambda sid, command, args:
                        calls.append((command, args)) or {"success": True, "data": {"url": args[0] if args else ""}})

    def create(validator=None):
        executor = service.NativeBrowserExecutor("synthetic", trusted_url_validator=validator)
        executor.pages = lambda: [{"url": "https://example.com/"}]
        return executor, calls
    return create


def test_public_open_uses_service_validator_with_local_dns_unavailable(executor_factory):
    checked = []
    executor, calls = executor_factory(lambda url: checked.append(url) or True)
    assert executor.execute("open", ["https://example.com/"])["success"]
    assert checked == ["https://example.com/", "https://example.com/"]
    assert calls == [("open", ["https://example.com/"])]


def test_standalone_default_still_fails_closed_on_dns_failure(executor_factory):
    executor, calls = executor_factory()
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("open", ["https://example.com/"])
    assert calls == []


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://10.0.0.1/", "http://169.254.169.254/",
    "http://100.100.100.200/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
    "http://metadata.google.internal/", "http://metadata.goog/", "file:///etc/passwd",
    "https://user:password@example.com/", "http://localhost/", "http://localhost./",
])
def test_trusted_validator_cannot_override_literal_floor(executor_factory, url):
    executor, calls = executor_factory(lambda _: True)
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("open", [url])
    assert calls == []


@pytest.mark.parametrize("result", [False, None, {}, "PUBLIC", 1])
def test_unavailable_or_non_boolean_validation_fails_closed(executor_factory, result):
    executor, calls = executor_factory(lambda _: result)
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("open", ["https://example.com/"])
    assert calls == []


def test_validator_exception_is_finite_and_private(executor_factory):
    def unavailable(_):
        raise TimeoutError("synthetic private URL must not appear")
    executor, calls = executor_factory(unavailable)
    with pytest.raises(BrowserBoundaryError) as error:
        executor.execute("open", ["https://example.com/"])
    assert str(error.value) == "command_denied"
    assert calls == []


def test_redirect_is_validated_before_observation(executor_factory):
    executor, _ = executor_factory(lambda url: url == "https://example.com/")
    executor.pages = lambda: [{"url": "https://private-name.example/"}]
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("snapshot", [])


def test_website_policy_is_preserved_for_open_and_redirect(executor_factory, monkeypatch):
    executor, calls = executor_factory(lambda _: True)
    monkeypatch.setattr(bt, "check_website_access", lambda _: "blocked by website policy")
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("open", ["https://example.com/"])
    assert calls == []
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.execute("snapshot", [])


def test_safe_return_uses_same_service_validator(executor_factory, monkeypatch):
    checked = []
    executor, _ = executor_factory(lambda url: checked.append(url) or False)
    monkeypatch.setattr(service, "clear_remote_clipboard", lambda _: None)
    assert executor.safe_return() is False
    assert checked == ["https://example.com/"]


def test_agent_rpc_cannot_replace_service_validator(executor_factory):
    executor, calls = executor_factory(lambda _: False)
    boundary = IsolatedBrowserBoundary(lease, executor.execute, executor.safe_return,
        agent_uid=os.geteuid()+1, management_uid=os.geteuid(),
        session_id="synthetic", expires_at=time.time()+300)
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        boundary.dispatch(boundary.agent_uid, {"session_id": "synthetic", "method": "browser.command",
            "command": "open", "args": ["https://example.com/"], "trusted_url_validator": True})
    assert calls == []


@pytest.mark.parametrize("supply_callback", [False, True])
def test_service_wires_only_constructor_callback_not_manifest(monkeypatch, supply_callback):
    callback = lambda _: True
    observed = []
    class StopAfterProvision(Exception):
        pass
    def executor(session_id, *, trusted_url_validator=None):
        observed.append((session_id, trusted_url_validator))
        raise StopAfterProvision()
    monkeypatch.setattr(service, "NativeBrowserExecutor", executor)
    monkeypatch.setattr(service.os, "geteuid", lambda: 10002)
    manifest = {"session_id": "synthetic", "expires_at": time.time()+300, "agent_uid": 10001,
        "management_uid": 0, "agent_socket": "/unused/agent", "management_socket": "/unused/management",
        "agent_group": 10001, "viewer_port": 4321, "trusted_url_validator": True}
    kwargs = {"trusted_url_validator": callback} if supply_callback else {}
    with pytest.raises(StopAfterProvision):
        asyncio.run(service.run(manifest, **kwargs))
    assert observed == [("synthetic", callback if supply_callback else None)]
