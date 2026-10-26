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
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
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
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
        executor.execute("open", [url])
    assert calls == []


@pytest.mark.parametrize("result", [False, None, {}, "PUBLIC", 1])
def test_unavailable_or_non_boolean_validation_fails_closed(executor_factory, result):
    executor, calls = executor_factory(lambda _: result)
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
        executor.execute("open", ["https://example.com/"])
    assert calls == []


def test_validator_exception_is_finite_and_private(executor_factory):
    def unavailable(_):
        raise TimeoutError("synthetic private URL must not appear")
    executor, calls = executor_factory(unavailable)
    with pytest.raises(BrowserBoundaryError) as error:
        executor.execute("open", ["https://example.com/"])
    assert str(error.value) == "blocked_target"
    assert calls == []


def test_redirect_is_validated_before_observation(executor_factory):
    executor, _ = executor_factory(lambda url: url == "https://example.com/")
    executor.pages = lambda: [{"url": "https://private-name.example/"}]
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
        executor.execute("snapshot", [])


def test_website_policy_is_preserved_for_open_and_redirect(executor_factory, monkeypatch):
    executor, calls = executor_factory(lambda _: True)
    monkeypatch.setattr(bt, "check_website_access", lambda _: "blocked by website policy")
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
        executor.execute("open", ["https://example.com/"])
    assert calls == []
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
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
    with pytest.raises(BrowserBoundaryError, match="blocked_target"):
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


# --- HPD-1722: the broker types the card; no process the model reaches can fetch it ---

FIELDS = {"number": "4242424242424242", "expiry": "12/28", "expiry_long": "12/2028", "cvc": "123", "name": "Max Muster"}


def _boundary(filler):
    boundary = IsolatedBrowserBoundary(lease, lambda *_: {"success": True}, lambda: True,
        agent_uid=os.geteuid() + 1, management_uid=os.geteuid(),
        session_id="synthetic", expires_at=time.time() + 300)
    boundary.card_fill = filler
    return boundary


def _card_request(**extra):
    request = {"session_id": "synthetic", "method": "card.fill", "approved_host": "shop.example.ch", "fields": dict(FIELDS)}
    request.update(extra)
    return request


def test_card_fill_is_a_management_method_only(monkeypatch):
    seen = []
    boundary = _boundary(lambda host, fields: seen.append((host, dict(fields))) or {"filled": ["number"], "missing": []})
    # The model's channel (agent uid) can never call it, even with valid fields.
    with pytest.raises(BrowserBoundaryError, match="caller_denied"):
        boundary.dispatch(boundary.agent_uid, _card_request())
    assert seen == []
    assert boundary.dispatch(boundary.management_uid, _card_request()) == {"filled": ["number"], "missing": []}
    assert seen == [("shop.example.ch", FIELDS)]


def test_card_fill_absent_without_a_trusted_service_filler():
    boundary = _boundary(None)
    with pytest.raises(BrowserBoundaryError, match="unknown_method"):
        boundary.dispatch(boundary.management_uid, _card_request())


@pytest.mark.parametrize("bad", [
    {"fields": {"number": "4242", "pan": "1"}},
    {"fields": {}},
    {"fields": {"number": ""}},
    {"fields": {"number": "4" * 65}},
    {"fields": {"number": 4242424242424242}},
    {"approved_host": 7},
    {"extra": "x"},
])
def test_card_fill_rejects_malformed_requests(bad):
    called = []
    boundary = _boundary(lambda *_: called.append(1))
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        boundary.dispatch(boundary.management_uid, _card_request(**bad))
    assert called == [] and boundary.active == 0


def test_card_fill_waits_for_nothing_while_a_human_holds_the_browser():
    called = []
    boundary = _boundary(lambda *_: called.append(1))
    lease.acquire("human-fixture")
    try:
        with pytest.raises(BrowserBoundaryError, match="human_has_control"):
            boundary.dispatch(boundary.management_uid, _card_request())
    finally:
        lease.release("human-fixture")
    assert called == []


class _Ws:
    """A scripted CDP page: records every Runtime.evaluate expression."""
    def __init__(self, page):
        self.page, self.pending, self.expressions = page, None, []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def send(self, data):
        import json
        message = json.loads(data)
        method, params = message["method"], message["params"]
        if method == "Page.getFrameTree":
            result = {"frameTree": {"frame": {"id": "main", "url": self.page["url"]},
                                    "childFrames": [{"frame": {"id": f["id"], "url": f["url"]}} for f in self.page.get("frames", [])]}}
        elif method == "Page.createIsolatedWorld":
            result = {"executionContextId": params["frameId"]}
        elif method == "Runtime.evaluate":
            self.expressions.append((params["contextId"], params["expression"]))
            if params["expression"] == "document.visibilityState":
                result = {"result": {"value": self.page.get("visibility", "visible")}}
            else:
                result = {"result": {"value": self.page["fill_result"](params["contextId"])}}
        else:
            result = {}
        self.pending = json.dumps({"id": message["id"], "result": result})

    def recv(self, timeout=None):
        return self.pending


def _filler(monkeypatch, pages):
    executor = object.__new__(service.NativeBrowserExecutor)
    executor.lock = __import__("threading").RLock()
    executor.pages = lambda: [{"url": p["url"], "webSocketDebuggerUrl": "ws://page/" + str(i)} for i, p in enumerate(pages)]
    sockets = {}
    def connect(url, **_):
        index = int(url.rsplit("/", 1)[1])
        return sockets.setdefault(index, _Ws(pages[index]))
    monkeypatch.setattr(service, "connect", connect)
    return executor, sockets


def test_fill_card_types_only_into_the_visible_page_on_the_approved_host(monkeypatch):
    page = {"url": "https://pay.shop.example.ch/checkout", "fill_result": lambda ctx: {"filled": ["number", "cvc", "name", "expiry"]},
            "frames": [{"id": "stripe", "url": "https://js.stripe.com/v3/elements"}]}
    executor, sockets = _filler(monkeypatch, [page])
    assert executor.fill_card("shop.example.ch", FIELDS) == {"filled": ["number", "expiry", "cvc", "name"], "missing": []}
    typed = [e for e in sockets[0].expressions if e[1] != "document.visibilityState"]
    # The main frame was typed into; the payment iframe on another host was not.
    assert [ctx for ctx, _ in typed] == ["main"]
    assert "4242424242424242" in typed[0][1] and '"shop.example.ch"' in typed[0][1]


def test_fill_card_refuses_a_page_on_another_host_and_types_nothing(monkeypatch):
    page = {"url": "https://shop.example.ch.evil.example/checkout", "fill_result": lambda ctx: {"filled": ["number"]}}
    executor, sockets = _filler(monkeypatch, [page])
    with pytest.raises(BrowserBoundaryError, match="host_mismatch"):
        executor.fill_card("shop.example.ch", FIELDS)
    assert all(e[1] == "document.visibilityState" for e in sockets[0].expressions)


def test_fill_card_refuses_plain_http_and_credentials_in_the_url(monkeypatch):
    for url in ("http://shop.example.ch/checkout", "https://user:pw@shop.example.ch/checkout"):
        executor, sockets = _filler(monkeypatch, [{"url": url, "fill_result": lambda ctx: {"filled": ["number"]}}])
        with pytest.raises(BrowserBoundaryError, match="host_mismatch"):
            executor.fill_card("shop.example.ch", FIELDS)
        assert all(e[1] == "document.visibilityState" for e in sockets[0].expressions)


def test_fill_card_page_side_host_recheck_stops_a_redirect_race(monkeypatch):
    page = {"url": "https://shop.example.ch/checkout", "fill_result": lambda ctx: {"host_mismatch": True}}
    executor, _ = _filler(monkeypatch, [page])
    with pytest.raises(BrowserBoundaryError, match="host_mismatch"):
        executor.fill_card("shop.example.ch", FIELDS)


def test_fill_card_ignores_a_background_tab_and_needs_a_visible_one(monkeypatch):
    hidden = {"url": "https://shop.example.ch/checkout", "visibility": "hidden", "fill_result": lambda ctx: {"filled": ["number"]}}
    executor, sockets = _filler(monkeypatch, [hidden])
    with pytest.raises(BrowserBoundaryError, match="no_visible_page"):
        executor.fill_card("shop.example.ch", FIELDS)
    assert all(e[1] == "document.visibilityState" for e in sockets[0].expressions)


def test_fill_card_reports_missing_fields_by_name_only(monkeypatch):
    page = {"url": "https://shop.example.ch/checkout", "fill_result": lambda ctx: {"filled": ["number"]}}
    executor, _ = _filler(monkeypatch, [page])
    result = executor.fill_card("shop.example.ch", FIELDS)
    assert result == {"filled": ["number"], "missing": ["expiry", "cvc", "name"]}
    assert "4242" not in repr(result)


@pytest.mark.parametrize("host", ["", "localhost", "10.0.0.1", "shop", "a b.example", "-x.example.ch", None, 5])
def test_fill_card_needs_a_plain_dns_host(monkeypatch, host):
    executor, _ = _filler(monkeypatch, [])
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.fill_card(host, FIELDS)


def test_service_connects_the_filler_to_the_native_boundary(monkeypatch):
    source = __import__("inspect").getsource(service.run)
    assert 'boundary.card_fill = getattr(executor, "fill_card", None)' in source


SNAPSHOT = ('- textbox "Card number" [ref=e1]: 4242424242424242\n- textbox "Security code (CVC)" [ref=e2]: 987\n'
            '- textbox "Name on card" [ref=e3]: Max Muster\n  - StaticText "4242424242424242"\n  - StaticText "987"\n'
            '- text: Total CHF 987.00\n- text: Paid with 4242 4242 4242 4242 and again 4242-4242-4242-4242\n'
            '- textbox "Coupon" [ref=e4]: SAVE10\n')


def test_agent_snapshot_after_typing_hides_what_was_typed(executor_factory, monkeypatch):
    executor, calls = executor_factory(lambda _: True)
    monkeypatch.setattr(bt, "_run_browser_command", lambda sid, command, args: {"success": True, "data": {"snapshot": SNAPSHOT}})
    before = executor.execute("snapshot", [])
    assert "4242424242424242" in before["data"]["snapshot"]  # nothing hidden before a fill
    fields = {"number": "4242424242424242", "expiry": "12/28", "expiry_long": "12/2028", "cvc": "987", "name": "Max Muster"}
    monkeypatch.setattr(service, "connect", lambda *_a, **_k: _Ws({"url": "https://shop.example.ch/", "fill_result": lambda ctx: {"filled": ["number", "cvc"]}}))
    executor.pages = lambda: [{"url": "https://shop.example.ch/", "webSocketDebuggerUrl": "ws://page/0"}]
    executor.fill_card("shop.example.ch", fields)
    after = executor.execute("snapshot", [])["data"]["snapshot"]
    for secret in ("4242424242424242", "4242 4242 4242 4242", "4242-4242-4242-4242", "Max Muster", "SAVE10"):
        assert secret not in after
    assert "Total CHF 987.00" in after  # a price that looks like the CVC stays readable
    assert 'StaticText "[hidden]"' in after


def test_card_mask_has_no_timeout(executor_factory, monkeypatch):
    executor, _ = executor_factory(lambda _: True)
    monkeypatch.setattr(bt, "_run_browser_command", lambda sid, command, args: {"success": True, "data": {"snapshot": SNAPSHOT}})
    executor._card_typed = {"numbers": ("4242424242424242",), "cvcs": ("987",), "strings": ("Max Muster",)}
    assert "SAVE10" not in executor.execute("snapshot", [])["data"]["snapshot"]
    # No timeout: the values stay in the page until a human hand-back empties it.
    monkeypatch.setattr(service.time, "monotonic", lambda: 10 ** 9)
    assert "SAVE10" not in executor.execute("snapshot", [])["data"]["snapshot"]
    monkeypatch.undo()


def test_luhn_valid_numbers_are_hidden_even_if_not_the_typed_one():
    typed = {"numbers": (), "cvcs": (), "strings": ()}
    masked = service._mask_card_text("order 4111 1111 1111 1111 ref 1234567890123", typed)
    assert "4111" not in masked and "1234567890123" in masked


def test_a_second_fill_adds_to_the_mask_and_value_keys_are_hidden(executor_factory, monkeypatch):
    executor, _ = executor_factory(lambda _: True)
    monkeypatch.setattr(service, "connect", lambda *_a, **_k: _Ws({"url": "https://shop.example.ch/", "fill_result": lambda ctx: {"filled": ["number"]}}))
    executor.pages = lambda: [{"url": "https://shop.example.ch/", "webSocketDebuggerUrl": "ws://page/0"}]
    executor.fill_card("shop.example.ch", {"number": "4111111111111111", "cvc": "321", "name": "Anna Beispiel"})
    executor.fill_card("shop.example.ch", {"number": "4242424242424242", "cvc": "987", "name": "Max Muster"})
    assert set(executor._card_typed["numbers"]) == {"4111111111111111", "4242424242424242"}
    monkeypatch.setattr(bt, "_run_browser_command", lambda sid, c, a: {"success": True, "data": {
        "snapshot": "- textbox [ref=e1]: Anna Beispiel 4111 1111 1111 1111", "refs": {"e1": {"role": "textbox", "value": "4242424242424242"}}}})
    seen = executor.execute("snapshot", [])["data"]
    assert "Anna" not in seen["snapshot"] and "4111" not in seen["snapshot"] and seen["refs"]["e1"]["value"] == "[hidden]"


@pytest.mark.parametrize("host", ["github.io", "evil.vercel.app", "co.uk", "com.au", "x.co.uk"])
def test_shared_hosting_and_public_suffixes_never_receive_a_card(monkeypatch, host):
    executor, _ = _filler(monkeypatch, [{"url": "https://" + host + "/", "fill_result": lambda ctx: {"filled": ["number"]}}])
    if host in ("evil.vercel.app", "x.co.uk"):
        executor.fill_card(host, FIELDS)  # a merchant's own registrable name under a shared suffix is allowed
        return
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        executor.fill_card(host, FIELDS)
