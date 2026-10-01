"""HPD-321 tests real native file leases and existing browser dispatch."""
import concurrent.futures
import json
import os
import threading
import time

import pytest

from tools.bot_desktop import lease
from tools.isolated_browser import BrowserBoundaryError, IsolatedBrowserBoundary


@pytest.fixture
def boundary():
    return IsolatedBrowserBoundary(lease, lambda command, args: {"success": True},
                                   lambda: True, agent_uid=os.geteuid() + 1,
                                   management_uid=os.geteuid(), session_id="test-run",
                                   expires_at=time.time() + 300)


def call(boundary, method, **values):
    return boundary.dispatch(boundary.management_uid, {
        "session_id": "test-run", "method": method, "epoch": lease.get().epoch, **values})


def agent(boundary, command="snapshot", args=None):
    return boundary.dispatch(boundary.agent_uid, {
        "session_id": "test-run", "method": "browser.command", "command": command, "args": args or []})


def test_same_native_lease_and_explicit_handback(boundary):
    assert agent(boundary)["success"]
    granted = call(boundary, "display.lease.acquire", viewer_id="viewer-a")
    assert granted["holder"] == "human" and granted["viewer_id"] is None
    assert lease.human_holds() and boundary.viewer_may_input("viewer-a")
    assert not boundary.viewer_may_input("viewer-b")
    with pytest.raises(BrowserBoundaryError, match="human_has_control"):
        agent(boundary)
    call(boundary, "display.lease.release", viewer_id="viewer-a")
    assert not lease.human_holds() and agent(boundary)["success"]


@pytest.mark.parametrize("method", ["display.status", "display.lease.acquire", "display.lease.release", "display.cancel"])
def test_agent_cannot_elevate_or_read_management_state(boundary, method):
    with pytest.raises(BrowserBoundaryError, match="caller_denied"):
        boundary.dispatch(boundary.agent_uid, {"session_id": "test-run", "method": method})


@pytest.mark.parametrize("command,args", [
    ("cookies", []), ("screenshot", ["/tmp/out.png"]), ("record", ["start"]),
    ("eval", ["document.cookie"]), ("eval", ["document.querySelector('input').value"]),
    ("console", []), ("errors", []), ("get", ["value", "@e1"]),
    ("press", ["Control+Shift+J"]), ("click", ["input[type=password]"]),
    ("open", ["--profile", "/private"]), ("snapshot", ["-d", "100"]),
    ("fill", ["@e1", "text", "-p", "external"]), ("type", ["-p"]), ("back", ["--compact"]),
    ("scroll", ["down", "10001"]),
])
def test_broker_denies_observation_and_cli_escape_paths(boundary, command, args):
    with pytest.raises(BrowserBoundaryError, match="command_denied"):
        agent(boundary, command, args)


def test_takeover_drains_admitted_action_before_human_can_input(boundary):
    started, finish = threading.Event(), threading.Event()
    def execute(command, args):
        started.set()
        assert finish.wait(3)
        return {"success": True, "data": {"snapshot": "late output must be discarded"}}
    boundary.execute = execute
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(agent, boundary)
        assert started.wait(2)
        takeover = pool.submit(call, boundary, "display.lease.acquire", viewer_id="viewer-a")
        deadline = time.monotonic() + 2
        while not lease.human_holds() and time.monotonic() < deadline:
            time.sleep(.005)
        assert lease.human_holds()
        assert not boundary.viewer_may_input("viewer-a")
        assert not takeover.done()
        finish.set()
        with pytest.raises(BrowserBoundaryError, match="human_has_control"):
            active.result(timeout=2)
        assert takeover.result(timeout=2)["holder"] == "human"
        assert boundary.viewer_may_input("viewer-a")


def test_unsafe_handback_retains_input_for_same_human_to_correct(boundary):
    call(boundary, "display.lease.acquire", viewer_id="viewer-a")
    boundary.safe_return = lambda: False
    with pytest.raises(BrowserBoundaryError, match="unsafe_return"):
        call(boundary, "display.lease.release", viewer_id="viewer-a")
    assert lease.human_holds() and boundary.viewer_may_input("viewer-a")
    boundary.safe_return = lambda: True
    assert call(boundary, "display.lease.release", viewer_id="viewer-a")["holder"] == "agent"


def test_wrong_viewer_stale_revision_and_missing_viewer_do_not_release(boundary):
    call(boundary, "display.lease.acquire", viewer_id="viewer-a")
    with pytest.raises(BrowserBoundaryError, match="viewer_mismatch"):
        call(boundary, "display.lease.release", viewer_id="viewer-b")
    with pytest.raises(BrowserBoundaryError, match="stale_lease"):
        boundary.dispatch(boundary.management_uid, {"session_id": "test-run", "method": "display.lease.release",
                                                    "epoch": 0, "viewer_id": "viewer-a"})
    assert lease.human_holds()


def test_disconnect_is_not_a_handback_and_cancel_cannot_resume(boundary):
    call(boundary, "display.lease.acquire", viewer_id="viewer-a")
    boundary.ready_viewer = None
    with pytest.raises(BrowserBoundaryError, match="viewer_mismatch"):
        call(boundary, "display.lease.release")
    call(boundary, "display.cancel")
    with pytest.raises(BrowserBoundaryError, match="session_expired"):
        agent(boundary)
    assert lease.human_holds() and not boundary.viewer_may_input("viewer-a")


def test_expiry_fences_native_lease_and_viewer(boundary):
    boundary.now = lambda: boundary.expires_at
    with pytest.raises(BrowserBoundaryError, match="session_expired"):
        agent(boundary)
    assert lease.human_holds() and not boundary.viewer_may_input("viewer-a")


def test_wrong_session_and_os_caller_are_rejected(boundary):
    with pytest.raises(BrowserBoundaryError, match="session_mismatch"):
        boundary.dispatch(boundary.agent_uid, {"session_id": "foreign-run"})
    with pytest.raises(BrowserBoundaryError, match="caller_denied"):
        boundary.dispatch(boundary.agent_uid + 100, {"session_id": "test-run"})


def test_existing_browser_dispatch_fails_closed_without_local_cloud_or_cdp(tmp_path, monkeypatch):
    from tools import browser_tool as bt
    from hermes_constants import get_hermes_home
    import yaml
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"browser": {
        "isolated_socket": str(tmp_path / "missing.sock"), "isolated_session_id": "test-run"}}))
    monkeypatch.setenv("CAMOFOX_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:1")
    def forbidden(*args, **kwargs):
        raise AssertionError("protected transport must not start another browser")
    monkeypatch.setattr(bt, "_find_agent_browser", forbidden)
    monkeypatch.setattr(bt, "_start_browser_cleanup_thread", forbidden)
    assert bt._is_camofox_mode() is False
    assert bt._get_cdp_override() == ""
    assert bt._get_cloud_provider() is None
    assert bt._get_session_info("run")["features"]["isolated"]
    result = bt._run_browser_command("run", "snapshot")
    assert result["success"] is False and result["code"] == "browser_unavailable"
    assert json.loads(bt._browser_eval("document.cookie", "run"))["success"] is False


def test_partial_isolated_config_does_not_fall_back(monkeypatch):
    from tools import browser_tool as bt
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("browser:\n  isolated_session_id: test-run\n")
    monkeypatch.setattr(bt, "_find_agent_browser", lambda: pytest.fail("no local fallback"))
    assert bt._run_browser_command("run", "snapshot")["code"] == "browser_unavailable"


def test_service_owned_visible_session_never_starts_the_ordinary_idle_reaper(monkeypatch):
    from tools import browser_tool as bt
    session = {"session_name": "owned", "externally_managed": True, "cdp_url": "http://127.0.0.1:1"}
    monkeypatch.setitem(bt._active_sessions, "owned-test", session)
    monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: pytest.fail("human browser must survive ordinary idle reaper"))
    assert bt._get_session_info("owned-test") is session
