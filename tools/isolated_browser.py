"""Protected process boundary around Hermes' native browser and Bot Screen.

The native desktop deliberately has a same-user, tool-level fence. HPD-321
needs a stronger boundary for credential entry: the agent has only a Unix RPC
socket, not the browser's profile, X/RFB sockets or network namespace. Run this
service as a distinct UID in a separate network namespace/container with no
shared profile volume. SO_PEERCRED makes agent and trusted management callers
different principals; no agent-readable bearer secret can elevate a caller.

This service executes the EXISTING browser command implementation and native
lease. It does not add a model tool, queue, browser executor or retry policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import threading
import time
import re
from typing import Callable

MAX_MESSAGE = 2 * 1024 * 1024


def valid_takeover_href(session_id, href):
    return (isinstance(session_id, str) and
            re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", session_id) is not None and
            href == f"/api/workspace/preview/4321/browser/{session_id}")


class BrowserBoundaryError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class IsolatedBrowserBoundary:
    """One broker process/profile/browser per product-bound session.

    Native lease remains authority. The condition tracks already admitted work
    only so a human grant waits for real quiescence, not a dropped late result.
    """

    def __init__(self, lease, execute: Callable, safe_return: Callable, *,
                 agent_uid: int, management_uid: int, session_id: str,
                 expires_at: float, now: Callable = time.time, takeover_href=None):
        if agent_uid == management_uid or agent_uid == os.geteuid():
            raise BrowserBoundaryError("distinct_agent_uid_required")
        if not session_id or not (now() < expires_at <= now() + 1800):
            raise BrowserBoundaryError("invalid_session")
        if takeover_href is not None and not valid_takeover_href(session_id, takeover_href):
            raise BrowserBoundaryError("invalid_takeover_href")
        self.takeover_href = takeover_href
        self.lease = lease
        self.execute = execute
        self.safe_return = safe_return
        self.agent_uid = agent_uid
        self.management_uid = management_uid
        self.session_id = session_id
        self.expires_at = expires_at
        self.now = now
        self.condition = threading.Condition(threading.RLock())
        self.active = 0
        self.revoked = False
        self.ready_viewer = None
        self.viewer_id = None
        self.transition = False

    def handoff_metadata(self, uid, request, response):
        with self.condition:
            if (not self.takeover_href or uid != self.agent_uid or not isinstance(request, dict) or
                    request.get("method") != "browser.command" or request.get("session_id") != self.session_id or
                    self.revoked or self.now() >= self.expires_at):
                return
            paused = response.get("code") in {"human_has_control", "handoff_pending"}
            if response.get("ok") is True and isinstance(response.get("result"), dict):
                target = response["result"]
                paused = target.get("success") is False
            elif paused:
                target = response
            else:
                return
            target.update(takeover_href=self.takeover_href, needs_user_takeover=paused)

    def _bound(self, request):
        if not isinstance(request, dict) or request.get("session_id") != self.session_id:
            raise BrowserBoundaryError("session_mismatch")
        if self.revoked or self.now() >= self.expires_at:
            # Expiry never gives the agent a credential-bearing screen.
            self.revoked = True
            self.lease.acquire("expired-session")
            self.ready_viewer = None
            raise BrowserBoundaryError("session_expired")

    def _revision(self, request):
        current = self.lease.get()
        epoch = request.get("epoch")
        if type(epoch) is not int or epoch != current.epoch:
            raise BrowserBoundaryError("stale_lease")
        return current

    def dispatch(self, uid: int, request: dict):
        with self.condition:
            if uid not in (self.agent_uid, self.management_uid):
                raise BrowserBoundaryError("caller_denied")
            self._bound(request)
            method = request.get("method")
            if uid == self.agent_uid:
                if method != "browser.command":
                    raise BrowserBoundaryError("caller_denied")
                if self.lease.get().holder != self.lease.AGENT:
                    raise BrowserBoundaryError("human_has_control")
                admitted = self.lease.assert_agent_may_act()
                if self.transition:
                    raise BrowserBoundaryError("handoff_pending")
                command, args = request.get("command"), request.get("args", [])
                # No profile/export/CDP attach, arbitrary filesystem paths,
                # recording or cookies through the agent channel.
                allowed = {"open", "snapshot", "click", "fill", "type", "scroll",
                           "back", "forward", "press", "get", "eval"}
                if command not in allowed or not isinstance(args, list) or any(
                    not isinstance(arg, str) for arg in args
                ) or any(arg.startswith("--") and arg not in (
                    "--interactive", "--compact"
                ) for arg in args):
                    raise BrowserBoundaryError("command_denied")
                arity = {"open": 1, "click": 1, "fill": 2, "type": 1, "press": 1, "back": 0, "forward": 0}
                if command in arity and len(args) != arity[command]:
                    raise BrowserBoundaryError("command_denied")
                if command not in {"snapshot", "eval"} and any(arg.startswith("-") for arg in args):
                    raise BrowserBoundaryError("command_denied")
                if command == "scroll" and (len(args) not in {1, 2} or args[0] not in {"up", "down", "left", "right"}
                    or len(args) == 2 and (not args[1].isdigit() or int(args[1]) > 10000)):
                    raise BrowserBoundaryError("command_denied")
                # Console buffers, arbitrary JS, form-value/HTML getters and
                # DevTools keyboard shortcuts can reveal past human input.
                # Agent-browser's structured navigation/snapshot/actions keep
                # working; the two internal URL/title reads are fixed queries.
                if command == "eval" and args not in (["window.location.href"], ["document.title"]):
                    raise BrowserBoundaryError("command_denied")
                if command == "get" and (len(args) != 1 or args[0] not in {"url", "title"}):
                    raise BrowserBoundaryError("command_denied")
                if command == "snapshot" and any(arg not in {"-c", "-i", "--compact", "--interactive"} for arg in args):
                    raise BrowserBoundaryError("command_denied")
                if command == "press" and (len(args) != 1 or args[0] not in {
                    "Enter", "Tab", "Escape", "Backspace", "Delete", "Space",
                    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"
                }):
                    raise BrowserBoundaryError("command_denied")
                if command in {"click", "fill"} and (not args or not re.fullmatch(r"@?e\d+", args[0])):
                    raise BrowserBoundaryError("command_denied")
                self.active += 1
            else:
                if method == "display.status":
                    return self.lease.public_view(self.lease.get())
                current = self._revision(request)
                if method == "display.lease.acquire":
                    viewer = request.get("viewer_id")
                    if not isinstance(viewer, str) or not viewer or len(viewer) > 256:
                        raise BrowserBoundaryError("invalid_viewer")
                    if self.viewer_id and self.viewer_id != viewer:
                        raise BrowserBoundaryError("another_viewer_holds")
                    if self.transition:
                        raise BrowserBoundaryError("handoff_pending")
                    self.transition = True
                    self.viewer_id = viewer
                    self.ready_viewer = None
                    acquired = self.lease.acquire(viewer)
                    deadline = time.monotonic() + 35
                    while self.active:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            self.transition = False
                            raise BrowserBoundaryError("action_drain_timeout")
                        self.condition.wait(left)
                        self._bound(request)
                    self.transition = False
                    if self.lease.get().epoch != acquired.epoch:
                        raise BrowserBoundaryError("stale_lease")
                    self.ready_viewer = viewer
                    return self.lease.public_view(acquired)
                if method == "display.lease.release":
                    if not self.ready_viewer or current.holder != self.lease.HUMAN or request.get("viewer_id") != self.ready_viewer:
                        raise BrowserBoundaryError("viewer_mismatch")
                    if self.transition:
                        raise BrowserBoundaryError("handoff_pending")
                    # Fence human input too while validating/clearing residual
                    # sensitive DOM state. Failure retains native human control.
                    previous_viewer = self.ready_viewer
                    self.ready_viewer = None
                    self.transition = True
                    try:
                        if self.safe_return() is not True:
                            raise BrowserBoundaryError("unsafe_return")
                        self._bound(request)
                        self._revision(request)
                        result = self.lease.release(self.viewer_id)
                        if result.holder != self.lease.AGENT:
                            raise BrowserBoundaryError("native_release_failed")
                        self.viewer_id = None
                        return self.lease.public_view(result)
                    finally:
                        self.transition = False
                        if (not self.revoked and self.now() < self.expires_at
                                and self.lease.viewer_may_send_input(previous_viewer)):
                            self.ready_viewer = previous_viewer
                if method == "display.cancel":
                    self.revoked = True
                    self.ready_viewer = None
                    self.lease.acquire("revoked-session")
                    return {"revoked": True}
                raise BrowserBoundaryError("unknown_method")

        try:
            result = self.execute(command, args)
            with self.condition:
                self._bound(request)
                if self.lease.get().epoch != admitted.epoch:
                    raise BrowserBoundaryError("human_has_control")
            return result
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def viewer_may_input(self, viewer_id: str) -> bool:
        with self.condition:
            if self.revoked or self.now() >= self.expires_at or self.transition:
                return False
            return self.ready_viewer == viewer_id and self.lease.viewer_may_send_input(viewer_id)


class BrowserRpcServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, boundary: IsolatedBrowserBoundary, *, group_id: int, role: str):
        peer_option = getattr(socket, "SO_PEERCRED", None)
        if not isinstance(peer_option, int):
            raise BrowserBoundaryError("linux_peer_credentials_required")
        self.peer_credential_option = peer_option
        if role not in {"agent", "management"}:
            raise BrowserBoundaryError("invalid_socket_role")
        self.boundary = boundary
        self.role = role
        # Provisioning owns a private directory. Never unlink a foreign socket
        # or accept an agent-writable parent where it can replace our endpoint.
        parent = Path(path).parent
        st = parent.stat()
        if st.st_uid != os.geteuid() or st.st_mode & 0o022:
            raise BrowserBoundaryError("unsafe_socket_directory")
        super().__init__(path, BrowserRpcHandler)
        try:
            os.chown(path, os.geteuid(), group_id)
            os.chmod(path, 0o660)
        except Exception:
            self.server_close()
            Path(path).unlink()
            raise


class BrowserRpcHandler(socketserver.StreamRequestHandler):
    server: BrowserRpcServer

    def handle(self):
        self.connection.settimeout(40)
        uid, request = None, None
        try:
            raw_uid = self.connection.getsockopt(socket.SOL_SOCKET, self.server.peer_credential_option, 12)
            _, uid, _ = struct.unpack("3i", raw_uid)
            expected_uid = self.server.boundary.agent_uid if self.server.role == "agent" else self.server.boundary.management_uid
            if uid != expected_uid:
                raise BrowserBoundaryError("caller_denied")
            raw = self.rfile.readline(MAX_MESSAGE + 1)
            if len(raw) > MAX_MESSAGE or not raw.endswith(b"\n"):
                raise BrowserBoundaryError("invalid_message")
            request = json.loads(raw)
            result = self.server.boundary.dispatch(uid, request)
            response = {"ok": True, "result": result}
        except BrowserBoundaryError as error:
            response = {"ok": False, "code": error.code}
        except Exception:
            # No raw website/native/JSON exception or stack trace on this wire.
            response = {"ok": False, "code": "browser_unavailable"}
        self.server.boundary.handoff_metadata(uid, request, response)
        encoded = json.dumps(response, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_MESSAGE:
            encoded = b'{"ok":false,"code":"response_too_large"}\n'
        try:
            self.wfile.write(encoded)
        except OSError:
            pass


def command_from_agent(socket_path: str, session_id: str, command: str, args: list):
    """No local/cloud fallback when the protected endpoint fails."""
    try:
        if not os.path.isabs(socket_path):
            raise ValueError("socket must be absolute")
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(40)
            client.connect(socket_path)
            body = json.dumps({"method": "browser.command", "session_id": session_id,
                               "command": command, "args": args}, separators=(",", ":")).encode() + b"\n"
            if len(body) > MAX_MESSAGE:
                raise ValueError("too large")
            client.sendall(body)
            with client.makefile("rb") as stream:
                raw = stream.readline(MAX_MESSAGE + 1)
            if len(raw) > MAX_MESSAGE:
                raise ValueError("too large")
            response = json.loads(raw)
            if response.get("ok") is True and isinstance(response.get("result"), dict):
                return response["result"]
            code = response.get("code")
            if code not in {"human_has_control", "session_expired", "command_denied", "handoff_pending"}:
                code = "browser_unavailable"
            result = {"success": False, "code": code,
                      "error": "Browser is paused or unavailable. Ask the user to take over or return control."}
            href = response.get("takeover_href")
            if valid_takeover_href(session_id, href) and response.get("needs_user_takeover") is True:
                result.update(takeover_href=href, needs_user_takeover=True)
                result["error"] += f" [Open private browser]({href})."
            return result
    except Exception:
        return {"success": False, "code": "browser_unavailable",
                "error": "Protected browser connection is unavailable."}
