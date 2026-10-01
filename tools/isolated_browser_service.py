"""Linux service for one isolated native browser session.

Provision the process with a distinct UID/network/mount namespace. Mount only
the agent socket directory into the agent container. The management socket,
profile, CDP and native desktop sockets remain private to this service.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import socket
import struct
import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.server import serve
from websockets.sync.client import connect

from tools.bot_desktop import browser, lease, runtime
from tools.isolated_browser import BrowserBoundaryError, BrowserRpcServer, IsolatedBrowserBoundary
from tools.isolated_browser_viewer import NativeRfbViewer


CLEAR_INPUTS = r"""(() => {
  const roots = [document];
  for (let i = 0; i < roots.length; i++) {
    for (const el of roots[i].querySelectorAll('*')) {
      if (el.shadowRoot) roots.push(el.shadowRoot);
      if (el instanceof HTMLInputElement && !['checkbox','radio','button','submit','file'].includes(el.type))
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
      if (el instanceof HTMLTextAreaElement)
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(el, '');
    }
  }
  return true;
})()"""


def public_url(url):
    parsed = urlsplit(url)
    # Credential-bearing redirects/fragment/query are never ordinary metadata.
    return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))


def clear_remote_clipboard(socket_path):
    """Clear the PRIVATE native Xvnc clipboard before handing back.

    A site's Paste button could otherwise reveal prior human clipboard data
    even though arbitrary agent JS/clipboard reads are fenced. Use the native
    RFB protocol, not a new dependency or model executor. Never return contents.
    """
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(str(socket_path))
        def read(n):
            out = bytearray()
            while len(out) < n:
                chunk = client.recv(n-len(out))
                if not chunk:
                    raise BrowserBoundaryError("unsafe_return")
                out.extend(chunk)
            return bytes(out)
        if read(12) != b"RFB 003.008\n":
            raise BrowserBoundaryError("unsafe_return")
        client.sendall(b"RFB 003.008\n")
        count = read(1)[0]
        if count == 0 or 1 not in read(count):
            raise BrowserBoundaryError("unsafe_return")
        client.sendall(b"\x01")
        if read(4) != b"\x00\x00\x00\x00":
            raise BrowserBoundaryError("unsafe_return")
        client.sendall(b"\x01")
        init = read(24)
        name_length = int.from_bytes(init[20:24], "big")
        if name_length > 4096:
            raise BrowserBoundaryError("unsafe_return")
        read(name_length)
        # Empty ClientCutText, then a one-pixel update request as an ordered
        # server round trip. Consume any server clipboard notification without
        # exporting, printing or storing it outside this private process.
        client.sendall(b"\x06\0\0\0\0\0\0\0" + struct.pack(">BBHHHH", 3, 0, 0, 0, 1, 1))
        for _ in range(8):
            kind = read(1)[0]
            if kind == 0:
                read(3)
                return
            if kind == 2:
                continue
            if kind == 3:
                size = abs(int.from_bytes(read(7)[3:7], "big", signed=True))
                if size > 256 * 1024:
                    raise BrowserBoundaryError("unsafe_return")
                read(size)
                continue
            raise BrowserBoundaryError("unsafe_return")
        raise BrowserBoundaryError("unsafe_return")


class NativeBrowserExecutor:
    def __init__(self, session_id):
        from tools import browser_tool as bt
        self.bt = bt
        self.session_id = session_id
        self.lock = threading.RLock()
        self.port = None
        runtime.start()
        os.environ.update(runtime.desktop_env())
        os.environ["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = "1800000"
        launch = browser.dock_launch()
        if launch is None:
            runtime.stop()
            raise BrowserBoundaryError("browser_start_failed")
        exe, profile = launch
        self.process = subprocess.Popen(browser.dock_argv(exe, profile) + ["about:blank"], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.port = browser.running_instance_cdp_port(profile)
            if self.port:
                break
            time.sleep(.1)
        if not self.port:
            self.process.terminate()
            runtime.stop()
            raise BrowserBoundaryError("browser_start_failed")
        # Use the existing native CLI dispatcher against the DOCK'S Chromium.
        # The broker alone can reach this private loopback CDP endpoint.
        bt._active_sessions[session_id] = {"session_name": session_id,
            "session_key": session_id, "cdp_url": f"http://127.0.0.1:{self.port}",
            "features": {"local": True}, "_first_nav": False, "externally_managed": True}
        bt._cloud_provider_resolved = True
        bt._cached_cloud_provider = None

    def pages(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=3) as response:
            return [p for p in json.load(response) if p.get("type") == "page"]

    def execute(self, command, args):
        from agent.redact import redact_sensitive_text
        with self.lock:
            if command == "open":
                if len(args) != 1 or urlsplit(args[0]).scheme not in {"http", "https"}:
                    raise BrowserBoundaryError("command_denied")
                if self.bt._is_always_blocked_url(args[0]) or not self.bt._is_safe_url(args[0]):
                    raise BrowserBoundaryError("command_denied")
                if self.bt.check_website_access(args[0]):
                    raise BrowserBoundaryError("command_denied")
            result = self.bt._run_browser_command(self.session_id, command, args)
            if not result.get("success"):
                return {"success": False, "error": "Browser action failed; user takeover may be required."}
            # Check redirects before returning any browser data. Network egress
            # is separately restricted by deployment, including DNS rebinding.
            pages = self.pages()
            for page in pages:
                url = page.get("url", "")
                if url != "about:blank" and (self.bt._is_always_blocked_url(url) or not self.bt._is_safe_url(url)):
                    raise BrowserBoundaryError("command_denied")
            data = result.get("data", {})
            if isinstance(data, dict) and isinstance(data.get("url"), str):
                data["url"] = public_url(data["url"])
            if command == "eval" and args == ["window.location.href"] and isinstance(data, dict):
                if isinstance(data.get("result"), str):
                    data["result"] = public_url(data["result"])
            return json.loads(redact_sensitive_text(json.dumps(result), force=True))

    def safe_return(self):
        # Runs only after input is fenced, under the broker's human lease.
        # Isolated-world DOM access avoids page-overridden JS getters/setters.
        # Never return input values or exception bodies over management RPC.
        try:
            with self.lock:
                clear_remote_clipboard(runtime.rfb_socket_path())
                deadline_all = time.monotonic() + 15
                for page in self.pages():
                    url = page.get("url", "")
                    if url == "about:blank":
                        continue
                    if (time.monotonic() >= deadline_all or urlsplit(url).scheme not in {"http", "https"}
                            or not self.bt._is_safe_url(url) or self.bt._sensitive_query_param_name(url)
                            or urlsplit(url).fragment):
                        return False
                    with connect(page["webSocketDebuggerUrl"], open_timeout=3, max_size=2*1024*1024) as ws:
                        seq = 0
                        def rpc(method, params=None):
                            nonlocal seq
                            seq += 1
                            ws.send(json.dumps({"id": seq, "method": method, "params": params or {}}))
                            deadline = time.monotonic() + 3
                            while time.monotonic() < deadline:
                                result = json.loads(ws.recv(timeout=max(.01, deadline-time.monotonic())))
                                if result.get("id") == seq:
                                    if "error" in result:
                                        raise BrowserBoundaryError("unsafe_return")
                                    return result.get("result", {})
                            raise BrowserBoundaryError("unsafe_return")
                        tree = rpc("Page.getFrameTree")["frameTree"]
                        frames = [tree]
                        for frame in frames:
                            frames.extend(frame.get("childFrames", []))
                            ctx = rpc("Page.createIsolatedWorld", {"frameId": frame["frame"]["id"],
                                                                   "worldName": "hermes-private-return"})["executionContextId"]
                            cleaned = rpc("Runtime.evaluate", {"contextId": ctx, "expression": CLEAR_INPUTS,
                                                                "returnByValue": True})
                            if cleaned.get("exceptionDetails") or cleaned.get("result", {}).get("value") is not True:
                                return False
                return bool(self.pages())
        except Exception:
            return False


class NativeSessionBoundary(IsolatedBrowserBoundary):
    viewer: NativeRfbViewer

    def dispatch(self, uid, request):
        if uid == self.management_uid and request.get("method") == "display.viewer.mint":
            with self.condition:
                self._bound(request)
                return self.viewer.mint(request.get("viewer_id"))
        if uid == self.management_uid and request.get("method") == "display.status":
            with self.condition:
                self._bound(request)
                from hermes_constants import get_hermes_home
                return {"supported": True, "installed": True, "running": runtime.status().running,
                        "profile_key": str(get_hermes_home()),
                        "lease": lease.public_view(lease.get()), "session_id": self.session_id}
        return super().dispatch(uid, request)


async def run(manifest):
    required = {"session_id", "expires_at", "agent_uid", "management_uid", "agent_socket", "management_socket",
                "agent_group", "viewer_port"}
    if not required.issubset(manifest) or not hasattr(os, "geteuid") or os.geteuid() == 0:
        raise BrowserBoundaryError("invalid_provisioning")
    executor = NativeBrowserExecutor(manifest["session_id"])
    boundary = NativeSessionBoundary(lease, executor.execute, executor.safe_return,
        agent_uid=manifest["agent_uid"], management_uid=manifest["management_uid"],
        session_id=manifest["session_id"], expires_at=manifest["expires_at"])
    viewer = boundary.viewer = NativeRfbViewer(boundary, runtime.rfb_socket_path())
    servers = []
    running_servers = []
    try:
        servers.append(BrowserRpcServer(manifest["agent_socket"], boundary, group_id=manifest["agent_group"], role="agent"))
        servers.append(BrowserRpcServer(manifest["management_socket"], boundary, group_id=os.getegid(), role="management"))
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            running_servers.append(server)
        async with serve(viewer.handle, "0.0.0.0", manifest["viewer_port"], max_size=MAX_WS_FRAME,
                         compression=None, logger=None):
            # Readiness is metadata only; errors never dump input or manifest.
            print(json.dumps({"ready": True, "session_id": manifest["session_id"]}), flush=True)
            while time.time() < boundary.expires_at and not boundary.revoked:
                await asyncio.sleep(.5)
    finally:
        boundary.revoked = True
        boundary.ready_viewer = None
        lease.acquire("session-ended")
        for server in running_servers:
            server.shutdown()
        for server in servers:
            server.server_close()
            Path(server.server_address).unlink(missing_ok=True)
        executor.process.terminate()
        runtime.stop()


MAX_WS_FRAME = 65536 + 28


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    path = Path(args.manifest)
    st = path.stat()
    if st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise SystemExit("Protected session manifest required")
    try:
        asyncio.run(run(json.loads(path.read_text(encoding="utf-8"))))
    except Exception:
        raise SystemExit("Protected browser service unavailable")


if __name__ == "__main__":
    main()
