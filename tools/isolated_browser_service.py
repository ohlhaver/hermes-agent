"""Linux service for one isolated native browser session.

Provision the process with a distinct UID/network/mount namespace. Mount only
the agent socket directory into the agent container. The management socket,
profile, CDP and native desktop sockets remain private to this service.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import struct
import threading
import time
import urllib.request
from typing import Callable
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
                # HPD-1097: Xvnc now sends its clipboard (-SendCutText=1), and its outgoing size is
                # not bounded by -MaxCutText. Discard it in pieces here (never kept, logged or
                # returned) so a large copy cannot fail the hand-back; 64 MiB stays a hard stop.
                size = abs(int.from_bytes(read(7)[3:7], "big", signed=True))
                if size > 64 * 1024 * 1024:
                    raise BrowserBoundaryError("unsafe_return")
                while size:
                    size -= len(read(min(size, 65536)))
                continue
            raise BrowserBoundaryError("unsafe_return")
        raise BrowserBoundaryError("unsafe_return")

CARD_FIELD_NAMES = frozenset({"number", "expiry", "cvc", "name"})


# Shared-hosting and public suffixes: a card is never typed for "an approved host" that
# anybody can register a subdomain under. Mirrors packages/core payment-card.ts in
# hey-hermes (sharedSuffixes); the plane refuses them at approval too.
CARD_SHARED_SUFFIXES = frozenset({
    "github.io", "gitlab.io", "vercel.app", "netlify.app", "netlify.com", "pages.dev", "workers.dev", "web.app", "firebaseapp.com",
    "herokuapp.com", "azurewebsites.net", "cloudfront.net", "appspot.com", "blogspot.com", "myshopify.com", "wixsite.com",
    "weebly.com", "wordpress.com", "squarespace.com", "onrender.com", "fly.dev", "railway.app", "repl.co", "glitch.me",
    "ngrok.io", "ngrok-free.app", "s3.amazonaws.com", "amazonaws.com", "cloudflare.net", "trycloudflare.com", "framer.app",
    "bubbleapps.io", "webflow.io", "gitbook.io", "notion.site", "carrd.co", "000webhostapp.com", "surge.sh", "now.sh",
    "deno.dev", "pythonanywhere.com", "ondigitalocean.app", "run.app", "cloudapp.net", "azurestaticapps.net",
    "elasticbeanstalk.com", "blob.core.windows.net", "r2.dev", "supabase.co", "godaddysites.com", "mystrikingly.com",
    "square.site", "wixstudio.io", "tumblr.com", "myftp.org", "duckdns.org", "no-ip.org", "dyndns.org",
})
_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"})


def _is_public_suffix(host):
    labels = host.split(".")
    return (host in CARD_SHARED_SUFFIXES
            or (len(labels) == 2 and labels[0] in _SECOND_LEVEL and len(labels[1]) == 2))


def _card_host(value):
    """Approved merchant host: a plain DNS name, nothing else may receive a card."""
    if not isinstance(value, str):
        return None
    host = value.strip().lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    labels = host.split(".")
    if (len(host) > 253 or len(labels) < 2 or any(not 0 < len(l) <= 63 or l.startswith("-") or l.endswith("-")
            or not all(c.isalnum() or c == "-" for c in l) for l in labels)):
        return None
    if _is_public_suffix(host):
        return None
    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        return host


def _card_url_matches(url, host):
    """True for an https URL on the approved host or one of its subdomains."""
    try:
        target = urlsplit(url)
        page = (target.hostname or "").lower().rstrip(".")
        return (target.scheme == "https" and target.username is None and target.password is None
                and (page == host or page.endswith("." + host)))
    except ValueError:
        return False


_TEXTBOX_VALUE = re.compile(r"(?m)^(\s*(?:- )?(?:textbox|spinbutton|searchbox|combobox)\b[^\n]*?\]): .*$")
_DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn(digits):
    total, flip = 0, False
    for char in reversed(digits):
        number = int(char)
        if flip:
            number = number * 2 - 9 if number > 4 else number * 2
        total += number
        flip = not flip
    return total % 10 == 0


def _mask_card_text(text, typed):
    """Hide what was typed from the agent's next observations.

    The agent's snapshot lists the value of every text field, so a typed number
    would come straight back to the model. Field values are hidden, the typed
    strings are replaced wherever they appear, and any Luhn-valid card number is
    hidden too, until a human hand-back has emptied the inputs. This keeps an
    honest page's echo of the card from the model; it is no defence against a
    page that disguises or re-encodes it, which is why the card is only ever
    typed on an approved, non-shared merchant host.
    """
    text = _TEXTBOX_VALUE.sub(r"\1: [hidden]", text)
    for secret in typed.get("strings", ()):
        if len(secret) >= 3:
            text = text.replace(secret, "[hidden]")
    for number in typed.get("numbers", ()):
        loose = r"[ -]?".join(re.escape(char) for char in number)
        text = re.sub(loose, "[hidden]", text)
    text = _DIGIT_RUN.sub(lambda m: "[hidden]" if _luhn(re.sub(r"\D", "", m.group(0))) else m.group(0), text)
    for cvc in typed.get("cvcs", ()):
        text = re.sub(r'(?m)^(\s*(?:- )?StaticText) "%s"$' % re.escape(cvc), r'\1 "[hidden]"', text)
    return text


def _mask_card_value(value, typed):
    if isinstance(value, str):
        return _mask_card_text(value, typed)
    if isinstance(value, list):
        return [_mask_card_value(item, typed) for item in value]
    if isinstance(value, dict):
        return {key: "[hidden]" if key == "value" and isinstance(item, str) else _mask_card_value(item, typed)
                for key, item in value.items()}
    return value


class _Cdp:
    def __init__(self, ws):
        self.ws, self.seq = ws, 0

    def call(self, method, params=None):
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params or {}}))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = json.loads(self.ws.recv(timeout=max(.01, deadline - time.monotonic())))
            if result.get("id") == self.seq:
                if "error" in result:
                    raise BrowserBoundaryError("card_fill_failed")
                return result.get("result", {})
        raise BrowserBoundaryError("card_fill_failed")


# HPD-1722: typed into the page by the broker, never by a process the model can
# reach. Runs in an isolated world (page scripts cannot override the setters or
# read the arguments); the host is checked again here, at the moment of typing.
CARD_FILL_JS = r"""((approvedHost, values) => {
  const host = location.hostname.toLowerCase().replace(/\.$/, '');
  if (location.protocol !== 'https:' || !(host === approvedHost || host.endsWith('.' + approvedHost))) return {host_mismatch: true};
  const setters = {
    input: Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set,
    textarea: Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set,
    select: Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set,
  };
  const pattern = {
    number: /(card.?(number|no\b|num)|cc.?(num|number)|kartennummer|karten-nr|num[eé]ro de carte|n[uú]mero de tarjeta|numero carta)/i,
    expiry: /(expir|exp\.?\s?date|mm\s*\/\s*(yy|jj|aa)|g[uü]ltig|valid (thru|until)|ablauf|date d.expiration)/i,
    cvc: /(cvc|cvv|csc|security.?code|verification.?(code|number)|pr[uü]fziffer|sicherheitscode|code de s[eé]curit)/i,
    name: /(name on card|card.?holder|cardholder|karteninhaber|inhaber|titulaire|titular)/i,
    month: /(month|monat|mois|\bmes\b|mese|\bmm\b)/i,
    year: /(year|jahr|ann[eé]e|a[nñ]o|anno|\byy(yy)?\b|\bjj\b)/i,
  };
  const roots = [document];
  const inputs = [];
  const selects = [];
  for (let i = 0; i < roots.length; i++) {
    for (const el of roots[i].querySelectorAll('*')) {
      if (el.shadowRoot) roots.push(el.shadowRoot);
      if (el instanceof HTMLInputElement && !['checkbox','radio','button','submit','file','hidden','image','reset'].includes(el.type)) inputs.push(el);
      if (el instanceof HTMLSelectElement) selects.push(el);
    }
  }
  const visible = (el) => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return !el.disabled && !el.readOnly && r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const label = (el) => {
    const parts = [el.name, el.id, el.placeholder, el.getAttribute('aria-label')];
    if (el.labels) for (const l of el.labels) parts.push(l.textContent);
    const by = el.getAttribute('aria-labelledby');
    if (by) for (const id of by.split(/\s+/)) { const n = document.getElementById(id); if (n) parts.push(n.textContent); }
    return parts.filter(Boolean).join(' ');
  };
  const byAuto = (token, list = inputs) => list.find((el) => visible(el) && (el.autocomplete || '').split(/\s+/).includes(token));
  const byText = (field) => inputs.find((el) => {
    if (!visible(el)) return false;
    const text = label(el);
    if (!pattern[field].test(text)) return false;
    return field === 'cvc' || field === 'expiry' || !pattern.cvc.test(text);
  });
  const put = (el, value) => {
    el.focus();
    (el instanceof HTMLTextAreaElement ? setters.textarea : setters.input).call(el, value);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    el.blur();
  };
  const choose = (el, wanted) => {
    const same = (text) => wanted.some((w) => text === w || new RegExp('^0*' + w.replace(/^0+/, '') + '(\\D|$)').test(text) || (w.length === 4 && text.endsWith(w)));
    for (const option of el.options) {
      if (same((option.value || '').trim()) || same((option.text || '').trim())) {
        el.focus();
        setters.select.call(el, option.value);
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.blur();
        return true;
      }
    }
    return false;
  };
  const filled = [];
  const targets = {
    number: byAuto('cc-number') || byText('number'),
    cvc: byAuto('cc-csc') || byText('cvc'),
    name: byAuto('cc-name') || byText('name'),
  };
  for (const field of ['number', 'cvc', 'name']) {
    if (targets[field] && values[field]) { put(targets[field], values[field]); filled.push(field); }
  }
  if (values.expiry) {
    const [mm, yy] = values.expiry.split('/');
    const yyyy = (values.expiry_long || '').split('/')[1] || '20' + yy;
    const combined = byAuto('cc-exp');
    const anyMonth = byAuto('cc-exp-month') || byAuto('cc-exp-month', selects) || selects.find((el) => visible(el) && pattern.month.test(label(el)) && !pattern.year.test(label(el)));
    const anyYear = byAuto('cc-exp-year') || byAuto('cc-exp-year', selects) || selects.find((el) => visible(el) && pattern.year.test(label(el)) && !pattern.month.test(label(el)));
    if (combined) {
      const wide = (combined.maxLength || 0) >= 7;
      put(combined, wide && values.expiry_long ? values.expiry_long : values.expiry);
      filled.push('expiry');
    } else if (anyMonth && anyYear) {
      const okMonth = anyMonth instanceof HTMLSelectElement ? choose(anyMonth, [mm, String(Number(mm))]) : (put(anyMonth, mm), true);
      const okYear = anyYear instanceof HTMLSelectElement ? choose(anyYear, [yyyy, yy]) : (put(anyYear, (anyYear.maxLength || 0) === 4 ? yyyy : yy), true);
      if (okMonth && okYear) filled.push('expiry');
    } else {
      const other = byText('expiry');
      if (other) { put(other, values.expiry); filled.push('expiry'); }
    }
  }
  return {filled};
})"""


class NativeBrowserExecutor:
    def __init__(self, session_id, *, trusted_url_validator: Callable[[str], bool] | None = None):
        # Only a trusted service caller can supply this check. Manifests and
        # agent/management RPC never select or replace it. The concrete consumer
        # resolves public targets in its existing protected egress namespace.
        if trusted_url_validator is not None and not callable(trusted_url_validator):
            raise BrowserBoundaryError("invalid_provisioning")
        from tools import browser_tool as bt
        self.bt = bt
        self._trusted_url_validator = trusted_url_validator
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

    def _url_allowed(self, url):
        try:
            target = urlsplit(url)
            if target.scheme not in {"http", "https"} or not target.hostname:
                return False
            if self.bt._is_always_blocked_url(url) or self.bt.check_website_access(url):
                return False
            if self._trusted_url_validator is None:
                return bool(self.bt._is_safe_url(url))
            # A service validator cannot relax literal private/metadata targets.
            # Name resolution belongs to the protected egress; each actual
            # connection still independently enforces its checked-address policy.
            host = target.hostname.lower().rstrip(".")
            if target.username is not None or target.password is not None or host == "localhost" or host.endswith(".localhost"):
                return False
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is not None:
                if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                    address = address.ipv4_mapped
                if not address.is_global or address.is_multicast or address.is_reserved or address.is_unspecified:
                    return False
            return self._trusted_url_validator(url) is True
        except Exception:
            # Do not export target URLs, resolver errors or private diagnostics.
            return False

    def execute(self, command, args):
        from agent.redact import redact_sensitive_text
        with self.lock:
            if command == "open":
                if len(args) != 1 or not self._url_allowed(args[0]):
                    raise BrowserBoundaryError("command_denied")
            result = self.bt._run_browser_command(self.session_id, command, args)
            if not result.get("success"):
                return {"success": False, "error": "Browser action failed; user takeover may be required."}
            # Check redirects before returning any browser data. Network egress
            # is separately restricted by deployment, including DNS rebinding.
            pages = self.pages()
            for page in pages:
                url = page.get("url", "")
                if url != "about:blank" and not self._url_allowed(url):
                    raise BrowserBoundaryError("command_denied")
            data = result.get("data", {})
            if isinstance(data, dict) and isinstance(data.get("url"), str):
                data["url"] = public_url(data["url"])
            if command == "eval" and args == ["window.location.href"] and isinstance(data, dict):
                if isinstance(data.get("result"), str):
                    data["result"] = public_url(data["result"])
            result = self._mask_typed_card(result)
            return json.loads(redact_sensitive_text(json.dumps(result), force=True))

    def _mask_typed_card(self, result):
        typed = getattr(self, "_card_typed", None)
        if not typed:
            return result
        return _mask_card_value(result, typed)


    def fill_card(self, approved_host, fields):
        """Type one approved payment's card into the visible page. Names only come back.

        The host check uses the page's real URL read from the browser here, never
        a value the caller supplies. Only the visible tab and its same-host frames
        are typed into; a frame on another host (a payment iframe) is left alone.
        """
        host = _card_host(approved_host)
        if host is None:
            raise BrowserBoundaryError("command_denied")
        values = {key: fields[key] for key in fields}
        # What the agent must not read back afterwards (see _mask_card_text); set
        # before typing so no observation can slip between the fill and the mask.
        with self.lock:
            # Added to what an earlier fill left in the page, never replacing it. The mask
            # ends only when a human hand-back has emptied every input (safe_return).
            previous = getattr(self, "_card_typed", None) or {"numbers": (), "cvcs": (), "strings": ()}
            self._card_typed = {
                "numbers": previous["numbers"] + ((values["number"],) if values.get("number") else ()),
                "cvcs": previous["cvcs"] + ((values["cvc"],) if values.get("cvc") else ()),
                "strings": previous["strings"] + tuple(v for k, v in values.items() if k in ("name", "expiry", "expiry_long")),
            }
        with self.lock:
            targets = []
            for page in self.pages():
                url = page.get("url", "")
                if url == "about:blank":
                    continue
                if not page.get("webSocketDebuggerUrl"):
                    continue
                targets.append(page)
            filled, seen_visible, mismatch = set(), False, False
            for page in targets:
                with connect(page["webSocketDebuggerUrl"], open_timeout=3, max_size=2 * 1024 * 1024) as ws:
                    cdp = _Cdp(ws)
                    tree = cdp.call("Page.getFrameTree")["frameTree"]
                    main_world = cdp.call("Page.createIsolatedWorld", {"frameId": tree["frame"]["id"],
                                                                       "worldName": "hermes-private-card"})["executionContextId"]
                    state = cdp.call("Runtime.evaluate", {"contextId": main_world, "returnByValue": True,
                                                         "expression": "document.visibilityState"})
                    if state.get("result", {}).get("value") != "visible":
                        continue
                    seen_visible = True
                    frames = [tree]
                    for frame in frames:
                        frames.extend(frame.get("childFrames", []))
                        info = frame["frame"]
                        if not _card_url_matches(info.get("url", ""), host):
                            if info["id"] == tree["frame"]["id"]:
                                mismatch = True
                                break
                            continue
                        ctx = cdp.call("Page.createIsolatedWorld", {"frameId": info["id"],
                                                                    "worldName": "hermes-private-card"})["executionContextId"]
                        outcome = cdp.call("Runtime.evaluate", {
                            "contextId": ctx, "returnByValue": True,
                            "expression": CARD_FILL_JS + "(" + json.dumps(host) + "," + json.dumps(values) + ")"})
                        value = outcome.get("result", {}).get("value")
                        if outcome.get("exceptionDetails") or not isinstance(value, dict):
                            continue
                        if value.get("host_mismatch"):
                            mismatch = True
                            break
                        filled.update(name for name in value.get("filled", []) if name in CARD_FIELD_NAMES)
                    break  # Only the one visible (focused) tab is ever typed into.
            values.clear()
        if mismatch and not filled:
            raise BrowserBoundaryError("host_mismatch")
        if not seen_visible:
            raise BrowserBoundaryError("no_visible_page")
        order = ("number", "expiry", "cvc", "name")
        return {"filled": [n for n in order if n in filled], "missing": [n for n in order if n not in filled]}

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
                            or not self._url_allowed(url) or self.bt._sensitive_query_param_name(url)
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
                cleared = bool(self.pages())
                if cleared:
                    self._card_typed = None  # Every input was emptied above; nothing left to hide.
                return cleared
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


async def run(manifest, *, trusted_url_validator: Callable[[str], bool] | None = None):
    required = {"session_id", "expires_at", "agent_uid", "management_uid", "agent_socket", "management_socket",
                "agent_group", "viewer_port"}
    if not required.issubset(manifest) or not hasattr(os, "geteuid") or os.geteuid() == 0:
        raise BrowserBoundaryError("invalid_provisioning")
    executor = NativeBrowserExecutor(manifest["session_id"], trusted_url_validator=trusted_url_validator)
    boundary = NativeSessionBoundary(lease, executor.execute, executor.safe_return,
        agent_uid=manifest["agent_uid"], management_uid=manifest["management_uid"],
        session_id=manifest["session_id"], expires_at=manifest["expires_at"], takeover_href=manifest.get("takeover_href"))
    boundary.card_fill = getattr(executor, "fill_card", None)
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
