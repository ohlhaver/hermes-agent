"""Bounded HPD-321 Linux probe against two explicitly named OWN test containers.

No provider/model calls, real credentials or customer profiles. All Docker/RPC
output is captured; stdout contains booleans only. Pass the dedicated Docker
context and the agent/browser container names. Broker must already be ready.
The fixture is placed on the public example.com page in the protected browser
by the trusted management tester, then driven through the encrypted RFB wire.
This proves real browser/lease/process behavior, not iPhone or LLM acceptance.
"""
import argparse
import base64
import json
import struct
import subprocess
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from websockets.sync.client import connect

from tools.isolated_browser_viewer import RfbCipher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--browser", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--broker-uid", type=int, default=10001)
    parser.add_argument("--agent-uid", type=int, default=10000)
    parser.add_argument("--port", type=int, default=18432)
    args = parser.parse_args()
    if "hpd321" not in args.agent or "hpd321" not in args.browser or "hermes-browse-" not in args.context:
        raise SystemExit("Dedicated owned HPD321 test containers/context required")
    checks = {}

    def docker_python(container, uid, code):
        result = subprocess.run(["docker", "--context", args.context, "exec", "-i", "-u", uid,
            "--workdir", "/opt/hpd321-src", container, "/opt/hermes/.venv/bin/python", "-"],
            input=code, text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError("probe_subprocess_failed")
        return json.loads(result.stdout)

    def management(method, **values):
        request = {"session_id": args.session, "method": method, **values}
        result = docker_python(args.browser, "0:10001", f"""
import socket,json
with socket.socket(socket.AF_UNIX) as s:
 s.settimeout(25);s.connect('/profile/management.sock')
 s.sendall(({json.dumps(request)!r}+'\\n').encode())
 r=json.loads(s.makefile('rb').readline())
 print(json.dumps(r))
""")
        if result.get("ok") is not True:
            raise RuntimeError("management_action_failed")
        return result["result"]

    def fixture(expression):
        return docker_python(args.browser, f"{args.broker_uid}:{args.broker_uid}", f"""
from pathlib import Path
import urllib.request,json,time
from websockets.sync.client import connect
p=Path('/profile/hermes/bot-desktop/browser-profile/DevToolsActivePort').read_text().splitlines()[0]
with urllib.request.urlopen('http://127.0.0.1:'+p+'/json/list',timeout=3) as r: pages=json.load(r)
page=next(p for p in pages if p.get('type')=='page' and p.get('url','').startswith('https://example.com'))
with connect(page['webSocketDebuggerUrl'],open_timeout=3) as ws:
 ws.send(json.dumps({{'id':1,'method':'Page.bringToFront'}}))
 while json.loads(ws.recv(timeout=3)).get('id')!=1: pass
 ws.send(json.dumps({{'id':2,'method':'Runtime.evaluate','params':{{'expression':{expression!r},'returnByValue':True}}}}))
 while True:
  r=json.loads(ws.recv(timeout=3))
  if r.get('id')==2:
   print(json.dumps({{'ok':not r.get('error') and not r.get('result',{{}}).get('exceptionDetails'),
                     'value':r.get('result',{{}}).get('result',{{}}).get('value')}}));break
""")

    def agent(command="snapshot", cmd_args=None):
        return docker_python(args.agent, f"{args.agent_uid}:{args.agent_uid}", f"""
import json
from tools.browser_tool import _run_browser_command
r=_run_browser_command('hpd321-probe',{command!r},{cmd_args or []!r})
print(json.dumps(r))
""")

    checks["agent_private_profile_absent"] = docker_python(args.agent, str(args.agent_uid), """
import json,os,socket
print(json.dumps(not os.path.exists('/profile/hermes') and not os.path.exists('/profile/management.sock')))
""")
    checks["native_navigation"] = agent("open", ["https://example.com"])["success"] is True
    checks["native_snapshot"] = agent()["success"] is True
    # Synthetic login + 2FA fixture. No real identity, website transaction or
    # real credential. Its input values stay inside the protected browser.
    setup = r"""(() => {
      document.body.innerHTML = '<form id="login"><label>Password<input id="pw" type="password" autocomplete="current-password"></label><button>Continue</button></form>';
      document.getElementById('login').onsubmit = e => {
        e.preventDefault();
        document.body.innerHTML = '<form id="verify"><label>Verification<input id="otp" autocomplete="one-time-code"></label><button>Verify</button></form>';
        document.getElementById('otp').focus();
        document.getElementById('verify').onsubmit = e => {
          e.preventDefault();document.cookie='hpd321fixture=complete;path=/;SameSite=Strict';
          document.body.innerHTML='<h1>Signed in</h1><button id="next">Continue task</button>';
        };
      };
      document.getElementById('pw').focus();return true;
    })()"""
    checks["fixture_ready"] = fixture(setup)["value"] is True
    viewer_id = "hpd321-probe-viewer"
    status = management("display.status")
    grant = management("display.lease.acquire", epoch=status["lease"]["epoch"], viewer_id=viewer_id)
    checks["same_native_human_lease"] = grant["holder"] == "human"
    blocked = agent()
    checks["agent_snapshot_fenced"] = blocked.get("code") == "human_has_control"
    blocked = agent("eval", ["document.cookie"])
    checks["agent_cdp_eval_fenced"] = blocked.get("code") == "human_has_control"
    info = management("display.viewer.mint", viewer_id=viewer_id)
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(serialization.Encoding.X962,
                                               serialization.PublicFormat.UncompressedPoint)
    cipher = RfbCipher(private, base64.b64decode(info["server_public_key"]),
                       session_id=args.session, viewer_id=viewer_id, server=False)
    with connect(f"ws://127.0.0.1:{args.port}",open_timeout=5,max_size=65564) as ws:
        ws.send(json.dumps({"ticket":info["ticket"],"client_public_key":base64.b64encode(public).decode()}))
        checks["encrypted_viewer_ready"] = json.loads(ws.recv(timeout=5)).get("ready") is True
        buffered = bytearray()
        def read(n):
            while len(buffered)<n:
                buffered.extend(cipher.decrypt(ws.recv(timeout=5)))
            value=bytes(buffered[:n]);del buffered[:n];return value
        def send(value):
            ws.send(cipher.encrypt(value))
        assert read(12) == b"RFB 003.008\n"
        send(b"RFB 003.008\n")
        n=read(1)[0];types=read(n);assert 1 in types;send(b"\x01")
        assert read(4)==b"\x00\x00\x00\x00"
        send(b"\x01")
        init=read(24);read(int.from_bytes(init[20:24],"big"))
        def key(sym):
            send(struct.pack(">BBHI",4,1,0,sym)+struct.pack(">BBHI",4,0,0,sym))
            time.sleep(.04)
        for char in "synthetic-private-entry": key(ord(char))
        time.sleep(.3)
        checks["real_rfb_password_input"] = fixture("document.getElementById('pw').value === 'synthetic-private-entry'")["value"] is True
        key(0xff0d)  # Enter submits the synthetic login.
        time.sleep(.3)
        for char in "123456": key(ord(char))
        time.sleep(.3)
        checks["real_rfb_2fa_input"] = fixture("document.getElementById('otp').value === '123456'")["value"] is True
        key(0xff0d)
        time.sleep(.3)
        checks["same_browser_login_completed"] = fixture("document.cookie.includes('hpd321fixture=complete') && !!document.getElementById('next')")["value"] is True
        returned = management("display.lease.release", epoch=grant["epoch"], viewer_id=viewer_id)
        checks["explicit_safe_handback"] = returned["holder"] == "agent"
        continued=agent()
        snapshot=continued.get("data",{}).get("snapshot","")
        checks["agent_continues_same_logged_in_page"] = continued.get("success") is True and "Continue task" in snapshot
        checks["no_synthetic_input_in_agent_output"] = "synthetic-private-entry" not in json.dumps(continued) and "123456" not in json.dumps(continued)
    # A real viewer connection closed WHILE human holds must stay human.
    status=management("display.status")
    management("display.lease.acquire",epoch=status["lease"]["epoch"],viewer_id=viewer_id)
    reconnect = management("display.viewer.mint", viewer_id=viewer_id)
    key = ec.generate_private_key(ec.SECP256R1())
    pub = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    with connect(f"ws://127.0.0.1:{args.port}",open_timeout=5) as ws:
        ws.send(json.dumps({"ticket":reconnect["ticket"],"client_public_key":base64.b64encode(pub).decode()}))
        assert json.loads(ws.recv(timeout=5)).get("ready") is True
    time.sleep(.2)
    checks["disconnect_never_resumes_agent"] = agent().get("code") == "human_has_control"
    status=management("display.status")
    management("display.cancel",epoch=status["lease"]["epoch"])
    checks["cancel_revokes_agent"] = agent().get("code") in {"session_expired","browser_unavailable"}
    print(json.dumps({"checks":checks,"all_passed":all(checks.values()),
                      "evidence":"real Linux browser and encrypted RFB; synthetic login; no model/iPhone"},sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("Isolated browser probe failed; no private payload printed")
