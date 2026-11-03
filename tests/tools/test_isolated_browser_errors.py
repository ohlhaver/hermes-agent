"""Distinct agent error texts, retry window, dialog hoisting (Uber Eats 2026-10-26)."""
import json
import socket
import threading
from unittest.mock import patch

import pytest

from tools import isolated_browser as ib
from tools import browser_tool as bt


def _serve(tmp_path, responses):
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(dir="/tmp"), "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(5)
    seen = []

    def run():
        for resp in responses:
            conn, _ = srv.accept()
            with conn:
                seen.append(conn.makefile("rb").readline())
                conn.sendall(json.dumps(resp).encode() + b"\n")
        srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return path, seen, t


def test_each_code_has_its_own_text(tmp_path):
    texts = {}
    for code in ("command_denied", "human_has_control", "session_expired"):
        path, _, _ = _serve(tmp_path, [{"ok": False, "code": code}])
        r = ib.command_from_agent(path, "sess", "snapshot", [])
        assert r["success"] is False and r["code"] == code
        texts[code] = r["error"]
    assert "browser is working" in texts["command_denied"]
    assert "browser_snapshot" in texts["command_denied"]
    assert "paused" not in texts["command_denied"].lower()
    assert "taken over" in texts["human_has_control"]
    assert len(set(texts.values())) == 3


def test_denied_does_not_say_user_holds_browser():
    assert "user" not in ib._AGENT_ERROR_TEXT["command_denied"].lower()


def test_unavailable_retries_then_reports(tmp_path):
    sleeps = []
    clock = [0.0]

    def mono():
        return clock[0]

    def sleep(n):
        sleeps.append(n)
        clock[0] += n

    with patch.object(ib.time, "monotonic", mono), patch.object(ib.time, "sleep", sleep):
        r = ib.command_from_agent(str(tmp_path / "missing.sock"), "s", "click", ["e1"])
    assert r["code"] == "browser_unavailable"
    assert "_retry_safe" not in r
    assert 18 <= sum(sleeps) <= 20  # connect failed: nothing sent, safe even for click
    assert "20 seconds" in r["error"]


def test_unavailable_recovers_during_wait(tmp_path):
    path, seen, t = _serve(tmp_path, [{"ok": False, "code": "handoff_pending"},
                                      {"ok": True, "result": {"success": True}}])
    with patch.object(ib.time, "sleep", lambda n: None):
        r = ib.command_from_agent(path, "s", "click", ["e1"])
    assert r == {"success": True}
    assert len(seen) == 2


def test_broker_error_on_click_is_not_repeated(tmp_path):
    path, seen, t = _serve(tmp_path, [{"ok": False, "code": "browser_unavailable"}])
    with patch.object(ib.time, "sleep", lambda n: pytest.fail("must not wait")):
        r = ib.command_from_agent(path, "s", "click", ["e1"])
    assert r["code"] == "browser_unavailable"
    assert len(seen) == 1


def test_denied_returns_without_waiting(tmp_path):
    path, seen, t = _serve(tmp_path, [{"ok": False, "code": "command_denied"}])
    with patch.object(ib.time, "sleep", lambda n: pytest.fail("must not wait")):
        ib.command_from_agent(path, "s", "screenshot", [])
    assert len(seen) == 1


def test_vision_and_console_not_offered_when_isolated():
    with patch.object(bt, "_isolated_browser_binding", return_value=("/x", "s")), \
         patch.object(bt, "check_browser_requirements", return_value=True):
        assert bt.check_browser_unprotected_only_requirements() is False
        assert bt.check_browser_vision_requirements() is False
    with patch.object(bt, "_isolated_browser_binding", return_value=None), \
         patch.object(bt, "check_browser_requirements", return_value=True):
        assert bt.check_browser_unprotected_only_requirements() is True


SNAP = "\n".join(
    ["- banner [ref=e1]:"] + [f"  - link \"Nav {i}\" [ref=e{i+2}]" for i in range(900)]
    + ['- dialog "Spaghetti-Eis" [ref=e2000]:',
       '  - radio "Gross" [ref=e2001]',
       '  - button "Add to order" [ref=e2002]',
       '- contentinfo [ref=e3000]']
)


def test_dialog_survives_truncation():
    assert len(SNAP) > bt.SNAPSHOT_SUMMARIZE_THRESHOLD
    out = bt._truncate_snapshot(SNAP)
    assert "Add to order" in out and "ref=e2002" in out
    assert out.index("Add to order") < out.index("Nav 0")


def test_no_dialog_unchanged_order():
    plain = "\n".join(f"- link \"L{i}\" [ref=e{i}]" for i in range(1500))
    assert bt._hoist_dialogs(plain) == plain


def test_modal_attribute_hoisted():
    snap = "- main [ref=e1]:\n  - link \"a\" [ref=e2]\n- region \"Cart\" [ref=e3, modal]:\n  - button \"Close\" [ref=e4]"
    out = bt._hoist_dialogs(snap)
    assert out.startswith("[OPEN DIALOG")
    assert out.index("Close") < out.index("link \"a\"")


def test_snapshot_and_navigate_pass_code():
    failure = {"success": False, "code": "command_denied", "error": "x"}
    with patch.object(bt, "_run_browser_command", return_value=failure), \
         patch.object(bt, "_is_camofox_mode", return_value=False):
        assert json.loads(bt.browser_snapshot(task_id="t"))["code"] == "command_denied"


def test_blocked_target_text():
    text = ib._AGENT_ERROR_TEXT["blocked_target"]
    assert "cannot be opened" in text and "browser is working" in text
    # HPD-1787: a refusal can be a brief check failure; the model tries once more.
    assert "check failed briefly" in text and "Try opening the page once more with browser_navigate" in text
    # Rules review 2026-10-28: never "the same step" (a click or submit that already ran).
    assert "same step" not in text


def test_navigate_settles_once_in_isolated_browser():
    calls = []

    def run(task, command, args=None, **kw):
        calls.append(command)
        if command == "open":
            return {"success": True, "data": {"title": "", "url": "https://sbb.ch/x"}}
        n = calls.count("snapshot")
        refs = {"e1": 1} if n == 1 else {f"e{i}": i for i in range(30)}
        return {"success": True, "data": {"snapshot": "- x", "refs": refs}}

    with patch.object(bt, "_run_browser_command", side_effect=run), \
         patch.object(bt, "_isolated_browser_binding", return_value=("/x", "s")), \
         patch.object(bt, "_is_camofox_mode", return_value=False), \
         patch.object(bt, "_get_session_info", return_value={"_first_nav": False}), \
         patch.object(bt, "check_website_access", return_value=None), \
         patch.object(bt, "_is_always_blocked_url", return_value=False), \
         patch.object(bt, "_is_local_backend", return_value=True), \
         patch.object(bt.time, "sleep") as sl:
        out = json.loads(bt.browser_navigate("https://sbb.ch/x", task_id="t"))
    assert calls.count("snapshot") == 2 and sl.called
    assert out["element_count"] == 30


def test_long_dialog_keeps_head_buttons_and_end():
    # r152 canary: a 6,000+ character options dialog lost its add button at the end.
    opts = [f'        - checkbox "Option {i} +CHF4.00" [checked=false, ref=e{1000 + i}]' for i in range(400)]
    opts[200] = '        - button "Größe wählen" [ref=e7777]'
    opts[2] = '        - radio "Gross" [checked=false, ref=e7778]'
    lines = (['- main [ref=e1]:', '  - link "nav" [ref=e2]', '- dialog "Pizza" [ref=e3]:',
              '    - heading "Pizza Margherita" [level=1, ref=e4]',
              '    - button "Mehr Info" [ref=e5]'] + opts
             + ['    - button "Hinzufügen 1 CHF18.90" [ref=e9001]', '    - button "Schließen" [ref=e9002]'])
    out = bt._hoist_dialogs('\n'.join(lines))
    head, _, rest = out.partition('[END DIALOG]')
    assert len(head) <= bt._DIALOG_BLOCK_MAX_CHARS + 200
    assert 'Pizza Margherita' in head and 'Hinzufügen 1 CHF18.90' in head and 'ref=e9001' in head
    assert 'Mehr Info' in head and 'lines of this dialog omitted' in head
    assert 'ref=e7777' in head and 'ref=e7778' in head  # a middle button and a middle radio survive
    assert 'link "nav"' in rest


def test_short_dialog_is_not_shrunk():
    lines = ['- dialog "d" [ref=e1]:', '  - button "OK" [ref=e2]']
    assert bt._shrink_dialog_block(lines) == '\n'.join(lines)


# HPD-1798 -----------------------------------------------------------------

def _navigate(snapshots, title=""):
    calls = []

    def run(task, command, args=None, **kw):
        calls.append(command)
        if command == "open":
            return {"success": True, "data": {"title": title, "url": "https://site.example/a"}}
        n = calls.count("snapshot")
        return {"success": True, "data": snapshots[min(n, len(snapshots)) - 1]}

    with patch.object(bt, "_run_browser_command", side_effect=run), \
         patch.object(bt, "_isolated_browser_binding", return_value=("/x", "s")), \
         patch.object(bt, "_is_camofox_mode", return_value=False), \
         patch.object(bt, "_get_session_info", return_value={"_first_nav": False}), \
         patch.object(bt, "check_website_access", return_value=None), \
         patch.object(bt, "_is_always_blocked_url", return_value=False), \
         patch.object(bt, "_is_local_backend", return_value=True), \
         patch.object(bt.time, "sleep"):
        out = json.loads(bt.browser_navigate("https://site.example/a", task_id="t"))
    return out, calls.count("snapshot")


_HTML = {"content_type": "text/html", "consent": False, "document_text": ""}
_FULL = {f"e{i}": i for i in range(30)}


def test_untitled_spa_is_waited_for_beyond_one_look():
    thin = {"snapshot": "- generic", "refs": {"e1": 1}, "origin": "https://site.example/a", "page_info": _HTML}
    full = {"snapshot": "- x", "refs": _FULL, "origin": "https://site.example/a", "page_info": _HTML}
    out, looks = _navigate([thin, thin, thin, full])
    assert looks == 4 and out["element_count"] == 30 and "page_notice" not in out


def test_error_document_is_named_and_not_waited_for():
    xml = {"content_type": "application/xml", "consent": False, "document_text": "AccessDenied Access Denied"}
    out, looks = _navigate([{"snapshot": "- generic", "refs": {"e1": 1}, "origin": "https://site.example/a", "page_info": xml}])
    assert looks == 1 and "not a web page" in out["page_notice"]
    assert "AccessDenied" not in out["page_notice"] and "AccessDenied" in out["page_document_text"]


def test_pdf_and_full_non_html_pages_are_not_called_errors():
    pdf = {"content_type": "application/pdf", "consent": False, "document_text": ""}
    out, _ = _navigate([{"snapshot": "- generic", "refs": {"e1": 1}, "origin": "https://site.example/a", "page_info": pdf}], title="T")
    assert "page_notice" not in out
    xml = {"content_type": "text/xml", "consent": False, "document_text": "feed"}
    out, _ = _navigate([{"snapshot": "- x", "refs": _FULL, "origin": "https://site.example/a", "page_info": xml}], title="T")
    assert "page_notice" not in out


def test_reported_url_after_a_redirect_carries_no_query():
    other = {"snapshot": "- x", "refs": _FULL, "origin": "https://other.example/cb?code=SECRET#frag", "page_info": _HTML}
    out, _ = _navigate([other, other], title="T")
    assert out["url"] == "https://other.example/cb"


def test_snapshot_of_another_page_is_taken_again_then_named():
    other = {"snapshot": "- x", "refs": _FULL, "origin": "https://other.example/b", "page_info": _HTML}
    right = {"snapshot": "- x", "refs": _FULL, "origin": "https://site.example/a?3", "page_info": _HTML}
    out, looks = _navigate([other, right], title="T")
    assert looks == 2 and out["url"] == "https://site.example/a"
    out, looks = _navigate([other, other], title="T")
    assert looks == 2 and out["url"] == "https://other.example/b"


def test_consent_banner_is_reported():
    info = dict(_HTML, consent=True)
    out, _ = _navigate([{"snapshot": "- x", "refs": _FULL, "origin": "https://site.example/a", "page_info": info}], title="T")
    assert "consent" in out["consent_notice"]
