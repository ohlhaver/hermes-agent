"""HPD-1901: browser_click sets a checkbox or radio with check/uncheck."""

import json
from unittest.mock import patch

import pytest

from tools import browser_tool
from tools.browser_tool import browser_click, browser_snapshot

REFS = {"e88": {"role": "checkbox", "name": "Mit dem Abschicken"},
        "e113": {"role": "radio", "name": "Kreditkarte & Twint"},
        "e48": {"role": "button", "name": "kostenpflichtig bestellen"}}


@pytest.fixture(autouse=True)
def _plain_backend(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(browser_tool, "_CONTROL_ROLES", {})


def _backend(checked=False, fail=(), checked_after=None):
    calls = []
    reads = []

    def run(session, command, args, **kwargs):
        calls.append((command, args))
        if command in ("is", "check", "uncheck"):
            assert kwargs.get("_engine_override") == "auto"
        if command in fail:
            return {"success": False, "error": "Not a checkbox or radio button"}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "- checkbox [ref=e88]", "refs": REFS}}
        if command == "is":
            reads.append(1)
            value = checked if len(reads) == 1 or checked_after is None else checked_after
            return {"success": True, "data": {"checked": value}}
        return {"success": True, "data": {}}
    return run, calls


def _click(ref, run):
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        browser_snapshot(task_id="t")
        return json.loads(browser_click(ref, task_id="t"))


def test_unchecked_checkbox_is_checked_not_clicked():
    run, calls = _backend(checked=False)
    assert _click("@e88", run) == {"success": True, "clicked": "@e88", "checked": True}
    assert ("check", ["@e88"]) in calls and not any(c[0] == "click" for c in calls)


def test_checked_checkbox_is_unchecked():
    run, calls = _backend(checked=True)
    assert _click("e88", run)["checked"] is False
    assert ("uncheck", ["@e88"]) in calls


def test_radio_is_checked_and_a_checked_radio_stays():
    run, calls = _backend(checked=False)
    assert _click("@e113", run)["checked"] is True
    assert ("check", ["@e113"]) in calls
    run, calls = _backend(checked=True)
    assert _click("@e113", run)["checked"] is True
    assert not any(c[0] in ("check", "uncheck", "click") for c in calls)


def test_other_refs_and_failed_checks_fall_back_to_click():
    run, calls = _backend()
    assert _click("@e48", run) == {"success": True, "clicked": "@e48"}
    assert [c[0] for c in calls] == ["snapshot", "click"]
    run, calls = _backend(fail=("check",))
    assert _click("@e88", run) == {"success": True, "clicked": "@e88"}
    assert calls[-1] == ("click", ["@e88"])


def test_without_a_snapshot_a_ref_is_clicked():
    run, calls = _backend()
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        assert json.loads(browser_click("@e88", task_id="t")) == {"success": True, "clicked": "@e88"}
    assert [c[0] for c in calls] == ["click"]


def test_a_check_that_changed_the_box_before_failing_is_not_clicked_again():
    run, calls = _backend(checked=False, fail=("check",), checked_after=True)
    assert _click("@e88", run) == {"success": True, "clicked": "@e88", "checked": True}
    assert not any(c[0] == "click" for c in calls)


def test_snapshot_names_the_page_address_without_query():
    def run(session, command, args, **kwargs):
        return {"success": True, "data": {"snapshot": "- button [ref=e1]", "refs": {},
                                          "origin": "https://www.saferpay.com/vt2/Api/PaymentPage/1/2?token=x#y"}}
    with patch("tools.browser_tool._run_browser_command", side_effect=run), \
            patch("tools.browser_tool._is_local_backend", return_value=True):
        result = json.loads(browser_snapshot(task_id="t"))
    assert result["url"] == "https://www.saferpay.com/vt2/Api/PaymentPage/1/2"


def test_snapshot_address_drops_login_details():
    def run(session, command, args, **kwargs):
        return {"success": True, "data": {"snapshot": "", "refs": {}, "origin": "https://alice:secret@shop.example:8443/pay?x=1"}}
    with patch("tools.browser_tool._run_browser_command", side_effect=run), \
            patch("tools.browser_tool._is_local_backend", return_value=True):
        assert json.loads(browser_snapshot(task_id="t"))["url"] == "https://shop.example:8443/pay"
