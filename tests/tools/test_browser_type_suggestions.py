"""HPD-1901: browser_type names a suggestion list that opened after typing."""

import json
from unittest.mock import patch

import pytest

from tools import browser_tool
from tools.browser_tool import browser_type
from tools.isolated_browser import SUGGESTION_COUNT_JS


@pytest.fixture(autouse=True)
def _plain_backend(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    monkeypatch.setattr(browser_tool, "_SUGGESTION_WAITS", (0, 0))


def _backend(counts, refs=None):
    counts = list(counts)
    calls = []

    def run(session, command, args, **kwargs):
        calls.append((command, args))
        if command == "eval":
            assert args == [SUGGESTION_COUNT_JS]
            return {"success": True, "data": {"result": counts.pop(0) if counts else 0}}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "- list [ref=e1]\n  - listitem \"8700 Küsnacht ZH\" [ref=e3]",
                                              "refs": refs or {}}}
        return {"success": True}
    return run, calls


def test_names_the_matching_suggestion_and_returns_current_refs():
    refs = {"e1": {"role": "list", "name": "8700 Küsnacht ZH"},
            "e3": {"role": "listitem", "name": "8700 Küsnacht ZH"},
            "e9": {"role": "listitem", "name": "Pizza Kuriere"}}
    run, calls = _backend([0, 1], refs)
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        result = json.loads(browser_type("@e86", "8700", task_id="t"))
    assert result["suggestions"] == (
        'The page shows suggestions for this field: "8700 Küsnacht ZH" (@e3). '
        "Many forms keep only a value picked from such a list. "
        "Click the matching suggestion with browser_click before you type the next field.")
    assert "[ref=e3]" in result["snapshot"]
    assert [c[0] for c in calls] == ["eval", "fill", "press", "eval", "snapshot"]
    assert ("press", ["End"]) in calls and ("snapshot", ["-i"]) in calls


def test_matching_ignores_accents_and_case():
    refs = {"e2": {"role": "listitem", "name": "Bühlstrasse"}}
    assert browser_tool._matching_suggestions("", refs, "buhl") == [("e2", "Bühlstrasse")]


def test_names_come_from_the_snapshot_when_refs_have_none():
    # agent-browser 0.26 on dieci.ch: cursor-clickable entries have an empty refs name.
    snapshot = ('- list "8700 Küsnacht ZH" [ref=e1] focusable [tabindex]\n'
                '  - listitem "8700 Küsnacht ZH" [level=1, ref=e3] clickable [cursor:pointer]')
    refs = {"e1": {"name": "", "role": "list"}, "e3": {"name": "", "role": "listitem"}}
    assert browser_tool._matching_suggestions(snapshot, refs, "8700") == [("e3", "8700 Küsnacht ZH")]


def test_no_new_suggestions_means_no_snapshot_and_no_notice():
    run, calls = _backend([2, 2, 2])
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        result = json.loads(browser_type("@e5", "Justus", task_id="t"))
    assert "suggestions" not in result and "snapshot" not in result
    assert "snapshot" not in [c[0] for c in calls]


def test_unrelated_new_entries_return_snapshot_without_notice():
    run, _ = _backend([0, 1], {"e4": {"role": "option", "name": "Something else"}})
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        result = json.loads(browser_type("@e5", "Zürich", task_id="t"))
    assert "suggestions" not in result
    assert "snapshot" in result


def test_secret_text_gets_no_suggestion_check():
    run, calls = _backend([1, 1])
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        browser_type("@e5", "sk-proj-ABCD1234567890EFGH", task_id="t")
    assert [c[0] for c in calls] == ["fill"]


def test_failed_fill_skips_the_check():
    calls = []

    def run(session, command, args, **kwargs):
        calls.append(command)
        return {"success": command == "eval", "data": {"result": 0}, "error": "x"}
    with patch("tools.browser_tool._run_browser_command", side_effect=run):
        result = json.loads(browser_type("@e5", "8700", task_id="t"))
    assert result["success"] is False
    assert calls == ["eval", "fill"]
