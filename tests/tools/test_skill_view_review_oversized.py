"""HPD-1167 follow-up: in the background review, skill_view does not return
the body of a skill whose SKILL.md is above AGENT_SKILL_MD_MAX_CHARS.

r122 acceptance on fc-e2e-2 (2026-10-11): the review read the pre-existing
69,286-char "hey-hermes-memory-stack" in full, and four review calls ran at
102-106k native tokens, although the 25k cap refuses every write that would
grow it. Chat-turn skill_view is unchanged.
"""

import json
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import AGENT_SKILL_MD_MAX_CHARS, skill_manage


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


def _place(skills_dir, name, total_chars):
    head = f"---\nname: {name}\ndescription: Size test skill.\n---\n# {name}\n\n"
    content = head + ("x" * max(0, total_chars - len(head)))
    (skills_dir / name).mkdir()
    (skills_dir / name / "SKILL.md").write_text(content, encoding="utf-8")
    refs = skills_dir / name / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("small reference\n", encoding="utf-8")
    return content


def _review():
    return patch("tools.skill_provenance.is_background_review", return_value=True)


def test_review_gets_notice_not_body_for_oversized_skill(isolate_skills, caplog):
    from tools.skills_tool import skill_view
    from tools.skill_manager_tool import _reset_background_review_read_marks

    _place(isolate_skills, "big-skill", 69_286)
    _reset_background_review_read_marks()
    with _review(), caplog.at_level("INFO"):
        raw = skill_view("big-skill")
    assert "skill_review_skip_oversize name=big-skill chars=69286" in caplog.text
    result = json.loads(raw)
    assert result["success"] is True
    assert "content" not in result
    assert result["name"] == "big-skill"
    assert result["description"] == "Size test skill."
    assert "size cap" in result["notice"]
    assert "leave it" in result["notice"]
    assert len(raw) < 2_000


def test_review_notice_does_not_authorize_a_write(isolate_skills):
    from tools.skills_tool import skill_view
    from tools.skill_manager_tool import _reset_background_review_read_marks

    content = _place(isolate_skills, "big-skill", 69_286)
    _reset_background_review_read_marks()
    with _review():
        skill_view("big-skill")
        result = json.loads(skill_manage(
            action="patch", name="big-skill",
            old_string="# big-skill\n\n" + "x" * 1_000, new_string="# big-skill\n\n",
        ))
    assert result["success"] is False
    assert (isolate_skills / "big-skill" / "SKILL.md").read_text(encoding="utf-8") == content
    _reset_background_review_read_marks()


def test_review_still_reads_skill_at_cap_in_full(isolate_skills):
    from tools.skills_tool import skill_view
    from tools.skill_manager_tool import _reset_background_review_read_marks

    _place(isolate_skills, "ok-skill", AGENT_SKILL_MD_MAX_CHARS)
    _reset_background_review_read_marks()
    with _review():
        result = json.loads(skill_view("ok-skill"))
    assert result["success"] is True
    assert len(result["content"]) > AGENT_SKILL_MD_MAX_CHARS - 1_000
    _reset_background_review_read_marks()


def test_chat_turn_still_reads_oversized_skill_in_full(isolate_skills):
    from tools.skills_tool import skill_view

    _place(isolate_skills, "big-skill", 69_286)
    result = json.loads(skill_view("big-skill"))
    assert result["success"] is True
    assert len(result["content"]) > 69_000
    assert "notice" not in result


def test_review_can_still_read_a_supporting_file_of_oversized_skill(isolate_skills):
    from tools.skills_tool import skill_view

    _place(isolate_skills, "big-skill", 69_286)
    with _review():
        result = json.loads(skill_view("big-skill", "references/notes.md"))
    assert result["success"] is True
    assert "small reference" in result["content"]
