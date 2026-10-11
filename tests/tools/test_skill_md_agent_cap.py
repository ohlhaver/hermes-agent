"""HPD-1167: agent-written SKILL.md is capped at AGENT_SKILL_MD_MAX_CHARS.

Diagnosis (HPD-1151, fc-e2e-2, 2026-10-11): the background skill review patched
the same agent-created skill on every pass; it grew from 62k to 72k chars and
every review loaded it in full, pushing review calls above 100k native tokens.
Growth beyond the cap must go to references/; an existing over-cap skill stays
as it is until its next edit, and an edit that does not grow it is allowed so
it can be split.
"""

import json

import pytest

from tools.skill_manager_tool import (
    AGENT_SKILL_MD_MAX_CHARS,
    MAX_SKILL_CONTENT_CHARS,
    skill_manage,
)


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


def _content(name: str, total_chars: int) -> str:
    head = f"---\nname: {name}\ndescription: Cap test skill.\n---\n# {name}\n\n"
    return head + ("x" * max(0, total_chars - len(head)))


def _hand_place(skills_dir, name: str, total_chars: int) -> str:
    content = _content(name, total_chars)
    (skills_dir / name).mkdir()
    (skills_dir / name / "SKILL.md").write_text(content, encoding="utf-8")
    return content


def test_cap_is_about_25k_and_below_the_general_limit():
    assert AGENT_SKILL_MD_MAX_CHARS == 25_000
    assert AGENT_SKILL_MD_MAX_CHARS < MAX_SKILL_CONTENT_CHARS


def test_create_at_cap_succeeds():
    content = _content("at-cap", AGENT_SKILL_MD_MAX_CHARS)
    assert len(content) == AGENT_SKILL_MD_MAX_CHARS
    result = json.loads(skill_manage(action="create", name="at-cap", content=content))
    assert result["success"] is True, result


def test_create_over_cap_refused_with_references_hint(isolate_skills):
    content = _content("too-big", AGENT_SKILL_MD_MAX_CHARS + 1)
    result = json.loads(skill_manage(action="create", name="too-big", content=content))
    assert result["success"] is False
    assert "25,000" in result["error"]
    assert "references/" in result["error"]
    assert not (isolate_skills / "too-big").exists()


def test_edit_growing_past_cap_refused(isolate_skills):
    small = _content("grower", 1_000)
    assert json.loads(skill_manage(action="create", name="grower", content=small))["success"]
    big = _content("grower", AGENT_SKILL_MD_MAX_CHARS + 500)
    result = json.loads(skill_manage(action="edit", name="grower", content=big))
    assert result["success"] is False
    assert "25,000" in result["error"]
    assert (isolate_skills / "grower" / "SKILL.md").read_text(encoding="utf-8") == small


def test_patch_growing_past_cap_refused(isolate_skills):
    near = _content("near", AGENT_SKILL_MD_MAX_CHARS - 10)
    assert json.loads(skill_manage(action="create", name="near", content=near))["success"]
    result = json.loads(skill_manage(
        action="patch", name="near",
        old_string="# near", new_string="# near\n" + ("y" * 100),
    ))
    assert result["success"] is False
    assert "25,000" in result["error"]
    assert (isolate_skills / "near" / "SKILL.md").read_text(encoding="utf-8") == near


def test_existing_over_cap_skill_is_left_untouched_and_still_readable(isolate_skills):
    from tools.skills_tool import skill_view

    content = _hand_place(isolate_skills, "legacy-big", 70_000)
    viewed = json.loads(skill_view("legacy-big"))
    assert viewed["success"] is True
    assert len(viewed["content"]) >= 69_000
    assert (isolate_skills / "legacy-big" / "SKILL.md").read_text(encoding="utf-8") == content


def test_existing_over_cap_skill_cannot_grow(isolate_skills):
    content = _hand_place(isolate_skills, "legacy-big", 70_000)
    result = json.loads(skill_manage(
        action="patch", name="legacy-big",
        old_string="# legacy-big", new_string="# legacy-big\n\nNew pitfall: more text.",
    ))
    assert result["success"] is False
    assert "references/" in result["error"]
    assert (isolate_skills / "legacy-big" / "SKILL.md").read_text(encoding="utf-8") == content


def test_existing_over_cap_skill_may_shrink_by_patch(isolate_skills):
    _hand_place(isolate_skills, "legacy-big", 70_000)
    result = json.loads(skill_manage(
        action="patch", name="legacy-big",
        old_string="# legacy-big\n\n" + "x" * 1_000,
        new_string="# legacy-big\n\nSee references/detail.md.\n",
    ))
    assert result["success"] is True, result
    after = (isolate_skills / "legacy-big" / "SKILL.md").read_text(encoding="utf-8")
    assert AGENT_SKILL_MD_MAX_CHARS < len(after) < 70_000


def test_existing_over_cap_skill_may_be_rewritten_smaller(isolate_skills):
    _hand_place(isolate_skills, "legacy-big", 70_000)
    smaller = _content("legacy-big", 40_000)
    result = json.loads(skill_manage(action="edit", name="legacy-big", content=smaller))
    assert result["success"] is True, result


def test_growth_goes_to_references(isolate_skills):
    near = _content("umbrella", AGENT_SKILL_MD_MAX_CHARS - 10)
    assert json.loads(skill_manage(action="create", name="umbrella", content=near))["success"]
    result = json.loads(skill_manage(
        action="write_file", name="umbrella",
        file_path="references/detail.md", file_content="z" * 60_000,
    ))
    assert result["success"] is True, result


def test_supporting_file_patch_keeps_general_limit(isolate_skills):
    small = _content("with-ref", 1_000)
    assert json.loads(skill_manage(action="create", name="with-ref", content=small))["success"]
    assert json.loads(skill_manage(
        action="write_file", name="with-ref",
        file_path="references/data.md", file_content="small",
    ))["success"]
    result = json.loads(skill_manage(
        action="patch", name="with-ref", file_path="references/data.md",
        old_string="small", new_string="w" * 50_000,
    ))
    assert result["success"] is True, result


def test_dashboard_write_path_is_not_capped(isolate_skills):
    """The dashboard editor calls _create_skill/_edit_skill directly; a user
    writing their own skill is not an agent write and keeps the general limit."""
    from tools.skill_manager_tool import _create_skill, _edit_skill

    big = _content("user-big", 40_000)
    assert _create_skill("user-big", big)["success"] is True
    bigger = _content("user-big", 60_000)
    assert _edit_skill("user-big", bigger)["success"] is True


def test_patch_with_explicit_skill_md_file_path_is_capped(isolate_skills):
    """Review finding on #27: file_path='SKILL.md' must not bypass the cap."""
    base = _content("explicit", 20_000)
    assert json.loads(skill_manage(action="create", name="explicit", content=base))["success"]
    result = json.loads(skill_manage(
        action="patch", name="explicit", file_path="SKILL.md",
        old_string="# explicit", new_string="# explicit\n" + ("y" * 30_000),
    ))
    assert result["success"] is False
    assert "25,000" in result["error"]
    assert (isolate_skills / "explicit" / "SKILL.md").read_text(encoding="utf-8") == base


def test_patch_with_explicit_skill_md_file_path_keeps_frontmatter_check(isolate_skills):
    base = _content("fm", 1_000)
    assert json.loads(skill_manage(action="create", name="fm", content=base))["success"]
    result = json.loads(skill_manage(
        action="patch", name="fm", file_path="SKILL.md",
        old_string="---\nname: fm", new_string="name: fm",
    ))
    assert result["success"] is False
    assert (isolate_skills / "fm" / "SKILL.md").read_text(encoding="utf-8") == base


def test_write_file_cannot_target_skill_md(isolate_skills):
    """Review finding on #27: write_file must not replace SKILL.md (use edit)."""
    base = _content("wf", 20_000)
    assert json.loads(skill_manage(action="create", name="wf", content=base))["success"]
    result = json.loads(skill_manage(
        action="write_file", name="wf", file_path="SKILL.md",
        file_content=_content("wf", 90_000),
    ))
    assert result["success"] is False
    assert "edit" in result["error"]
    assert (isolate_skills / "wf" / "SKILL.md").read_text(encoding="utf-8") == base


def test_over_cap_error_names_current_size_for_existing_skill(isolate_skills):
    _hand_place(isolate_skills, "legacy-big", 70_000)
    result = json.loads(skill_manage(
        action="patch", name="legacy-big",
        old_string="# legacy-big", new_string="# legacy-big\n\nMore.",
    ))
    assert result["success"] is False
    assert "70,000" in result["error"]
    assert "may not grow" in result["error"]
