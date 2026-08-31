import pytest

from skill_memory.registry import (
    SkillRegistry,
    DuplicateSkillError,
    CapacityExhaustedError,
    UnknownSkillError,
)


class FakeSkill:
    """Stand-in for skill-cloning's Skill -- registry.py doesn't care what this is."""
    def __init__(self, name):
        self.name = name


def test_register_and_get():
    reg = SkillRegistry(max_skills=4)
    s = FakeSkill("addition")
    record = reg.register("addition", s, origin="scratch")
    assert reg.contains("addition")
    assert reg.get("addition") is record
    assert record.skill is s
    assert record.creation_order == 1


def test_reject_duplicate():
    reg = SkillRegistry(max_skills=4)
    reg.register("addition", FakeSkill("addition"))
    with pytest.raises(DuplicateSkillError):
        reg.register("addition", FakeSkill("addition"))


def test_unknown_skill_lookup():
    reg = SkillRegistry(max_skills=4)
    with pytest.raises(UnknownSkillError):
        reg.get("does_not_exist")


def test_capacity_tracking_and_exhaustion():
    reg = SkillRegistry(max_skills=2)
    assert reg.available_capacity() == 2
    reg.register("addition", FakeSkill("addition"))
    assert reg.available_capacity() == 1
    reg.register("multiplication", FakeSkill("multiplication"))
    assert reg.available_capacity() == 0
    with pytest.raises(CapacityExhaustedError):
        reg.register("subtraction", FakeSkill("subtraction"))


def test_list_and_skills_dict():
    reg = SkillRegistry(max_skills=4)
    a, m = FakeSkill("addition"), FakeSkill("multiplication")
    reg.register("addition", a)
    reg.register("multiplication", m)
    assert set(reg.list_skills()) == {"addition", "multiplication"}
    assert reg.skills_dict() == {"addition": a, "multiplication": m}
    assert reg.allocated_capacity() == 2
