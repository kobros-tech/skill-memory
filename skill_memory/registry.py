"""
registry.py

Metadata store for acquired skills.

Deliberately generic: it does not import anything from skill-cloning, and it
does not know or care what kind of object `skill` is, as long as callers are
consistent (in this project, it's a skill-cloning `Skill` instance wrapping a
`TinyMLP`). This keeps the registry testable without training any networks.

Note on "capacity": unlike the original block-allocator design, skills here
are NOT slices of one shared network -- each skill is its own independent
network (matching how skill-cloning actually works; see bootstrap.py's
docstring for why). So "capacity" is a soft budget on the *number* of
registered skills, not a partition of a fixed neuron pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class DuplicateSkillError(Exception):
    """Raised when registering a skill name that already exists."""


class CapacityExhaustedError(Exception):
    """Raised when the registry is at its skill-count budget."""


class UnknownSkillError(KeyError):
    """Raised when looking up a skill name that isn't registered."""


@dataclass
class SkillRecord:
    name: str
    skill: Any                      # skill-cloning Skill instance
    parent: Optional[str] = None
    origin: str = "scratch"         # "scratch" | "clone" | "reuse"
    acquisition_steps: int = 0
    solved: bool = False
    final_accuracy: float = 0.0
    compatibility_score: float = 0.0
    seed: Optional[int] = None
    creation_order: int = 0


class SkillRegistry:
    def __init__(self, max_skills: int = 8):
        self.max_skills = max_skills
        self._records: dict[str, SkillRecord] = {}
        self._order = 0

    def register(self, name: str, skill: Any, **meta) -> SkillRecord:
        if name in self._records:
            raise DuplicateSkillError(f"skill '{name}' is already registered")
        if len(self._records) >= self.max_skills:
            raise CapacityExhaustedError(
                f"registry is at capacity ({self.max_skills} skills)"
            )
        self._order += 1
        record = SkillRecord(name=name, skill=skill, creation_order=self._order, **meta)
        self._records[name] = record
        return record

    def get(self, name: str) -> SkillRecord:
        try:
            return self._records[name]
        except KeyError:
            raise UnknownSkillError(name) from None

    def contains(self, name: str) -> bool:
        return name in self._records

    def list_skills(self) -> list[str]:
        return list(self._records.keys())

    def available_capacity(self) -> int:
        return self.max_skills - len(self._records)

    def allocated_capacity(self) -> int:
        return len(self._records)

    def skills_dict(self) -> dict[str, Any]:
        """{name: Skill}, the shape skill-cloning's compatibility.py expects."""
        return {name: rec.skill for name, rec in self._records.items()}
