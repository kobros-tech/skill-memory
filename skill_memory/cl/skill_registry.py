# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Bookkeeping and skill-state snapshot storage.

The registry deliberately separates two concepts:

* ``SkillMemory`` stores model ``state_dict`` snapshots by reserved slot.
* ``ExperienceClassMap`` stores the semantic bookkeeping needed to route
  classes back to the skill that mastered them.

An experience is only a container.  A skill can solve multiple classes, and
one experience can therefore map to multiple ``(skill, classes)`` groups.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


class SkillMemory:
    """Bounded, index-addressed storage for independent skill snapshots."""

    def __init__(self, max_skills: int = 200):
        if max_skills < 1:
            raise ValueError("max_skills must be positive")
        self.max_skills = max_skills
        self._states: dict[int, dict[str, Tensor]] = {}
        self._metadata: dict[int, dict] = {}

    def allocate(self) -> int:
        """Reserve and return the lowest free slot index.

        Raises `RuntimeError` once every slot up to `max_skills` is in use.
        """
        for slot in range(self.max_skills):
            if slot not in self._states:
                return slot
        raise RuntimeError(f"skill memory is at capacity ({self.max_skills})")

    def store(
        self,
        slot: int,
        state_dict: Mapping[str, Tensor],
        metadata: dict | None = None,
    ) -> None:
        """Overwrite `slot` with a detached CPU copy of `state_dict`.

        Copying prevents later changes to the live model from leaking into the
        stored snapshot. The snapshot can intentionally be replaced when
        `reuse_is_mutable=True` updates a canonical skill.
        """
        if not 0 <= slot < self.max_skills:
            raise ValueError(f"invalid skill slot {slot}")
        self._states[slot] = {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        }
        self._metadata[slot] = dict(metadata or {})

    def state(self, slot: int) -> dict[str, Tensor]:
        """Return the stored `state_dict` for `slot` (raises `KeyError` if empty)."""
        return self._states[slot]

    def metadata(self, slot: int) -> dict:
        """Return a copy of the metadata dict passed to `store` for `slot`."""
        return dict(self._metadata.get(slot, {}))

    def slots(self) -> set[int]:
        """Return the set of currently occupied slot indices."""
        return set(self._states)

    def __len__(self) -> int:
        return len(self._states)


@dataclass(frozen=True)
class ClassRecord:
    """Outcome for one class in one logical training experience."""

    experience_index: int
    class_id: int
    decision: str
    skill: int
    new_score: float = 0.0
    old_accuracy: float = 0.0
    new_accuracy: float = 0.0


class ExperienceClassMap:
    """Map experiences to class/skill assignments and classes to skills.

    Invariant: a class has one canonical skill for the lifetime of the
    strategy.  Seeing that class again therefore reuses that skill instead
    of allowing the probe heuristic to silently assign it to another slot.
    """

    def __init__(self):
        self._by_experience: dict[int, dict[int, ClassRecord]] = {}
        self._class_to_skill: dict[int, int] = {}
        self._by_skill: dict[int, set[int]] = {}

    def record(self, record: ClassRecord) -> None:
        """Register one class's outcome for one experience.

        Raises `RuntimeError` if this class was already mapped to a
        *different* skill, either globally (the class-to-skill invariant
        this class exists to enforce) or within the same experience.
        """
        previous = self._class_to_skill.get(record.class_id)
        if previous is not None and previous != record.skill:
            raise RuntimeError(
                f"class {record.class_id} is already mapped to skill {previous}; "
                f"cannot remap it to skill {record.skill}"
            )

        existing = self._by_experience.setdefault(record.experience_index, {}).get(
            record.class_id
        )
        if existing is not None and existing.skill != record.skill:
            raise RuntimeError(
                f"class {record.class_id} has conflicting assignments in "
                f"experience {record.experience_index}"
            )

        self._by_experience[record.experience_index][record.class_id] = record
        self._class_to_skill[record.class_id] = record.skill
        self._by_skill.setdefault(record.skill, set()).add(record.class_id)

    def classes_for_experience(self, experience_index: int) -> dict[int, ClassRecord]:
        """Return `{class_id: ClassRecord}` for every class seen in this experience."""
        return dict(self._by_experience.get(experience_index, {}))

    def skill_for_class(self, experience_index: int, class_id: int) -> int | None:
        """Return the skill assigned to `class_id` within `experience_index`.

        `None` if that (experience, class) pair was never recorded. Unlike
        `find_skill_for_class_anywhere`, this does not search other
        experiences.
        """
        record = self._by_experience.get(experience_index, {}).get(class_id)
        return record.skill if record else None

    def find_skill_for_class_anywhere(self, class_id: int) -> int | None:
        """Return the canonical skill for `class_id`, searching all experiences."""
        return self._class_to_skill.get(class_id)

    def classes_for_skill(self, skill: int) -> set[int]:
        """Return every class currently mapped to `skill`."""
        return set(self._by_skill.get(skill, set()))

    def skills_for_experience(
        self, experience_index: int
    ) -> list[tuple[int, set[int]]]:
        """Return ``[(skill, {classes}), ...]`` for one experience."""
        grouped: dict[int, set[int]] = {}
        for class_id, record in self._by_experience.get(experience_index, {}).items():
            grouped.setdefault(record.skill, set()).add(class_id)
        return list(grouped.items())

    def class_skill_for_experience(self, experience_index: int) -> dict[int, int]:
        """Return the complete ``class -> skill`` mapping for one experience."""
        return {
            class_id: record.skill
            for class_id, record in self._by_experience.get(
                experience_index, {}
            ).items()
        }
