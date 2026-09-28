# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Skill Memory: continual learning with one verifier skill per class group.

Typical use::

    from skill_memory import SkillMemoryStrategy

    strategy = SkillMemoryStrategy(model=..., optimizer=..., criterion=...,
                                   update_mode="replay")
    for experience in benchmark.train_stream:
        strategy.train(experience)
    results = strategy.eval(benchmark.test_stream)

Optional, opt-in diagnostics live in :mod:`skill_memory.diagnostics`.
"""

from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .evaluation.cl_evaluator import CLEvaluationPlugin
from .strategy import SkillMemoryStrategy

__all__ = [
    "CLEvaluationPlugin",
    "ClassRecord",
    "ExperienceClassMap",
    "SkillMemory",
    "SkillMemoryPlugin",
    "SkillMemoryStrategy",
]
