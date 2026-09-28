# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Skill Memory public API.

The package exposes the single native Skill Memory CL training/evaluation
path. Optional diagnostics remain under skill_memory.diagnostics.
"""

from .cl.decision import find_best_skill
from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .evaluation.behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    compare_binary_behavior,
    identify_binary_behavior,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y,
    reverse_engineer_y_from_weights,
)
from .evaluation.cl_evaluator import CLEvaluationPlugin
from .evaluation.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin
from .evaluation.memory import EvaluationMemory, EvaluationMemoryPlugin
from .evaluation.reverse_engineering import CandidateParameters, NormalMLReverseEngineer
from .evaluation.routing import RoutingResult
from .strategy import SkillMemoryStrategy

__all__ = [
    "BehaviorFingerprintCache",
    "CandidateParameters",
    "ClassRecord",
    "ClassBehaviorRecord",
    "CLEvaluationPlugin",
    "EvaluationMemory",
    "EvaluationMemoryPlugin",
    "ExperienceClassMap",
    "RoutingResult",
    "SkillMemory",
    "SkillMemoryPlugin",
    "PersistentFingerprintSkillMemoryPlugin",
    "NormalMLReverseEngineer",
    "compare_binary_behavior",
    "identify_binary_behavior",
    "reverse_engineer_scores_from_weights",
    "reverse_engineer_y",
    "reverse_engineer_y_from_weights",
    "find_best_skill",
    "SkillMemoryStrategy",
]
