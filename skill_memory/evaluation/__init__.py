# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Machine-learning evaluation and anonymous routing components."""

from .reverse_engineering import (
    CandidateParameters as CandidateParameters,
)
from .reverse_engineering import (
    NormalMLReverseEngineer as NormalMLReverseEngineer,
)
from .routing import RoutingResult as RoutingResult
from .routing import score_skill_compatibility as score_skill_compatibility
from .routing import select_skill_from_scores as select_skill_from_scores

# `cl_evaluator` (CLEvaluationPlugin) and `memory` (EvaluationMemoryPlugin) are
# deliberately NOT re-exported here (they are exported from the top-level
# `skill_memory` package, and always importable directly, e.g.
# `skill_memory.evaluation.cl_evaluator`). `memory` subclasses
# `skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin`, and `cl/`'s own
# modules trigger this package's `__init__` before `cl` itself has finished
# loading. Importing them here would import `cl` back before it is ready - a
# real circular import, not just a lint warning. See `skill_memory/__init__.py`,
# where `cl` is already fully loaded by the time they are imported.
