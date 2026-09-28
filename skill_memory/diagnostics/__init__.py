# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Every diagnostic in this package, in one place, gated by ``diagnose=True``.

**Everything here is deliberately not part of `strategy.eval()`.**
Production evaluation goes through the stored Skill Memory CL evaluator,
which never uses a true label for routing. This package exists to answer a
different kind of question -- *could Skill Memory's own stored skills
reproduce that accuracy?*, *where did the time in this run actually go?*,
*does each skill's own classifier still have a column for every class it
owns?* -- using tools (an oracle label, a per-skill forward-pass sweep, a
raw-weight inspection) that would be a leak, or an unbudgeted cost, if
they ran automatically.

**The rule, enforced, not just documented:** every public function in
this package takes ``diagnose`` as a required keyword argument with no
default. There are two variants of what it checks, matched to what each
function actually risks:

- Functions that could use a true label, or that always cost real
  forward-pass time regardless of how anything was configured
  (:func:`find_best_routing_skill`, :func:`route_probe_logits`,
  :func:`evaluate_skill_memory`, :func:`evaluate_class_oracle`,
  :func:`diagnose_evaluator_probe`, :func:`routing_rank_diagnostics`,
  :func:`class_index_alignment_report`) require the caller to pass
  ``diagnose=True`` at that exact call site, regardless of how the
  strategy involved was built. A leak here can never be explained away
  as "the strategy happened to be built the wrong way" -- the call
  itself has to say so.
- :func:`timing_report` and :func:`reset_timing` instead check
  ``strategy.diagnose`` -- since a `SkillMemoryStrategy` built with
  ``diagnose=False`` (the default) never records timing at all
  (`TimingAccumulator.track` is then a true no-op, not just an unread
  one), there is no separate leak to gate against; the check exists so
  the failure mode is a clear error instead of a silently empty report.

Auditing whether any diagnostic-only computation or ground truth could
have reached a production number is therefore one grep for
``diagnose=True`` across the codebase, not a review of every module that
might have forgotten to check a flag.
"""

from __future__ import annotations

from .alignment import class_index_alignment_report, routing_rank_diagnostics
from .evaluation import (
    evaluate_class_oracle,
    evaluate_skill_memory,
)
from .leakage import (
    assert_no_split_overlap,
    audit_split_overlap,
    audit_strategy_leakage,
)
from .old_scores import measure_old_class_scores
from .routing import find_best_routing_skill, route_probe_logits
from .timing import TimingAccumulator, reset_timing, timing_report

__all__ = [
    "TimingAccumulator",
    "assert_no_split_overlap",
    "audit_split_overlap",
    "audit_strategy_leakage",
    "measure_old_class_scores",
    "class_index_alignment_report",
    "diagnose_evaluator_probe",
    "evaluate_class_oracle",
    "evaluate_skill_memory",
    "find_best_routing_skill",
    "reset_timing",
    "route_probe_logits",
    "routing_rank_diagnostics",
    "timing_report",
]
