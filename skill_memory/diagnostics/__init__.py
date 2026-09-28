# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Opt-in diagnostics -- never part of ``strategy.eval()``.

Production evaluation (``CLEvaluationPlugin``) never uses a true label. The
tools here answer *other* questions, and some of them would leak ground
truth or cost unbudgeted time if they ran implicitly:

=============================  ==============================================
function                       question it answers
=============================  ==============================================
``evaluate_class_oracle``      upper bound: route every sample with its TRUE
                               label (a leak by construction)
``evaluate_skill_memory``      accuracy of the stored skills with oracle or
                               anonymous ``"probe"`` routing
``replay_provenance_report``   which historical data did each training call
                               consume? (checks the replay invariants)
``timing_report``              where did the run time go?
``audit_split_overlap`` ...    exact-content train/test overlap (diagnostic,
                               not a proof of no leakage)
=============================  ==============================================

**The gate.** Every function that could use a label, or that always costs real
forward passes, takes ``diagnose`` as a required keyword with no default and
raises unless it is ``True`` -- so auditing whether a diagnostic could have
reached a production number is one ``grep diagnose=True``. ``timing_report``
and ``reset_timing`` instead check ``strategy.diagnose``, because a strategy
built with ``diagnose=False`` records no timing at all.
"""

from __future__ import annotations

from .evaluation import (
    evaluate_class_oracle,
    evaluate_skill_memory,
)
from .leakage import (
    assert_no_split_overlap,
    audit_split_overlap,
    audit_strategy_leakage,
)
from .replay import replay_provenance_report
from .routing import RoutingResult, find_best_routing_skill, route_probe_logits
from .timing import TimingAccumulator, reset_timing, timing_report

__all__ = [
    "RoutingResult",
    "TimingAccumulator",
    "assert_no_split_overlap",
    "audit_split_overlap",
    "audit_strategy_leakage",
    "evaluate_class_oracle",
    "evaluate_skill_memory",
    "find_best_routing_skill",
    "replay_provenance_report",
    "reset_timing",
    "route_probe_logits",
    "timing_report",
]
