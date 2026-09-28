# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Replay provenance audit: *prove* what historical data was trained on.

Every training call made by the plugin appends one provenance record to
``plugin.training_log`` (counts per data source, see
:class:`skill_memory.cl.training.TrainingProvenance`).  This module turns the
log into a report and **checks the replay invariants** of
:mod:`skill_memory.cl.replay`:

.. math::

    |H_c| = 0 \ (\texttt{new\_class}),\qquad
    |H_c| \le K \ (K=\texttt{replay\_samples\_per\_class}),\qquad
    |H_c| \le m \ (m=\texttt{memory\_per\_class}),

and that refresh calls only occur under ``update_mode="refresh"``; here
:math:`H_c` is the retained data replayed for old class :math:`c`.  It also
separates the optimiser work spent on *class training* from the work spent on
*refreshing existing skills*, which explains most run-time differences.
"""

from __future__ import annotations

from typing import Any

from ._gate import require_diagnose


def replay_provenance_report(strategy, *, diagnose: bool) -> dict[str, Any]:
    """Summarise and verify the replay provenance of a finished run.

    Parameters
    ----------
    strategy:
        A :class:`~skill_memory.strategy.SkillMemoryStrategy` (or anything
        with a ``skill_memory_plugin`` attribute) that has been trained.
    diagnose:
        Must be ``True`` (required keyword, no default).

    Returns
    -------
    dict with keys

    ``mode`` / ``replay_samples_per_class``
        The complete update policy configuration in force.
    ``calls``
        The raw per-call provenance records.
    ``class_training`` / ``refresh``
        ``{"calls", "optimizer_steps", "historical_examples",
        "current_examples"}`` aggregates for each training kind.
    ``violations``
        Human-readable strings; **empty iff every invariant holds**.
    """
    require_diagnose(diagnose, "replay_provenance_report")
    plugin = getattr(strategy, "skill_memory_plugin", strategy)
    policy = plugin.update_policy
    log = list(plugin.training_log)

    violations: list[str] = []
    for entry in log:
        label = (
            f"experience {entry['experience_index']} {entry['kind']} "
            f"{entry['target_classes']}"
        )
        if not policy.uses_history and entry["historical_total"]:
            violations.append(
                f"{label}: new_class used {entry['historical_total']} "
                "historical examples"
            )
        if entry["kind"] == "refresh" and not policy.refreshes_existing_skills:
            violations.append(f"{label}: refresh call outside update_mode='refresh'")
        for class_id, count in entry["retained"].items():
            cap = policy.per_class or plugin.memory_per_class
            if count > cap:
                violations.append(
                    f"{label}: class {class_id} replayed {count} > cap {cap}"
                )

    def aggregate(kind: str) -> dict[str, int]:
        rows = [entry for entry in log if entry["kind"] == kind]
        return {
            "calls": len(rows),
            "optimizer_steps": sum(entry["optimizer_steps"] for entry in rows),
            "historical_examples": sum(entry["historical_total"] for entry in rows),
            "current_examples": sum(sum(entry["current"].values()) for entry in rows),
        }

    return {
        "mode": policy.mode,
        "replay_samples_per_class": policy.per_class,
        "calls": log,
        "class_training": aggregate("class"),
        "refresh": aggregate("refresh"),
        "violations": violations,
    }
