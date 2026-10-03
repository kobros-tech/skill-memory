# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Replay provenance audit: *prove* what historical data was trained on.

Every training call made by the plugin appends one provenance record to
``plugin.training_log`` (counts per data source, see
:class:`skill_memory.cl.training.TrainingProvenance`).  This module turns the
log into a report and **checks the replay invariants** of
:mod:`skill_memory.cl.replay`:

.. math::

    |H_c| = 0 \quad(\text{new\_class}),\qquad
    |H_c| \le K \quad(\text{small\_replay}),\qquad
    |H_c| \le |R_c| \quad(\text{replay}),

where :math:`H_c` is the historical (retained + offline-pool) data replayed
for class :math:`c`.  It also separates the optimiser work spent on
*class training* from the work spent on *refreshing existing skills*, which is
the quantity that explains most run-time differences between configurations.
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

    ``mode`` / ``replay_per_class`` / ``refresh_existing_skills``
        The configuration in force.
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
    policy = plugin.replay_policy
    log = list(plugin.training_log)

    violations: list[str] = []
    for entry in log:
        label = (
            f"experience {entry['experience_index']} {entry['kind']} "
            f"{entry['target_classes']}"
        )
        historical = entry["historical_total"]
        if not policy.uses_history and historical:
            violations.append(
                f"{label}: new_class used {historical} historical examples"
            )
        cap = policy.historical_limit
        if cap is not None:
            for source in ("retained", "offline_pool"):
                for class_id, count in entry[source].items():
                    if count > cap:
                        violations.append(
                            f"{label}: class {class_id} replayed {count} "
                            f"> cap {cap} from {source}"
                        )
        if entry["offline_pool"] and not plugin.allow_offline_negative_pool:
            violations.append(f"{label}: offline pool used without opt-in")

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
        "replay_per_class": policy.per_class,
        "refresh_existing_skills": plugin.refresh_policy.enabled,
        "calls": log,
        "class_training": aggregate("class"),
        "refresh": aggregate("refresh"),
        "violations": violations,
    }
