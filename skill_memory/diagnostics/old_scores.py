# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""On-demand old-class probability scores for a stored skill.

During training the safety check judges a skill on old-class *accuracy*
alone, so the mean true-class probability ("score") of its old classes is
never computed on the hot path. It is only interesting for inspection, so it
lives here, behind ``diagnose=True``.

The probes are drawn exactly as the safety stage draws them (same pooled
experiences, batch shape and per-class seed), so for a seeded strategy the
numbers are what the decision stage would have measured.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..cl.decision import _first_experience_with_class, _probe_class_across
from ..utils.probing import evaluate_state
from ._gate import require_diagnose


def measure_old_class_scores(
    strategy,
    skill: int,
    *,
    diagnose: bool,
) -> dict[str, Any]:
    """Measure loss/score/accuracy of stored `skill` on each of its old classes.

    Returns ``{"per_class": {class_id: {"loss", "score", "accuracy"}},
    "old_score": min score, "old_accuracy": min accuracy}``. Classes whose
    training data is no longer among the seen experiences are skipped; if none
    can be probed, ``old_score`` and ``old_accuracy`` are ``None``.
    """
    require_diagnose(diagnose, "measure_old_class_scores")

    plugin = strategy.skill_memory_plugin
    state = plugin.memory.state(skill)
    old_classes = sorted(plugin.class_map.classes_for_skill(skill))
    seen = plugin._seen_experiences
    seed = plugin.probe_seed
    loss_fn = nn.CrossEntropyLoss()

    per_class: dict[int, dict[str, float]] = {}
    for old_class in old_classes:
        old_experience = _first_experience_with_class(seen, old_class)
        if old_experience is None:
            continue
        try:
            x, y = _probe_class_across(
                seen,
                old_class,
                plugin.probe_batch_size,
                plugin.probe_batches,
                None if seed is None else seed + 100003 + old_class,
            )
        except RuntimeError:
            continue
        loss, score, accuracy = evaluate_state(
            strategy.model,
            state,
            x,
            y,
            loss_fn,
            old_experience,
            seed=seed,
        )
        per_class[old_class] = {"loss": loss, "score": score, "accuracy": accuracy}

    if not per_class:
        return {"per_class": {}, "old_score": None, "old_accuracy": None}
    return {
        "per_class": per_class,
        "old_score": min(m["score"] for m in per_class.values()),
        "old_accuracy": min(m["accuracy"] for m in per_class.values()),
    }
