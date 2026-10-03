# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Anonymous routing from each skill's own raw response -- no evaluator, no label.

Not part of `strategy.eval()`: this answers "if I had no evaluator at
all, could each skill's own raw output alone pick the right skill?",
which is a deliberately expensive question to ask over a whole stream
(one forward pass per stored skill, per batch). Every public function
here therefore requires ``diagnose=True`` (see
:mod:`skill_memory.diagnostics._gate`).
"""

from __future__ import annotations

import torch

from ..evaluation.routing import RoutingResult, _normalize_routing_scores
from ._gate import require_diagnose


def _skill_own_score(logits: torch.Tensor, owned_classes) -> torch.Tensor:
    r"""Score one skill's own raw logits at its own owned class columns.

    Owning zero classes scores as all-zero (handled by
    ``_normalize_routing_scores``'s uniform-fallback branch). A skill whose
    raw output has only one column (a genuine single-class head) cannot use
    a softmax at all -- softmax over one column is identically :math:`1` for
    every sample and carries no information -- so that case uses the
    column's own sigmoid instead:

    .. math::

        \text{score} =
        \begin{cases}
            0 & \text{no owned classes} \\[4pt]
            \sigma(z_0) & \text{single-column head } (C = 1) \\[4pt]
            \sum_{c \in \text{owned}} \operatorname{softmax}(z)_c
                & \text{otherwise}
        \end{cases}
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    owned = sorted(int(c) for c in owned_classes)
    if not owned:
        return torch.zeros(logits.shape[0])

    width = logits.shape[1]
    out_of_range = [c for c in owned if not 0 <= c < width]
    if out_of_range:
        raise RuntimeError(
            f"skill owns classes {out_of_range} but its own raw output is "
            f"only {width}-wide; class bookkeeping and the model have "
            "drifted apart"
        )

    if width == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=1)[:, owned].sum(dim=1)


def find_best_routing_skill(
    logits_by_skill,
    states,
    classes_by_skill,
    *,
    temperature: float = 1.0,
    diagnose: bool,
) -> RoutingResult:
    """Route each sample to one skill using only each skill's own response.

    `logits_by_skill[i]` is skill ``i``'s own raw forward-pass output for
    the batch (shape ``[batch, classes_i]``); `classes_by_skill[i]` is the
    set of global class ids skill ``i`` owns. No label, task id, or
    experience id is required or used -- see :func:`_skill_own_score` for
    the per-skill scoring rule and
    :func:`skill_memory.evaluation.routing.select_skill_from_scores` for how
    the resulting per-skill scores become a probability and a selection.
    `states` (each skill's stored ``state_dict``) is accepted for interface
    symmetry with the rest of this module's state-aware helpers; it is not
    required for this computation.

    Requires ``diagnose=True`` (see
    `skill_memory.diagnostics.require_diagnose`): this is one forward pass
    per stored skill, and is never something `strategy.eval()` should pay
    for automatically.
    """
    require_diagnose(diagnose, "find_best_routing_skill")
    del states
    scores = torch.stack(
        [
            _skill_own_score(logits, owned)
            for logits, owned in zip(logits_by_skill, classes_by_skill, strict=True)
        ],
        dim=0,
    )
    probabilities = _normalize_routing_scores(scores, temperature)
    skill_indices = probabilities.argmax(dim=0)
    if probabilities.shape[0] == 1:
        best_probability = probabilities[0]
        second_probability = torch.zeros_like(best_probability)
    else:
        top2 = torch.topk(probabilities, k=2, dim=0).values
        best_probability, second_probability = top2[0], top2[1]
    return RoutingResult(
        skill_indices=skill_indices,
        probabilities=probabilities,
        best_probability=best_probability,
        second_probability=second_probability,
        confidence_gap=best_probability - second_probability,
    )


def route_probe_logits(
    logits_by_skill,
    states,
    classes_by_skill,
    *,
    temperature: float = 1.0,
    diagnose: bool,
) -> torch.Tensor:
    """Return just the chosen skill index per sample.

    See :func:`find_best_routing_skill` (including the required
    ``diagnose=True``). Kept for callers that only need the routing
    decision, not the full
    :class:`~skill_memory.evaluation.routing.RoutingResult`.
    """
    return find_best_routing_skill(
        logits_by_skill,
        states,
        classes_by_skill,
        temperature=temperature,
        diagnose=diagnose,
    ).skill_indices
