# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Direct Skill Memory evaluation: could the stored skills reproduce the accuracy?

Not part of `strategy.eval()`. These functions are explicit diagnostics for
inspecting stored Skill Memory states and anonymous routing. Everything here answers
a different, diagnostic question instead, and one of the two ways to
answer it (``routing="oracle"``) uses each sample's *true label* to pick
a skill, which would be a real information leak if it ever reached a
production metric. Every public function here therefore requires
``diagnose=True`` (see :mod:`skill_memory.diagnostics._gate`) -- a hard,
per-call acknowledgement that does not depend on how the strategy itself
was configured.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.probing import expand_skill_logits, predict_logits
from ._gate import require_diagnose
from .routing import find_best_routing_skill


@torch.no_grad()
def evaluate_skill_memory(
    model: nn.Module,
    skill_memory_plugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    routing: str = "oracle",
    batch_size: int,
    device,
    diagnose: bool,
) -> dict[int, dict[str, float]]:
    """Evaluate Skill Memory's own stored skills directly, per class.

    Unlike
    :func:`~skill_memory.evaluation.cl_evaluator.evaluate_model_by_class`
    (which evaluates one independent evaluator model), this applies each
    *routed* skill's own frozen weights before scoring, so it measures
    whether Skill Memory's stored skills -- not an auxiliary learner --
    would reproduce the reported accuracy on their own. This is
    intentionally expensive (one forward pass per sample, or per sample per
    candidate skill) and is never called from the normal
    ``strategy.eval()`` lifecycle.

    ``routing="oracle"`` looks up each sample's canonical skill directly
    from its true label via
    ``skill_memory_plugin.class_map.find_skill_for_class_anywhere`` -- an
    upper bound on accuracy that presupposes knowing the label. Requires
    ``diagnose=True`` for exactly this reason: a true label must never
    reach a routing decision outside this explicitly-opted-into check.
    ``routing="probe"`` instead routes anonymously with
    :func:`~skill_memory.diagnostics.routing.find_best_routing_skill`,
    using only every stored skill's own raw response to the very same
    input -- no label used for routing, but still gated the same way,
    since the per-sample-per-skill cost is the same either way.

    Returns ``{class_id: {"accuracy": ..., "loss": ...}}`` for every class
    with at least one scored sample.
    """
    require_diagnose(diagnose, "evaluate_skill_memory")
    if routing not in ("oracle", "probe"):
        raise ValueError("routing must be one of 'oracle' or 'probe'")
    if up_to_index < 0:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    memory = skill_memory_plugin.memory
    class_map = skill_memory_plugin.class_map
    slot_ids = sorted(memory.slots())
    if not slot_ids:
        return {}
    owned_by_slot = [sorted(class_map.classes_for_skill(slot)) for slot in slot_ids]

    class_loss: dict[int, float] = {}
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    for experience_index in range(up_to_index + 1):
        experience = test_stream[experience_index]
        loader = DataLoader(experience.dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            if routing == "oracle":
                chosen_skill_ids: list[int | None] = [
                    class_map.find_skill_for_class_anywhere(int(label))
                    for label in y.tolist()
                ]
            else:
                per_skill_raw = [
                    predict_logits(model, memory.state(slot), x) for slot in slot_ids
                ]
                per_skill_states = [memory.state(slot) for slot in slot_ids]
                result = find_best_routing_skill(
                    per_skill_raw,
                    per_skill_states,
                    owned_by_slot,
                    diagnose=diagnose,
                )
                chosen_skill_ids = [
                    slot_ids[index] for index in result.skill_indices.tolist()
                ]

            for sample_index, skill_id in enumerate(chosen_skill_ids):
                if skill_id is None:
                    continue
                slot_index = slot_ids.index(skill_id)
                owned = owned_by_slot[slot_index]
                sample_x = x[sample_index : sample_index + 1]
                label = y[sample_index : sample_index + 1]

                raw_logits = predict_logits(model, memory.state(skill_id), sample_x)
                expanded = expand_skill_logits(
                    raw_logits.to(device),
                    memory.state(skill_id),
                    owned,
                    num_classes,
                )
                loss = nn.functional.cross_entropy(expanded, label)
                correct = int(expanded.argmax(dim=1).item() == int(label.item()))

                class_id = int(label.item())
                class_loss[class_id] = class_loss.get(class_id, 0.0) + float(
                    loss.item()
                )
                class_correct[class_id] = class_correct.get(class_id, 0) + correct
                class_total[class_id] = class_total.get(class_id, 0) + 1

    return {
        class_id: {
            "loss": class_loss[class_id] / class_total[class_id],
            "accuracy": class_correct[class_id] / class_total[class_id],
        }
        for class_id in sorted(class_total)
    }


def evaluate_class_oracle(
    model: nn.Module,
    skill_memory_plugin,
    test_stream,
    up_to_index: int,
    *,
    num_classes: int,
    batch_size: int,
    device,
    diagnose: bool,
) -> dict[int, dict[str, float]]:
    """Convenience wrapper: :func:`evaluate_skill_memory` with oracle routing.

    Requires ``diagnose=True``, propagated straight through to
    :func:`evaluate_skill_memory` -- see its docstring for why oracle
    routing in particular must never run unacknowledged.
    """
    require_diagnose(diagnose, "evaluate_class_oracle")
    return evaluate_skill_memory(
        model,
        skill_memory_plugin,
        test_stream,
        up_to_index,
        num_classes=num_classes,
        routing="oracle",
        batch_size=batch_size,
        device=device,
        diagnose=diagnose,
    )
