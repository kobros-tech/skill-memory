# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Per-class Skill Memory decisions."""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.utils.data import ConcatDataset

from ..utils.probing import (
    FunctionalStateCache,
    _sample_batches,
    class_subset,
    evaluate_state,
    evaluate_state_accuracy,
    evaluate_states_batch,
    experience_has_class,
    incremental_out_features,
    probe_class,
)

#: Default number of top-ranked candidate skills whose old classes are
#: verified. ``None`` verifies every stored skill (exact, but scales with
#: ``skills x old classes`` per new class).
DEFAULT_MAX_SAFETY_CANDIDATES = 5


class DecisionProbeCache:
    """Per-experience caches that make repeated probing cheaper without
    changing any probing result.

    * ``states`` -- expanded functional skill states
      (see :class:`~skill_memory.utils.probing.FunctionalStateCache`).
    * old-class probe batches -- for a *seeded* run the probe drawn for an
      old class is a pure function of ``(class, the experiences that contain
      that class, batch shape, seed)``, so it is drawn once and reused by
      every later new-class decision -- across experiences too, until a
      later experience that contains the same class widens its pool --
      instead of being re-decoded from the dataset each time.

    Unseeded runs (``probe_seed=None``) are deliberately never cached: their
    probes are random by design.
    """

    def __init__(self) -> None:
        self.states = FunctionalStateCache()
        self._old_probes: dict[tuple, tuple | None] = {}

    def clear(self) -> None:
        """Drop everything (skill states AND old-class probes)."""
        self.states.clear()
        self._old_probes.clear()

    def end_of_experience(self) -> None:
        """Release skill states; keep old-class probes.

        Expanded skill states are memory-heavy (one parameter dict per skill)
        and skills are re-stored as training proceeds, so they are dropped
        every experience. Old-class probes are small ``(x, y)`` batches whose
        keys already encode the exact pool they were drawn from, so they stay
        valid and are what makes later experiences cheap.
        """
        self.states.clear()

    def invalidate_skill(self, slot: int) -> None:
        self.states.invalidate(slot)

    def old_probe(self, key: tuple, build):
        if key in self._old_probes:
            return self._old_probes[key]
        value = build()
        self._old_probes[key] = value
        return value


def _probe_class_across(experiences, target_class, batch_size, n_batches, seed=None):
    """Pool only ``target_class`` samples from every matching experience."""
    subsets = []
    for experience in experiences:
        try:
            subsets.append(class_subset(experience, target_class))
        except RuntimeError:
            continue
    if not subsets:
        raise RuntimeError(f"class {target_class} not found in seen experiences")
    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    return _sample_batches(dataset, batch_size, n_batches, seed)


def _first_experience_with_class(experiences, target_class):
    for experience in experiences:
        if experience_has_class(experience, target_class):
            return experience
    return None


def _pool_key(experiences, target_class) -> tuple:
    """Identify exactly which experiences an old-class probe pools from.

    ``_probe_class_across`` only ever reads the experiences that contain
    ``target_class``, so a later experience that lacks the class cannot change
    its probe. If the datasets cannot be inspected the key conservatively
    falls back to the whole pool (never a wrong hit, at worst a miss).
    """
    try:
        return tuple(
            id(experience)
            for experience in experiences
            if experience_has_class(experience, target_class)
        )
    except (AttributeError, TypeError):
        return tuple(id(experience) for experience in experiences)


def _strongest_candidates(results, key, floor):
    ranked = sorted(results, key=lambda result: result[key], reverse=True)
    if not ranked:
        return set()
    if len(ranked) == 1:
        return {ranked[0]["skill"]} if ranked[0][key] > floor else set()

    values = [result[key] for result in ranked]
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    split = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[split] <= 0:
        return set()

    return {result["skill"] for result in ranked[: split + 1] if result[key] > floor}


def find_best_skill(
    imagination_results: list[dict[str, Any]],
    forgetting_margin: float,
    score_floor: float | None = 0.9,
):
    """Select a new-class skill only when all its old classes remain safe."""
    if not imagination_results:
        return None

    safe_results = [
        result
        for result in imagination_results
        if result["old_accuracy"] > result["chance"] + forgetting_margin
    ]
    if not safe_results:
        return None

    floor_score = (
        max(result["chance"] for result in safe_results)
        if score_floor is None
        else score_floor
    )
    floor_accuracy = max(result["chance"] for result in safe_results)

    score_candidates = _strongest_candidates(safe_results, "new_score", floor_score)
    accuracy_candidates = _strongest_candidates(
        safe_results, "new_accuracy", floor_accuracy
    )
    intersection = score_candidates & accuracy_candidates
    if not intersection:
        return None

    candidates = [result for result in safe_results if result["skill"] in intersection]
    return max(candidates, key=lambda r: (r["new_score"], r["new_accuracy"]))


def score_class_against_skills(
    strategy,
    experience,
    target_class: int,
    memory,
    class_map,
    probe_batch_size: int,
    probe_batches: int,
    probe_seed: int | None,
    seen_experiences: list,
    max_safety_candidates: int | None = DEFAULT_MAX_SAFETY_CANDIDATES,
    forgetting_margin: float | None = None,
    probe_cache: DecisionProbeCache | None = None,
    batch_stage1: bool = False,
    stage1_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """Probe skills against a new class, then verify only top candidates.

    The first stage measures the new-class compatibility of every stored skill.
    The second stage measures the *real* old-class accuracy for the
    ``max_safety_candidates`` strongest candidates (default 5; ``None``
    verifies every skill exactly).  A finite cap is an explicit performance
    approximation: it can miss a reusable lower-ranked skill.

    Safety is judged on old-class *accuracy* alone, so the second stage uses
    the accuracy-only :func:`~skill_memory.utils.probing.evaluate_state_accuracy`.
    The old-class probability *score* is deliberately not computed here; it
    was metadata only. Use
    :func:`skill_memory.diagnostics.measure_old_class_scores` (requires
    ``diagnose=True``) to obtain it for a stored skill on demand.

    When ``forgetting_margin`` is given, a candidate's old classes are
    evaluated one at a time and evaluation stops at the first class whose
    accuracy is ``<= chance + forgetting_margin``. Such a skill is already
    unsafe under :func:`find_best_skill` (which requires the *worst* old class
    to clear that threshold), so the REUSE/SCRATCH outcome is identical; only
    the unsafe skill's ``old_metrics`` list is partial
    (``result["safety_complete"]`` is ``False``).  ``forgetting_margin=None``
    evaluates every old class.

    Both stages apply each candidate skill's weights functionally (see
    ``skill_memory.utils.probing.evaluate_state``) rather than mutating a
    model in place -- so, unlike the original loop this replaced, none of
    this needs its own copy of ``strategy.model``: ``strategy.model`` is
    read from directly and is never modified by probing a class.

    ``batch_stage1=True`` runs stage 1 (which, unlike stage 2, always
    scores *every* stored skill and has no bound) through
    :func:`~skill_memory.utils.probing.evaluate_states_batch` instead of one
    ``evaluate_state`` call per skill. It computes the exact same thing --
    same candidate skills, same expanded skill states, same forward-pass
    math -- just batched via ``torch.vmap`` where skills' expanded states
    happen to share a shape (see ``evaluate_states_batch`` for the grouping
    rule and why it is on by opt-in: batching helps or hurts depending on
    the model and device, so it is not a safe default). ``stage1_chunk_size``
    bounds memory when batching; see the same docstring.
    """
    new_x, new_y = probe_class(
        experience, target_class, probe_batch_size, probe_batches, probe_seed
    )
    probe_model = strategy.model

    # Stage 1: new-class imagination for every skill.
    slots_with_classes = [
        (slot, sorted(class_map.classes_for_skill(slot)))
        for slot in sorted(memory.slots())
    ]
    slots_with_classes = [
        (slot, mastered) for slot, mastered in slots_with_classes if mastered
    ]

    state_cache_for_stage1 = None if probe_cache is None else probe_cache.states
    if batch_stage1:
        stage1_metrics = evaluate_states_batch(
            probe_model,
            [(slot, memory.state(slot)) for slot, _ in slots_with_classes],
            new_x,
            new_y,
            experience,
            seed=probe_seed,
            cache=state_cache_for_stage1,
            chunk_size=stage1_chunk_size,
        )
    else:
        stage1_metrics = {
            slot: evaluate_state(
                probe_model,
                memory.state(slot),
                new_x,
                new_y,
                nn.functional.cross_entropy,
                experience,
                seed=probe_seed,
                cache=state_cache_for_stage1,
                slot=slot,
            )
            for slot, _ in slots_with_classes
        }

    candidates = []
    for slot, mastered_classes in slots_with_classes:
        new_loss, new_score, new_accuracy = stage1_metrics[slot]
        out_features = incremental_out_features(strategy.model, memory.state(slot))
        chance = 1.0 / out_features if out_features else 0.0
        candidates.append(
            {
                "skill": slot,
                "class": target_class,
                "old_classes": mastered_classes,
                "new_loss": new_loss,
                "new_score": new_score,
                "new_accuracy": new_accuracy,
                "chance": chance,
            }
        )

    # Stage 2: verify the strongest candidates on their real old classes.
    candidates.sort(key=lambda r: (r["new_score"], r["new_accuracy"]), reverse=True)
    if max_safety_candidates is None:
        safety_candidates = candidates
    else:
        safety_candidates = candidates[:max_safety_candidates]

    local_old_probes: dict[int, tuple | None] = {}
    state_cache = None if probe_cache is None else probe_cache.states

    def build_old_probe(old_class: int) -> tuple | None:
        old_experience = _first_experience_with_class(seen_experiences, old_class)
        if old_experience is None:
            return None
        try:
            return (
                old_experience,
                *_probe_class_across(
                    seen_experiences,
                    old_class,
                    probe_batch_size,
                    probe_batches,
                    None if probe_seed is None else probe_seed + 100003 + old_class,
                ),
            )
        except RuntimeError:
            return None

    def old_probe(old_class: int) -> tuple | None:
        if probe_seed is not None and probe_cache is not None:
            key = (
                old_class,
                _pool_key(seen_experiences, old_class),
                probe_batch_size,
                probe_batches,
                int(probe_seed),
            )
            return probe_cache.old_probe(key, lambda: build_old_probe(old_class))
        if old_class not in local_old_probes:
            local_old_probes[old_class] = build_old_probe(old_class)
        return local_old_probes[old_class]

    results = []
    for result in safety_candidates:
        skill_state = memory.state(result["skill"])
        threshold = (
            None if forgetting_margin is None else result["chance"] + forgetting_margin
        )
        old_metrics = []
        safety_complete = True
        for old_class in result["old_classes"]:
            cached = old_probe(old_class)
            if cached is None:
                old_metrics = []
                break
            old_experience, old_x, old_y = cached
            old_accuracy = evaluate_state_accuracy(
                probe_model,
                skill_state,
                old_x,
                old_y,
                old_experience,
                seed=probe_seed,
                cache=state_cache,
                slot=result["skill"],
            )
            old_metrics.append({"class": old_class, "accuracy": old_accuracy})
            if threshold is not None and old_accuracy <= threshold:
                # Already unsafe: the worst old class can only get worse.
                safety_complete = len(old_metrics) == len(result["old_classes"])
                break

        if not old_metrics:
            continue

        result = dict(result)
        result["old_metrics"] = old_metrics
        result["safety_complete"] = safety_complete
        # REAL measured accuracy on the stored skill's old classes. Use the
        # worst mastered class so one forgotten class cannot be hidden.
        result["old_accuracy"] = min(m["accuracy"] for m in old_metrics)
        results.append(result)

    # Keep the result order deterministic and put the strongest candidate first.
    results.sort(key=lambda r: (r["new_score"], r["new_accuracy"]), reverse=True)
    return results


def known_class_decision(target_class: int, skill: int) -> dict[str, Any]:
    """Return the deterministic decision for a previously mastered class."""
    return {
        "class": target_class,
        "decision": "reuse",
        "skill": skill,
        "new_score": 0.0,
        "old_accuracy": 0.0,
        "new_accuracy": 0.0,
        "results": [],
        "known_class": True,
    }


def decide_class(
    strategy,
    experience,
    target_class: int,
    memory,
    class_map,
    probe_batch_size: int,
    probe_batches: int,
    probe_seed: int | None,
    seen_experiences: list,
    forgetting_margin: float,
    score_floor: float | None,
    force_decision: str | None,
    logger_fn,
    max_safety_candidates: int | None = DEFAULT_MAX_SAFETY_CANDIDATES,
    probe_cache: DecisionProbeCache | None = None,
    batch_stage1: bool = False,
    stage1_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Decide for one class, never for an entire multi-class experience.

    If the class was already mastered, its canonical class->skill mapping
    wins.  The generic imagination search is only for genuinely new classes.
    """
    known_skill = class_map.find_skill_for_class_anywhere(target_class)
    if known_skill is not None and force_decision != "scratch":
        decision = known_class_decision(target_class, known_skill)
        logger_fn(
            f"Class {target_class}: REUSE known skill {known_skill} "
            "(canonical class mapping)"
        )
        return decision

    results = score_class_against_skills(
        strategy,
        experience,
        target_class,
        memory,
        class_map,
        probe_batch_size,
        probe_batches,
        probe_seed,
        seen_experiences,
        max_safety_candidates=max_safety_candidates,
        forgetting_margin=forgetting_margin,
        probe_cache=probe_cache,
        batch_stage1=batch_stage1,
        stage1_chunk_size=stage1_chunk_size,
    )

    logger_fn(f"\nImagination for class {target_class}:")
    for result in results:
        old_detail = ", ".join(
            f"class {m['class']}: acc={m['accuracy']:.3f}"
            for m in result.get("old_metrics", [])
        )
        logger_fn(
            f"  skill {result['skill']} (classes={result['old_classes']}): "
            f"old_acc={result['old_accuracy']:.3f}, "
            f"new_score={result['new_score']:.3f}, "
            f"new_acc={result['new_accuracy']:.3f}"
        )
        logger_fn(f"    old-by-class: {old_detail}")

    best = None
    if force_decision is None:
        best = find_best_skill(results, forgetting_margin, score_floor)
    elif force_decision == "reuse" and results:
        best = max(results, key=lambda r: (r["new_score"], r["new_accuracy"]))

    decision: dict[str, Any] = {
        "class": target_class,
        "decision": "scratch",
        "skill": None,
        "new_score": 0.0,
        "old_accuracy": 0.0,
        "new_accuracy": 0.0,
        "results": results,
        "known_class": False,
    }

    if best is not None:
        decision.update(
            {
                "decision": "reuse",
                "skill": best["skill"],
                "new_score": best["new_score"],
                "old_accuracy": best["old_accuracy"],
                "new_accuracy": best["new_accuracy"],
            }
        )
        logger_fn(
            f"Class {target_class}: REUSE skill {best['skill']} "
            f"(score={best['new_score']:.3f}, accuracy={best['new_accuracy']:.3f})"
        )
    else:
        logger_fn(f"Class {target_class}: no compatible skill -> SCRATCH")

    return decision
