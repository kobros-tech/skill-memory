# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Low-level probing, state-application, and IncrementalClassifier helpers.

This module has three responsibilities, all deliberately kept independent of
Avalanche's strategy/plugin machinery:

1. **Dataset probing** -- reading small, deterministic samples of a single
   class out of an Avalanche experience (:func:`classes_in_experience`,
   :func:`class_indices`, :func:`class_subset`, :func:`probe_class`).
   Class-label lookups are cached per dataset object (see
   :func:`_dataset_labels`), because the decision logic in
   :mod:`skill_memory.cl.decision` queries the same experience once per
   ``(candidate skill, mastered class)`` pair -- without caching, that is
   ``O(skills * classes)`` full dataset scans.

2. **Exact state application** -- loading a frozen ``state_dict`` snapshot
   into a live model, including resizing Avalanche's
   :class:`~avalanche.models.dynamic_modules.IncrementalClassifier` head(s)
   first so every tensor shape matches exactly
   (:func:`apply_skill_state_exact`, :func:`restore_initial_state`,
   :func:`resize_incremental_classifiers_for_state`, :func:`predict_logits`).

3. **IncrementalClassifier introspection** -- reading a stored snapshot's
   output width and active-unit mask without instantiating a model
   (:func:`incremental_out_features`, :func:`incremental_active_units`), and
   reshaping a skill's own raw response into a shared global class space
   (:func:`expand_skill_logits`).

Avalanche's ``IncrementalClassifier`` indexes its output units directly by
the raw (global) class label rather than by a per-skill compacted position:
if a skill has only ever seen classes ``{3, 7}``, its classifier still has
one unused column for every label below 7. Every function here that touches
a classifier's width has to respect that convention, since it is exactly
what lets a skill's own stored weights (see
:func:`skill_memory.evaluation.behavior.build_weight_behavior_statistics`)
be compared directly, column-for-column, against another skill's.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from avalanche.models.dynamic_modules import (
    IncrementalClassifier,
    avalanche_model_adaptation,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Dataset probing
# ---------------------------------------------------------------------------

# Per-dataset cache of every sample's integer label, keyed by the dataset
# object itself (not its id()) so that a garbage-collected dataset's cache
# entry is dropped automatically instead of risking an id() collision with
# an unrelated, later dataset. Building this list is the only place that
# ever calls `dataset[index]` for label-only queries; every subsequent
# `classes_in_experience`/`class_indices` call for the same dataset object
# reuses it.
_LABEL_CACHE: weakref.WeakKeyDictionary[Any, list[int]] = weakref.WeakKeyDictionary()


def _dataset_labels(dataset: Dataset) -> list[int]:
    """Return every sample's integer label, scanning `dataset` at most once.

    Prefers a `.targets` shortcut (present on most Avalanche/torchvision
    datasets) over decoding every sample; falls back to `dataset[index][1]`
    otherwise. Datasets that cannot be held by a weak reference (rare) are
    simply not cached -- correctness never depends on the cache.
    """
    try:
        cached = _LABEL_CACHE.get(dataset)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        labels = [int(target) for target in targets]
    else:
        labels = [int(dataset[index][1]) for index in range(len(dataset))]

    try:
        _LABEL_CACHE[dataset] = labels
    except TypeError:
        pass
    return labels


_CLASS_SET_CACHE: weakref.WeakKeyDictionary[Any, frozenset[int]] = (
    weakref.WeakKeyDictionary()
)


def _dataset_class_set(dataset: Dataset) -> frozenset[int]:
    """Return the set of labels present in `dataset` (cached per dataset)."""
    try:
        cached = _CLASS_SET_CACHE.get(dataset)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    classes = frozenset(_dataset_labels(dataset))
    try:
        _CLASS_SET_CACHE[dataset] = classes
    except TypeError:
        pass
    return classes


def experience_has_class(experience, target_class: int) -> bool:
    """Return whether `experience.dataset` contains any `target_class` sample.

    O(1) after the first call per dataset, unlike ``class_indices`` which
    rebuilds an index list on every call.
    """
    return int(target_class) in _dataset_class_set(experience.dataset)


def classes_in_experience(experience) -> list[int]:
    """Return the sorted, unique class labels actually present in `experience`.

    Reads `experience.dataset`'s samples rather than assuming a fixed
    per-experience class layout, since generic Avalanche benchmarks do not
    guarantee one.
    """
    return sorted(set(_dataset_labels(experience.dataset)))


def class_indices(experience, target_class: int) -> list[int]:
    """Return `experience.dataset` indices whose label equals `target_class`."""
    target_class = int(target_class)
    labels = _dataset_labels(experience.dataset)
    return [index for index, label in enumerate(labels) if label == target_class]


def class_subset(experience, target_class: int) -> Subset:
    """Return a `Subset` of `experience.dataset` containing only `target_class`.

    Raises `RuntimeError` if `target_class` has no samples in this
    experience, so callers that probe several experiences looking for one
    class can treat "not here" as an expected, catchable outcome (see
    `skill_memory.cl.decision._first_experience_with_class`).
    """
    indices = class_indices(experience, target_class)
    if not indices:
        raise RuntimeError(f"class {target_class} not found in this experience")
    return Subset(experience.dataset, indices)


def _sample_batches(
    dataset: Dataset,
    batch_size: int,
    n_batches: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw `n_batches` random minibatches of `dataset`, concatenated as one pair.

    Sampling is with replacement across batches (the loader restarts
    whenever it is exhausted), so this always returns exactly
    ``batch_size * n_batches`` samples -- even probing a class with fewer
    raw examples than that -- with a fixed `seed` making the draw
    reproducible.
    """
    if len(dataset) == 0:
        raise RuntimeError("cannot sample batches from an empty dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    iterator = iter(loader)
    for _ in range(max(1, n_batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        xs.append(batch[0])
        ys.append(batch[1])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def probe_class(
    experience,
    target_class: int,
    batch_size: int,
    n_batches: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw a deterministic `(x, y)` probe of only `target_class` from `experience`.

    A thin composition of :func:`class_subset` and :func:`_sample_batches`;
    kept as its own public function because it is the one most call sites
    actually need (probing one class in one experience), while
    `skill_memory.cl.decision._probe_class_across` composes the two lower
    -level pieces itself to pool one class across several experiences.
    """
    dataset = class_subset(experience, target_class)
    return _sample_batches(dataset, batch_size, n_batches, seed)


def origin_experience(experience):
    """Return the original, undivided experience behind a sub-experience.

    Avalanche's online/continuous-task scenarios split one logical
    experience into several ``OnlineCLExperience`` sub-experiences, each
    exposing ``origin_experience`` back to the whole, undivided experience.
    Benchmarks that never split experiences simply lack that attribute, in
    which case `experience` is already the one to keep.
    """
    return getattr(experience, "origin_experience", None) or experience


# ---------------------------------------------------------------------------
# Classifier introspection (IncrementalClassifier-aware)
# ---------------------------------------------------------------------------


def _incremental_classifiers(model: nn.Module):
    """Yield `(dotted_name, module)` for every `IncrementalClassifier` in `model`.

    `dotted_name` is empty when `model` itself is the classifier (matching
    `nn.Module.named_modules()`'s own convention), which only happens for
    unusual architectures where the whole model *is* the classifier head.
    """
    for name, module in model.named_modules():
        if isinstance(module, IncrementalClassifier):
            yield name, module


def _weight_key(name: str) -> str:
    return f"{name}.classifier.weight" if name else "classifier.weight"


def _bias_key(name: str) -> str:
    return f"{name}.classifier.bias" if name else "classifier.bias"


def _active_units_key(name: str) -> str:
    return f"{name}.active_units" if name else "active_units"


def incremental_out_features(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> int | None:
    """Return the classifier output width recorded in a stored skill state.

    Reads the width from `state_dict` (a frozen snapshot), not from
    `model`'s current, possibly-since-grown classifier -- `model` is only
    used to locate *which* key in `state_dict` is the classifier's weight.
    Returns `None` when `model` has no `IncrementalClassifier` head (e.g. a
    fixed-size classifier that never grows), in which case a uniform
    "chance" baseline is undefined and callers should treat it as such.
    """
    for name, _module in _incremental_classifiers(model):
        key = _weight_key(name)
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def incremental_active_units(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> Tensor | None:
    """Return the stored `active_units` mask for a skill snapshot, if any.

    `active_units` is Avalanche's own record of which output columns of an
    `IncrementalClassifier` have actually been trained on (as opposed to
    merely allocated when the head grew to cover a *different* class); see
    `class_index_alignment_report` for how this is used to detect a skill
    whose owned classes and trained columns have drifted apart.
    """
    for name, _module in _incremental_classifiers(model):
        key = _active_units_key(name)
        if key in state_dict:
            return state_dict[key]
    return None


def expand_skill_logits(
    raw_logits: Tensor,
    state: Mapping[str, Tensor],
    owned_classes,
    output_dim: int,
) -> Tensor:
    """Place one skill's raw response into the shared global-class space.

    A skill may store either a global Avalanche classifier head or a compact
    head containing exactly its owned classes. Global heads use class IDs as
    output columns; compact heads use the sorted owned-class order. Every
    non-owned global class receives a finite suppression value so routed
    logits remain safe for downstream metrics.
    """
    del state
    if raw_logits.ndim != 2:
        raise ValueError("raw_logits must have shape [batch, classes]")
    if output_dim < 1:
        raise ValueError("output_dim must be positive")

    owned = sorted({int(class_id) for class_id in owned_classes})
    if not owned:
        return raw_logits.new_full(
            (raw_logits.shape[0], output_dim),
            -20.0,
        )

    width = raw_logits.shape[1]
    if width > len(owned):
        # Avalanche's IncrementalClassifier uses global class IDs directly.
        if max(owned) >= width:
            raise RuntimeError(
                f"skill owns class {max(owned)}, but its classifier has only "
                f"{width} output columns"
            )
        source_indices = owned
    elif width == len(owned):
        # Compact skill heads map local row i to the i-th owned global class.
        source_indices = list(range(width))
    else:
        raise RuntimeError(
            f"skill owns {len(owned)} classes but its classifier has only "
            f"{width} output columns"
        )

    expanded = raw_logits.new_full(
        (raw_logits.shape[0], output_dim),
        -20.0,
    )
    for source_index, global_class in zip(source_indices, owned, strict=True):
        if global_class >= output_dim:
            raise RuntimeError(
                f"skill owns class {global_class}, outside global output "
                f"dimension {output_dim}"
            )
        expanded[:, global_class] = raw_logits[:, source_index]
    return expanded


# ---------------------------------------------------------------------------
# Exact state application
# ---------------------------------------------------------------------------


def resize_incremental_classifiers_for_state(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> None:
    """Resize every `IncrementalClassifier` head in `model` to match `state_dict`.

    Resizing happens in place. `nn.Module.load_state_dict` requires an
    exact shape match for every parameter, but Avalanche's
    `IncrementalClassifier` grows during training. A model whose head has
    since grown past (or never reached) a stored snapshot's recorded width
    must therefore be resized to that exact width *before* the real values
    are loaded. The freshly allocated linear layer's weights are
    irrelevant here, since the caller is always about to overwrite them
    with `load_state_dict` immediately afterwards.
    """
    for name, module in _incremental_classifiers(model):
        weight_key = _weight_key(name)
        if weight_key not in state_dict:
            continue
        target_out, target_in = state_dict[weight_key].shape
        current = module.classifier
        if current.out_features == target_out and current.in_features == target_in:
            continue
        device = current.weight.device
        dtype = current.weight.dtype
        module.classifier = nn.Linear(target_in, target_out).to(
            device=device, dtype=dtype
        )
        active_units_key = _active_units_key(name)
        if active_units_key in state_dict:
            module.active_units = torch.zeros_like(state_dict[active_units_key])
        else:
            module.active_units = torch.zeros(
                target_out, dtype=torch.int8, device=device
            )


def apply_skill_state_exact(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    """Load a frozen skill snapshot into `model`, exactly and in place.

    Resizes any `IncrementalClassifier` head first (see
    :func:`resize_incremental_classifiers_for_state`) so `load_state_dict`
    never fails on a shape mismatch between the live model's current
    classifier width and the width the snapshot was stored at.
    """
    resize_incremental_classifiers_for_state(model, state_dict)
    model.load_state_dict(state_dict)


def restore_initial_state(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    """Restore `model` to a previously captured `state_dict` snapshot.

    Mechanically identical to :func:`apply_skill_state_exact`; kept as a
    separate name because call sites use it for a different purpose --
    undoing a SCRATCH skill's adaptation, or an evaluation phase's
    temporary routing state -- rather than switching in a stored skill.
    """
    apply_skill_state_exact(model, state_dict)


def _functional_growth_for_experience(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    experience,
    *,
    seed: int | None = None,
) -> dict[str, Tensor]:
    """Return `state_dict`, expanded to cover `experience`'s classes.

    `model` itself is never touched.

    Reproduces Avalanche's ``IncrementalClassifier.adaptation`` growth
    rule: the output width becomes ``max(old width,
    max(experience's classes) + 1)``; old rows and active-unit flags are
    preserved, and only the classes in the experience are activated. Newly
    added rows use the same ``nn.Linear`` initialization as Avalanche. When
    ``seed`` is provided, that temporary initialization is deterministic and
    the caller's global RNG state is left unchanged. This returns a plain,
    detached parameter/buffer dict for ``torch.func.functional_call`` rather
    than replacing any module in place."""
    curr_classes = list(getattr(experience, "classes_in_this_experience", ()))
    if not curr_classes:
        return dict(state_dict)

    target_min_width = max(int(c) for c in curr_classes) + 1
    params = dict(state_dict)

    for name, _module in _incremental_classifiers(model):
        weight_key = _weight_key(name)
        if weight_key not in params:
            continue
        old_weight = params[weight_key]
        old_out, in_features = old_weight.shape
        new_out = max(old_out, target_min_width)

        active_key = _active_units_key(name)
        old_active = params.get(active_key)
        if old_active is not None:
            new_active = torch.zeros(
                new_out, dtype=old_active.dtype, device=old_active.device
            )
            new_active[: old_active.shape[0]] = old_active
        else:
            new_active = torch.zeros(new_out, dtype=torch.int8)

        # Match Avalanche: activate exactly the classes in this experience,
        # not every newly allocated column between old_out and max(class)+1.
        new_active[list(int(c) for c in curr_classes)] = 1
        params[active_key] = new_active

        if new_out == old_out:
            continue

        if seed is None:
            fresh = nn.Linear(in_features, new_out)
        else:
            # Keep probe initialization deterministic without advancing the
            # caller's global RNG state.
            with torch.random.fork_rng():
                torch.manual_seed(int(seed))
                fresh = nn.Linear(in_features, new_out)

        fresh = fresh.to(device=old_weight.device, dtype=old_weight.dtype)
        with torch.no_grad():
            fresh.weight[:old_out].copy_(old_weight)
            bias_key = _bias_key(name)
            if bias_key in params:
                fresh.bias[:old_out].copy_(params[bias_key])
        params[weight_key] = fresh.weight.detach()
        bias_key = _bias_key(name)
        if bias_key in params:
            params[bias_key] = fresh.bias.detach()

    return params


def prepare_for_experience(model: nn.Module, experience) -> None:
    """Adapt `model`'s dynamic modules (if any) for `experience`, without training.

    A thin wrapper around Avalanche's own
    ``avalanche_model_adaptation``, which grows every
    `IncrementalClassifier` in `model` to cover `experience`'s classes,
    exactly as Avalanche's own strategy templates do before training --
    but callable on demand, since `SkillMemoryPlugin` bypasses Avalanche's
    normal per-experience training loop (see
    `skill_memory.cl.skill_memory_plugin`). This *does* mutate `model` in
    place; :func:`predict_logits` and :func:`evaluate_state` need the same
    growth rule for a probe forward pass and deliberately avoid this
    function (see :func:`_functional_growth_for_experience`) so that
    probing many skills never touches the caller's live model at all.
    """
    avalanche_model_adaptation(model, experience)


def prepare_for_classes(model: nn.Module, classes: set[int]) -> None:
    """Grow incremental classifier heads to cover explicit global classes."""
    if not classes:
        return

    class _ClassSetExperience:
        classes_in_this_experience = sorted(classes)

    prepare_for_experience(model, _ClassSetExperience())


def predict_logits(
    model: nn.Module, state_dict: Mapping[str, Tensor], x: Tensor
) -> Tensor:
    """Return `model`'s logits for `x` under a frozen skill snapshot.

    Uses :func:`torch.func.functional_call` to substitute `state_dict`'s
    values for the duration of one forward pass, so `model` itself is
    never mutated, never needs its classifier resized, and never needs
    restoring afterwards -- unlike the otherwise-equivalent
    ``apply_skill_state_exact`` + forward pass. This is what makes probing
    many stored skills against the same small batch cheap: swapping in a
    different skill costs nothing more than a dict lookup, not a full
    `load_state_dict` copy of every parameter tensor.
    """
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        params = {key: value.to(device) for key, value in state_dict.items()}
        with torch.no_grad():
            return torch.func.functional_call(model, params, (x.to(device),)).detach()
    finally:
        model.train(was_training)


@dataclass
class _CachedState:
    """One cached functional parameter dict plus the snapshot it came from."""

    source: Mapping[str, Tensor]
    params: dict[str, Tensor]


class FunctionalStateCache:
    """Cache of functionally-expanded skill states, valid for one decision phase.

    :func:`_functional_growth_for_experience` rebuilds a full parameter dict
    (and, when the head must grow, a fresh ``nn.Linear``) for every probe.
    Within one training experience the same stored skill is probed many
    times against experiences that need the *same* classifier growth, so
    the result is identical every time. This cache stores it once, already
    moved to the compute device.

    Exactness rules -- the cache never changes a result:

    * Entries are keyed by ``(slot, sorted experience classes, seed)`` and
      each entry remembers the *identity* of the snapshot it was built
      from. ``SkillMemory.store`` always installs a brand-new dict, so an
      updated skill can never be served a stale entry.
    * Only seeded probes are cached. With ``seed=None`` the temporary
      classifier rows are random *by design*, so every call bypasses the
      cache and behaves exactly as before.
    * The cached tensors are only ever read by ``functional_call``.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, _CachedState] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def invalidate(self, slot: int) -> None:
        """Drop every entry built from `slot` (called when it is re-stored)."""
        for key in [key for key in self._entries if key[0] == slot]:
            del self._entries[key]

    def get(
        self,
        model: nn.Module,
        slot: int,
        state_dict: Mapping[str, Tensor],
        experience,
        *,
        seed: int | None,
    ) -> dict[str, Tensor]:
        """Return device-resident probe parameters for `slot` under `experience`."""
        device = next(model.parameters()).device
        if seed is None:
            params = _functional_growth_for_experience(
                model, state_dict, experience, seed=None
            )
            return {key: value.to(device) for key, value in params.items()}

        classes = tuple(
            sorted(
                int(c) for c in getattr(experience, "classes_in_this_experience", ())
            )
        )
        key = (slot, classes, int(seed), str(device))
        entry = self._entries.get(key)
        if entry is not None and entry.source is state_dict:
            self.hits += 1
            return entry.params

        self.misses += 1
        params = _functional_growth_for_experience(
            model, state_dict, experience, seed=seed
        )
        params = {name: value.to(device) for name, value in params.items()}
        self._entries[key] = _CachedState(source=state_dict, params=params)
        return params


def _probe_params(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    experience,
    seed: int | None,
    cache: FunctionalStateCache | None,
    slot: int | None,
) -> dict[str, Tensor]:
    if cache is not None and slot is not None:
        return cache.get(model, slot, state_dict, experience, seed=seed)
    params = _functional_growth_for_experience(model, state_dict, experience, seed=seed)
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in params.items()}


def _forward_metrics(logits: Tensor, targets: Tensor) -> tuple[float, float, float]:
    """(loss, score, accuracy) for one skill's logits -- shared by every evaluator.

    Kept as one function so the sequential and batched code paths can never
    silently drift apart: both end up calling exactly this.
    """
    loss = torch.nn.functional.cross_entropy(logits, targets)
    probabilities = torch.softmax(logits, dim=1)
    score = probabilities.gather(1, targets.view(-1, 1)).mean()
    accuracy = logits.argmax(dim=1).eq(targets).float().mean()
    return float(loss.item()), float(score.item()), float(accuracy.item())


def evaluate_state_accuracy(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    x: Tensor,
    y: Tensor,
    experience,
    *,
    seed: int | None = None,
    cache: FunctionalStateCache | None = None,
    slot: int | None = None,
) -> float:
    """Return only top-1 accuracy of a frozen skill snapshot on a probe batch.

    Numerically identical to the ``accuracy`` that :func:`evaluate_state`
    returns, but skips the loss and softmax/score computation. Used for
    old-class *safety* checks, whose decision rule reads accuracy only.
    """
    params = _probe_params(model, state_dict, experience, seed, cache, slot)
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = torch.func.functional_call(model, params, (x.to(device),))
            targets = y.to(device)
            return float(logits.argmax(dim=1).eq(targets).float().mean().item())
    finally:
        model.train(was_training)


def _param_shape_signature(params: Mapping[str, Tensor]) -> tuple:
    """A hashable fingerprint of every tensor's `(shape, dtype)` in `params`.

    Two skills with the same signature have parameter/buffer dicts that can
    be `torch.stack`-ed key-for-key: identical keys, and every tensor at each
    key has the identical shape and dtype (values may of course differ).
    Skills whose growth (see `_functional_growth_for_experience`) leaves a
    *wider* `IncrementalClassifier` than others -- which happens whenever a
    stored skill was last updated after a later, larger-class-id experience
    than the one currently being probed -- fall into a different group and
    are never stacked with the rest.
    """
    return tuple(
        (key, tuple(value.shape), value.dtype) for key, value in sorted(params.items())
    )


def evaluate_states_batch(
    model: nn.Module,
    states: list[tuple[Any, Mapping[str, Tensor]]],
    x: Tensor,
    y: Tensor,
    experience,
    *,
    seed: int | None = None,
    cache: FunctionalStateCache | None = None,
    chunk_size: int | None = None,
) -> dict[Any, tuple[float, float, float]]:
    """Batched, GPU-parallel equivalent of calling :func:`evaluate_state` once
    per ``(key, state_dict)`` pair in `states`, all against the same `(x, y)`.

    Returns ``{key: (loss, score, accuracy)}``, one entry per input pair, with
    values that are the *same computation* as the sequential per-skill
    ``evaluate_state`` loop -- not an approximation. Skills are grouped by
    :func:`_param_shape_signature` (see there for why two skills' expanded
    states can differ in shape) and each group of 2+ same-shaped skills is
    evaluated in one call via ``torch.func.functional_call`` stacked over the
    skill dimension and mapped with :func:`torch.vmap`, following the
    ensembling pattern in the PyTorch ``torch.func`` migration guide
    (https://docs.pytorch.org/docs/stable/func.migrating.html). A group of
    exactly 1 skill is evaluated with the ordinary sequential path -- vmap has
    per-call overhead of its own that a single skill cannot amortize.

    Numerically, results agree with the sequential path up to ordinary
    float32 matmul-reassociation noise (empirically ~1e-6 absolute on logits
    for a small batchnorm+conv model; see
    ``tests/test_stage1_batching.py::test_batched_matches_sequential_reference``).
    This does not change routing decisions: `score_class_against_skills` only
    ever compares these values against each other and against fixed
    thresholds several orders of magnitude coarser than that noise floor.

    `chunk_size` bounds how many skills' worth of activations exist at once
    (forwarded to ``torch.vmap``); pass a smaller value if a group is large
    enough to risk exhausting device memory. `None` (default) runs the whole
    group as a single vmap call.

    Whether this is actually *faster* than the sequential loop is
    hardware- and kernel-dependent -- see
    ``skill_memory/benchmarks/stage1_batching.py``. It is not a formality:
    functorch's batching rules do not always map to fused kernels, so this
    should be benchmarked on the target device before being enabled by
    default (see ``SkillMemoryPlugin(batch_stage1=...)``).
    """
    device = next(model.parameters()).device
    targets = y.to(device)
    x = x.to(device)

    groups: dict[tuple, list[tuple[Any, dict[str, Tensor]]]] = {}
    for key, state_dict in states:
        params = _probe_params(model, state_dict, experience, seed, cache, key)
        groups.setdefault(_param_shape_signature(params), []).append((key, params))

    results: dict[Any, tuple[float, float, float]] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for members in groups.values():
                if len(members) == 1:
                    key, params = members[0]
                    logits = torch.func.functional_call(model, params, (x,))
                    results[key] = _forward_metrics(logits, targets)
                    continue

                keys = [key for key, _ in members]
                param_names = members[0][1].keys()
                stacked = {
                    name: torch.stack([params[name] for _, params in members])
                    for name in param_names
                }

                def call(one_skill_params: dict[str, Tensor]) -> Tensor:
                    return torch.func.functional_call(model, one_skill_params, (x,))

                batched_logits = torch.vmap(call, chunk_size=chunk_size)(stacked)
                n_group = batched_logits.shape[0]
                n_samples = targets.shape[0]
                flat_logits = batched_logits.reshape(n_group * n_samples, -1)
                flat_targets = targets.unsqueeze(0).expand(n_group, -1).reshape(-1)
                losses = (
                    torch.nn.functional.cross_entropy(
                        flat_logits, flat_targets, reduction="none"
                    )
                    .view(n_group, n_samples)
                    .mean(dim=1)
                )
                probabilities = torch.softmax(batched_logits, dim=2)
                scores = (
                    probabilities.gather(
                        2, targets.view(1, -1, 1).expand(n_group, -1, 1)
                    )
                    .squeeze(2)
                    .mean(dim=1)
                )
                accuracies = (
                    (batched_logits.argmax(dim=2) == targets.view(1, -1))
                    .float()
                    .mean(dim=1)
                )
                for index, key in enumerate(keys):
                    results[key] = (
                        float(losses[index].item()),
                        float(scores[index].item()),
                        float(accuracies[index].item()),
                    )
    finally:
        model.train(was_training)

    return results


def evaluate_state(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    x: Tensor,
    y: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    experience,
    *,
    seed: int | None = None,
    cache: FunctionalStateCache | None = None,
    slot: int | None = None,
) -> tuple[float, float, float]:
    r"""Score one frozen skill snapshot against a probe batch, with no training.

    This is what :mod:`skill_memory.cl.decision` calls "imagination":
    measuring how a stored skill's *existing* representation would perform
    on a probe batch, including one it may never have trained on, without
    taking a single gradient step -- and, like :func:`predict_logits`,
    without mutating `model` at all: the classifier growth that
    `experience` may require is computed as a plain tensor dict (see
    :func:`_functional_growth_for_experience`) and substituted in only for
    this one forward pass via :func:`torch.func.functional_call`. Probing
    ``S`` stored skills against the same class therefore uses one
    functional forward per skill, without mutating or copying the caller's
    live model between skills.

    The growth rule gives the classifier a column for every class
    `experience` actually contains and activates exactly those classes.
    If the stored state already has enough columns, no weights are changed;
    only the active-unit mask is updated. Newly added classifier rows use
    Avalanche's normal random initialization, optionally seeded by `seed`.
    Given the resulting logits
    :math:`z \in \mathbb{R}^{N \times C}` and labels
    :math:`y \in \{0, \dots, C-1\}^N`:

    .. math::

        \text{loss} = \texttt{loss\_fn}(z, y), \qquad
        \text{score} = \frac{1}{N} \sum_{i=1}^{N} \operatorname{softmax}(z_i)_{y_i},
        \qquad
        \text{accuracy} = \frac{1}{N} \sum_{i=1}^{N}
            \mathbb{1}\!\left[\operatorname*{arg\,max}_c z_{i,c} = y_i\right]

    `score` -- the mean probability mass the *unmodified* skill already
    places on the true label -- is what `find_best_skill`'s `score_floor`
    thresholds; `accuracy` is a plain top-1 accuracy over the same batch.
    """
    params = _probe_params(model, state_dict, experience, seed, cache, slot)
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = torch.func.functional_call(model, params, (x.to(device),))
            targets = y.to(device)
            loss = loss_fn(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            score = probabilities.gather(1, targets.view(-1, 1)).mean()
            accuracy = logits.argmax(dim=1).eq(targets).float().mean()
        return float(loss.item()), float(score.item()), float(accuracy.item())
    finally:
        model.train(was_training)
