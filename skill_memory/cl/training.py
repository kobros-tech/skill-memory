# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Class-level training loops for Skill Memory.

Two training objectives are available (``mode``):

``multiclass``
    Only target-class samples are loaded and ordinary cross-entropy is used.

``binary_one_vs_rest``
    The target class :math:`c` becomes an explicit YES/NO verifier.  With
    :math:`z_c(x)` the raw logit of class :math:`c` and
    :math:`y_c = \mathbb 1[y = c]`,

    .. math::

        \mathcal L(x, y) = -\,y_c \log\sigma(z_c(x))
                           -(1-y_c)\log\bigl(1-\sigma(z_c(x))\bigr).

    Only column :math:`c` of the logits participates in the loss.

Everything stochastic in this module is driven by explicit
:class:`torch.Generator` objects, so the *same seeds reproduce the same
training run* irrespective of the global RNG state:

* the validation hold-out split uses ``validation_seed``;
* historical replay selection uses ``validation_seed + 1543 + class_id``;
* mini-batch order, balanced re-sampling **and stochastic layers such as
  dropout** use ``sampler_seed`` (the optimisation loop runs in a forked RNG
  scope that is restored afterwards).

Every call returns a :class:`TrainingResult` whose ``provenance`` states
exactly how many examples came from each data source, so the replay policy
(see :mod:`skill_memory.cl.replay`) is *auditable*, not just documented.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    TensorDataset,
    WeightedRandomSampler,
)

from ..utils.probing import _dataset_labels, class_subset
from .replay import select_historical_samples

VALID_CLASS_TRAIN_MODES = ("multiclass", "binary_one_vs_rest")

#: Salt separating the replay-selection stream from the hold-out stream.
_REPLAY_SEED_OFFSET = 1543
#: Salt reproducing the historical domain-update hold-out stream.
_DOMAIN_SEED_OFFSET = 7919

# Kept for backwards compatibility with code importing the private name.
_select_historical_samples = select_historical_samples


def derive_seed(base: int, *parts: int) -> int:
    """Deterministically mix ``base`` with integer ``parts`` into a seed.

    A simple polynomial hash; collisions between distinct ``parts`` tuples are
    irrelevant here -- the goal is only to decorrelate per-class / per-skill /
    per-experience streams while staying a pure function of the inputs.
    """
    value = int(base) & 0x7FFFFFFF
    for part in parts:
        value = (value * 1_000_003 + int(part) + 1) & 0x7FFFFFFFFFFF
    return value


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingProvenance:
    """Where every training example of one training call came from.

    ``current`` / ``retained`` / ``offline_pool`` map ``class_id -> count``.
    ``historical_total`` (= retained + offline pool) is ``0`` under
    ``cl_update_mode='new_class'`` -- the invariant asserted by the tests.
    """

    kind: str  # "class" or "refresh"
    target_classes: tuple[int, ...]
    objective: str
    current: dict[int, int] = field(default_factory=dict)
    retained: dict[int, int] = field(default_factory=dict)
    offline_pool: dict[int, int] = field(default_factory=dict)
    sampler_seed: int | None = None
    optimizer_steps: int = 0

    @property
    def current_total(self) -> int:
        return sum(self.current.values())

    @property
    def historical_total(self) -> int:
        return sum(self.retained.values()) + sum(self.offline_pool.values())

    @property
    def dataset_size(self) -> int:
        return self.current_total + self.historical_total

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target_classes": list(self.target_classes),
            "objective": self.objective,
            "current": dict(self.current),
            "retained": dict(self.retained),
            "offline_pool": dict(self.offline_pool),
            "historical_total": self.historical_total,
            "dataset_size": self.dataset_size,
            "sampler_seed": self.sampler_seed,
            "optimizer_steps": self.optimizer_steps,
        }


class TrainingResult(NamedTuple):
    """Return value of :func:`train_on_class` / :func:`train_skill_on_domain`.

    ``validation_*`` is the **calibration hold-out**: examples that were
    *excluded* from training and are later used only to fit the Platt
    calibration of the skill's verifier.  They are never replay data.
    """

    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    provenance: TrainingProvenance


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_budgets(
    validation_fraction: float,
    samples_per_class: int | None,
    historical_samples_per_class: int | None,
) -> None:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if historical_samples_per_class is not None and historical_samples_per_class <= 0:
        raise ValueError("historical_samples_per_class must be positive")


def split_holdout(
    labels: Sequence[int],
    *,
    validation_fraction: float,
    samples_per_class: int | None,
    seed: int,
) -> tuple[list[int], list[int], dict[int, list[int]]]:
    """Stratified, seeded split into ``(train, validation, by_class)`` indices.

    For each class with :math:`n` samples,
    :math:`n_{val} = \\min(\\lfloor n f\\rfloor \\lor 1,\\; n-1)` (the ``\\lor 1``
    applies only when ``f > 0`` and ``n > 1``), and at most
    ``samples_per_class`` samples *after* the hold-out are trained on.
    """
    generator = torch.Generator().manual_seed(int(seed))
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in by_class.values():
        shuffled = torch.tensor(indices, dtype=torch.long)
        if len(indices) > 1:
            shuffled = shuffled[torch.randperm(len(indices), generator=generator)]
        n_validation = int(len(indices) * validation_fraction)
        if validation_fraction > 0.0 and n_validation == 0 and len(indices) > 1:
            n_validation = 1
        n_validation = min(n_validation, max(0, len(indices) - 1))
        validation_indices.extend(shuffled[:n_validation].tolist())
        stop = None if samples_per_class is None else n_validation + samples_per_class
        train_indices.extend(shuffled[n_validation:stop].tolist())
    return train_indices, validation_indices, by_class


def _materialize(dataset, indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode ``dataset[i]`` for ``i in indices`` into stacked CPU tensors."""
    inputs, targets = [], []
    for index in indices:
        sample = dataset[index]
        inputs.append(torch.as_tensor(sample[0]).detach().cpu())
        targets.append(int(sample[1]))
    if not inputs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.stack(inputs), torch.tensor(targets, dtype=torch.long)


def _collect_history(
    sources: Sequence[tuple[str, Iterable, int | None]],
    *,
    exclude: set[int],
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, dict[int, int]]]:
    """Gather capped per-class historical examples from named sources.

    ``sources`` is a sequence of ``(name, items, limit)``.  A class already
    contributed by an earlier source, or listed in ``exclude`` (the current
    classes), is skipped -- so the same class is never replayed twice.
    """
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    counts: dict[str, dict[int, int]] = {name: {} for name, _, _ in sources}
    taken: set[int] = set()
    for name, items, limit in sources:
        for item in items or ():
            class_id = int(item.class_id)
            if class_id in exclude or class_id in taken:
                continue
            taken.add(class_id)
            x, y = select_historical_samples(
                item.inputs.detach().cpu(),
                item.targets.detach().cpu(),
                limit=limit,
                seed=seed,
                class_id=class_id,
            )
            inputs.append(x)
            targets.append(y)
            counts[name][class_id] = int(len(x))
    return inputs, targets, counts


def balanced_weights(
    labels: Sequence[int], positive_classes: Sequence[int]
) -> torch.Tensor:
    r"""Sampling weights giving every owned class and the negatives equal mass.

    A positive of owned class :math:`c` with :math:`n_c` examples gets weight
    :math:`\tfrac1{2n_c}`; each of the :math:`N^-` negatives gets
    :math:`\tfrac1{2N^-}`.  For one owned class the draw is positive with
    probability exactly :math:`\tfrac12`; for :math:`|O|` owned classes it is
    :math:`|O|/(|O|+1)`.  Raises if any owned class or the negatives are empty.
    """
    owned = [int(c) for c in positive_classes]
    values = [int(v) for v in labels]
    counts = {c: values.count(c) for c in owned}
    negatives = len(values) - sum(counts.values())
    if negatives <= 0 or any(n <= 0 for n in counts.values()):
        raise RuntimeError(
            "balanced sampling requires positive and negative samples "
            "for every owned class"
        )
    return torch.tensor(
        [0.5 / counts[v] if v in counts else 0.5 / negatives for v in values],
        dtype=torch.double,
    )


def _seeded_loader(
    dataset,
    batch_size: int,
    *,
    seed: int,
    weights: torch.Tensor | None,
) -> DataLoader:
    """DataLoader whose shuffling/re-sampling depends only on ``seed``."""
    generator = torch.Generator().manual_seed(int(seed))
    batch = min(batch_size, len(dataset))
    if weights is None:
        return DataLoader(dataset, batch_size=batch, shuffle=True, generator=generator)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch, sampler=sampler)


@contextlib.contextmanager
def _isolated_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Run a block with RNG state derived from ``seed`` only.

    Stochastic layers (``Dropout``, noise, ...) draw from the global RNG, so
    without this the trained weights would depend on whatever else consumed
    randomness earlier.  The previous global state is restored on exit, so
    training neither depends on nor disturbs the caller's RNG stream.
    """
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def _optimise(strategy, loader, epochs: int, loss_fn, *, seed: int) -> int:
    """Run ``epochs`` passes of SGD-style updates; return the step count."""
    device = next(strategy.model.parameters()).device
    steps = 0
    strategy.model.train()
    with _isolated_rng(seed, device):
        for _ in range(epochs):
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                strategy.optimizer.zero_grad()
                loss = loss_fn(strategy.model(x), y)
                loss.backward()
                strategy.optimizer.step()
                # This custom loop bypasses Avalanche's iteration events, so
                # the clock must be advanced by hand; JSONLogger uses it to
                # tell evaluation checkpoints apart.
                strategy.clock.train_iterations += 1
                steps += 1
    return steps


def _empty_result(provenance: TrainingProvenance) -> TrainingResult:
    return TrainingResult(torch.empty(0), torch.empty(0, dtype=torch.long), provenance)


# ---------------------------------------------------------------------------
# Per-class training
# ---------------------------------------------------------------------------


def train_on_class(
    strategy,
    experience,
    target_class: int,
    epochs: int,
    batch_size: int,
    *,
    mode: str = "multiclass",
    validation_fraction: float = 0.2,
    validation_seed: int = 0,
    retained_memory=None,
    negative_pool=None,
    samples_per_class: int | None = None,
    historical_samples_per_class: int | None = None,
    sampler_seed: int = 0,
) -> TrainingResult:
    r"""Train one target class with multiclass or one-vs-rest supervision.

    Data sources (binary mode)
    --------------------------
    * **current** -- every class of the current experience, each capped at
      ``samples_per_class`` after the calibration hold-out.  Target-class
      samples are positives, all others negatives.
    * **retained** -- ``retained_memory``: previously seen classes, each
      capped at ``historical_samples_per_class`` (``None`` = all retained).
      The caller passes ``None`` for ``cl_update_mode='new_class'``.
    * **offline pool** -- ``negative_pool``: explicit *oracle* negatives for
      offline ablations; capped by the same ``historical_samples_per_class``.

    Future classes are never read implicitly.  The returned
    :class:`TrainingResult` carries the per-source counts.

    Balanced sampling
    -----------------
    With :math:`P` positives and :math:`N` negatives in the assembled dataset
    each sample is drawn with probability :math:`\tfrac{1}{2P}` (positive) or
    :math:`\tfrac{1}{2N}` (negative), so positive and negative evidence have
    equal expected mass regardless of how much history is replayed.
    """
    _check_budgets(validation_fraction, samples_per_class, historical_samples_per_class)
    if mode not in VALID_CLASS_TRAIN_MODES:
        raise ValueError(
            f"invalid class training mode {mode!r}; "
            f"expected one of {VALID_CLASS_TRAIN_MODES}"
        )

    empty = TrainingProvenance(
        kind="class",
        target_classes=(int(target_class),),
        objective=mode,
        sampler_seed=int(sampler_seed),
    )
    positive_dataset = class_subset(experience, target_class)
    if len(positive_dataset) == 0:
        raise RuntimeError(f"class {target_class} has no samples to train on")
    if epochs < 1:
        return _empty_result(empty)

    # A deterministic hold-out stays completely outside the training data.
    # It is later used only to calibrate the verification gate.
    full_dataset = experience.dataset
    labels = _dataset_labels(full_dataset)
    training_indices, validation_indices, by_class = split_holdout(
        labels,
        validation_fraction=validation_fraction,
        samples_per_class=samples_per_class,
        seed=int(validation_seed) + target_class,
    )

    retained_counts: dict[int, int] = {}
    pool_counts: dict[int, int] = {}
    weights = None

    if mode == "multiclass":
        target_indices = [i for i in training_indices if labels[i] == target_class]
        dataset = Subset(full_dataset, target_indices)
        current_counts = {int(target_class): len(target_indices)}
        train_labels = [int(target_class)] * len(target_indices)
    else:
        # Materialise current-experience samples first so every item is a
        # uniform TensorDataset sample even once retained tensors are added.
        current_x, current_y = _materialize(full_dataset, training_indices)
        if len(current_x) == 0:
            raise RuntimeError(f"class {target_class} has no training samples")
        dataset = TensorDataset(current_x, current_y)
        current_counts = {
            int(c): int((current_y == c).sum()) for c in torch.unique(current_y)
        }

        hist_x, hist_y, counts = _collect_history(
            [
                ("retained", retained_memory, historical_samples_per_class),
                ("offline_pool", negative_pool, historical_samples_per_class),
            ],
            exclude=set(by_class) | {int(target_class)},
            seed=int(validation_seed) + _REPLAY_SEED_OFFSET,
        )
        retained_counts, pool_counts = counts["retained"], counts["offline_pool"]
        if hist_x:
            dataset = ConcatDataset(
                [dataset, TensorDataset(torch.cat(hist_x), torch.cat(hist_y))]
            )

        # Balanced sampler built from the *assembled* dataset.
        train_labels = [int(dataset[i][1]) for i in range(len(dataset))]
        weights = balanced_weights(train_labels, [int(target_class)])

    if len(dataset) == 0:
        raise RuntimeError(f"class {target_class} has no training samples")

    criterion = getattr(strategy, "_criterion", None)
    if criterion is None:
        criterion = torch.nn.functional.cross_entropy

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if mode == "multiclass":
            return criterion(logits, y)
        if target_class < 0 or target_class >= logits.shape[1]:
            raise RuntimeError(
                f"target class {target_class} is outside the model "
                f"classifier width {logits.shape[1]}"
            )
        column = logits[:, target_class]
        return torch.nn.functional.binary_cross_entropy_with_logits(
            column, y.eq(target_class).to(dtype=column.dtype)
        )

    loader = _seeded_loader(dataset, batch_size, seed=sampler_seed, weights=weights)
    steps = _optimise(strategy, loader, epochs, loss_fn, seed=sampler_seed)

    provenance = TrainingProvenance(
        kind="class",
        target_classes=(int(target_class),),
        objective=mode,
        current=current_counts,
        retained=retained_counts,
        offline_pool=pool_counts,
        sampler_seed=int(sampler_seed),
        optimizer_steps=steps,
    )
    if not validation_indices:
        return _empty_result(provenance)
    val_x, val_y = _materialize(full_dataset, validation_indices)
    return TrainingResult(val_x, val_y, provenance)


# ---------------------------------------------------------------------------
# Skill-domain refresh
# ---------------------------------------------------------------------------


def train_skill_on_domain(
    strategy,
    experience,
    positive_classes: set[int],
    observed_classes: set[int],
    epochs: int,
    batch_size: int,
    *,
    validation_fraction: float = 0.2,
    validation_seed: int = 0,
    retained_memory=None,
    samples_per_class: int | None = None,
    historical_samples_per_class: int | None = None,
    training_counts: dict[int, int] | None = None,
    sampler_seed: int = 0,
) -> TrainingResult:
    r"""Refresh one multi-label binary skill on the full observed domain.

    A skill owning classes :math:`O` is trained against the observed domain
    :math:`D` with a multi-label BCE over the owned columns,

    .. math::

        \mathcal L = \frac1{|O|}\sum_{c\in O}\operatorname{BCE}
            \bigl(z_c(x),\ \mathbb 1[y=c]\bigr).

    Sampling weights are :math:`\tfrac1{2n_c}` for a positive of owned class
    :math:`c` (``n_c`` = its count) and :math:`\tfrac1{2N}` for each of the
    :math:`N` non-owned samples.  Hence positive mass is
    :math:`|O|/(|O|+1)` of the draws and negatives get :math:`1/(|O|+1)`;
    for a single-class skill this is exactly the balanced 1/2 : 1/2 split.
    """
    if not positive_classes:
        raise ValueError("positive_classes must not be empty")
    if not positive_classes <= observed_classes:
        raise ValueError("positive_classes must be a subset of observed_classes")
    _check_budgets(validation_fraction, samples_per_class, historical_samples_per_class)

    owned_classes = sorted(positive_classes)
    base = TrainingProvenance(
        kind="refresh",
        target_classes=tuple(owned_classes),
        objective="binary_one_vs_rest",
        sampler_seed=int(sampler_seed),
    )
    if epochs < 1:
        return _empty_result(base)

    full_dataset = experience.dataset
    labels = _dataset_labels(full_dataset)
    training_indices, validation_indices, by_class = split_holdout(
        labels,
        validation_fraction=validation_fraction,
        samples_per_class=samples_per_class,
        seed=int(validation_seed) + _DOMAIN_SEED_OFFSET,
    )
    current_classes = set(by_class)

    pools: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    current_counts: dict[int, int] = {}
    for class_id in sorted(current_classes):
        x, y = _materialize(
            full_dataset, [i for i in training_indices if labels[i] == class_id]
        )
        if len(x):
            pools[class_id] = (x, y)
            current_counts[class_id] = len(x)

    retained_counts: dict[int, int] = {}
    retained_by_class = {int(item.class_id): item for item in (retained_memory or ())}
    for class_id in sorted(observed_classes - current_classes):
        item = retained_by_class.get(class_id)
        if item is None:
            continue
        x, y = select_historical_samples(
            item.inputs.detach().cpu(),
            item.targets.detach().cpu(),
            limit=historical_samples_per_class,
            seed=int(validation_seed) + _REPLAY_SEED_OFFSET,
            class_id=class_id,
        )
        pools[class_id] = (x, y)
        retained_counts[class_id] = len(x)

    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for class_id in sorted(pools):
        x, y = pools[class_id]
        if training_counts is not None:
            training_counts[class_id] = len(x)
        inputs.append(x)
        targets.append(y)
    if not inputs:
        raise RuntimeError("skill-domain update has no training samples")

    all_targets = torch.cat(targets)
    dataset = TensorDataset(torch.cat(inputs), all_targets)
    weights = balanced_weights(all_targets.tolist(), owned_classes)

    width_needed = max(observed_classes) + 1
    owned_index = torch.tensor(owned_classes, dtype=torch.long)

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if width_needed > logits.shape[1]:
            raise RuntimeError(
                "skill-domain update requires a classifier head covering "
                f"observed class {width_needed - 1}; width={logits.shape[1]}"
            )
        columns = logits[:, owned_index.to(logits.device)]
        wanted = torch.stack([y.eq(c) for c in owned_classes], dim=1)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            columns, wanted.to(dtype=columns.dtype)
        )

    loader = _seeded_loader(dataset, batch_size, seed=sampler_seed, weights=weights)
    steps = _optimise(strategy, loader, epochs, loss_fn, seed=sampler_seed)

    provenance = TrainingProvenance(
        kind="refresh",
        target_classes=tuple(owned_classes),
        objective="binary_one_vs_rest",
        current=current_counts,
        retained=retained_counts,
        sampler_seed=int(sampler_seed),
        optimizer_steps=steps,
    )
    if not validation_indices:
        return _empty_result(provenance)
    val_x, val_y = _materialize(full_dataset, validation_indices)
    return TrainingResult(val_x, val_y, provenance)
