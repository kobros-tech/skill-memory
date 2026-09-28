# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Class-level training loops for Skill Memory."""

from __future__ import annotations

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    TensorDataset,
    WeightedRandomSampler,
)

from ..utils.probing import class_subset

VALID_CLASS_TRAIN_MODES = ("multiclass", "binary_one_vs_rest")


def _select_historical_samples(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    limit: int | None,
    seed: int,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a deterministic per-class historical replay subset."""
    if limit is None or len(inputs) <= limit:
        return inputs, targets
    if limit <= 0:
        raise ValueError("limit must be positive or None")

    generator = torch.Generator().manual_seed(int(seed) + int(class_id))
    indices = torch.randperm(len(inputs), generator=generator)[:limit]
    return inputs[indices], targets[indices]


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train one target class with multiclass or one-vs-rest supervision.

    ``multiclass`` preserves the original objective: only target-class
    samples are loaded and ordinary cross-entropy is applied.

    ``binary_one_vs_rest`` turns the target class into an explicit YES/NO
    verifier. Target-class samples are positive (1), every other sample in
    the current experience is negative (0). When retained memory is supplied,
    retained samples from previously seen classes are added as additional
    negatives. If ``negative_pool`` is supplied, it is an explicit offline
    experiment source containing all requested non-target classes. Future
    classes are never read implicitly by this training loop. Only the
    The target-class logit is trained relative to the log-sum-exp of the
    negative-class logits, so the negative classes receive gradients too.

    ``BCEWithLogitsLoss`` consumes raw logits and applies sigmoid internally.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if mode not in VALID_CLASS_TRAIN_MODES:
        raise ValueError(
            f"invalid class training mode {mode!r}; "
            f"expected one of {VALID_CLASS_TRAIN_MODES}"
        )
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if (
        historical_samples_per_class is not None
        and historical_samples_per_class <= 0
    ):
        raise ValueError("historical_samples_per_class must be positive")

    positive_dataset = class_subset(experience, target_class)
    if len(positive_dataset) == 0:
        raise RuntimeError(f"class {target_class} has no samples to train on")
    if epochs < 1:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    # Keep a deterministic holdout completely outside the training dataset.
    # It is later used only to calibrate the Skill Memory verification gate.
    full_dataset = experience.dataset
    labels = [int(full_dataset[index][1]) for index in range(len(full_dataset))]

    generator = torch.Generator().manual_seed(int(validation_seed) + target_class)
    validation_indices: list[int] = []
    training_indices: list[int] = []
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)

    for _class_id, indices in by_class.items():
        shuffled = torch.tensor(indices, dtype=torch.long)
        if len(indices) > 1:
            shuffled = shuffled[torch.randperm(len(indices), generator=generator)]
        n_validation = int(len(indices) * validation_fraction)
        if validation_fraction > 0.0 and n_validation == 0 and len(indices) > 1:
            n_validation = 1
        n_validation = min(n_validation, max(0, len(indices) - 1))
        validation_indices.extend(shuffled[:n_validation].tolist())
        training_indices.extend(
            shuffled[
                n_validation : (
                    n_validation + samples_per_class
                    if samples_per_class is not None
                    else None
                )
            ].tolist()
        )

    current_classes = set(by_class)

    if mode == "multiclass":
        target_indices = [
            index for index in training_indices if labels[index] == target_class
        ]
        dataset = Subset(full_dataset, target_indices)
    else:
        # Materialize current-experience samples before adding retained memory.
        # This keeps every item a uniform TensorDataset sample; mixing an
        # Avalanche Subset with retained tensors can otherwise make the
        # default DataLoader collate different sample representations.
        current_inputs = []
        current_targets = []
        for index in training_indices:
            sample = full_dataset[index]
            current_inputs.append(torch.as_tensor(sample[0]).detach().cpu())
            current_targets.append(int(sample[1]))

        dataset = TensorDataset(
            torch.stack(current_inputs),
            torch.tensor(current_targets, dtype=torch.long),
        )

        # Add retained examples from previously seen classes as negatives.
        # Future classes are never read, preserving the CL protocol.
        if retained_memory or negative_pool:
            prior_inputs = []
            prior_targets = []
            source_items = [
                (item, historical_samples_per_class)
                for item in (retained_memory or [])
            ] + [
                (item, samples_per_class) for item in (negative_pool or [])
            ]
            seen_source_classes = set()
            for item, source_limit in source_items:
                class_id = int(item.class_id)
                if class_id in current_classes or class_id == target_class:
                    continue
                if class_id in seen_source_classes:
                    continue
                seen_source_classes.add(class_id)
                item_inputs = item.inputs.detach().cpu()
                item_targets = item.targets.detach().cpu()
                item_inputs, item_targets = _select_historical_samples(
                    item_inputs,
                    item_targets,
                    limit=source_limit,
                    seed=int(validation_seed) + 1543,
                    class_id=class_id,
                )
                prior_inputs.append(item_inputs)
                prior_targets.append(item_targets)
            if prior_inputs:
                prior_dataset = TensorDataset(
                    torch.cat(prior_inputs, dim=0),
                    torch.cat(prior_targets, dim=0),
                )
                dataset = ConcatDataset([dataset, prior_dataset])

    if mode == "binary_one_vs_rest":
        # Use every assembled example exactly once per epoch. The previous
        # replacement sampler could repeatedly train on a small subset while
        # never seeing other examples, making fresh SCRATCH skills unstable.
        # Keep the one-vs-rest objective, but balance it through BCE itself.
        train_labels = [int(dataset[index][1]) for index in range(len(dataset))]
    else:
        train_labels = [labels[index] for index in training_indices]
    positive_count = sum(label == target_class for label in train_labels)
    negative_count = len(train_labels) - positive_count
    if len(dataset) == 0:
        raise RuntimeError(f"class {target_class} has no training samples")
    if positive_count == 0:
        raise RuntimeError(f"class {target_class} has no positive training samples")
    if mode == "binary_one_vs_rest" and negative_count == 0:
        raise RuntimeError(
            f"class {target_class} has no negative training samples"
        )

    device = next(strategy.model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    criterion = getattr(strategy, "_criterion", None)
    if criterion is None:
        criterion = torch.nn.functional.cross_entropy

    strategy.model.train()
    for _ in range(epochs):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            strategy.optimizer.zero_grad()
            logits = strategy.model(x)

            if mode == "multiclass":
                loss = criterion(logits, y)
            else:
                if target_class < 0 or target_class >= logits.shape[1]:
                    raise RuntimeError(
                        f"target class {target_class} is outside the model "
                        f"classifier width {logits.shape[1]}"
                    )
                negative_classes = sorted(set(train_labels) - {target_class})
                if not negative_classes:
                    raise RuntimeError(
                        f"class {target_class} has no negative classes"
                    )
                if max(negative_classes) >= logits.shape[1]:
                    raise RuntimeError(
                        "binary one-vs-rest training requires a classifier head "
                        f"covering negative class {max(negative_classes)}; "
                        f"width={logits.shape[1]}"
                    )
                negative_logits = logits[:, negative_classes]
                binary_logits = logits[:, target_class] - torch.logsumexp(
                    negative_logits, dim=1
                )
                binary_targets = y.eq(target_class).to(dtype=binary_logits.dtype)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    binary_logits,
                    binary_targets,
                )

            loss.backward()
            strategy.optimizer.step()

            # This custom loop bypasses Avalanche's normal training
            # iteration events, so BaseStrategy cannot advance the clock.
            # JSONLogger uses this clock to distinguish evaluation
            # checkpoints. Without it, later evaluations overwrite earlier
            # records because they receive the same mb_index.
            strategy.clock.train_iterations += 1

    if not validation_indices:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    validation_inputs = []
    validation_targets = []
    for index in validation_indices:
        sample = full_dataset[index]
        validation_inputs.append(sample[0].detach().cpu())
        validation_targets.append(int(sample[1]))

    return torch.stack(validation_inputs), torch.tensor(
        validation_targets, dtype=torch.long
    )


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update one binary skill against the complete observed class domain."""
    if not positive_classes:
        raise ValueError("positive_classes must not be empty")
    if not positive_classes <= observed_classes:
        raise ValueError("positive_classes must be a subset of observed_classes")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if historical_samples_per_class is not None and historical_samples_per_class <= 0:
        raise ValueError("historical_samples_per_class must be positive")
    if epochs < 1:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    full_dataset = experience.dataset
    labels = [int(full_dataset[index][1]) for index in range(len(full_dataset))]
    generator = torch.Generator().manual_seed(int(validation_seed) + 7919)
    validation_indices: list[int] = []
    training_indices: list[int] = []
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)

    for indices in by_class.values():
        shuffled = torch.tensor(indices, dtype=torch.long)
        if len(indices) > 1:
            shuffled = shuffled[torch.randperm(len(indices), generator=generator)]
        n_validation = int(len(indices) * validation_fraction)
        if validation_fraction > 0.0 and n_validation == 0 and len(indices) > 1:
            n_validation = 1
        n_validation = min(n_validation, max(0, len(indices) - 1))
        validation_indices.extend(shuffled[:n_validation].tolist())
        training_indices.extend(
            shuffled[
                n_validation : (
                    n_validation + samples_per_class
                    if samples_per_class is not None
                    else None
                )
            ].tolist()
        )

    inputs = []
    targets = []
    current_classes = set(by_class)
    replay_limit = historical_samples_per_class

    class_pools: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for class_id in sorted(current_classes):
        class_indices = [
            index for index in training_indices if labels[index] == class_id
        ]
        class_inputs = []
        class_targets = []
        for index in class_indices:
            sample = full_dataset[index]
            class_inputs.append(torch.as_tensor(sample[0]).detach().cpu())
            class_targets.append(int(sample[1]))
        if class_inputs:
            class_pools[class_id] = (
                torch.stack(class_inputs),
                torch.tensor(class_targets, dtype=torch.long),
            )

    retained_by_class = {int(item.class_id): item for item in (retained_memory or [])}
    for class_id in sorted(observed_classes - current_classes):
        item = retained_by_class.get(class_id)
        if item is None:
            continue
        class_pools[class_id] = (
            item.inputs.detach().cpu(),
            item.targets.detach().cpu(),
        )

    for class_id in sorted(class_pools):
        class_inputs, class_targets = class_pools[class_id]
        if class_id not in current_classes:
            class_inputs, class_targets = _select_historical_samples(
                class_inputs,
                class_targets,
                limit=replay_limit,
                seed=int(validation_seed) + 1543,
                class_id=class_id,
            )
        if training_counts is not None:
            training_counts[class_id] = len(class_inputs)
        inputs.extend(list(class_inputs))
        targets.extend(int(value) for value in class_targets.tolist())

    if not inputs:
        raise RuntimeError("skill-domain update has no training samples")

    dataset = TensorDataset(
        torch.stack(inputs),
        torch.tensor(targets, dtype=torch.long),
    )
    owned_classes = sorted(positive_classes)
    positive_counts = torch.tensor(
        [sum(label == class_id for label in targets) for class_id in owned_classes],
        dtype=torch.float32,
    )
    negative_counts = len(targets) - positive_counts
    if torch.any(positive_counts <= 0) or torch.any(negative_counts <= 0):
        raise RuntimeError(
            "skill-domain update requires positive and negative samples "
            "for every owned class"
        )

    device = next(strategy.model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )
    # Balance positive and negative evidence through sampling rather than
    # changing the BCE objective with pos_weight.
    negative_classes = sorted(set(targets) - set(owned_classes))
    if not negative_classes:
        raise RuntimeError("skill-domain update requires negative samples")
    # Balance the binary objective independently of the replay policy.
    # small_replay controls how much historical data enters the dataset; it
    # must not change the positive-vs-negative objective itself.
    negative_total = len(targets) - int(positive_counts.sum().item())
    if negative_total <= 0:
        raise RuntimeError("skill-domain update requires negative samples")
    sample_weights = [
        (
            0.5 / int(positive_counts[owned_classes.index(label)].item())
            if label in owned_classes
            else 0.5 / negative_total
        )
        for label in targets
    ]
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        sampler=WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        ),
    )

    strategy.model.train()
    for _ in range(epochs):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            strategy.optimizer.zero_grad()
            logits = strategy.model(x)
            if max(observed_classes) >= logits.shape[1]:
                raise RuntimeError(
                    "skill-domain update requires a classifier head covering "
                    f"observed class {max(observed_classes)}; width={logits.shape[1]}"
                )
            binary_logits = logits[:, owned_classes]
            binary_targets = torch.stack(
                [y.eq(class_id) for class_id in owned_classes],
                dim=1,
            ).to(dtype=binary_logits.dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logits,
                binary_targets,
            )
            loss.backward()
            strategy.optimizer.step()
            strategy.clock.train_iterations += 1

    validation_inputs = []
    validation_targets = []
    for index in validation_indices:
        sample = full_dataset[index]
        validation_inputs.append(torch.as_tensor(sample[0]).detach().cpu())
        validation_targets.append(int(sample[1]))
    if not validation_inputs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.stack(validation_inputs), torch.tensor(
        validation_targets, dtype=torch.long
    )
