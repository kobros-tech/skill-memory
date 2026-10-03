# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Numerical checks of the formulas in docs/MATHEMATICS.md (sections 3, 5, 6)."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from skill_memory.cl.training import (
    balanced_weights,
    split_holdout,
    train_on_class,
    train_skill_on_domain,
)
from skill_memory.evaluation.memory import EvaluationMemory


def test_balanced_weights_give_positive_probability_one_half():
    labels = [0] * 3 + [1] * 5 + [2] * 40
    weights = balanced_weights(labels, [1])
    probability = weights / weights.sum()
    assert probability[3:8].sum().item() == pytest.approx(0.5)  # class 1
    negatives = probability[:3].sum() + probability[8:].sum()
    assert negatives.item() == pytest.approx(0.5)
    assert probability.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("owned", [[1], [1, 2], [0, 1, 2]])
def test_balanced_weights_positive_mass_is_o_over_o_plus_one(owned):
    labels = [0] * 4 + [1] * 6 + [2] * 9 + [3] * 50
    weights = balanced_weights(labels, owned)
    probability = weights / weights.sum()
    positive = sum(probability[i].item() for i, y in enumerate(labels) if y in owned)
    assert positive == pytest.approx(len(owned) / (len(owned) + 1))
    # each owned class gets an equal share
    shares = [
        sum(probability[i].item() for i, y in enumerate(labels) if y == c)
        for c in owned
    ]
    assert shares == pytest.approx([shares[0]] * len(owned))


def test_balanced_weights_reject_empty_sides():
    with pytest.raises(RuntimeError):
        balanced_weights([1, 1, 1], [1])  # no negatives
    with pytest.raises(RuntimeError):
        balanced_weights([0, 0], [1])  # no positives


@pytest.mark.parametrize("n", [1, 2, 4, 5, 10, 33])
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.2, 0.5])
def test_holdout_size_matches_the_documented_formula(n, fraction):
    labels = [7] * n
    train, validation, _ = split_holdout(
        labels, validation_fraction=fraction, samples_per_class=None, seed=0
    )
    expected = min(
        max(math.floor(fraction * n), 1 if fraction > 0 and n > 1 else 0), n - 1
    )
    assert len(validation) == expected
    assert len(train) == n - expected
    assert not set(train) & set(validation)


def test_samples_per_class_caps_training_after_the_holdout():
    labels = [0] * 50 + [1] * 50
    train, validation, _ = split_holdout(
        labels, validation_fraction=0.2, samples_per_class=7, seed=0
    )
    assert len(validation) == 20
    assert len(train) == 14  # 7 per class, drawn after the 10-per-class hold-out


class _Experience:
    def __init__(self, n_per_class=12, classes=(0, 1)):
        xs = torch.arange(len(classes) * n_per_class, dtype=torch.float32).reshape(
            -1, 1
        )
        ys = torch.tensor([c for c in classes for _ in range(n_per_class)])
        self.dataset = TensorDataset(xs, ys)
        self.dataset.targets = ys.tolist()


def _strategy(width=4):
    model = torch.nn.Linear(1, width)
    return SimpleNamespace(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        clock=SimpleNamespace(train_iterations=0),
    )


@pytest.mark.parametrize(("epochs", "batch"), [(1, 4), (3, 5), (2, 100)])
def test_step_count_is_epochs_times_ceil_n_over_b(epochs, batch):
    experience = _Experience(12, (0, 1))
    memory = [
        EvaluationMemory(torch.zeros(6, 1), torch.full((6,), 2), 2),
        EvaluationMemory(torch.zeros(6, 1), torch.full((6,), 3), 3),
    ]
    result = train_on_class(
        _strategy(),
        experience,
        1,
        epochs,
        batch,
        mode="binary_one_vs_rest",
        validation_fraction=0.25,
        retained_memory=memory,
        historical_samples_per_class=4,
    )
    p = result.provenance
    assert p.current == {0: 9, 1: 9}  # 12 - 3 held out each
    assert p.retained == {2: 4, 3: 4}
    n = p.dataset_size
    assert n == 18 + 8
    assert p.optimizer_steps == epochs * math.ceil(n / batch)
    assert len(result.validation_targets) == 6


def test_refresh_step_count_and_provenance():
    experience = _Experience(10, (2, 3))
    memory = [
        EvaluationMemory(torch.zeros(8, 1), torch.full((8,), 0), 0),
        EvaluationMemory(torch.zeros(8, 1), torch.full((8,), 1), 1),
    ]
    result = train_skill_on_domain(
        _strategy(),
        experience,
        {0},
        {0, 1, 2, 3},
        2,
        6,
        validation_fraction=0.0,
        retained_memory=memory,
        historical_samples_per_class=5,
    )
    p = result.provenance
    assert p.kind == "refresh"
    assert p.retained == {0: 5, 1: 5}
    assert p.current == {2: 10, 3: 10}
    assert p.optimizer_steps == 2 * math.ceil(p.dataset_size / 6)
    assert len(result.validation_targets) == 0


def test_refresh_without_positives_of_an_owned_class_fails_loudly():
    experience = _Experience(10, (2, 3))
    with pytest.raises(RuntimeError, match="positive and negative"):
        train_skill_on_domain(
            _strategy(),
            experience,
            {0},
            {0, 2, 3},
            1,
            4,
            validation_fraction=0.0,
            retained_memory=None,  # class 0 has no history -> no positives
        )


def test_new_class_hands_no_history_to_the_trainer():
    experience = _Experience(10, (0, 1))
    result = train_on_class(
        _strategy(),
        experience,
        1,
        1,
        4,
        mode="binary_one_vs_rest",
        validation_fraction=0.0,
        retained_memory=None,
    )
    assert result.provenance.historical_total == 0
