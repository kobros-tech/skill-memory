# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Shared, download-free fixtures for the replay / evaluator test modules."""

from __future__ import annotations

import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from torch.utils.data import TensorDataset

from skill_memory import SkillMemoryStrategy

N_FEATURES = 6


def class_data(n_classes: int, n_per_class: int, seed: int):
    """Well separated classes: class ``c`` shifts its own feature dimension."""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for class_id in range(n_classes):
        offsets = torch.zeros(N_FEATURES)
        offsets[class_id % N_FEATURES] = 6.0
        noise = torch.randn(n_per_class, N_FEATURES, generator=generator) * 0.5
        xs.append(noise + offsets)
        ys.append(torch.full((n_per_class,), class_id, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def make_benchmark(n_classes=6, n_experiences=3, n_per_class=30):
    x_train, y_train = class_data(n_classes, n_per_class, seed=1)
    x_test, y_test = class_data(n_classes, n_per_class, seed=2)
    return nc_benchmark(
        TensorDataset(x_train, y_train),
        TensorDataset(x_test, y_test),
        n_experiences=n_experiences,
        task_labels=False,
        seed=0,
        shuffle=False,
        fixed_class_order=list(range(n_classes)),
    )


def make_strategy(n_classes=6, *, model_seed=0, **kwargs) -> SkillMemoryStrategy:
    """Binary one-vs-rest strategy on a tiny MLP (override via ``kwargs``)."""
    torch.manual_seed(model_seed)
    model = SimpleMLP(input_size=N_FEATURES, hidden_size=8, num_classes=n_classes)
    options = {
        "class_train_mode": "binary_one_vs_rest",
        "max_skills": 10,
        "eval_memory_per_class": 10,
        "train_mb_size": 16,
        "train_epochs": 1,
        "eval_mb_size": 16,
        "probe_seed": 0,
        "verbose": False,
    }
    options.update(kwargs)
    return SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        **options,
    )


def train_all(strategy, benchmark):
    for experience in benchmark.train_stream:
        strategy.train(experience)
    return strategy
