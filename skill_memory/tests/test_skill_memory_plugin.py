# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Regression tests for Skill Memory's dynamic-head and optimizer behavior."""

from types import SimpleNamespace

import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training import Naive
from torch import nn
from torch.utils.data import TensorDataset

from skill_memory.cl.skill_memory_plugin import SkillMemoryPlugin


class TinyIncrementalModel(nn.Module):
    """Small model whose classifier grows with Avalanche class labels."""

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 8)
        self.classifier = IncrementalClassifier(
            8,
            initial_out_features=1,
        )

    def forward(self, x):
        return self.classifier(torch.relu(self.features(x)))


def _two_experience_benchmark():
    torch.manual_seed(0)
    x0 = torch.randn(20, 4)
    x7 = torch.randn(20, 4)
    y0 = torch.zeros(20, dtype=torch.long)
    y7 = torch.full((20,), 7, dtype=torch.long)
    train = TensorDataset(torch.cat([x0, x7]), torch.cat([y0, y7]))
    test = TensorDataset(torch.cat([x0, x7]), torch.cat([y0, y7]))
    return nc_benchmark(
        train,
        test,
        n_experiences=2,
        task_labels=False,
        seed=0,
        shuffle=False,
    )


def test_new_class_reuse_grows_incremental_head_before_training():
    benchmark = _two_experience_benchmark()
    model = TinyIncrementalModel()
    plugin = SkillMemoryPlugin(
        max_skills=4,
        force_decision="reuse",
        class_train_epochs=1,
        class_train_batch_size=8,
        probe_batch_size=8,
        probe_batches=1,
        verbose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=8,
        train_epochs=1,
        eval_mb_size=8,
        plugins=[plugin],
    )

    strategy.train(benchmark.train_stream[0])
    strategy.train(benchmark.train_stream[1])

    decision = plugin.last_class_decisions[1][7]
    assert decision["decision"] == plugin.REUSE
    assert decision["known_class"] is False
    assert decision["skill"] == 0
    assert model.classifier.classifier.out_features >= 8


def test_optimizer_reset_preserves_multiple_parameter_groups():
    model = TinyIncrementalModel()
    optimizer = torch.optim.SGD(
        [
            {"params": model.features.parameters(), "lr": 0.1},
            {"params": model.classifier.parameters(), "lr": 0.01},
        ]
    )
    strategy = SimpleNamespace(model=model, optimizer=optimizer)
    plugin = SkillMemoryPlugin(verbose=False)

    group_by_name = plugin._capture_optimizer_groups(strategy)

    model.classifier = IncrementalClassifier(
        8,
        initial_out_features=8,
    )
    plugin._reset_optimizer(strategy, group_by_name)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.01
    assert set(optimizer.param_groups[0]["params"]) == set(model.features.parameters())
    assert set(optimizer.param_groups[1]["params"]) == set(
        model.classifier.parameters()
    )


def test_diagnose_false_does_not_call_perf_counter(monkeypatch):
    benchmark = _two_experience_benchmark()
    model = TinyIncrementalModel()
    plugin = SkillMemoryPlugin(
        max_skills=4,
        force_decision="scratch",
        class_train_epochs=1,
        class_train_batch_size=8,
        probe_batch_size=8,
        probe_batches=1,
        verbose=False,
        diagnose=False,
    )
    strategy = Naive(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=8,
        train_epochs=1,
        eval_mb_size=8,
        plugins=[plugin],
    )

    def fail_perf_counter():
        raise AssertionError("production timing called time.perf_counter()")

    monkeypatch.setattr(
        "skill_memory.cl.skill_memory_plugin.time.perf_counter",
        fail_perf_counter,
    )
    strategy.train(benchmark.train_stream[0])


def test_cl_update_modes_are_covered_end_to_end():
    """The old fake-object test was replaced by real runs.

    See ``test_replay_isolation.py``: provenance of every training call is
    audited for each of ``new_class`` / ``small_replay`` / ``replay``.
    """
    from skill_memory.cl.replay import REPLAY_MODES

    assert set(REPLAY_MODES) == {"new_class", "small_replay", "replay"}
