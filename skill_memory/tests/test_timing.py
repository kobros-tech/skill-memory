# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from torch.utils.data import TensorDataset

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import reset_timing, timing_report


def _tiny_strategy():
    torch.manual_seed(0)
    x = torch.randn(80, 6)
    y = torch.randint(0, 4, (80,))
    benchmark = nc_benchmark(
        TensorDataset(x, y),
        TensorDataset(x, y),
        n_experiences=2,
        task_labels=False,
        seed=0,
        shuffle=False,
    )
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=4)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        eval_memory_per_class=5,
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        verbose=False,
        diagnose=True,
    )
    return strategy, benchmark


def test_timing_report_has_the_expected_buckets_after_one_train_eval_cycle():
    strategy, benchmark = _tiny_strategy()

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

    report = timing_report(strategy)

    assert set(report) == {
        "skill_memory_decision_probing",
        "skill_memory_class_training",
        "cl_evaluation",
    }
    for metrics in report.values():
        assert metrics["calls"] >= 1
        assert metrics["total_seconds"] >= 0.0
        assert metrics["mean_seconds"] == metrics["total_seconds"] / metrics["calls"]

    # One decision + one training pass per class introduced in the stream.
    assert report["skill_memory_decision_probing"]["calls"] == 4
    assert report["skill_memory_class_training"]["calls"] == 4
    # One strategy.eval() call per experience trained so far.
    assert report["cl_evaluation"]["calls"] == 2


def test_refresh_bucket_appears_only_when_refresh_is_enabled():
    from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all

    benchmark = make_benchmark(4, 2, 24)
    plain = train_all(make_strategy(4, diagnose=True), benchmark)
    assert "skill_memory_domain_refresh" not in timing_report(plain)

    refreshed = train_all(
        make_strategy(4, diagnose=True, refresh_existing_skills=True), benchmark
    )
    report = timing_report(refreshed)
    assert report["skill_memory_domain_refresh"]["calls"] >= 1


def test_reset_timing_clears_every_bucket():
    strategy, benchmark = _tiny_strategy()

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

    assert timing_report(strategy)

    reset_timing(strategy)

    assert timing_report(strategy) == {}
