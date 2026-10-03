# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""End-to-end tests for complete replay update policies."""

import pytest
import torch
from torch.utils.data import TensorDataset

from skill_memory.diagnostics import replay_provenance_report
from skill_memory.evaluation.memory import EvaluationMemory
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all

PER_CLASS = 10
K = 3


def _run(mode, replay_budget=None, **kwargs):
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(
        6,
        update_mode=mode,
        replay_samples_per_class=replay_budget,
        **kwargs,
    )
    train_all(strategy, benchmark)
    return strategy, replay_provenance_report(strategy, diagnose=True)


def _class_calls(report):
    return [call for call in report["calls"] if call["kind"] == "class"]


def test_new_class_uses_zero_historical_examples():
    _, report = _run("new_class")
    assert report["violations"] == []
    assert all(call["historical_total"] == 0 for call in report["calls"])
    assert report["refresh"]["calls"] == 0


def test_replay_is_capped_when_a_budget_is_set():
    _, report = _run("replay", K)
    assert report["violations"] == []
    later = [call for call in _class_calls(report) if call["experience_index"] > 0]
    assert later
    for call in later:
        assert call["retained"]
        assert all(0 < n <= K for n in call["retained"].values())


def test_replay_uses_all_currently_retained_examples_without_a_cap():
    _, report = _run("replay")
    assert report["violations"] == []
    later = [call for call in _class_calls(report) if call["experience_index"] > 0]
    for call in later:
        assert call["retained"]
        assert all(n == PER_CLASS for n in call["retained"].values())


def test_history_grows_from_new_class_to_bounded_to_full_replay():
    totals = {
        "new_class": _run("new_class")[1]["class_training"]["historical_examples"],
        "replay(K)": _run("replay", K)[1]["class_training"]["historical_examples"],
        "replay(all)": _run("replay")[1]["class_training"]["historical_examples"],
    }
    assert totals["new_class"] == 0
    assert 0 < totals["replay(K)"] < totals["replay(all)"]


def test_refresh_is_explicit_and_adds_refresh_work():
    _, plain = _run("replay", K)
    _, refreshed = _run("refresh", K)
    assert plain["violations"] == []
    assert refreshed["violations"] == []
    assert plain["refresh"]["calls"] == 0
    assert refreshed["refresh"]["calls"] > 0
    assert plain["class_training"] == refreshed["class_training"]
    assert refreshed["refresh"]["optimizer_steps"] > 0
    assert refreshed["refresh_enabled"] is True


def test_refresh_respects_the_replay_cap():
    _, report = _run("refresh", K)
    for call in report["calls"]:
        if call["kind"] == "refresh":
            assert all(n <= K for n in call["retained"].values())


def test_refresh_requires_binary_training():
    with pytest.raises(ValueError, match="binary_one_vs_rest"):
        make_strategy(
            6,
            update_mode="refresh",
            class_train_mode="multiclass",
        )


def _pool():
    return [
        EvaluationMemory(
            inputs=torch.randn(8, 6), targets=torch.full((8,), 5), class_id=5
        )
    ]


def test_offline_pool_cannot_leak_into_new_class():
    with pytest.raises(ValueError, match="new_class"):
        make_strategy(
            6,
            update_mode="new_class",
            binary_negative_pool=_pool(),
            allow_offline_negative_pool=True,
        )


def test_offline_pool_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_offline_negative_pool"):
        make_strategy(6, update_mode="replay", binary_negative_pool=_pool())


def test_offline_pool_is_governed_by_the_replay_cap_and_audited():
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(
        6,
        update_mode="replay",
        replay_samples_per_class=K,
        binary_negative_pool=_pool(),
        allow_offline_negative_pool=True,
    )
    train_all(strategy, benchmark)
    report = replay_provenance_report(strategy, diagnose=True)
    pooled = [call for call in report["calls"] if call["offline_pool"]]
    assert pooled
    assert all(
        n <= K for call in pooled for n in call["offline_pool"].values()
    )
    assert report["violations"] == []


def test_audit_catches_a_tampered_log():
    strategy, _ = _run("new_class")
    strategy.skill_memory_plugin.training_log.append(
        {
            "kind": "class",
            "target_classes": [0],
            "experience_index": 9,
            "retained": {1: 4},
            "offline_pool": {},
            "historical_total": 4,
            "current": {0: 1},
            "optimizer_steps": 1,
        }
    )
    report = replay_provenance_report(strategy, diagnose=True)
    assert any("new_class used 4" in violation for violation in report["violations"])


def test_audit_requires_diagnose_true():
    strategy, _ = _run("new_class")
    with pytest.raises(RuntimeError, match="diagnose=True"):
        replay_provenance_report(strategy, diagnose=False)


def test_calibration_holdout_is_disjoint_from_training_data():
    benchmark = make_benchmark(n_classes=2, n_experiences=1, n_per_class=30)
    strategy = make_strategy(
        2,
        update_mode="replay",
        validation_fraction=0.2,
        train_samples_per_class=100,
    )
    train_all(strategy, benchmark)
    memory = strategy.skill_memory
    from skill_memory.cl.skill_registry import CALIBRATION_EXAMPLES_KEY

    dataset: TensorDataset = benchmark.train_stream[0].dataset
    train_rows = {
        tuple(dataset[index][0].tolist()) for index in range(len(dataset))
    }
    for skill in memory.slots():
        held_out = memory.metadata(skill)[CALIBRATION_EXAMPLES_KEY]
        assert held_out
        for inputs, _ in held_out.values():
            assert len(inputs) == 6
            assert {tuple(row.tolist()) for row in inputs} <= train_rows
    call = strategy.skill_memory_plugin.training_log[0]
    assert call["current"][0] == 24
