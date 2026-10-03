# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""End-to-end: the three replay modes isolate replay *quantity* cleanly.

Review invariant (PR #13)::

    new_class     -> historical replay = 0
    small_replay  -> <= K retained examples per old class
    replay        -> all currently retained examples per old class

and ``refresh_existing_skills`` is an independent switch.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from skill_memory.diagnostics import replay_provenance_report
from skill_memory.evaluation.memory import EvaluationMemory
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all

PER_CLASS = 10  # eval_memory_per_class used by make_strategy
K = 3


def _run(mode, **kwargs):
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(6, cl_update_mode=mode, cl_replay_per_class=K, **kwargs)
    train_all(strategy, benchmark)
    return strategy, replay_provenance_report(strategy, diagnose=True)


def _class_calls(report):
    return [c for c in report["calls"] if c["kind"] == "class"]


def test_new_class_uses_zero_historical_examples():
    _, report = _run("new_class")
    assert report["violations"] == []
    assert all(call["historical_total"] == 0 for call in report["calls"])
    assert report["refresh"]["calls"] == 0


def test_small_replay_is_capped_at_k_per_old_class():
    _, report = _run("small_replay")
    assert report["violations"] == []
    later = [c for c in _class_calls(report) if c["experience_index"] > 0]
    assert later
    for call in later:
        assert call["retained"], "history must actually be replayed"
        assert all(0 < n <= K for n in call["retained"].values())
    first = [c for c in _class_calls(report) if c["experience_index"] == 0]
    assert all(c["historical_total"] == 0 for c in first)


def test_replay_uses_all_currently_retained_examples():
    _, report = _run("replay")
    assert report["violations"] == []
    later = [c for c in _class_calls(report) if c["experience_index"] > 0]
    for call in later:
        # Retained memory holds eval_memory_per_class (= PER_CLASS) frozen
        # examples per class -- "all retained", not "all historical data".
        assert call["retained"]
        assert all(n == PER_CLASS for n in call["retained"].values())


def test_history_grows_monotonically_with_replay_mode():
    totals = {
        m: _run(m)[1]["class_training"]["historical_examples"]
        for m in ("new_class", "small_replay", "replay")
    }
    assert totals["new_class"] == 0
    assert 0 < totals["small_replay"] < totals["replay"]


def test_refresh_is_off_by_default_for_every_mode():
    for mode in ("small_replay", "replay"):
        _, report = _run(mode)
        assert report["refresh"]["calls"] == 0
        assert report["refresh_existing_skills"] is False


@pytest.mark.parametrize("mode", ["small_replay", "replay"])
def test_refresh_is_an_independent_switch(mode):
    _, plain = _run(mode)
    _, refreshed = _run(mode, refresh_existing_skills=True)
    assert refreshed["violations"] == []
    assert refreshed["refresh"]["calls"] > 0
    # Class-training work is unchanged by the refresh switch...
    assert plain["class_training"] == refreshed["class_training"]
    # ...the refresh adds optimiser work on top of it.
    assert refreshed["refresh"]["optimizer_steps"] > 0


def test_refresh_respects_the_replay_cap():
    _, report = _run("small_replay", refresh_existing_skills=True)
    for call in report["calls"]:
        if call["kind"] == "refresh":
            assert all(n <= K for n in call["retained"].values())


def test_new_class_refresh_combination_is_rejected():
    with pytest.raises(ValueError, match="new_class"):
        make_strategy(6, cl_update_mode="new_class", refresh_existing_skills=True)


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
            cl_update_mode="new_class",
            binary_negative_pool=_pool(),
            allow_offline_negative_pool=True,
        )


def test_offline_pool_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_offline_negative_pool"):
        make_strategy(6, cl_update_mode="replay", binary_negative_pool=_pool())


def test_offline_pool_is_governed_by_the_replay_cap_and_audited():
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(
        6,
        cl_update_mode="small_replay",
        cl_replay_per_class=K,
        binary_negative_pool=_pool(),
        allow_offline_negative_pool=True,
    )
    train_all(strategy, benchmark)
    report = replay_provenance_report(strategy, diagnose=True)
    pooled = [c for c in report["calls"] if c["offline_pool"]]
    assert pooled, "the opted-in pool should be consumed"
    assert all(n <= K for c in pooled for n in c["offline_pool"].values())
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
    assert any("new_class used 4" in v for v in report["violations"])


def test_audit_requires_diagnose_true():
    strategy, _ = _run("new_class")
    with pytest.raises(RuntimeError, match="diagnose=True"):
        replay_provenance_report(strategy, diagnose=False)


def test_calibration_holdout_is_disjoint_from_training_data():
    """Calibration examples are excluded from training, never replayed."""
    benchmark = make_benchmark(n_classes=2, n_experiences=1, n_per_class=30)
    strategy = make_strategy(
        2,
        cl_update_mode="replay",
        validation_fraction=0.2,
        skill_train_samples_per_class=100,
    )
    train_all(strategy, benchmark)
    memory = strategy.skill_memory
    from skill_memory.cl.skill_registry import CALIBRATION_EXAMPLES_KEY

    dataset: TensorDataset = benchmark.train_stream[0].dataset
    train_rows = {tuple(dataset[i][0].tolist()) for i in range(len(dataset))}
    for skill in memory.slots():
        held_out = memory.metadata(skill)[CALIBRATION_EXAMPLES_KEY]
        assert held_out
        for inputs, _ in held_out.values():
            assert len(inputs) == 6  # 20% of 30 samples of that class
            assert {tuple(row.tolist()) for row in inputs} <= train_rows
    # Disjointness from what was *trained on* is asserted via provenance:
    call = strategy.skill_memory_plugin.training_log[0]
    assert call["current"][0] == 24  # 30 - 6 held out
