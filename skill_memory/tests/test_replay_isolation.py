# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""End-to-end: each ``update_mode`` consumes exactly the history it promises.

* ``new_class`` -- zero historical examples, existing skills frozen;
* ``replay``    -- all currently retained examples (or at most K per old class);
* ``refresh``   -- the same history, plus one retraining of every old skill.
"""

import pytest

from skill_memory.cl.skill_registry import CALIBRATION_EXAMPLES_KEY
from skill_memory.diagnostics import replay_provenance_report
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all

MEMORY = 10  # memory_per_class used by make_strategy
K = 3


def _run(mode, cap=None, **kwargs):
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(
        6, update_mode=mode, replay_samples_per_class=cap, **kwargs
    )
    train_all(strategy, benchmark)
    return strategy, replay_provenance_report(strategy, diagnose=True)


def _class_calls(report, *, later_only=False):
    return [
        c
        for c in report["calls"]
        if c["kind"] == "class" and (not later_only or c["experience_index"] > 0)
    ]


def test_new_class_uses_zero_historical_examples():
    _, report = _run("new_class")
    assert report["violations"] == []
    assert all(call["historical_total"] == 0 for call in report["calls"])
    assert report["refresh"]["calls"] == 0


def test_replay_with_a_cap_replays_at_most_k_per_old_class():
    _, report = _run("replay", K)
    assert report["violations"] == []
    later = _class_calls(report, later_only=True)
    assert later
    for call in later:
        assert all(0 < n <= K for n in call["retained"].values())
    assert all(
        c["historical_total"] == 0
        for c in _class_calls(report)
        if c["experience_index"] == 0
    )


def test_replay_without_a_cap_uses_all_retained_examples():
    _, report = _run("replay")
    assert report["violations"] == []
    for call in _class_calls(report, later_only=True):
        # "all retained", i.e. memory_per_class -- not "all historical data".
        assert call["retained"]
        assert all(n == MEMORY for n in call["retained"].values())


def test_history_grows_from_new_class_to_capped_to_full_replay():
    used = {
        label: _run(mode, cap)[1]["class_training"]["historical_examples"]
        for label, (mode, cap) in {
            "new_class": ("new_class", None),
            "capped": ("replay", K),
            "full": ("replay", None),
        }.items()
    }
    assert used["new_class"] == 0
    assert 0 < used["capped"] < used["full"]


def test_only_refresh_mode_retrains_existing_skills():
    _, replay = _run("replay", K)
    _, refresh = _run("refresh", K)
    assert replay["refresh"]["calls"] == 0
    assert refresh["refresh"]["calls"] > 0
    assert refresh["refresh"]["optimizer_steps"] > 0
    assert refresh["violations"] == []
    # The new-class training work is identical; refresh only adds work on top.
    assert replay["class_training"] == refresh["class_training"]


def test_refresh_respects_the_replay_cap():
    _, report = _run("refresh", K)
    refreshes = [c for c in report["calls"] if c["kind"] == "refresh"]
    assert refreshes
    assert all(n <= K for c in refreshes for n in c["retained"].values())


def test_audit_catches_a_tampered_log():
    strategy, _ = _run("new_class")
    strategy.skill_memory_plugin.training_log.append(
        {
            "kind": "class",
            "target_classes": [0],
            "experience_index": 9,
            "retained": {1: 4},
            "historical_total": 4,
            "current": {0: 1},
            "optimizer_steps": 1,
        }
    )
    report = replay_provenance_report(strategy, diagnose=True)
    assert any("new_class used 4" in v for v in report["violations"])


def test_audit_catches_a_refresh_outside_refresh_mode():
    strategy, _ = _run("replay")
    strategy.skill_memory_plugin.training_log.append(
        {
            "kind": "refresh",
            "target_classes": [0],
            "experience_index": 9,
            "retained": {},
            "historical_total": 0,
            "current": {},
            "optimizer_steps": 1,
        }
    )
    report = replay_provenance_report(strategy, diagnose=True)
    assert any("outside update_mode" in v for v in report["violations"])


def test_audit_requires_diagnose_true():
    strategy, _ = _run("new_class")
    with pytest.raises(RuntimeError, match="diagnose=True"):
        replay_provenance_report(strategy, diagnose=False)


def test_calibration_holdout_is_never_trained_on():
    """The hold-out only calibrates the evaluator: it is excluded from training."""
    benchmark = make_benchmark(n_classes=2, n_experiences=1, n_per_class=30)
    strategy = make_strategy(2, validation_fraction=0.2, train_samples_per_class=100)
    train_all(strategy, benchmark)
    memory = strategy.skill_memory_plugin.memory
    for skill in memory.slots():
        held_out = memory.metadata(skill)[CALIBRATION_EXAMPLES_KEY]
        assert held_out
        assert all(len(inputs) == 6 for inputs, _ in held_out.values())  # 20% of 30
    first_call = strategy.skill_memory_plugin.training_log[0]
    assert first_call["current"][0] == 24  # 30 - 6 held out
