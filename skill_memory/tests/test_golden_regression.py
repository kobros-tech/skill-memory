# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The refactored package must reproduce the validated v3 engine.

``data/golden_v3.json`` was produced by the reference implementation that
generated the published demo results, on the offline synthetic benchmark below
(seed 3, dropout MLP, 3 experiences x 2 classes).  It pins, per update policy:

* **exact** quantities, independent of float noise -- the provenance of every
  training call (how many examples came from where, how many optimiser steps),
  the REUSE/SCRATCH decisions and the class -> skill assignment;
* **approximate** quantities -- stored-skill checksums and accuracies, compared
  with a tolerance that absorbs PyTorch-version differences.

If one of these tests fails after a change, the change altered *what is
computed*, not just how the code is organised.
"""

import json
from pathlib import Path

import pytest
import torch
from avalanche.models import SimpleMLP

from skill_memory import SkillMemoryStrategy
from skill_memory.tests._helpers import make_benchmark

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_v3.json").read_text())

#: golden name -> (update_mode, replay_samples_per_class)
POLICIES = {
    "new_class": ("new_class", None),
    "replay": ("replay", None),
    "replay_k3": ("replay", 3),
    "refresh": ("refresh", None),
    "refresh_k3": ("refresh", 3),
}


def _run(name):
    mode, cap = POLICIES[name]
    benchmark = make_benchmark(n_classes=6, n_experiences=3, n_per_class=40)
    torch.manual_seed(3)
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=6)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        update_mode=mode,
        replay_samples_per_class=cap,
        memory_per_class=10,
        train_samples_per_class=10,
        max_skills=10,
        train_mb_size=64,  # the reference trained with batch size 64
        class_train_epochs=2,
        eval_mb_size=16,
        seed=3,
        verbose=False,
    )
    for experience in benchmark.train_stream:
        strategy.train(experience)
    return strategy, strategy.eval(benchmark.test_stream)


@pytest.fixture(scope="module", params=sorted(POLICIES))
def run(request):
    strategy, results = _run(request.param)
    return GOLDEN[request.param], strategy, results


def test_provenance_of_every_training_call_is_unchanged(run):
    golden, strategy, _ = run
    keys = ("kind", "target_classes", "current", "retained", "historical_total")
    actual = [
        {k: v for k, v in entry.items() if k in keys or k in ("optimizer_steps",)}
        for entry in strategy.skill_memory_plugin.training_log
    ]
    expected = [
        {
            k: ({int(c): n for c, n in v.items()} if isinstance(v, dict) else v)
            for k, v in entry.items()
            if k in keys or k == "optimizer_steps"
        }
        for entry in golden["provenance"]
    ]
    assert actual == expected


def test_decisions_and_class_to_skill_assignment_are_unchanged(run):
    golden, strategy, _ = run
    plugin = strategy.skill_memory_plugin
    decisions = {
        str(e): {str(c): [d["decision"], int(d["skill"])] for c, d in cs.items()}
        for e, cs in plugin.last_class_decisions.items()
    }
    assert decisions == golden["decisions"]
    skills = {
        str(s): sorted(plugin.class_map.classes_for_skill(s))
        for s in sorted(strategy.skill_memory.slots())
    }
    assert skills == golden["skills"]


def test_stored_skill_weights_are_numerically_unchanged(run):
    golden, strategy, _ = run
    memory = strategy.skill_memory
    for skill, expected in golden["state_checksum"].items():
        state = memory.state(int(skill))
        checksum = float(sum(v.double().abs().sum() for v in state.values()))
        assert checksum == pytest.approx(expected, rel=1e-3)


def test_evaluation_results_are_unchanged(run):
    golden, _, results = run
    assert results["raw_mean_final_accuracy"] == pytest.approx(
        golden["raw_mean_final_accuracy"], abs=0.03
    )
    assert results["mean_final_accuracy"] == pytest.approx(
        golden["mean_final_accuracy"], abs=0.03
    )
    for class_id, accuracy in golden["final_class_accuracy"].items():
        assert results["final_class_accuracy"][int(class_id)] == pytest.approx(
            accuracy, abs=0.10
        )
