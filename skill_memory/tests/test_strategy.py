# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Tests for `skill_memory.strategy.SkillMemoryStrategy`.

The central property: a plain ``strategy.train(experience)`` /
``strategy.eval(test_stream)`` loop -- the pattern used for ``Naive``,
``Replay``, ``ER-ACE`` and every other Avalanche strategy -- must work with
no extra method calls, and Avalanche's own accuracy/loss metrics must reflect
the stored-skill CL evaluator's predictions.
"""

import pytest
import torch
from avalanche.models import SimpleMLP

from skill_memory.diagnostics import evaluate_skill_memory
from skill_memory.strategy import SkillMemoryStrategy
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all


def test_standard_train_eval_loop_requires_no_extra_method_calls():
    benchmark = make_benchmark(n_classes=4, n_experiences=2, n_per_class=30)
    strategy = make_strategy(4)

    for experience in benchmark.train_stream:
        strategy.train(experience)
        results = strategy.eval(benchmark.test_stream)

    accuracy_keys = [k for k in results if k.startswith("Top1_Acc_Stream")]
    loss_keys = [k for k in results if k.startswith("Loss_Stream")]
    assert accuracy_keys, f"no stream accuracy metric in {sorted(results)}"
    assert loss_keys, f"no stream loss metric in {sorted(results)}"
    for key in accuracy_keys:
        assert 0.0 <= results[key] <= 1.0
    for key in ("mean_final_accuracy", "final_class_accuracy", "final_class_loss"):
        assert key in results


def test_skill_memory_diagnostic_is_separate_from_strategy_eval():
    """The oracle diagnostic reads the stored skills; strategy.eval() does not."""
    benchmark = make_benchmark(n_classes=4, n_experiences=2, n_per_class=40)
    strategy = train_all(make_strategy(4), benchmark)

    results = evaluate_skill_memory(
        strategy.model,
        strategy.skill_memory_plugin,
        benchmark.test_stream,
        1,
        num_classes=4,
        routing="oracle",
        batch_size=16,
        device=strategy.device,
        diagnose=True,
    )
    assert set(results) == {0, 1, 2, 3}
    for metrics in results.values():
        assert "accuracy" in metrics
        assert "loss" in metrics


def test_training_epochs_are_forwarded():
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=2)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_epochs=3,
    )
    assert strategy.train_epochs == 3
    assert strategy.skill_memory_plugin.class_train_epochs == 3


def test_public_properties_expose_underlying_components():
    strategy = make_strategy(2)
    assert strategy.memory is strategy.skill_memory_plugin.memory
    assert strategy.skill_memory is strategy.memory
    assert strategy.evaluation_plugin is strategy.cl_evaluation_plugin
    assert strategy.cl_evaluation_plugin.evaluator_model is None


def test_replay_configuration_is_exposed_and_forwarded():
    strategy = make_strategy(
        2, cl_update_mode="small_replay", cl_replay_per_class=7, training_seed=3
    )
    plugin = strategy.skill_memory_plugin
    assert strategy.cl_update_mode == plugin.cl_update_mode == "small_replay"
    assert strategy.cl_replay_per_class == plugin.cl_replay_per_class == 7
    assert plugin.replay_policy.historical_limit == 7
    assert plugin.refresh_policy.enabled is False
    assert plugin.training_seed == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cl_update_mode": "everything"},
        {"cl_update_mode": "small_replay", "cl_replay_per_class": 0},
        {"cl_update_mode": "new_class", "refresh_existing_skills": True},
        {"class_train_mode": "multiclass", "refresh_existing_skills": True},
    ],
)
def test_invalid_replay_configuration_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        make_strategy(2, **kwargs)


def test_strategy_rejects_removed_ml_evaluator_arguments():
    """The independent ML evaluator is gone; stale kwargs must not be accepted."""
    with pytest.raises(TypeError):
        make_strategy(2, evaluator_model_factory=lambda: None)
