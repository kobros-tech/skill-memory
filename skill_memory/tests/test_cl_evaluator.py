# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""CLEvaluationPlugin: width handling, chunking, calibration, debug gating."""

import numpy as np
import pytest
import torch

from skill_memory.evaluation.cl_evaluator import (
    DEFAULT_EVAL_CHUNK_SIZE,
    CLEvaluationPlugin,
    _class_logit,
    _fit_platt_calibrator,
)
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all


def test_eval_on_future_classes_does_not_crash_and_scores_them_zero():
    """Regression: the score matrix must cover not-yet-trained target classes."""
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    strategy = make_strategy(6)
    strategy.train(benchmark.train_stream[0])  # classes {0, 1} only

    results = strategy.eval(benchmark.test_stream)  # includes classes 2..5

    accuracy = results["final_class_accuracy"]
    assert set(accuracy) == set(range(6))
    assert all(accuracy[c] == 0.0 for c in (2, 3, 4, 5))
    assert accuracy[0] > 0.5 and accuracy[1] > 0.5


def test_results_expose_raw_and_calibrated_accuracy():
    benchmark = make_benchmark(n_classes=4, n_experiences=2)
    strategy = train_all(make_strategy(4), benchmark)
    results = strategy.eval(benchmark.test_stream)
    assert 0.0 <= results["raw_mean_final_accuracy"] <= 1.0
    assert 0.0 <= results["mean_final_accuracy"] <= 1.0
    assert np.all(np.isfinite(results["diagonal_accuracy"]))


def test_eval_chunk_size_changes_speed_not_predictions():
    benchmark = make_benchmark(n_classes=6, n_experiences=3)
    outputs = []
    for chunk in (1, 2, DEFAULT_EVAL_CHUNK_SIZE, 64):
        strategy = train_all(make_strategy(6, eval_chunk_size=chunk), benchmark)
        results = strategy.eval(benchmark.test_stream)
        outputs.append(
            (results["final_class_accuracy"], results["raw_mean_final_accuracy"])
        )
    assert all(o == outputs[0] for o in outputs[1:])


def test_eval_chunk_size_is_validated_and_defaults_to_a_named_constant():
    assert DEFAULT_EVAL_CHUNK_SIZE == 8
    with pytest.raises(ValueError, match="eval_chunk_size"):
        CLEvaluationPlugin(memory_plugin=None, eval_chunk_size=0)
    strategy = make_strategy(2)
    assert strategy.cl_evaluation_plugin.eval_chunk_size == DEFAULT_EVAL_CHUNK_SIZE


def test_score_debug_output_is_opt_in(capsys):
    benchmark = make_benchmark(n_classes=2, n_experiences=1)
    quiet = train_all(make_strategy(2, verbose=True), benchmark)
    quiet.eval(benchmark.test_stream)
    assert "score election debug" not in capsys.readouterr().out

    loud = train_all(make_strategy(2, verbose=True, debug_scores=True), benchmark)
    loud.eval(benchmark.test_stream)
    assert "score election debug" in capsys.readouterr().out


def test_class_logit_uses_the_global_class_column():
    logits = torch.arange(12.0).reshape(2, 6)
    assert torch.equal(_class_logit(logits, 4, [1, 4]), logits[:, 4])
    with pytest.raises(RuntimeError, match="not owned"):
        _class_logit(logits, 3, [1, 4])
    with pytest.raises(RuntimeError, match="outside"):
        _class_logit(logits, 9, [9])
    with pytest.raises(RuntimeError, match="shape"):
        _class_logit(torch.zeros(3), 0, [0])


def test_platt_calibration_is_monotone_and_falls_back_to_identity():
    scores = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    scale, bias = _fit_platt_calibrator(scores, targets)
    assert scale >= 1e-3
    assert (scale * scores + bias).argsort().tolist() == scores.argsort().tolist()
    # Degenerate hold-out (no negatives) -> identity calibration.
    assert _fit_platt_calibrator(scores, torch.ones(6)) == (1.0, 0.0)
    assert _fit_platt_calibrator(scores[:1], targets[:1]) == (1.0, 0.0)


def test_repeated_evaluation_is_idempotent():
    benchmark = make_benchmark(n_classes=4, n_experiences=2)
    strategy = train_all(make_strategy(4), benchmark)
    plugin = strategy.cl_evaluation_plugin
    strategy.eval(benchmark.test_stream)
    first = plugin.results()["raw_mean_final_accuracy"]
    strategy.eval(benchmark.test_stream)
    assert plugin.results()["raw_mean_final_accuracy"] == first
