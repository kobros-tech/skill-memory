# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Leakage test suite.

These tests try to *break* the protocol instead of checking the happy path:

* **Future data** -- datasets are "poisoned" so that reading a sample of a
  class before its experience has been reached raises immediately.
* **Test data during training / training data during evaluation** -- the
  same poisoning, per split.
* **Labels at evaluation** -- predictions must be bit-identical when the
  test labels are changed (labels may only be used *after* prediction).
* **Train/eval overlap** -- the content audit must be clean on a disjoint
  split and must catch a deliberately leaky one.
* **Probing side effects** -- probing must not mutate stored skills, the
  live model, or the global RNG stream.
* **Optimization exactness** -- caches / short-circuiting must not change
  any decision or stored skill relative to the uncached reference path.
"""

from types import SimpleNamespace

import pytest
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training.plugins import SupervisedPlugin
from torch import nn
from torch.utils.data import TensorDataset

from skill_memory.cl import decision as decision_module
from skill_memory.diagnostics import (
    assert_no_split_overlap,
    audit_split_overlap,
    audit_strategy_leakage,
)
from skill_memory.strategy import SkillMemoryStrategy
from skill_memory.utils import probing
from skill_memory.utils.protocol_guard import (
    ProtocolViolation,
    assert_evaluation_experiences,
    assert_memory_classes_match,
    assert_training_experience,
)

N_FEATURES = 6


# ---------------------------------------------------------------------------
# Poisoned benchmark
# ---------------------------------------------------------------------------


class AccessGuard:
    """Shared switchboard deciding which classes each split may read now."""

    def __init__(self):
        self.armed = False
        self.allowed = {"train": set(), "test": set()}
        self.reads = {"train": 0, "test": 0}


class PoisonedDataset(TensorDataset):
    """TensorDataset that raises when read outside what the guard allows."""

    def __init__(self, x, y, role, guard):
        super().__init__(x, y)
        self.role = role
        self.guard = guard

    def __getitem__(self, index):
        guard = self.guard
        if guard.armed:
            label = int(self.tensors[1][int(index)])
            if label not in guard.allowed[self.role]:
                raise AssertionError(
                    f"LEAK: read a {self.role!r} sample of class {label} but only "
                    f"{sorted(guard.allowed[self.role])} may be read now"
                )
            guard.reads[self.role] += 1
        return super().__getitem__(index)


def _class_data(n_classes, n_per_class, seed):
    torch.manual_seed(seed)
    xs, ys = [], []
    for class_id in range(n_classes):
        offsets = torch.zeros(N_FEATURES)
        offsets[class_id % N_FEATURES] = 6.0
        xs.append(torch.randn(n_per_class, N_FEATURES) * 0.5 + offsets)
        ys.append(torch.full((n_per_class,), class_id, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def _benchmark(
    *,
    n_classes=6,
    n_experiences=3,
    n_per_class=24,
    guard=None,
    test_labels=None,
    leak_train_rows_into_test=0,
):
    x_train, y_train = _class_data(n_classes, n_per_class, seed=1)
    x_test, y_test = _class_data(n_classes, n_per_class, seed=2)  # independent draw
    if leak_train_rows_into_test:
        k = leak_train_rows_into_test
        x_test = torch.cat([x_test, x_train[:k]])
        y_test = torch.cat([y_test, y_train[:k]])
    if test_labels is not None:
        y_test = test_labels(y_test)
    guard = guard or AccessGuard()
    benchmark = nc_benchmark(
        PoisonedDataset(x_train, y_train, "train", guard),
        PoisonedDataset(x_test, y_test, "test", guard),
        n_experiences=n_experiences,
        task_labels=False,
        seed=0,
        shuffle=False,
        fixed_class_order=list(range(n_classes)),
    )
    return benchmark, guard


def _strategy(n_classes=6, *, strict=True, per_class=None, **kw):
    model = SimpleMLP(input_size=N_FEATURES, hidden_size=8, num_classes=n_classes)
    return SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=nn.CrossEntropyLoss(),
        max_skills=10,
        eval_memory_per_class=per_class or 10,
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        probe_seed=0,
        verbose=False,
        strict_protocol=strict,
        **kw,
    )


def _classes(experience):
    return {int(c) for c in experience.classes_in_this_experience}


# ---------------------------------------------------------------------------
# 1. Future data / cross-split reads
# ---------------------------------------------------------------------------


def test_no_future_or_cross_split_reads_during_full_run():
    benchmark, guard = _benchmark()
    strategy = _strategy()
    all_test_classes = set(range(6))
    seen: set[int] = set()

    guard.armed = True  # everything below runs under the poison
    for experience in benchmark.train_stream:
        seen |= _classes(experience)

        # TRAINING: only classes of experiences <= t; test data untouchable.
        guard.allowed = {"train": set(seen), "test": set()}
        strategy.train(experience)

        # EVALUATION: training data untouchable (evaluator uses retained
        # tensors); test data of the whole stream is the thing being scored.
        guard.allowed = {"train": set(), "test": set(all_test_classes)}
        strategy.eval(benchmark.test_stream)

    # Sanity: the poison actually observed traffic on both splits, so the
    # assertions above were not vacuous.
    assert guard.reads["train"] > 0
    assert guard.reads["test"] > 0


def test_poison_guard_really_fires():
    """Meta-test: prove the poisoned dataset would catch a real leak."""
    benchmark, guard = _benchmark()
    guard.armed = True
    guard.allowed = {"train": {0, 1}, "test": set()}
    future_experience = benchmark.train_stream[2]  # classes 4, 5
    with pytest.raises(AssertionError, match="LEAK"):
        future_experience.dataset[0]


def test_old_class_probes_only_draw_from_already_trained_experiences(monkeypatch):
    benchmark, _ = _benchmark()
    strategy = _strategy()
    plugin = strategy.skill_memory_plugin
    seen_sizes = []
    original = decision_module.decide_class

    def spy(*args, **kwargs):
        seen_experiences = args[8]  # seen_experiences positional argument
        seen_sizes.append(len(seen_experiences))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "skill_memory.cl.skill_memory_plugin.decide_class",
        spy,
    )
    for index, experience in enumerate(benchmark.train_stream):
        strategy.train(experience)
        assert len(plugin._seen_experiences) == index + 1
    # experience t's decisions saw exactly the t experiences trained BEFORE it
    per_experience = {}
    for size in seen_sizes:
        per_experience.setdefault(size, 0)
        per_experience[size] += 1
    assert sorted(per_experience) == [0, 1, 2]


def test_eval_memory_holds_only_classes_trained_so_far():
    benchmark, _ = _benchmark()
    strategy = _strategy()
    seen: set[int] = set()
    for experience in benchmark.train_stream:
        strategy.train(experience)
        seen |= _classes(experience)
        retained = {m.class_id for m in strategy.skill_memory_plugin.eval_memory}
        assert retained == seen


# ---------------------------------------------------------------------------
# 2. Labels may only be used AFTER prediction
# ---------------------------------------------------------------------------


class _OutputRecorder(SupervisedPlugin):
    """Record the model output for every evaluated input, keyed by the input.

    Avalanche orders each experience's samples by class label, so changing
    labels can legitimately reorder rows; keying by input *content* makes the
    comparison independent of order.
    """

    def __init__(self):
        super().__init__()
        self.by_input = {}

    def after_eval_forward(self, strategy, **kwargs):
        inputs = strategy.mbatch[0].detach().cpu()
        outputs = strategy.mb_output.detach().cpu()
        for row, out in zip(inputs, outputs, strict=True):
            self.by_input[row.numpy().tobytes()] = out.clone()


def _swap_within_pairs(y):
    """Relabel 0<->1, 2<->3, 4<->5 : same x per experience, different labels."""
    return y ^ 1


def test_predictions_do_not_depend_on_test_labels():
    def run(test_labels):
        benchmark, _ = _benchmark(n_experiences=3, test_labels=test_labels)
        recorder = _OutputRecorder()
        strategy = _strategy(plugins=[recorder])
        for experience in benchmark.train_stream:
            strategy.train(experience)
        recorder.by_input.clear()
        results = strategy.eval(benchmark.test_stream)
        return recorder.by_input, results

    outputs_true, results_true = run(None)
    outputs_swapped, results_swapped = run(_swap_within_pairs)

    assert outputs_true.keys() == outputs_swapped.keys()
    assert len(outputs_true) == 6 * 24
    for key, out in outputs_true.items():
        assert torch.allclose(
            out,
            outputs_swapped[key],
            rtol=1e-6,
            atol=1e-6,
        ), "model output for an input changed when only the labels changed"
    # ...while the *metric* did react to the labels, so the test is not vacuous.
    assert (
        results_true["final_class_accuracy"] != results_swapped["final_class_accuracy"]
    )


# ---------------------------------------------------------------------------
# 3. Runtime protocol guards
# ---------------------------------------------------------------------------


def test_training_on_a_test_stream_experience_is_rejected():
    benchmark, _ = _benchmark()
    strategy = _strategy()
    with pytest.raises(ProtocolViolation, match="test"):
        strategy.train(benchmark.test_stream[0])


def test_evaluating_on_the_train_stream_is_rejected():
    benchmark, _ = _benchmark()
    strategy = _strategy()
    strategy.train(benchmark.train_stream[0])
    with pytest.raises(ProtocolViolation, match="train"):
        strategy.eval(benchmark.train_stream)


def test_strict_protocol_false_is_an_explicit_opt_out():
    benchmark, _ = _benchmark()
    strategy = _strategy(strict=False)
    strategy.train(benchmark.train_stream[0])
    strategy.eval(benchmark.train_stream)  # allowed only because opted out


def test_guard_rejects_shared_dataset_objects():
    dataset = object()
    trained = [SimpleNamespace(dataset=dataset)]
    with pytest.raises(ProtocolViolation, match="shares its dataset"):
        assert_evaluation_experiences([SimpleNamespace(dataset=dataset)], trained)
    assert_evaluation_experiences([SimpleNamespace(dataset=object())], trained)


def test_guards_are_silent_when_stream_provenance_is_unknown():
    assert_training_experience(SimpleNamespace())
    assert_evaluation_experiences([SimpleNamespace(dataset=object())], [])


def test_memory_class_guard_rejects_foreign_classes():
    experience = SimpleNamespace(classes_in_this_experience=[0, 1])
    assert_memory_classes_match([0, 1], experience)
    with pytest.raises(ProtocolViolation, match=r"\[7\]"):
        assert_memory_classes_match([0, 7], experience)


# ---------------------------------------------------------------------------
# 4. Content-level overlap audit
# ---------------------------------------------------------------------------


def _train_all(strategy, benchmark):
    for experience in benchmark.train_stream:
        strategy.train(experience)


def test_overlap_audit_is_clean_on_a_disjoint_split():
    benchmark, _ = _benchmark()
    strategy = _strategy(per_class=24)  # retain every training sample
    _train_all(strategy, benchmark)
    report = audit_strategy_leakage(strategy, benchmark.test_stream, diagnose=True)
    assert report["n_memory_samples"] == 6 * 24
    assert report["n_test_samples"] == 6 * 24
    assert report["memory_test_overlaps"] == 0
    assert report["train_test_overlaps"] == 0


def test_overlap_audit_catches_a_leaky_split():
    benchmark, _ = _benchmark(leak_train_rows_into_test=7)
    strategy = _strategy(per_class=24)
    _train_all(strategy, benchmark)
    report = audit_strategy_leakage(strategy, benchmark.test_stream, diagnose=True)
    assert report["memory_test_overlaps"] == 7
    assert report["train_test_overlaps"] == 7
    with pytest.raises(ProtocolViolation, match="overlap"):
        assert_no_split_overlap(
            eval_memory=strategy.skill_memory_plugin.eval_memory,
            test_stream=benchmark.test_stream,
            train_experiences=strategy.skill_memory_plugin._seen_experiences,
            diagnose=True,
        )


def test_overlap_audit_requires_explicit_diagnose():
    with pytest.raises(RuntimeError, match="diagnose=True"):
        audit_split_overlap(eval_memory=[], test_stream=[], diagnose=False)


# ---------------------------------------------------------------------------
# 5. Probing has no side effects
# ---------------------------------------------------------------------------


def test_probing_does_not_mutate_state_model_or_global_rng():
    from avalanche.models.dynamic_modules import IncrementalClassifier

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(N_FEATURES, 8)
            self.classifier = IncrementalClassifier(8, initial_out_features=2)

        def forward(self, x):
            return self.classifier(torch.relu(self.features(x)))

    torch.manual_seed(0)
    model = Net()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state_before = {k: v.clone() for k, v in state.items()}
    live_before = {k: v.clone() for k, v in model.state_dict().items()}
    experience = SimpleNamespace(classes_in_this_experience=[5, 6])
    x = torch.randn(16, N_FEATURES)
    y = torch.randint(0, 7, (16,))
    cache = probing.FunctionalStateCache()

    rng_before = torch.get_rng_state().clone()
    for use_cache in (False, True, True):
        probing.evaluate_state(
            model,
            state,
            x,
            y,
            nn.functional.cross_entropy,
            experience,
            seed=3,
            cache=cache if use_cache else None,
            slot=0,
        )
        probing.evaluate_state_accuracy(
            model,
            state,
            x,
            y,
            experience,
            seed=3,
            cache=cache if use_cache else None,
            slot=0,
        )

    assert torch.equal(torch.get_rng_state(), rng_before), "probe advanced global RNG"
    for key, value in state.items():
        assert torch.equal(value, state_before[key]), f"stored skill mutated: {key}"
    for key, value in model.state_dict().items():
        assert torch.equal(value, live_before[key]), f"live model mutated: {key}"


# ---------------------------------------------------------------------------
# 6. Optimization exactness (end to end)
# ---------------------------------------------------------------------------


def _run_pipeline(*, optimized: bool, monkeypatch):
    """Run the whole seeded pipeline; `optimized=False` is the reference path:
    no probe cache and no safety short-circuit."""
    if not optimized:
        original = decision_module.score_class_against_skills

        def reference(*args, **kwargs):
            kwargs["probe_cache"] = None
            kwargs["forgetting_margin"] = None
            return original(*args, **kwargs)

        monkeypatch.setattr(decision_module, "score_class_against_skills", reference)

    torch.manual_seed(7)
    benchmark, _ = _benchmark(n_classes=6, n_experiences=3)
    strategy = _strategy(max_safety_candidates=None)
    for experience in benchmark.train_stream:
        strategy.train(experience)
    plugin = strategy.skill_memory_plugin
    decisions = {
        (e, c): (d["decision"], d["skill"])
        for e, per in plugin.last_class_decisions.items()
        for c, d in per.items()
    }
    states = {slot: plugin.memory.state(slot) for slot in sorted(plugin.memory.slots())}
    return decisions, states


def test_caches_and_short_circuit_do_not_change_decisions_or_skills(monkeypatch):
    fast_decisions, fast_states = _run_pipeline(optimized=True, monkeypatch=monkeypatch)
    monkeypatch.undo()
    ref_decisions, ref_states = _run_pipeline(optimized=False, monkeypatch=monkeypatch)

    assert fast_decisions == ref_decisions
    assert fast_states.keys() == ref_states.keys()
    for slot in fast_states:
        for key in fast_states[slot]:
            assert torch.equal(fast_states[slot][key], ref_states[slot][key]), (
                slot,
                key,
            )


def test_each_old_class_probe_is_sampled_at_most_once_per_pool(monkeypatch):
    """Cross-experience reuse: no later experience contains an old class, so its
    probe pool never widens and it must be drawn exactly once for the whole run."""
    builds = []
    original = decision_module._probe_class_across

    def counting(seen, old_class, *args, **kwargs):
        builds.append(old_class)
        return original(seen, old_class, *args, **kwargs)

    monkeypatch.setattr(decision_module, "_probe_class_across", counting)
    benchmark, _ = _benchmark(n_classes=6, n_experiences=3)
    strategy = _strategy(max_safety_candidates=None)
    for experience in benchmark.train_stream:
        strategy.train(experience)
    assert builds, "expected the safety stage to probe old classes"
    assert len(builds) == len(set(builds)), f"old-class probes re-sampled: {builds}"


def test_eval_memory_matches_full_scan_reference_and_decodes_only_what_it_keeps():
    from skill_memory.evaluation.memory import EvaluationMemoryPlugin

    benchmark, guard = _benchmark(n_classes=6, n_experiences=3, n_per_class=24)
    plugin = EvaluationMemoryPlugin(eval_memory_per_class=10, verbose=False)
    experience = benchmark.train_stream[1]  # classes 2, 3

    # Reference: the original algorithm, decoding every sample to read labels.
    dataset = experience.dataset
    by_class: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        by_class.setdefault(int(dataset[index][1]), []).append(index)
    generator = torch.Generator()
    generator.manual_seed(plugin.eval_memory_seed + int(experience.current_experience))
    expected = {}
    for class_id in sorted(by_class):
        indices = by_class[class_id]
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        chosen = [indices[p] for p in permutation[:10]]
        expected[class_id] = torch.stack([dataset[i][0] for i in chosen])

    guard.armed = True
    guard.allowed = {"train": {2, 3}, "test": set()}
    guard.reads["train"] = 0
    memories = plugin._build_evaluation_memory(experience)

    assert [m.class_id for m in memories] == sorted(expected)
    for memory in memories:
        assert torch.equal(memory.inputs, expected[memory.class_id])
    # Only the retained samples were decoded (2 classes x 10), not all 48.
    assert guard.reads["train"] == 2 * 10


def test_guard_fires_when_eval_is_called_with_a_single_experience():
    """`strategy.eval(one_experience)` (not a list) must still be checked."""
    benchmark, _ = _benchmark()
    strategy = _strategy()
    strategy.train(benchmark.train_stream[0])
    with pytest.raises(ProtocolViolation, match="train"):
        strategy.eval(benchmark.train_stream[0])
    # the test-stream equivalent is allowed
    strategy.eval(benchmark.test_stream[0])
