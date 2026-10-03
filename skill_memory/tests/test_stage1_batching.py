# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Tests for the batched stage-1 evaluator (`evaluate_states_batch`).

Contract under test:

* the batched path computes the *same* thing as the sequential
  `evaluate_state` loop -- same candidate skills, same expanded skill
  states (same `FunctionalStateCache` keys), same forward-pass math --
  within ordinary float32 matmul-reassociation tolerance, never an
  approximation;
* skills whose expanded (post-growth) parameter shapes differ are never
  stacked together (`_param_shape_signature` grouping), including the case
  where different skills need *different* classifier widths for the same
  probe;
* `batch_stage1=True` end-to-end (through `score_class_against_skills` and
  a full seeded `SkillMemoryStrategy` run) makes the same REUSE/SCRATCH/skill
  choices, on the same stored skills, as `batch_stage1=False`;
* the existing safety-stage optimizations (cap, short-circuit,
  accuracy-only, both caches) are untouched by any of this -- stage 1 is the
  only thing `batch_stage1` changes.
"""

from types import SimpleNamespace

import pytest
import torch
from avalanche.models.dynamic_modules import IncrementalClassifier
from torch import nn

from skill_memory.cl import decision as decision_module
from skill_memory.cl.decision import score_class_against_skills
from skill_memory.utils import probing

# ---------------------------------------------------------------------------
# A small conv + BatchNorm model: batching a plain Linear model can't catch
# BatchNorm-buffer or convolution batching-rule mistakes.
# ---------------------------------------------------------------------------


class _ConvNet(nn.Module):
    def __init__(self, initial_out_features: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = IncrementalClassifier(
            12, initial_out_features=initial_out_features
        )

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _perturbed_state(model, seed):
    torch.manual_seed(seed)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for key, value in state.items():
        if value.is_floating_point():
            state[key] = value + torch.randn_like(value) * 0.05
    return state


def _probe_batch(n=24, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 3, 8, 8), torch.randint(0, 6, (n,))


@pytest.fixture
def conv_model():
    torch.manual_seed(0)
    return _ConvNet()


# ---------------------------------------------------------------------------
# Correctness: batched == sequential, including with real conv + BatchNorm
# ---------------------------------------------------------------------------


def test_batched_matches_sequential_reference(conv_model):
    """The scenario most likely to break: same-shaped skills, BatchNorm
    running stats differ per skill, classifier needs to grow for the probe."""
    states = [(slot, _perturbed_state(conv_model, seed=slot)) for slot in range(6)]
    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()

    sequential = {
        slot: probing.evaluate_state(
            conv_model, state, x, y, nn.functional.cross_entropy, experience, seed=11
        )
        for slot, state in states
    }
    batched = probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=11
    )

    assert batched.keys() == sequential.keys()
    for slot in sequential:
        for a, b in zip(sequential[slot], batched[slot], strict=True):
            assert a == pytest.approx(b, abs=1e-4), (
                slot,
                sequential[slot],
                batched[slot],
            )


def test_batched_matches_sequential_with_chunking(conv_model):
    states = [(slot, _perturbed_state(conv_model, seed=slot)) for slot in range(10)]
    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()

    full = probing.evaluate_states_batch(conv_model, states, x, y, experience, seed=3)
    chunked = probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=3, chunk_size=3
    )
    assert full.keys() == chunked.keys()
    for slot in full:
        for a, b in zip(full[slot], chunked[slot], strict=True):
            assert a == pytest.approx(b, abs=1e-4)


def test_single_skill_group_uses_sequential_fallback_and_still_matches(conv_model):
    """A group of exactly 1 skips vmap entirely (see docstring); still exact."""
    state = _perturbed_state(conv_model, seed=0)
    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()

    sequential = probing.evaluate_state(
        conv_model, state, x, y, nn.functional.cross_entropy, experience, seed=7
    )
    (batched,) = probing.evaluate_states_batch(
        conv_model, [(0, state)], x, y, experience, seed=7
    ).values()
    for a, b in zip(sequential, batched, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


# ---------------------------------------------------------------------------
# Heterogeneous classifier widths: skills must NOT be force-stacked
# ---------------------------------------------------------------------------


def test_shape_signature_distinguishes_dtype_not_just_shape():
    a = {"w": torch.zeros(3, 3, dtype=torch.float32)}
    b = {"w": torch.zeros(3, 3, dtype=torch.float64)}
    assert probing._param_shape_signature(a) != probing._param_shape_signature(b)


def test_skills_with_different_growth_widths_are_never_stacked_but_still_match(
    conv_model,
):
    """Simulates skills captured at different points of classifier growth:
    one skill's stored head is already wider than this probe's target width,
    so post-growth widths genuinely differ across skills for this call."""
    narrow = _perturbed_state(conv_model, seed=1)  # classifier: 2 columns

    wide_model = _ConvNet(initial_out_features=9)
    wide_model.load_state_dict(
        {k: v for k, v in conv_model.state_dict().items() if "classifier" not in k},
        strict=False,
    )
    wide = _perturbed_state(wide_model, seed=2)  # classifier: 9 columns

    experience = SimpleNamespace(classes_in_this_experience=[4, 5])  # target width 6
    x, y = _probe_batch()
    states = [(0, narrow), (1, wide)]

    batched = probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=5
    )
    sequential = {
        slot: probing.evaluate_state(
            conv_model, state, x, y, nn.functional.cross_entropy, experience, seed=5
        )
        for slot, state in states
    }
    for slot in sequential:
        for a, b in zip(sequential[slot], batched[slot], strict=True):
            assert a == pytest.approx(b, abs=1e-4)

    # And prove the two really did end up in different shape groups.
    params0 = probing._functional_growth_for_experience(
        conv_model, narrow, experience, seed=5
    )
    params1 = probing._functional_growth_for_experience(
        conv_model, wide, experience, seed=5
    )
    assert probing._param_shape_signature(params0) != probing._param_shape_signature(
        params1
    )
    assert params0["classifier.classifier.weight"].shape[0] == 6
    assert params1["classifier.classifier.weight"].shape[0] == 9


def test_three_shape_groups_all_resolve_correctly(conv_model):
    """Two skills share a width, a third is alone -- exercises the grouped
    (size >= 2) path and the singleton fallback in the same call."""
    a = _perturbed_state(conv_model, seed=1)
    b = _perturbed_state(conv_model, seed=2)
    wide_model = _ConvNet(initial_out_features=9)
    wide_model.load_state_dict(
        {k: v for k, v in conv_model.state_dict().items() if "classifier" not in k},
        strict=False,
    )
    c = _perturbed_state(wide_model, seed=3)

    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()
    states = [(0, a), (1, b), (2, c)]

    batched = probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=9
    )
    sequential = {
        slot: probing.evaluate_state(
            conv_model, state, x, y, nn.functional.cross_entropy, experience, seed=9
        )
        for slot, state in states
    }
    for slot in sequential:
        for m1, m2 in zip(sequential[slot], batched[slot], strict=True):
            assert m1 == pytest.approx(m2, abs=1e-4)


# ---------------------------------------------------------------------------
# Cache integration: the batched path must use the SAME FunctionalStateCache
# keys/semantics as the sequential path (req: preserve existing caching).
# ---------------------------------------------------------------------------


def test_batched_path_uses_and_fills_the_functional_state_cache(conv_model):
    states = [(slot, _perturbed_state(conv_model, seed=slot)) for slot in range(4)]
    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()
    cache = probing.FunctionalStateCache()

    probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=2, cache=cache
    )
    assert cache.misses == 4 and cache.hits == 0

    probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=2, cache=cache
    )
    assert cache.misses == 4 and cache.hits == 4  # second call served from cache


def test_batched_path_never_caches_unseeded_probes(conv_model):
    states = [(slot, _perturbed_state(conv_model, seed=slot)) for slot in range(3)]
    experience = SimpleNamespace(classes_in_this_experience=[4, 5])
    x, y = _probe_batch()
    cache = probing.FunctionalStateCache()

    probing.evaluate_states_batch(
        conv_model, states, x, y, experience, seed=None, cache=cache
    )
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# End to end: batch_stage1 changes nothing about routing decisions
# ---------------------------------------------------------------------------


N_SKILLS = 8
OLD_PER_SKILL = 3


class _FakeMemory:
    def slots(self):
        return set(range(N_SKILLS))

    def state(self, slot):
        return {}


class _FakeClassMap:
    def classes_for_skill(self, skill):
        return {10 * skill + i for i in range(OLD_PER_SKILL)}


def test_score_class_against_skills_batch_stage1_matches_sequential(monkeypatch):
    """Same fakes as test_safety_optimizations.py's `world` fixture, but
    checking batch_stage1=True vs False through the real decision function
    (not the probing layer), including stage 2 (safety) untouched by it."""
    monkeypatch.setattr(
        decision_module, "probe_class", lambda *a, **k: (torch.tensor([[99.0]]), None)
    )
    monkeypatch.setattr(
        decision_module, "_first_experience_with_class", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        decision_module,
        "_probe_class_across",
        lambda seen, old_class, *a, **k: (
            torch.tensor([[float(old_class)]]),
            torch.tensor([0]),
        ),
    )

    def fake_evaluate_state(model, state, x, y, loss_fn, experience, **kwargs):
        skill = kwargs["slot"]
        score = 1.0 - 0.01 * skill
        return 0.0, score, score

    def fake_evaluate_states_batch(model, states, x, y, experience, **kwargs):
        return {
            slot: fake_evaluate_state(model, state, x, y, None, experience, slot=slot)
            for slot, state in states
        }

    def fake_accuracy(model, state, x, y, experience, **kwargs):
        return 0.9

    monkeypatch.setattr(decision_module, "evaluate_state", fake_evaluate_state)
    monkeypatch.setattr(
        decision_module, "evaluate_states_batch", fake_evaluate_states_batch
    )
    monkeypatch.setattr(decision_module, "evaluate_state_accuracy", fake_accuracy)

    def run(batch_stage1):
        return score_class_against_skills(
            SimpleNamespace(model=nn.Linear(1, 1)),
            object(),
            7,
            _FakeMemory(),
            _FakeClassMap(),
            probe_batch_size=1,
            probe_batches=1,
            probe_seed=0,
            seen_experiences=[object()],
            max_safety_candidates=None,
            batch_stage1=batch_stage1,
        )

    sequential = run(False)
    batched = run(True)
    assert [(r["skill"], r["new_score"], r["old_accuracy"]) for r in sequential] == [
        (r["skill"], r["new_score"], r["old_accuracy"]) for r in batched
    ]


def test_full_strategy_run_same_decisions_with_and_without_batch_stage1():
    """A real, small, seeded end-to-end run: identical REUSE/SCRATCH/skill
    choices and identical stored skill weights, batch_stage1 on vs off."""
    from skill_memory.tests.test_leakage import _benchmark, _strategy

    def run(batch_stage1):
        torch.manual_seed(7)
        benchmark, _ = _benchmark(n_classes=6, n_experiences=3)
        strategy = _strategy(max_safety_candidates=None, batch_stage1=batch_stage1)
        for experience in benchmark.train_stream:
            strategy.train(experience)
        plugin = strategy.skill_memory_plugin
        decisions = {
            (e, c): (d["decision"], d["skill"])
            for e, per in plugin.last_class_decisions.items()
            for c, d in per.items()
        }
        states = {
            slot: plugin.memory.state(slot) for slot in sorted(plugin.memory.slots())
        }
        return decisions, states

    off_decisions, off_states = run(False)
    on_decisions, on_states = run(True)

    assert off_decisions == on_decisions
    assert off_states.keys() == on_states.keys()
    for slot in off_states:
        for key in off_states[slot]:
            assert torch.allclose(
                off_states[slot][key], on_states[slot][key], atol=1e-4
            ), (slot, key)


def test_batch_stage1_flag_actually_selects_the_batched_code_path(monkeypatch):
    """Guards against the wiring silently no-op'ing: `batch_stage1=True` must
    really call `evaluate_states_batch`, and `False` must really not."""
    from skill_memory.tests.test_leakage import _benchmark, _strategy

    def run(batch_stage1):
        calls = SimpleNamespace(batched=0)
        original = decision_module.evaluate_states_batch

        def spy(*args, **kwargs):
            calls.batched += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(decision_module, "evaluate_states_batch", spy)
        torch.manual_seed(7)
        benchmark, _ = _benchmark(n_classes=6, n_experiences=3)
        strategy = _strategy(max_safety_candidates=None, batch_stage1=batch_stage1)
        for experience in benchmark.train_stream:
            strategy.train(experience)
        monkeypatch.undo()
        return calls.batched

    assert run(True) > 0
    assert run(False) == 0


# ---------------------------------------------------------------------------
# Preserved: safety-stage features are untouched by batch_stage1
# ---------------------------------------------------------------------------


def test_batch_stage1_does_not_change_safety_stage_behavior(monkeypatch):
    """Stage 2's cap/short-circuit/accuracy-only/cache behavior (see
    test_safety_optimizations.py) must be identical whether stage 1 is
    batched or not -- batch_stage1 only ever touches stage 1."""
    calls = SimpleNamespace(accuracy_only=0)
    monkeypatch.setattr(
        decision_module, "probe_class", lambda *a, **k: (torch.tensor([[99.0]]), None)
    )
    monkeypatch.setattr(
        decision_module, "_first_experience_with_class", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        decision_module,
        "_probe_class_across",
        lambda seen, old_class, *a, **k: (
            torch.tensor([[float(old_class)]]),
            torch.tensor([0]),
        ),
    )

    def fake_evaluate_state(model, state, x, y, loss_fn, experience, **kwargs):
        skill = kwargs["slot"]
        score = 1.0 - 0.01 * skill
        return 0.0, score, score

    def fake_evaluate_states_batch(model, states, x, y, experience, **kwargs):
        return {
            slot: fake_evaluate_state(model, state, x, y, None, experience, slot=slot)
            for slot, state in states
        }

    def fake_accuracy(model, state, x, y, experience, **kwargs):
        calls.accuracy_only += 1
        return 0.9

    monkeypatch.setattr(decision_module, "evaluate_state", fake_evaluate_state)
    monkeypatch.setattr(
        decision_module, "evaluate_states_batch", fake_evaluate_states_batch
    )
    monkeypatch.setattr(decision_module, "evaluate_state_accuracy", fake_accuracy)

    results = score_class_against_skills(
        SimpleNamespace(model=nn.Linear(1, 1)),
        object(),
        7,
        _FakeMemory(),
        _FakeClassMap(),
        probe_batch_size=1,
        probe_batches=1,
        probe_seed=0,
        seen_experiences=[object()],
        batch_stage1=True,  # default cap of 5 still applies to stage 2
    )
    assert len(results) == 5  # DEFAULT_MAX_SAFETY_CANDIDATES, unaffected
    assert calls.accuracy_only == 5 * OLD_PER_SKILL
