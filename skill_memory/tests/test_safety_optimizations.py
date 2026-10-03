# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Tests for the safety-stage optimizations.

Contract under test:

* the default safety cap is 5 candidates; ``None`` is the exact/full mode;
* short-circuiting a failed safety check NEVER changes the REUSE/SCRATCH
  outcome (it is an exact optimization), it only saves forward passes;
* old-class safety uses the accuracy-only evaluator and never the full one;
* the functional-state / old-probe caches never change a probe's result and
  are never served stale after a skill is re-stored.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from skill_memory.cl import decision as decision_module
from skill_memory.cl.decision import (
    DEFAULT_MAX_SAFETY_CANDIDATES,
    DecisionProbeCache,
    find_best_skill,
    score_class_against_skills,
)
from skill_memory.cl.skill_registry import SkillMemory
from skill_memory.utils import probing

# ---------------------------------------------------------------------------
# A fake world: 8 skills; skill k owns old classes {10k, 10k+1, 10k+2}.
# New-class score decreases with k (skill 0 ranks first).
# ---------------------------------------------------------------------------

N_SKILLS = 8
OLD_PER_SKILL = 3


class FakeMemory:
    def slots(self):
        return set(range(N_SKILLS))

    def state(self, slot):
        return {}


class FakeClassMap:
    def classes_for_skill(self, skill):
        return {10 * skill + i for i in range(OLD_PER_SKILL)}


@pytest.fixture
def world(monkeypatch):
    """Patch the probing layer with call-counting fakes."""
    calls = SimpleNamespace(full=0, accuracy_only=0, per_skill=[], fail=set())

    monkeypatch.setattr(
        decision_module, "probe_class", lambda *a, **k: (torch.tensor([[99.0]]), None)
    )
    monkeypatch.setattr(
        decision_module, "_first_experience_with_class", lambda *a, **k: object()
    )
    # old-class probe x encodes the old class id so the fakes can look it up
    monkeypatch.setattr(
        decision_module,
        "_probe_class_across",
        lambda seen, old_class, *a, **k: (
            torch.tensor([[float(old_class)]]),
            torch.tensor([0]),
        ),
    )

    def fake_full(model, state, x, y, loss_fn, experience, **kwargs):
        # only the new-class (stage 1) probe may use the full evaluator
        assert x.item() == 99.0, "old-class safety must not use evaluate_state"
        calls.full += 1
        skill = kwargs["slot"]
        score = 1.0 - 0.01 * skill
        return 0.0, score, score

    def fake_accuracy(model, state, x, y, experience, **kwargs):
        calls.accuracy_only += 1
        skill = kwargs["slot"]
        calls.per_skill.append((skill, int(x.item())))
        old_class = int(x.item())
        # `FAIL[(skill, class)]` -> a forgotten class; everything else is fine
        return 0.05 if (skill, old_class) in calls.fail else 0.9

    monkeypatch.setattr(decision_module, "evaluate_state", fake_full)
    monkeypatch.setattr(decision_module, "evaluate_state_accuracy", fake_accuracy)
    return calls


def _score(max_safety_candidates=DEFAULT_MAX_SAFETY_CANDIDATES, margin=None):
    return score_class_against_skills(
        SimpleNamespace(model=nn.Linear(1, 1)),
        object(),
        7,
        FakeMemory(),
        FakeClassMap(),
        probe_batch_size=1,
        probe_batches=1,
        probe_seed=0,
        seen_experiences=[object()],
        max_safety_candidates=max_safety_candidates,
        forgetting_margin=margin,
    )


def test_default_safety_cap_is_five(world):
    assert DEFAULT_MAX_SAFETY_CANDIDATES == 5
    results = _score()
    assert len(results) == 5
    # the five kept are the five strongest new-class candidates
    assert {r["skill"] for r in results} == {0, 1, 2, 3, 4}
    # stage 1 still probes every skill; only stage 2 is bounded
    assert world.full == N_SKILLS


def test_none_verifies_every_skill_exactly(world):
    results = _score(max_safety_candidates=None)
    assert len(results) == N_SKILLS
    assert world.accuracy_only == N_SKILLS * OLD_PER_SKILL


def test_short_circuit_stops_at_first_unsafe_class(world):
    world.fail = {(0, 1)}  # skill 0 forgets its 2nd old class (id 1)
    results = _score(max_safety_candidates=None, margin=0.05)
    skill0 = next(r for r in results if r["skill"] == 0)
    # classes 0 and 1 evaluated; class 2 skipped
    assert [m["class"] for m in skill0["old_metrics"]] == [0, 1]
    assert skill0["safety_complete"] is False
    assert (0, 2) not in world.per_skill
    # a fully-safe skill is fully evaluated
    skill1 = next(r for r in results if r["skill"] == 1)
    assert skill1["safety_complete"] is True
    assert len(skill1["old_metrics"]) == OLD_PER_SKILL


def test_short_circuit_never_changes_the_decision(world):
    # Several skills forget different old classes.
    world.fail = {(0, 2), (1, 10), (3, 31)}
    for margin in (0.0, 0.05, 0.5):
        full = _score(max_safety_candidates=None, margin=None)
        short = _score(max_safety_candidates=None, margin=margin)
        for score_floor in (0.5, 0.9, None):
            a = find_best_skill(full, margin, score_floor)
            b = find_best_skill(short, margin, score_floor)
            assert (a and a["skill"]) == (b and b["skill"])

        # the safe/unsafe verdict of every individual skill is identical
        def safe(rs, margin=margin):
            return {r["skill"]: r["old_accuracy"] > r["chance"] + margin for r in rs}

        assert safe(full) == safe(short)


def test_short_circuit_saves_forward_passes(world):
    world.fail = {(s, 10 * s) for s in range(N_SKILLS)}  # every skill fails 1st class
    _score(max_safety_candidates=None, margin=0.05)
    short_calls = world.accuracy_only
    world.accuracy_only = 0
    _score(max_safety_candidates=None, margin=None)
    assert short_calls == N_SKILLS  # 1 per skill instead of 3
    assert world.accuracy_only == N_SKILLS * OLD_PER_SKILL


def test_old_score_is_not_part_of_the_decision_path(world):
    for result in _score():
        assert "old_score" not in result
        assert all("score" not in m and "loss" not in m for m in result["old_metrics"])


def test_old_probes_are_reused_across_decisions_when_seeded(world, monkeypatch):
    builds = []
    original = decision_module._probe_class_across

    def counting(seen, old_class, *a, **k):
        builds.append(old_class)
        return original(seen, old_class, *a, **k)

    monkeypatch.setattr(decision_module, "_probe_class_across", counting)
    cache = DecisionProbeCache()
    kwargs = dict(
        seen_experiences=[object()],
        max_safety_candidates=None,
        probe_cache=cache,
    )
    args = (
        SimpleNamespace(model=nn.Linear(1, 1)),
        object(),
        7,
        FakeMemory(),
        FakeClassMap(),
        1,
        1,
        0,
    )
    score_class_against_skills(*args, **kwargs)
    first = len(builds)
    score_class_against_skills(*args, **kwargs)  # a second new class decision
    assert first == N_SKILLS * OLD_PER_SKILL
    assert len(builds) == first  # nothing re-sampled

    # a larger seen-experience pool must NOT be served the old probes
    kwargs["seen_experiences"] = [object(), object()]
    score_class_against_skills(*args, **kwargs)
    assert len(builds) == 2 * first


def test_unseeded_probes_are_never_cached(world, monkeypatch):
    builds = []
    original = decision_module._probe_class_across
    monkeypatch.setattr(
        decision_module,
        "_probe_class_across",
        lambda *a, **k: builds.append(1) or original(*a, **k),
    )
    cache = DecisionProbeCache()
    for _ in range(2):
        score_class_against_skills(
            SimpleNamespace(model=nn.Linear(1, 1)),
            object(),
            7,
            FakeMemory(),
            FakeClassMap(),
            1,
            1,
            None,  # unseeded
            [object()],
            None,
            probe_cache=cache,
        )
    assert len(builds) == 2 * N_SKILLS * OLD_PER_SKILL


# ---------------------------------------------------------------------------
# Real probing layer: accuracy-only + cache exactness
# ---------------------------------------------------------------------------


class _Growing(nn.Module):
    def __init__(self):
        super().__init__()
        from avalanche.models.dynamic_modules import IncrementalClassifier

        self.features = nn.Linear(6, 12)
        self.classifier = IncrementalClassifier(12, initial_out_features=2)

    def forward(self, x):
        return self.classifier(torch.relu(self.features(x)))


def _state_and_experience():
    torch.manual_seed(3)
    model = _Growing()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # An experience that needs the head to grow from 2 -> 8 columns
    experience = SimpleNamespace(classes_in_this_experience=[6, 7])
    x = torch.randn(32, 6)
    y = torch.randint(0, 8, (32,))
    return model, state, experience, x, y


def test_accuracy_only_matches_full_evaluator_accuracy():
    model, state, experience, x, y = _state_and_experience()
    _, _, full_acc = probing.evaluate_state(
        model, state, x, y, nn.functional.cross_entropy, experience, seed=11
    )
    acc = probing.evaluate_state_accuracy(model, state, x, y, experience, seed=11)
    assert acc == full_acc


def test_state_cache_is_bit_exact_and_hits():
    model, state, experience, x, y = _state_and_experience()
    cache = probing.FunctionalStateCache()
    uncached = probing.evaluate_state(
        model, state, x, y, nn.functional.cross_entropy, experience, seed=11
    )
    for _ in range(3):
        cached = probing.evaluate_state(
            model,
            state,
            x,
            y,
            nn.functional.cross_entropy,
            experience,
            seed=11,
            cache=cache,
            slot=0,
        )
        assert cached == uncached
    assert (cache.misses, cache.hits) == (1, 2)


def test_state_cache_is_not_served_stale_after_restore():
    model, state, experience, x, y = _state_and_experience()
    memory = SkillMemory(max_skills=2)
    memory.store(0, state)
    cache = probing.FunctionalStateCache()

    def run():
        return probing.evaluate_state_accuracy(
            model, memory.state(0), x, y, experience, seed=5, cache=cache, slot=0
        )

    before = run()
    # Re-store the skill with very different weights (what a mutable REUSE does)
    new_state = {k: v.detach().clone() + 5.0 for k, v in state.items()}
    memory.store(0, new_state)
    after = run()
    fresh = probing.evaluate_state_accuracy(
        model, memory.state(0), x, y, experience, seed=5
    )
    assert after == fresh
    assert cache.misses == 2  # rebuilt, not served from the old snapshot
    assert before == before  # (sanity: first call succeeded)


def test_state_cache_bypassed_without_seed():
    model, state, experience, x, y = _state_and_experience()
    cache = probing.FunctionalStateCache()
    for _ in range(2):
        probing.evaluate_state_accuracy(
            model, state, x, y, experience, seed=None, cache=cache, slot=0
        )
    assert len(cache) == 0 and cache.hits == 0


# ---------------------------------------------------------------------------
# Old-class probe cache key: exactly the pool the probe is drawn from
# ---------------------------------------------------------------------------


def _exp(labels):
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(len(labels), 2), torch.tensor(labels))
    return SimpleNamespace(
        dataset=dataset, classes_in_this_experience=sorted(set(labels))
    )


def test_pool_key_ignores_experiences_without_the_class():
    a, b, c = _exp([0, 0, 1, 1]), _exp([2, 2, 3, 3]), _exp([0, 5, 5])
    key_a = decision_module._pool_key([a], 0)
    # b has no class 0, so it cannot change class 0's probe pool
    assert decision_module._pool_key([a, b], 0) == key_a
    # c also contains class 0, so the pool widened and the key must change
    assert decision_module._pool_key([a, b, c], 0) != key_a
    # and a class that only b has pools from b alone
    assert decision_module._pool_key([a, b, c], 2) == (id(b),)


def test_pool_key_falls_back_to_whole_pool_when_uninspectable():
    x, y = object(), object()
    assert decision_module._pool_key([x, y], 0) == (id(x), id(y))


# ---------------------------------------------------------------------------
# old_score lives in diagnostics, not in the decision path
# ---------------------------------------------------------------------------


def _trained_strategy():

    from skill_memory.tests.test_leakage import _benchmark, _strategy

    benchmark, _ = _benchmark(n_classes=6, n_experiences=3)
    strategy = _strategy(max_safety_candidates=None)
    for experience in benchmark.train_stream:
        strategy.train(experience)
    return strategy


def test_decisions_and_class_records_carry_no_old_score():
    from dataclasses import fields

    from skill_memory.cl.skill_registry import ClassRecord

    strategy = _trained_strategy()
    plugin = strategy.skill_memory_plugin
    assert "old_score" not in {f.name for f in fields(ClassRecord)}
    for per_class in plugin.last_class_decisions.values():
        for decision in per_class.values():
            assert "old_score" not in decision


def test_measure_old_class_scores_requires_diagnose():
    from skill_memory.diagnostics import measure_old_class_scores

    with pytest.raises(RuntimeError, match="diagnose=True"):
        measure_old_class_scores(object(), 0, diagnose=False)


def test_measure_old_class_scores_matches_direct_evaluation():
    from skill_memory.diagnostics import measure_old_class_scores

    strategy = _trained_strategy()
    plugin = strategy.skill_memory_plugin
    # any skill that owns classes (all 6 classes are in this run)
    skill = next(iter(sorted(plugin.memory.slots())))
    report = measure_old_class_scores(strategy, skill, diagnose=True)

    old_classes = sorted(plugin.class_map.classes_for_skill(skill))
    assert set(report["per_class"]) == set(old_classes)
    assert report["old_score"] == min(m["score"] for m in report["per_class"].values())
    assert report["old_accuracy"] == min(
        m["accuracy"] for m in report["per_class"].values()
    )
    for metrics in report["per_class"].values():
        assert 0.0 <= metrics["score"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    # deterministic for a seeded strategy, and side-effect free
    before = {k: v.clone() for k, v in plugin.memory.state(skill).items()}
    again = measure_old_class_scores(strategy, skill, diagnose=True)
    assert again == report
    for key, value in plugin.memory.state(skill).items():
        assert torch.equal(value, before[key])
