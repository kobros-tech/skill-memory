# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Same seeds -> same run, whatever the global RNG does in between."""

import pytest
import torch

from skill_memory.cl.training import derive_seed, train_on_class
from skill_memory.tests._helpers import make_benchmark, make_strategy, train_all


def _states(strategy):
    memory = strategy.skill_memory
    return {
        slot: {k: v.clone() for k, v in memory.state(slot).items()}
        for slot in sorted(memory.slots())
    }


def _run(mode, *, perturb_global_rng, **kwargs):
    benchmark = make_benchmark(n_classes=4, n_experiences=2, n_per_class=24)
    strategy = make_strategy(4, cl_update_mode=mode, **kwargs)
    # Burn a *different* amount of global randomness each run.
    for _ in range(perturb_global_rng):
        torch.rand(7)
    train_all(strategy, benchmark)
    return _states(strategy), strategy


@pytest.mark.parametrize("mode", ["new_class", "small_replay", "replay"])
def test_stored_skills_are_independent_of_the_global_rng(mode):
    first, _ = _run(mode, perturb_global_rng=0)
    second, _ = _run(mode, perturb_global_rng=17)
    assert first.keys() == second.keys()
    for slot in first:
        for name in first[slot]:
            assert torch.equal(first[slot][name], second[slot][name]), (slot, name)


def test_refresh_is_also_reproducible():
    first, _ = _run("replay", perturb_global_rng=0, refresh_existing_skills=True)
    second, _ = _run("replay", perturb_global_rng=5, refresh_existing_skills=True)
    for slot in first:
        for name in first[slot]:
            assert torch.equal(first[slot][name], second[slot][name])


def test_training_seed_changes_the_run():
    first, _ = _run("replay", perturb_global_rng=0, training_seed=0)
    other, _ = _run("replay", perturb_global_rng=0, training_seed=1)
    assert any(
        not torch.equal(first[slot][name], other[slot][name])
        for slot in first
        for name in first[slot]
    )


def test_evaluation_results_are_reproducible():
    benchmark = make_benchmark(n_classes=4, n_experiences=2, n_per_class=24)
    results = []
    for burn in (0, 11):
        strategy = make_strategy(4)
        for _ in range(burn):
            torch.rand(3)
        train_all(strategy, benchmark)
        results.append(strategy.eval(benchmark.test_stream))
    assert results[0]["mean_final_accuracy"] == results[1]["mean_final_accuracy"]
    assert (
        results[0]["raw_mean_final_accuracy"] == results[1]["raw_mean_final_accuracy"]
    )


def test_sampler_seed_controls_minibatch_stream_directly():
    from types import SimpleNamespace

    from torch.utils.data import TensorDataset

    class Experience:
        dataset = TensorDataset(
            torch.arange(20, dtype=torch.float32).reshape(20, 1),
            torch.tensor([0] * 10 + [1] * 10),
        )
        dataset.targets = [0] * 10 + [1] * 10

    def run(seed, mode):
        torch.manual_seed(seed * 0 + torch.randint(0, 10_000, (1,)).item())
        model = torch.nn.Linear(1, 2)
        torch.nn.init.constant_(model.weight, 0.1)
        torch.nn.init.constant_(model.bias, 0.0)
        strategy = SimpleNamespace(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            clock=SimpleNamespace(train_iterations=0),
        )
        train_on_class(
            strategy,
            Experience(),
            1,
            2,
            4,
            mode=mode,
            validation_fraction=0.0,
            sampler_seed=seed,
        )
        return model.weight.detach().clone()

    for mode in ("multiclass", "binary_one_vs_rest"):
        assert torch.equal(run(7, mode), run(7, mode))
        assert not torch.equal(run(7, mode), run(8, mode))


def test_derive_seed_is_pure_and_separates_streams():
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)
    assert derive_seed(0, 1, 2) != derive_seed(0, 2, 1)
    assert derive_seed(0, 1, 2) != derive_seed(1, 1, 2)
