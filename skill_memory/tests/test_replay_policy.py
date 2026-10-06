# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""`UpdatePolicy` semantics and the rules for combining update parameters."""

import pytest
import torch

from skill_memory.cl.replay import UPDATE_MODES, UpdatePolicy, select_historical_samples
from skill_memory.tests._helpers import make_strategy


def test_modes_are_ordered_from_least_to_most_work():
    assert UPDATE_MODES == ("new_class", "replay", "refresh")


@pytest.mark.parametrize(
    ("mode", "history", "refreshes"),
    [("new_class", False, False), ("replay", True, False), ("refresh", True, True)],
)
def test_policy_semantics(mode, history, refreshes):
    policy = UpdatePolicy(mode)
    assert policy.uses_history is history
    assert policy.refreshes_existing_skills is refreshes


def test_new_class_never_hands_retained_memory_down():
    sentinel = object()
    assert UpdatePolicy("new_class").history_for_training(sentinel) is None
    assert UpdatePolicy("replay").history_for_training(sentinel) is sentinel
    assert UpdatePolicy("refresh", 3).history_for_training(sentinel) is sentinel


def test_policy_rejects_inconsistent_arguments():
    with pytest.raises(ValueError, match="update_mode"):
        UpdatePolicy("everything")
    with pytest.raises(ValueError, match="new_class"):
        UpdatePolicy("new_class", 3)
    with pytest.raises(ValueError, match="positive"):
        UpdatePolicy("replay", 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"replay_samples_per_class": 3, "update_mode": "new_class"}, "new_class"),
        ({"replay_samples_per_class": 11}, "cannot exceed memory_per_class"),
        ({"update_mode": "refresh", "reuse_is_mutable": False}, "reuse_is_mutable"),
        ({"stage1_chunk_size": 4}, "batch_stage1"),
        ({"force_decision": "maybe"}, "force_decision"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
        ({"memory_per_class": 0}, "memory_per_class"),
        ({"class_train_epochs": 0}, "class_train_epochs"),
        ({"train_samples_per_class": 0}, "train_samples_per_class"),
    ],
)
def test_incompatible_strategy_parameters_fail_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_strategy(2, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"update_mode": "replay", "replay_samples_per_class": 10},
        {"update_mode": "refresh", "replay_samples_per_class": 3},
        {"batch_stage1": True, "stage1_chunk_size": 4},
        {"force_decision": "scratch"},
    ],
)
def test_documented_valid_combinations_are_accepted(kwargs):
    make_strategy(2, **kwargs)


def test_removed_parameters_are_rejected():
    for name in (
        "class_train_mode",
        "cl_update_mode",
        "refresh_existing_skills",
        "binary_negative_pool",
        "debug_scores",
        "probe_seed",
        "training_seed",
        "class_train_batch_size",
        "eval_every",
    ):
        with pytest.raises(TypeError):
            make_strategy(2, **{name: 1})


def test_selection_is_a_pure_function_of_seed_and_class():
    inputs = torch.arange(40).reshape(20, 2)
    targets = torch.zeros(20, dtype=torch.long)
    first = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    torch.manual_seed(12345)  # the global RNG must be irrelevant
    again = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    other = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=9)
    expected = torch.randperm(20, generator=torch.Generator().manual_seed(5))[:4]
    assert torch.equal(first[0], again[0])
    assert torch.equal(first[0], inputs[expected])
    assert not torch.equal(first[0], other[0])
    full = select_historical_samples(inputs, targets, limit=None, seed=0, class_id=0)
    assert full[0] is inputs
    with pytest.raises(ValueError):
        select_historical_samples(inputs, targets, limit=0, seed=0, class_id=0)
