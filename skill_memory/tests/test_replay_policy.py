# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The replay / refresh policy objects and their validation rules."""

import pytest
import torch

from skill_memory.cl.replay import (
    REPLAY_MODES,
    RefreshPolicy,
    ReplayPolicy,
    select_historical_samples,
    validate_policies,
)


def test_modes_are_ordered_from_least_to_most_history():
    assert REPLAY_MODES == ("new_class", "small_replay", "replay")


@pytest.mark.parametrize(
    ("mode", "uses_history", "limit"),
    [("new_class", False, None), ("small_replay", True, 5), ("replay", True, None)],
)
def test_policy_semantics(mode, uses_history, limit):
    policy = ReplayPolicy(mode, 5)
    assert policy.uses_history is uses_history
    assert policy.historical_limit == limit


def test_new_class_never_hands_retained_memory_down():
    sentinel = object()
    assert ReplayPolicy("new_class").retained_for_training(sentinel) is None
    assert ReplayPolicy("small_replay").retained_for_training(sentinel) is sentinel
    assert ReplayPolicy("replay").retained_for_training(sentinel) is sentinel


@pytest.mark.parametrize(
    ("mode", "available", "expected"),
    [
        ("new_class", 20, 0),
        ("small_replay", 20, 5),
        ("small_replay", 3, 3),
        ("replay", 20, 20),
        ("replay", 0, 0),
    ],
)
def test_expected_count_matches_the_documented_formula(mode, available, expected):
    assert ReplayPolicy(mode, 5).expected_count(available) == expected


def test_invalid_policy_arguments_are_rejected():
    with pytest.raises(ValueError, match="cl_update_mode"):
        ReplayPolicy("everything")
    with pytest.raises(ValueError, match="cl_replay_per_class"):
        ReplayPolicy("small_replay", 0)


def _validate(
    replay, refresh=False, *, mode="binary_one_vs_rest", pool=False, ok=False
):
    validate_policies(
        ReplayPolicy(replay),
        RefreshPolicy(refresh),
        class_train_mode=mode,
        has_offline_pool=pool,
        allow_offline_negative_pool=ok,
    )


def test_refresh_is_independent_of_replay_but_needs_history_and_binary_mode():
    _validate("replay", refresh=True)
    _validate("small_replay", refresh=True)
    with pytest.raises(ValueError, match="new_class"):
        _validate("new_class", refresh=True)
    with pytest.raises(ValueError, match="binary_one_vs_rest"):
        _validate("replay", refresh=True, mode="multiclass")


def test_offline_pool_needs_opt_in_and_cannot_combine_with_new_class():
    with pytest.raises(ValueError, match="allow_offline_negative_pool"):
        _validate("replay", pool=True)
    with pytest.raises(ValueError, match="new_class"):
        _validate("new_class", pool=True, ok=True)
    with pytest.raises(ValueError, match="binary_one_vs_rest"):
        _validate("replay", pool=True, ok=True, mode="multiclass")
    _validate("replay", pool=True, ok=True)
    _validate("small_replay", pool=True, ok=True)


def test_selection_is_pure_function_of_seed_and_class():
    inputs = torch.arange(40).reshape(20, 2)
    targets = torch.zeros(20, dtype=torch.long)
    first = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    torch.manual_seed(12345)  # global RNG must be irrelevant
    again = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    other = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=9)
    assert torch.equal(first[0], again[0])
    assert not torch.equal(first[0], other[0])
    with pytest.raises(ValueError):
        select_historical_samples(inputs, targets, limit=0, seed=0, class_id=0)
