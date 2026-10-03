# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The complete Skill Memory update policies and their validation rules."""

import pytest
import torch

from skill_memory.cl.replay import (
    REPLAY_MODES,
    REFRESH,
    ReplayPolicy,
    select_historical_samples,
    validate_policies,
)


def test_modes_are_complete_update_policies():
    assert REPLAY_MODES == ("new_class", "replay", "refresh")


@pytest.mark.parametrize(
    ("mode", "uses_history", "limit"),
    [
        ("new_class", False, None),
        ("replay", True, 5),
        ("replay", True, None),
        ("refresh", True, 5),
    ],
)
def test_policy_semantics(mode, uses_history, limit):
    policy = ReplayPolicy(mode, limit)
    assert policy.uses_history is uses_history
    assert policy.historical_limit == limit


def test_new_class_never_hands_retained_memory_down():
    sentinel = object()
    assert ReplayPolicy("new_class").retained_for_training(sentinel) is None
    assert ReplayPolicy("replay", 5).retained_for_training(sentinel) is sentinel
    assert ReplayPolicy("refresh", 5).retained_for_training(sentinel) is sentinel


@pytest.mark.parametrize(
    ("mode", "available", "limit", "expected"),
    [
        ("new_class", 20, None, 0),
        ("replay", 20, 5, 5),
        ("replay", 3, 5, 3),
        ("replay", 20, None, 20),
        ("refresh", 20, 5, 5),
        ("refresh", 0, None, 0),
    ],
)
def test_expected_count_matches_the_documented_formula(
    mode, available, limit, expected
):
    assert ReplayPolicy(mode, limit).expected_count(available) == expected


def test_invalid_policy_arguments_are_rejected():
    with pytest.raises(ValueError, match="update_mode"):
        ReplayPolicy("everything")
    with pytest.raises(ValueError, match="replay_samples_per_class"):
        ReplayPolicy("replay", 0)


def test_refresh_is_a_complete_policy_and_requires_binary_mode():
    assert ReplayPolicy(REFRESH).mode == "refresh"
    validate_policies(
        ReplayPolicy("refresh", 5),
        class_train_mode="binary_one_vs_rest",
        has_offline_pool=False,
        allow_offline_negative_pool=False,
    )
    with pytest.raises(ValueError, match="binary_one_vs_rest"):
        validate_policies(
            ReplayPolicy("refresh", 5),
            class_train_mode="multiclass",
            has_offline_pool=False,
            allow_offline_negative_pool=False,
        )


def test_offline_pool_needs_opt_in_and_cannot_combine_with_new_class():
    with pytest.raises(ValueError, match="allow_offline_negative_pool"):
        validate_policies(
            ReplayPolicy("replay"),
            class_train_mode="binary_one_vs_rest",
            has_offline_pool=True,
            allow_offline_negative_pool=False,
        )
    with pytest.raises(ValueError, match="new_class"):
        validate_policies(
            ReplayPolicy("new_class"),
            class_train_mode="binary_one_vs_rest",
            has_offline_pool=True,
            allow_offline_negative_pool=True,
        )
    with pytest.raises(ValueError, match="binary_one_vs_rest"):
        validate_policies(
            ReplayPolicy("replay"),
            class_train_mode="multiclass",
            has_offline_pool=True,
            allow_offline_negative_pool=True,
        )
    validate_policies(
        ReplayPolicy("replay"),
        class_train_mode="binary_one_vs_rest",
        has_offline_pool=True,
        allow_offline_negative_pool=True,
    )


def test_selection_is_pure_function_of_seed_and_class():
    inputs = torch.arange(40).reshape(20, 2)
    targets = torch.zeros(20, dtype=torch.long)
    first = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    torch.manual_seed(12345)
    again = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=2)
    other = select_historical_samples(inputs, targets, limit=4, seed=3, class_id=9)
    assert torch.equal(first[0], again[0])
    assert not torch.equal(first[0], other[0])
    with pytest.raises(ValueError):
        select_historical_samples(inputs, targets, limit=0, seed=0, class_id=0)


def test_strategy_rejects_redundant_replay_parameters():
    from skill_memory.tests._helpers import make_strategy

    with pytest.raises(ValueError, match="only valid for replay or refresh"):
        make_strategy(
            4,
            update_mode="new_class",
            replay_samples_per_class=2,
        )

    with pytest.raises(ValueError, match="cannot exceed memory_per_class"):
        make_strategy(
            4,
            update_mode="replay",
            replay_samples_per_class=11,
        )

    with pytest.raises(ValueError, match="requires reuse_is_mutable"):
        make_strategy(
            4,
            update_mode="refresh",
            reuse_is_mutable=False,
        )
