# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import torch

from skill_memory.evaluation.candidate_routing import route_skill_candidates


def test_native_candidate_routing_preserves_canonical_argmax():
    scores = torch.tensor([[0.1, 2.0, 0.5, 1.0]])
    result = route_skill_candidates(
        scores,
        {0: 0, 1: 1, 2: 0, 3: 1},
        candidate_k=3,
    )

    assert result.winner == 1
    assert [candidate.class_id for candidate in result.candidates] == [1, 3, 2]
    assert all(candidate.verified for candidate in result.candidates)


def test_native_candidate_routing_filters_unmapped_candidates():
    scores = torch.tensor([[0.1, 2.0, 1.5, 1.0]])
    result = route_skill_candidates(
        scores,
        {1: 7, 3: 8},
        candidate_k=3,
    )

    assert result.winner == 1
    assert result.candidates[1].class_id == 2
    assert not result.candidates[1].verified


def test_candidate_k_is_bounded_by_class_count():
    scores = torch.tensor([[1.0, 2.0]])
    result = route_skill_candidates(scores, {0: 0, 1: 1}, candidate_k=10)

    assert len(result.candidates) == 2
