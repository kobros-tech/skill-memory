# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Replay policy: the *one* place that defines what "historical data" means.

Skill Memory can train a class on four conceptually different data sources:

1. **current**  -- the samples of the experience being trained;
2. **retained** -- the bounded per-class memory kept by
   :class:`~skill_memory.evaluation.memory.EvaluationMemoryPlugin`
   (at most ``eval_memory_per_class`` frozen examples per class);
3. **offline pool** -- an optional, explicitly opted-in *oracle* negative pool
   used only for offline ablations (never a continual-learning setting);
4. **refresh** -- retraining of *existing* skills on the enlarged domain
   (a separate, independent switch, see :class:`RefreshPolicy` below).

The ``cl_update_mode`` flag controls **only** how much of source 2 (and, when
opted in, source 3) is consumed.  With :math:`\mathcal R_t` the retained
memory of all classes seen strictly before experience :math:`t`,
:math:`R_c \subseteq \mathcal R_t` the retained examples of class :math:`c`,
and :math:`K` = ``cl_replay_per_class``:

.. math::

    H_c =
    \begin{cases}
        \varnothing                        & \text{new\_class}\\
        \text{Sample}(R_c,\ \min(K,|R_c|)) & \text{small\_replay}\\
        R_c                                & \text{replay}
    \end{cases}

so the invariant ``|H_c|`` is ``0``, ``min(K, |R_c|)`` and ``|R_c|``
respectively.  ``replay`` therefore means *all currently retained history*, not
*all historical training data*: :math:`R_c` is already bounded.

``Sample`` is a deterministic seeded draw, see :func:`select_historical_samples`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NEW_CLASS = "new_class"
SMALL_REPLAY = "small_replay"
REPLAY = "replay"

#: Valid values for ``cl_update_mode`` (ordered from least to most history).
REPLAY_MODES = (NEW_CLASS, SMALL_REPLAY, REPLAY)


def select_historical_samples(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    limit: int | None,
    seed: int,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Select a deterministic per-class historical replay subset.

    Returns all of ``inputs`` when ``limit`` is ``None`` or ``len(inputs) <=
    limit``; otherwise draws ``limit`` rows without replacement using a
    generator seeded with ``seed + class_id``::

        idx = randperm(n, generator=Generator().manual_seed(seed + class_id))[:limit]

    The selection depends only on ``(seed, class_id, n)`` -- not on the global
    RNG stream -- so two runs with the same seed replay the same examples.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    if limit is None or len(inputs) <= limit:
        return inputs, targets

    generator = torch.Generator().manual_seed(int(seed) + int(class_id))
    indices = torch.randperm(len(inputs), generator=generator)[:limit]
    return inputs[indices], targets[indices]


@dataclass(frozen=True)
class ReplayPolicy:
    """Immutable description of the historical-data policy.

    Parameters
    ----------
    mode:
        One of :data:`REPLAY_MODES`.
    per_class:
        ``K`` -- the per-class cap used by ``small_replay`` (ignored, but
        still validated, by the other modes).
    """

    mode: str = REPLAY
    per_class: int = 5

    def __post_init__(self) -> None:
        if self.mode not in REPLAY_MODES:
            raise ValueError(
                f"cl_update_mode must be one of {REPLAY_MODES}, got {self.mode!r}"
            )
        if int(self.per_class) < 1:
            raise ValueError("cl_replay_per_class must be positive")
        object.__setattr__(self, "per_class", int(self.per_class))

    # -- semantics -----------------------------------------------------

    @property
    def uses_history(self) -> bool:
        """``True`` iff any historical example may enter training."""
        return self.mode != NEW_CLASS

    @property
    def historical_limit(self) -> int | None:
        """Per-class cap on historical examples (``None`` = no cap)."""
        return self.per_class if self.mode == SMALL_REPLAY else None

    def retained_for_training(self, retained_memory):
        """Return the retained memory this policy allows into training.

        ``new_class`` returns ``None`` -- the retained memory is *not even
        passed down*, so it cannot be consumed by accident.
        """
        return retained_memory if self.uses_history else None

    def expected_count(self, available: int) -> int:
        r"""``|H_c|`` for a class with ``available`` retained examples."""
        if not self.uses_history:
            return 0
        limit = self.historical_limit
        return available if limit is None else min(available, limit)

    def describe(self) -> str:
        """Short human-readable description used in logs."""
        if self.mode == NEW_CLASS:
            return "new_class (historical replay = 0)"
        if self.mode == SMALL_REPLAY:
            return f"small_replay (<= {self.per_class} retained examples/class)"
        return "replay (all retained examples/class)"


@dataclass(frozen=True)
class RefreshPolicy:
    r"""Whether *existing* skills are retrained on the enlarged class domain.

    This is deliberately **independent** of :class:`ReplayPolicy`.  When
    enabled, after each logical experience every pre-existing skill :math:`s`
    owning classes :math:`O_s` is retrained once on the observed domain
    :math:`D_t`; the cost is :math:`\mathcal O(\#\text{skills})` extra
    training passes *per experience* (see ``docs/MATHEMATICS.md``).

    A refresh needs positive examples of every owned class, which only exist
    when history is available, so ``refresh`` requires a replay mode other
    than ``new_class`` (validated by :func:`validate_policies`).
    """

    enabled: bool = False


def validate_policies(
    replay: ReplayPolicy,
    refresh: RefreshPolicy,
    *,
    class_train_mode: str,
    has_offline_pool: bool,
    allow_offline_negative_pool: bool,
) -> None:
    """Reject inconsistent combinations at construction time."""
    if refresh.enabled:
        if class_train_mode != "binary_one_vs_rest":
            raise ValueError(
                "refresh_existing_skills=True requires "
                "class_train_mode='binary_one_vs_rest'"
            )
        if not replay.uses_history:
            raise ValueError(
                "refresh_existing_skills=True needs historical data; "
                "it cannot be combined with cl_update_mode='new_class'"
            )
    if has_offline_pool:
        if not allow_offline_negative_pool:
            raise ValueError(
                "binary_negative_pool is an offline *oracle* source, not a "
                "continual-learning mode; pass allow_offline_negative_pool=True "
                "to use it explicitly"
            )
        if not replay.uses_history:
            raise ValueError(
                "binary_negative_pool contradicts cl_update_mode='new_class' "
                "(historical replay must be 0); choose 'small_replay' or 'replay'"
            )
        if class_train_mode != "binary_one_vs_rest":
            raise ValueError(
                "binary_negative_pool requires class_train_mode='binary_one_vs_rest'"
            )
