# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Update policy: the *one* place that defines what "historical data" means.

A skill is trained on up to three data sources:

1. **current**  -- samples of the experience being trained;
2. **retained** -- the bounded per-class replay memory (:class:`ReplayMemory`,
   at most ``memory_per_class`` frozen examples of every class seen so far);
3. **refresh**  -- retraining of *existing* skills on the enlarged domain.

``update_mode`` is one complete policy over these sources.  With
:math:`R_c` the retained examples of an old class :math:`c`, :math:`K` =
``replay_samples_per_class`` (``None`` = no cap) and :math:`H_c` the historical
data replayed for :math:`c`:

======== ======================================= =============================
mode     :math:`H_c`                             existing skills
======== ======================================= =============================
new_class :math:`\varnothing`                    frozen
replay   :math:`R_c` (or :math:`K` of them)      frozen
refresh  :math:`R_c` (or :math:`K` of them)      retrained once per experience
======== ======================================= =============================

so :math:`|H_c|` is ``0``, ``|R_c|`` or ``min(K, |R_c|)``.  ``replay`` therefore
means *all currently retained history*, not *all historical training data*:
:math:`R_c` is bounded.  The :math:`K` examples are a deterministic seeded draw
(:func:`select_historical_samples`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NEW_CLASS = "new_class"
REPLAY = "replay"
REFRESH = "refresh"

#: Valid values of ``update_mode`` (ordered from least to most history/work).
UPDATE_MODES = (NEW_CLASS, REPLAY, REFRESH)


@dataclass
class ReplayMemory:
    """Frozen raw examples ``(x, y)`` retained for one class."""

    inputs: torch.Tensor
    targets: torch.Tensor
    class_id: int

    @property
    def size(self) -> int:
        return int(self.targets.numel())


def select_historical_samples(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    limit: int | None,
    seed: int,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically pick at most ``limit`` rows (all rows if ``None``).

    The draw is ``randperm(n, Generator().manual_seed(seed + class_id))[:limit]``:
    a pure function of ``(seed, class_id, n)``, independent of the global RNG.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    if limit is None or len(inputs) <= limit:
        return inputs, targets
    generator = torch.Generator().manual_seed(int(seed) + int(class_id))
    indices = torch.randperm(len(inputs), generator=generator)[:limit]
    return inputs[indices], targets[indices]


@dataclass(frozen=True)
class UpdatePolicy:
    """Immutable, validated description of ``update_mode`` + the replay cap."""

    mode: str = REPLAY
    per_class: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in UPDATE_MODES:
            raise ValueError(
                f"update_mode must be one of {UPDATE_MODES}, got {self.mode!r}"
            )
        if self.per_class is not None:
            if self.mode == NEW_CLASS:
                raise ValueError(
                    "replay_samples_per_class is meaningless with "
                    "update_mode='new_class' (no history is replayed)"
                )
            if int(self.per_class) < 1:
                raise ValueError("replay_samples_per_class must be positive or None")
            object.__setattr__(self, "per_class", int(self.per_class))

    @property
    def uses_history(self) -> bool:
        """``True`` iff any retained example may enter training."""
        return self.mode != NEW_CLASS

    @property
    def refreshes_existing_skills(self) -> bool:
        return self.mode == REFRESH

    def history_for_training(self, retained):
        """The retained memory this policy lets into training.

        ``new_class`` returns ``None``: the memory is not even passed down, so
        it cannot be consumed by accident.
        """
        return retained if self.uses_history else None

    def describe(self) -> str:
        """Short human-readable description used in logs."""
        if not self.uses_history:
            return "new_class (no history; existing skills frozen)"
        history = (
            "all retained examples/class"
            if self.per_class is None
            else f"<= {self.per_class} retained examples/class"
        )
        refresh = "refreshed" if self.refreshes_existing_skills else "frozen"
        return f"{self.mode} ({history}; existing skills {refresh})"
