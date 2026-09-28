# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Frozen raw evaluation memory retained alongside Skill Memory training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cl.skill_memory_plugin import SkillMemoryPlugin
from ..utils.probing import _dataset_labels
from ..utils.protocol_guard import assert_memory_classes_match


@dataclass
class EvaluationMemory:
    """Frozen raw examples retained for one class."""

    inputs: torch.Tensor
    targets: torch.Tensor
    class_id: int

    @property
    def size(self) -> int:
        """Return the number of retained samples."""
        return int(self.targets.numel())


class EvaluationMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory plugin with independent evaluation-memory retention.

    Skill Memory remains responsible for:

    - REUSE/SCRATCH decisions
    - skill allocation
    - class-to-skill bookkeeping
    - training
    - storing skill states

    This subclass additionally retains a bounded set of raw examples for
    every class encountered by the training experience.

    The retained data contains only:

        x, y

    It contains no Skill Memory weights or derived representations.
    """

    def __init__(
        self,
        *args,
        eval_memory_per_class: int = 20,
        eval_memory_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")
        self.eval_memory_per_class = int(eval_memory_per_class)
        self.eval_memory_seed = int(eval_memory_seed)
        self.eval_memory: list[EvaluationMemory] = []

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Run Skill Memory lifecycle, then retain evaluation examples.

        ``SkillMemoryPlugin.after_training_exp`` is called first so the
        Skill Memory bookkeeping is completed before evaluation memory is
        captured.
        """
        super().after_training_exp(
            strategy,
            **kwargs,
        )
        experience = strategy.experience
        memories = self._build_evaluation_memory(experience)
        if self.strict_protocol:
            # Evaluation memory may only hold classes of THIS (already
            # trained) experience -- never another, in particular future, one.
            assert_memory_classes_match(
                [memory.class_id for memory in memories], experience
            )
        self.eval_memory.extend(memories)
        if self.verbose:
            experience_index = int(
                getattr(
                    experience,
                    "current_experience",
                    0,
                )
            )

            print(
                f"Evaluation memory {experience_index}: "
                f"{sum(memory.size for memory in memories)} "
                f"samples, "
                f"classes={[memory.class_id for memory in memories]}"
            )

    def _build_evaluation_memory(
        self,
        experience,
    ) -> list[EvaluationMemory]:
        """Retain a deterministic bounded sample for each actual class.

        Classes are discovered from the actual samples in the experience
        dataset rather than from an assumed experience layout.

        This is important for generic Avalanche benchmarks where the number
        of classes per experience is not necessarily fixed.
        """
        dataset = experience.dataset
        samples_by_class: dict[int, list[int]] = {}

        # Labels come from the dataset's `.targets` when it has them (cached,
        # no decoding); only the handful of samples actually retained below
        # are ever decoded. Scanning `dataset[i]` for every sample just to
        # read its label used to dominate the run time.
        for index, target in enumerate(_dataset_labels(dataset)):
            samples_by_class.setdefault(target, []).append(index)

        experience_index = int(
            getattr(
                experience,
                "current_experience",
                0,
            )
        )
        generator = torch.Generator()
        generator.manual_seed(self.eval_memory_seed + experience_index)
        memories: list[EvaluationMemory] = []

        for class_id in sorted(samples_by_class):
            indices = samples_by_class[class_id]

            if len(indices) > self.eval_memory_per_class:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()

                indices = [
                    indices[position]
                    for position in permutation[: self.eval_memory_per_class]
                ]

            inputs: list[torch.Tensor] = []
            targets: list[int] = []

            for index in indices:
                sample = dataset[index]
                if len(sample) < 2:
                    raise RuntimeError(
                        "Evaluation dataset samples must contain (input, target)."
                    )
                input_tensor = sample[0]

                if not isinstance(
                    input_tensor,
                    torch.Tensor,
                ):
                    input_tensor = torch.as_tensor(input_tensor)

                inputs.append(input_tensor.detach().cpu())
                targets.append(int(sample[1]))

            if not inputs:
                raise RuntimeError(
                    f"Class {class_id} produced an empty evaluation memory."
                )

            memories.append(
                EvaluationMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.tensor(
                        targets,
                        dtype=torch.long,
                    ),
                    class_id=class_id,
                )
            )

        return memories
