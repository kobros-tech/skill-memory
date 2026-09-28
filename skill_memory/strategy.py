# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""High-level Avalanche strategy with Skill Memory and ML evaluation.

The strategy integrates two distinct learning/evaluation processes:

1. Skill Memory
   - class-level REUSE/SCRATCH decisions
   - skill allocation and storage
   - class-to-skill bookkeeping

2. Anonymous ML evaluator
   - receives only x at prediction time
   - learns x -> y from frozen examples retained by Skill Memory
   - is trained on all accumulated evaluation memory
   - evaluates all classes seen so far
   - provides the methodology used to measure non-forgetting

The ML evaluator is intentionally independent from the Skill Memory model.
Its purpose is to measure whether an independently trained classifier can
recover the class identity of anonymous samples from the accumulated
retained data after continual training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate

from .cl.decision import DEFAULT_MAX_SAFETY_CANDIDATES
from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import SkillMemory
from .cl.training import VALID_CLASS_TRAIN_MODES
from .diagnostics.timing import TimingAccumulator
from .evaluation.cl_evaluator import CLEvaluationPlugin
from .evaluation.memory import EvaluationMemoryPlugin


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy integrating Skill Memory and anonymous ML evaluation.

    The Avalanche strategy is responsible for lifecycle integration and for
    exposing the complete experiment through one public object.

    train_epochs controls the number of epochs used for each explicit
    class-training pass; Avalanche's mixed-experience training loop is disabled
    by SkillMemoryPlugin after that custom training has been scheduled.

    Skill Memory training and evaluation-memory retention remain owned by
    ``EvaluationMemoryPlugin``, which extends ``SkillMemoryPlugin``. The
    independent ML evaluator is the sole evaluation methodology used by the
    normal ``strategy.eval()`` lifecycle. Direct Skill Memory diagnostics
    (``skill_memory.diagnostics``) are intentionally separate from this
    strategy and never run as part of it.

    ``diagnose=False`` (the default) means `self.timing` and the skill
    memory plugin's own `self.timing` never record anything -- every
    `self.timing.track(...)` call site in the codebase becomes a true
    no-op. Set ``diagnose=True`` to have `skill_memory.diagnostics.timing_report`
    return real numbers; it has no effect on which evaluation methodology
    `strategy.eval()` uses, or on whether `skill_memory.diagnostics`'
    oracle-routed functions can be called -- those always separately
    require ``diagnose=True`` at their own call site regardless of this
    flag.
    """

    #: Bucket name used with `self.timing` (see
    #: `skill_memory.diagnostics.timing_report`).
    TIMING_EVALUATION = "cl_evaluation"

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: EvaluationPlugin | None = None,
        plugins: list[SupervisedPlugin] | None = None,
        eval_every: int = -1,
        peval_mode: str = "epoch",
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        probe_seed: int | None = None,
        max_safety_candidates: int | None = DEFAULT_MAX_SAFETY_CANDIDATES,
        class_train_batch_size: int = 64,
        class_train_mode: str = "multiclass",
        validation_fraction: float = 0.2,
        validation_seed: int = 0,
        reuse_is_mutable: bool = True,
        force_decision: str | None = None,
        eval_memory_per_class: int = 20,
        skill_train_samples_per_class: int | None = None,
        eval_memory_seed: int = 0,
        train_mb_size: int = 64,
        train_epochs: int = 1,
        eval_mb_size: int = 64,
        device: torch.device | str | None = None,
        verbose: bool = True,
        cl_update_mode: str = "replay",
        cl_replay_per_class: int = 5,
        diagnose: bool = False,
        strict_protocol: bool = True,
        binary_negative_pool=None,
        batch_stage1: bool = False,
        stage1_chunk_size: int | None = None,
    ) -> None:
        if eval_memory_per_class <= 0:
            raise ValueError("eval_memory_per_class must be positive")

        if skill_train_samples_per_class is None:
            skill_train_samples_per_class = eval_memory_per_class
        if skill_train_samples_per_class <= 0:
            raise ValueError("skill_train_samples_per_class must be positive")

        if train_epochs < 1:
            raise ValueError("train_epochs must be at least 1")

        if class_train_mode not in VALID_CLASS_TRAIN_MODES:
            raise ValueError(
                f"class_train_mode must be one of {VALID_CLASS_TRAIN_MODES}"
            )

        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        self.verbose = verbose
        self.cl_update_mode = cl_update_mode
        self.cl_replay_per_class = int(cl_replay_per_class)
        self.train_epochs = train_epochs
        self.diagnose = bool(diagnose)
        self.timing = TimingAccumulator(enabled=self.diagnose)

        # ------------------------------------------------------------------
        # Skill Memory
        # ------------------------------------------------------------------

        self.memory = SkillMemory(max_skills=max_skills)

        # EvaluationMemoryPlugin extends SkillMemoryPlugin. Therefore there
        # is exactly one Skill Memory plugin in the Avalanche plugin list.
        self.plugin = EvaluationMemoryPlugin(
            memory=self.memory,
            max_skills=max_skills,
            forgetting_margin=forgetting_margin,
            score_floor=score_floor,
            probe_batch_size=probe_batch_size,
            probe_batches=probe_batches,
            probe_seed=probe_seed,
            max_safety_candidates=max_safety_candidates,
            class_train_epochs=train_epochs,
            class_train_batch_size=class_train_batch_size,
            class_train_mode=class_train_mode,
            samples_per_class=skill_train_samples_per_class,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            reuse_is_mutable=reuse_is_mutable,
            force_decision=force_decision,
            eval_memory_per_class=eval_memory_per_class,
            eval_memory_seed=eval_memory_seed,
            verbose=verbose,
            diagnose=self.diagnose,
            strict_protocol=strict_protocol,
            binary_negative_pool=binary_negative_pool,
            cl_update_mode=cl_update_mode,
            cl_replay_per_class=cl_replay_per_class,
            batch_stage1=batch_stage1,
            stage1_chunk_size=stage1_chunk_size,
        )

        self.cl_evaluation_plugin = CLEvaluationPlugin(
            memory_plugin=self.plugin,
            verbose=verbose,
            strict_protocol=strict_protocol,
        )
        self.evaluation_plugin = self.cl_evaluation_plugin

        strategy_plugins: list[SupervisedPlugin] = [
            self.plugin,
            self.evaluation_plugin,
        ]

        if plugins:
            strategy_plugins.extend(plugins)

        if evaluator is None:
            evaluator = EvaluationPlugin(
                accuracy_metrics(stream=True),
                loss_metrics(stream=True),
            )

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=evaluator,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            eval_every=eval_every,
            peval_mode=peval_mode,
            device=device,
            plugins=strategy_plugins,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, exp_list, **kwargs):
        """Run normal Avalanche evaluation using stored Skill Memory states."""
        with self.timing.track(self.TIMING_EVALUATION):
            avalanche_results = super().eval(exp_list, **kwargs)
        avalanche_results.update(self.evaluation_plugin.results())
        return avalanche_results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> dict[str, Any]:
        """Return the Skill Memory CL evaluation results."""
        return self.evaluation_plugin.results()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def skill_memory(self) -> SkillMemory:
        """Return the underlying Skill Memory."""
        return self.memory

    @property
    def skill_memory_plugin(self) -> SkillMemoryPlugin:
        """Return the underlying Skill Memory plugin."""
        return self.plugin
