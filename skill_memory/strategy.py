# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""High-level Avalanche strategy: Skill Memory training + CL evaluation.

Two cooperating components, both owned by :class:`SkillMemoryStrategy`:

1. **Skill Memory training** (``EvaluationMemoryPlugin``, a
   ``SkillMemoryPlugin`` subclass)
   - class-level REUSE/SCRATCH decisions by functional probing,
   - skill allocation, storage and class->skill bookkeeping,
   - a bounded per-class *retained memory* of frozen examples,
   - class training under an explicit replay policy.

2. **CL evaluation** (``CLEvaluationPlugin``)
   - receives only ``x`` at prediction time,
   - scores every canonical class with the stored skill that owns it,
   - applies a Platt calibration fitted on each skill's calibration hold-out,
   - **trains no evaluator model** and consults no external classifier.

Historical-data semantics are defined once, in :mod:`skill_memory.cl.replay`:
``update_mode`` (``new_class`` | ``small_replay`` | ``replay``) controls how
much *retained* history enters class training, and the separate
``update_mode="refresh"`` controls whether existing skills are
retrained on the enlarged domain.  ``docs/MATHEMATICS.md`` gives the formulas.
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
from .evaluation.cl_evaluator import DEFAULT_EVAL_CHUNK_SIZE, CLEvaluationPlugin
from .evaluation.memory import EvaluationMemoryPlugin


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy integrating Skill Memory training and CL evaluation.

    The Avalanche strategy is responsible for lifecycle integration and for
    exposing the complete experiment through one public object.

    train_epochs controls the number of epochs used for each explicit
    class-training pass; Avalanche's mixed-experience training loop is disabled
    by SkillMemoryPlugin after that custom training has been scheduled.

    Skill Memory training and evaluation-memory retention remain owned by
    ``EvaluationMemoryPlugin``, which extends ``SkillMemoryPlugin``.
    ``CLEvaluationPlugin`` is the sole evaluation methodology used by the
    normal ``strategy.eval()`` lifecycle. Direct Skill Memory diagnostics
    (``skill_memory.diagnostics``) are intentionally separate from this
    strategy and never run as part of it.

    Replay / refresh parameters
    ---------------------------
    ``update_mode``
        ``"new_class"``: current-class data only (historical replay = 0);
        ``"small_replay"``: current class + at most ``replay_samples_per_class``
        retained examples per old class; ``"replay"``: current class + *all
        currently retained* examples per old class.  The retained memory is
        bounded by ``memory_per_class``, so ``replay`` is not "all
        historical training data".
    ``update_mode="refresh"``
        Replay incoming classes and additionally retrain every pre-existing skill
        on the enlarged domain after each experience.
    ``binary_negative_pool`` / ``allow_offline_negative_pool``
        Offline oracle negatives for ablations; rejected unless explicitly
        allowed, and never combined with ``new_class``.
    ``training_seed``
        Seeds mini-batch order and balanced re-sampling for reproducibility.
    ``eval_chunk_size``, ``debug_scores``
        Evaluator speed knob and verbose score printout (see
        :class:`~skill_memory.evaluation.cl_evaluator.CLEvaluationPlugin`).

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
        memory_per_class: int = 20,
        train_samples_per_class: int | None = None,
        memory_seed: int = 0,
        train_mb_size: int = 64,
        class_train_epochs: int = 1,
        eval_mb_size: int = 64,
        device: torch.device | str | None = None,
        verbose: bool = True,
        update_mode: str = "replay",
        replay_samples_per_class: int | None = None,
        diagnose: bool = False,
        strict_protocol: bool = True,
        binary_negative_pool=None,
        allow_offline_negative_pool: bool = False,
        training_seed: int = 0,
        batch_stage1: bool = False,
        stage1_chunk_size: int | None = None,
        eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE,
        debug_scores: bool = False,
    ) -> None:
        if memory_per_class <= 0:
            raise ValueError("memory_per_class must be positive")

        if train_samples_per_class is None:
            train_samples_per_class = memory_per_class
        if train_samples_per_class <= 0:
            raise ValueError("train_samples_per_class must be positive")

        if replay_samples_per_class is not None and update_mode == "new_class":
            raise ValueError(
                "replay_samples_per_class is only valid for replay or refresh"
            )
        if (
            replay_samples_per_class is not None
            and replay_samples_per_class > memory_per_class
        ):
            raise ValueError(
                "replay_samples_per_class cannot exceed memory_per_class"
            )

        if update_mode == "refresh" and not reuse_is_mutable:
            raise ValueError("update_mode=refresh requires reuse_is_mutable=True")

        if class_train_epochs < 1:
            raise ValueError("class_train_epochs must be at least 1")

        if class_train_mode not in VALID_CLASS_TRAIN_MODES:
            raise ValueError(
                f"class_train_mode must be one of {VALID_CLASS_TRAIN_MODES}"
            )

        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        self.verbose = verbose
        self.train_epochs = class_train_epochs
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
            class_train_epochs=class_train_epochs,
            class_train_batch_size=class_train_batch_size,
            class_train_mode=class_train_mode,
            samples_per_class=train_samples_per_class,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            reuse_is_mutable=reuse_is_mutable,
            force_decision=force_decision,
            memory_per_class=memory_per_class,
            memory_seed=memory_seed,
            verbose=verbose,
            diagnose=self.diagnose,
            strict_protocol=strict_protocol,
            binary_negative_pool=binary_negative_pool,
            allow_offline_negative_pool=allow_offline_negative_pool,
            update_mode=update_mode,
            replay_samples_per_class=replay_samples_per_class,
            training_seed=training_seed,
            batch_stage1=batch_stage1,
            stage1_chunk_size=stage1_chunk_size,
        )

        self.cl_evaluation_plugin = CLEvaluationPlugin(
            memory_plugin=self.plugin,
            verbose=verbose,
            strict_protocol=strict_protocol,
            eval_chunk_size=eval_chunk_size,
            debug_scores=debug_scores,
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
            train_epochs=class_train_epochs,
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
    def update_mode(self) -> str:
        """Complete update policy in force."""
        return self.plugin.update_mode

    @property
    def replay_samples_per_class(self) -> int | None:
        """Optional retained-history cap per old class."""
        return self.plugin.replay_samples_per_class

    @property
    def skill_memory(self) -> SkillMemory:
        """Return the underlying Skill Memory."""
        return self.memory

    @property
    def skill_memory_plugin(self) -> SkillMemoryPlugin:
        """Return the underlying Skill Memory plugin."""
        return self.plugin
