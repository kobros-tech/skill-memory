# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""High-level Avalanche strategy: Skill Memory training + CL evaluation.

Two cooperating plugins, both owned by :class:`SkillMemoryStrategy`:

1. :class:`~skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin` -- trains.
   For every new class it decides REUSE or SCRATCH by functional probing,
   trains a YES/NO verifier under the chosen ``update_mode``, keeps a bounded
   replay memory and the class -> skill bookkeeping.
2. :class:`~skill_memory.evaluation.cl_evaluator.CLEvaluationPlugin` --
   evaluates.  It sees only ``x``, scores every class with the stored skill
   that owns it and applies a Platt calibration fitted on held-out examples.
   No evaluator network is trained.

``docs/MATHEMATICS.md`` gives every formula; the parameter compatibility rules
are listed in :class:`~skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin`.
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
from .diagnostics.timing import TimingAccumulator
from .evaluation.cl_evaluator import DEFAULT_EVAL_CHUNK_SIZE, CLEvaluationPlugin


class SkillMemoryStrategy(SupervisedTemplate):
    """Avalanche strategy integrating Skill Memory training and CL evaluation.

    Use it like any Avalanche strategy::

        strategy.train(experience)          # per experience
        results = strategy.eval(test_stream)

    ``eval`` returns Avalanche's stream metrics plus ``mean_final_accuracy``
    (calibrated), ``raw_mean_final_accuracy`` (uncalibrated -- the primary
    diagnostic), ``final_class_accuracy`` and ``final_class_loss``.

    Avalanche's own mixed-experience loop is disabled (the plugin trains class
    by class), so ``train_mb_size`` is the mini-batch size of every
    class-training pass and ``class_train_epochs`` its epoch count.

    Parameter groups (details and compatibility rules in
    :class:`~skill_memory.cl.skill_memory_plugin.SkillMemoryPlugin`):

    * policy -- ``update_mode``, ``replay_samples_per_class``;
    * data -- ``memory_per_class``, ``train_samples_per_class``,
      ``validation_fraction``;
    * decision -- ``max_skills``, ``forgetting_margin``, ``score_floor``,
      ``probe_batch_size``, ``probe_batches``, ``max_safety_candidates``,
      ``force_decision``, ``reuse_is_mutable``, ``batch_stage1``,
      ``stage1_chunk_size``;
    * run -- ``seed``, ``memory_seed``, ``device``, ``verbose``, ``diagnose``,
      ``strict_protocol``, ``eval_chunk_size``.

    ``diagnose=False`` (default) means no timing is ever recorded; it never
    changes results.
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
        # policy
        update_mode: str = "replay",
        replay_samples_per_class: int | None = None,
        # data
        memory_per_class: int = 20,
        train_samples_per_class: int | None = None,
        validation_fraction: float = 0.2,
        class_train_epochs: int = 1,
        train_mb_size: int = 64,
        eval_mb_size: int = 64,
        # decision
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        max_safety_candidates: int | None = DEFAULT_MAX_SAFETY_CANDIDATES,
        force_decision: str | None = None,
        reuse_is_mutable: bool = True,
        batch_stage1: bool = False,
        stage1_chunk_size: int | None = None,
        # run
        seed: int = 0,
        memory_seed: int = 0,
        device: torch.device | str | None = None,
        verbose: bool = True,
        diagnose: bool = False,
        strict_protocol: bool = True,
        eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE,
    ) -> None:
        device = (
            next(model.parameters()).device if device is None else torch.device(device)
        )
        self.verbose = verbose
        self.train_epochs = class_train_epochs
        self.diagnose = bool(diagnose)
        self.timing = TimingAccumulator(enabled=self.diagnose)

        # All parameter validation lives in the plugin (single source of truth).
        memory = SkillMemory(max_skills=max_skills)
        self.plugin = SkillMemoryPlugin(
            memory=memory,
            max_skills=max_skills,
            forgetting_margin=forgetting_margin,
            score_floor=score_floor,
            probe_batch_size=probe_batch_size,
            probe_batches=probe_batches,
            max_safety_candidates=max_safety_candidates,
            batch_stage1=batch_stage1,
            stage1_chunk_size=stage1_chunk_size,
            force_decision=force_decision,
            reuse_is_mutable=reuse_is_mutable,
            update_mode=update_mode,
            replay_samples_per_class=replay_samples_per_class,
            memory_per_class=memory_per_class,
            train_samples_per_class=train_samples_per_class,
            validation_fraction=validation_fraction,
            class_train_epochs=class_train_epochs,
            batch_size=train_mb_size,
            seed=seed,
            memory_seed=memory_seed,
            verbose=verbose,
            diagnose=self.diagnose,
            strict_protocol=strict_protocol,
        )
        self.cl_evaluation_plugin = CLEvaluationPlugin(
            memory_plugin=self.plugin,
            verbose=verbose,
            strict_protocol=strict_protocol,
            eval_chunk_size=eval_chunk_size,
        )
        self.evaluation_plugin = self.cl_evaluation_plugin

        strategy_plugins: list[SupervisedPlugin] = [self.plugin, self.evaluation_plugin]
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
            eval_every=-1,
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
        """The skill store (``skill_memory_plugin.memory``)."""
        return self.plugin.memory

    @property
    def skill_memory_plugin(self) -> SkillMemoryPlugin:
        """Return the underlying Skill Memory plugin."""
        return self.plugin
