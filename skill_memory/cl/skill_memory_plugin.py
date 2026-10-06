# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Avalanche plugin for class-level, probe-based Skill Memory.

An Avalanche experience is only a container.  The strategy extracts the
classes actually present in that experience and handles them independently:

    experience -> class -> REUSE/SCRATCH -> train only that class

Bookkeeping is explicit:

    experience -> [(skill, {classes})]
    class      -> canonical skill
    skill      -> all classes it currently masters

A class that has already been mastered is never re-assigned by the generic
probe heuristic. It deterministically returns to its canonical skill, and
REUSE may update the same reserved skill slot when `reuse_is_mutable=True`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from ..diagnostics.timing import TimingAccumulator
from ..utils.probing import (
    _dataset_labels,
    apply_skill_state_exact,
    classes_in_experience,
    origin_experience,
    prepare_for_classes,
    prepare_for_experience,
    restore_initial_state,
)
from ..utils.protocol_guard import (
    assert_memory_classes_match,
    assert_training_experience,
)
from .decision import DEFAULT_MAX_SAFETY_CANDIDATES, DecisionProbeCache, decide_class
from .replay import ReplayMemory, UpdatePolicy
from .skill_registry import (
    CALIBRATION_EXAMPLES_KEY,
    ClassRecord,
    ExperienceClassMap,
    SkillMemory,
)
from .training import (
    TrainingResult,
    derive_seed,
    train_on_class,
    train_skill_on_domain,
)

logger = logging.getLogger(__name__)


class SkillMemoryPlugin(SupervisedPlugin):
    """Class-level Skill Memory plugin with explicit class bookkeeping."""

    REUSE, SCRATCH = "reuse", "scratch"

    #: Bucket names used with `self.timing` (see
    #: `skill_memory.diagnostics.timing_report`).
    TIMING_DECISION = "skill_memory_decision_probing"
    TIMING_CLASS_TRAINING = "skill_memory_class_training"
    TIMING_DOMAIN_REFRESH = "skill_memory_domain_refresh"

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        # --- REUSE / SCRATCH decision --------------------------------------
        max_skills: int = 200,
        forgetting_margin: float = 0.05,
        score_floor: float | None = 0.9,
        probe_batch_size: int = 64,
        probe_batches: int = 5,
        max_safety_candidates: int | None = DEFAULT_MAX_SAFETY_CANDIDATES,
        batch_stage1: bool = False,
        stage1_chunk_size: int | None = None,
        force_decision: str | None = None,
        reuse_is_mutable: bool = True,
        # --- what a class is trained on ------------------------------------
        update_mode: str = "replay",
        replay_samples_per_class: int | None = None,
        memory_per_class: int = 20,
        train_samples_per_class: int | None = None,
        validation_fraction: float = 0.2,
        class_train_epochs: int = 1,
        batch_size: int = 64,
        # --- reproducibility / run control ---------------------------------
        seed: int = 0,
        memory_seed: int = 0,
        verbose: bool = True,
        diagnose: bool = False,
        strict_protocol: bool = True,
    ):
        """Configure per-class REUSE/SCRATCH decisions and class training.

        Which parameters work together
        ------------------------------
        * ``update_mode`` is one complete policy: ``new_class`` (no history,
          existing skills frozen), ``replay`` (retained history for each new
          class) or ``refresh`` (replay + retrain every existing skill once per
          experience; needs ``reuse_is_mutable=True``).
        * ``replay_samples_per_class`` caps the retained examples per old class
          (``None`` = all retained).  It is an error with ``new_class`` and
          must not exceed ``memory_per_class``.
        * ``train_samples_per_class`` caps the *current* examples per class
          (default: ``memory_per_class``); ``validation_fraction`` of each
          class is held out first and used only to calibrate the evaluator.
        * ``force_decision`` ("scratch" / "reuse") bypasses probing, so
          ``forgetting_margin``, ``score_floor``, ``probe_*``,
          ``max_safety_candidates`` and ``batch_stage1`` then have no effect.
          ``stage1_chunk_size`` requires ``batch_stage1=True``.
        * ``seed`` seeds the probe batches, the calibration hold-out, replay
          selection and all training randomness (mini-batch order, balanced
          re-sampling, dropout); ``memory_seed`` alone seeds which examples
          are retained (default ``0`` keeps the replay memory identical
          across experiment seeds).
        * ``diagnose`` only switches timing on; ``strict_protocol`` enables the
          cheap leakage guards of :mod:`skill_memory.utils.protocol_guard`.
        """
        super().__init__()
        if force_decision not in (None, self.REUSE, self.SCRATCH):
            raise ValueError("force_decision must be None, 'reuse' or 'scratch'")
        if max_safety_candidates is not None and max_safety_candidates < 1:
            raise ValueError("max_safety_candidates must be positive or None")
        if stage1_chunk_size is not None and not batch_stage1:
            raise ValueError("stage1_chunk_size requires batch_stage1=True")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        for name, value in (
            ("memory_per_class", memory_per_class),
            ("class_train_epochs", class_train_epochs),
            ("batch_size", batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if train_samples_per_class is None:
            train_samples_per_class = memory_per_class
        if train_samples_per_class < 1:
            raise ValueError("train_samples_per_class must be positive")

        self.update_policy = UpdatePolicy(update_mode, replay_samples_per_class)
        if (
            replay_samples_per_class is not None
            and replay_samples_per_class > memory_per_class
        ):
            raise ValueError("replay_samples_per_class cannot exceed memory_per_class")
        if self.update_policy.refreshes_existing_skills and not reuse_is_mutable:
            raise ValueError("update_mode='refresh' requires reuse_is_mutable=True")

        self.memory = memory if memory is not None else SkillMemory(max_skills)
        self.class_map = ExperienceClassMap()
        self.forgetting_margin = forgetting_margin
        self.score_floor = score_floor
        self.probe_batch_size = probe_batch_size
        self.probe_batches = probe_batches
        self.max_safety_candidates = (
            None if max_safety_candidates is None else int(max_safety_candidates)
        )
        self.batch_stage1 = bool(batch_stage1)
        self.stage1_chunk_size = stage1_chunk_size
        self.force_decision = force_decision
        self.reuse_is_mutable = reuse_is_mutable
        self.memory_per_class = int(memory_per_class)
        self.train_samples_per_class = int(train_samples_per_class)
        self.validation_fraction = float(validation_fraction)
        self.class_train_epochs = int(class_train_epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.memory_seed = int(memory_seed)
        self.verbose = verbose
        self.diagnose = bool(diagnose)
        self.strict_protocol = bool(strict_protocol)

        #: Bounded frozen examples of every class seen so far (replay source).
        self.replay_memory: list[ReplayMemory] = []
        #: One ``TrainingProvenance.as_dict()`` (+ context) per training call.
        self.training_log: list[dict[str, Any]] = []
        self.last_class_decisions: dict[int, dict[int, dict[str, Any]]] = {}
        self.timing = TimingAccumulator(enabled=self.diagnose)
        self._probe_cache = DecisionProbeCache()
        self._initial_state: dict | None = None
        self._task_active = False
        self._seen_experiences: list = []
        self._training_experience_count = 0
        self._current_training_experience_index: int | None = None
        self._original_train_epochs: int | None = None
        self._new_skills_this_experience: set[int] = set()
        self._pre_eval_state: dict | None = None
        self._eval_active = False

    # Read-only conveniences kept as properties so there is a single source of truth.
    @property
    def update_mode(self) -> str:
        return self.update_policy.mode

    @property
    def replay_samples_per_class(self) -> int | None:
        return self.update_policy.per_class

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        else:
            logger.info(message)

    @staticmethod
    def _snapshot(model) -> dict:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _is_first_subexp(experience) -> bool:
        return getattr(experience, "is_first_subexp", True)

    @staticmethod
    def _is_last_subexp(experience) -> bool:
        return getattr(experience, "is_last_subexp", True)

    @staticmethod
    def _capture_optimizer_groups(strategy) -> dict[str, int]:
        """Map parameter names to their optimizer parameter-group indices."""
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is None:
            return {}

        group_by_id = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                group_by_id[id(parameter)] = group_index

        return {
            name: group_by_id[id(parameter)]
            for name, parameter in strategy.model.named_parameters()
            if id(parameter) in group_by_id
        }

    def _reset_optimizer(
        self,
        strategy,
        group_by_name: dict[str, int] | None = None,
    ) -> None:
        """Rebind optimizer parameters and reset optimizer history.

        Optimizer state is intentionally not part of a stored Skill Memory
        snapshot. Switching skills therefore starts with a clean optimizer
        state instead of leaking momentum or adaptive moments from another
        skill into the restored weights.
        """
        optimizer = getattr(strategy, "optimizer", None)
        if optimizer is None:
            return

        params_by_name = dict(strategy.model.named_parameters())
        if not optimizer.param_groups:
            optimizer.add_param_group({"params": list(params_by_name.values())})
            optimizer.state.clear()
            return

        group_by_name = group_by_name or {}
        grouped_params = [[] for _ in optimizer.param_groups]
        for name, parameter in params_by_name.items():
            group_index = group_by_name.get(name, 0)
            if not 0 <= group_index < len(grouped_params):
                group_index = 0
            grouped_params[group_index].append(parameter)

        for group, parameters in zip(
            optimizer.param_groups,
            grouped_params,
            strict=True,
        ):
            group["params"] = parameters

        optimizer.state.clear()

    def _scratch_reset(self, strategy, experience) -> None:
        if self._initial_state is None:
            raise RuntimeError("Initial model state has not been captured")
        group_by_name = self._capture_optimizer_groups(strategy)
        restore_initial_state(strategy.model, self._initial_state)
        # A fresh skill must have the head required by the current
        # Avalanche experience before its single-class training pass.
        prepare_for_experience(strategy.model, experience)
        self._reset_optimizer(strategy, group_by_name)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def before_training_exp(self, strategy, **kwargs) -> None:
        """Decide REUSE/SCRATCH for, and train, every class of this experience.

        Avalanche may split a logical experience into sub-experiences; every
        one of them is processed (the first opens the logical experience, the
        last closes it in :meth:`after_training_exp`).
        """
        experience = strategy.experience
        if self.strict_protocol:
            assert_training_experience(experience)

        experience_index = self._open_experience(strategy, experience)
        classes = classes_in_experience(experience)
        self._log(
            f"Experience {experience_index} "
            f"subexp(first={self._is_first_subexp(experience)}, "
            f"last={self._is_last_subexp(experience)}): classes={classes}"
        )
        if not classes:
            self._log(f"Experience {experience_index}: empty sub-experience")
            return

        policy = self.update_policy
        history = policy.history_for_training(self.replay_memory)
        self._log(f"Update policy: {policy.describe()}")
        for target_class in classes:
            decision = self._decide(strategy, experience, target_class)
            if decision["decision"] == self.REUSE:
                self._reuse_skill(
                    strategy,
                    experience,
                    experience_index,
                    target_class,
                    decision,
                    history,
                )
            else:
                self._scratch_skill(
                    strategy,
                    experience,
                    experience_index,
                    target_class,
                    decision,
                    history,
                )
            self._record_decision(experience_index, target_class, decision)

    def _open_experience(self, strategy, experience) -> int:
        """Start (first sub-experience) or continue a logical experience.

        Returns its index. The first sub-experience also disables Avalanche's
        own mixed-experience loop, so it never retrains after our explicit
        class-by-class loop.
        """
        if self._is_first_subexp(experience):
            if self._task_active:
                raise RuntimeError(
                    "A new first sub-experience arrived while the previous "
                    "logical experience is still active"
                )
            self._task_active = True
            self._current_training_experience_index = self._training_experience_count
            self._training_experience_count += 1
            index = self._current_training_experience_index
            if self._initial_state is None:
                self._initial_state = self._snapshot(strategy.model)
            self.last_class_decisions[index] = {}
            self._new_skills_this_experience = set()
            self._original_train_epochs = getattr(strategy, "train_epochs", None)
            if self._original_train_epochs is not None:
                strategy.train_epochs = 0
            return index
        if not self._task_active or self._current_training_experience_index is None:
            raise RuntimeError(
                "Received a non-first sub-experience without an active "
                "logical training experience"
            )
        return self._current_training_experience_index

    def _decide(self, strategy, experience, target_class: int) -> dict[str, Any]:
        """REUSE or SCRATCH for one class (functional probing, no training)."""
        started = time.perf_counter() if self.diagnose else None
        with self.timing.track(self.TIMING_DECISION):
            decision = decide_class(
                strategy,
                experience,
                target_class,
                self.memory,
                self.class_map,
                self.probe_batch_size,
                self.probe_batches,
                self.seed,
                self._seen_experiences,
                self.forgetting_margin,
                self.score_floor,
                self.force_decision,
                self._log,
                max_safety_candidates=self.max_safety_candidates,
                probe_cache=self._probe_cache,
                batch_stage1=self.batch_stage1,
                stage1_chunk_size=self.stage1_chunk_size,
            )
        if started is not None:
            self._log(
                f"Class {target_class}: imagination+decision "
                f"time={time.perf_counter() - started:.2f}s"
            )
        return decision

    def _reuse_skill(
        self, strategy, experience, experience_index, target_class, decision, history
    ) -> None:
        """Restore the chosen skill and (if mutable) train the class into it."""
        skill = decision["skill"]
        self._log(
            f"Class {target_class}: REUSE skill {skill} "
            f"(mutable={self.reuse_is_mutable})"
        )
        # Restore the snapshot exactly: re-adapting the model to the current
        # sub-experience can traverse incompatible FlatData indices.
        groups = self._capture_optimizer_groups(strategy)
        apply_skill_state_exact(strategy.model, self.memory.state(skill))
        # A new class may reuse a skill whose head is narrower: grow the live
        # head first. Known-class reuse stays an exact snapshot restore.
        if not decision.get("known_class", False):
            prepare_for_experience(strategy.model, experience)
        self._reset_optimizer(strategy, groups)

        if not self.reuse_is_mutable:
            self._log(f"Class {target_class}: skill {skill} left unchanged")
            return
        result = self._train_class(
            strategy, experience, experience_index, target_class, history
        )
        previous = self.memory.metadata(skill)
        self.memory.store(
            skill,
            strategy.model.state_dict(),
            metadata={
                **previous,
                "last_updated_class": target_class,
                "last_updated_experience": experience_index,
                CALIBRATION_EXAMPLES_KEY: self._merge_calibration(
                    previous.get(CALIBRATION_EXAMPLES_KEY), result
                ),
            },
        )
        self._probe_cache.invalidate_skill(skill)
        self._log(f"Class {target_class}: skill {skill} updated in place")

    def _scratch_skill(
        self, strategy, experience, experience_index, target_class, decision, history
    ) -> None:
        """Allocate a new skill from the initial state and train the class."""
        skill = self.memory.allocate()
        self._log(f"Class {target_class}: SCRATCH -> new skill {skill}")
        self._scratch_reset(strategy, experience)
        result = self._train_class(
            strategy, experience, experience_index, target_class, history
        )
        self.memory.store(
            skill,
            strategy.model.state_dict(),
            metadata={
                "acquisition_decision": self.SCRATCH,
                "experience_index": experience_index,
                "last_updated_class": target_class,
                "probe_batch_size": self.probe_batch_size,
                "probe_batches": self.probe_batches,
                "seed": self.seed,
                CALIBRATION_EXAMPLES_KEY: self._merge_calibration(None, result),
            },
        )
        decision["skill"] = skill
        self._new_skills_this_experience.add(skill)

    def _record_decision(self, experience_index, target_class, decision) -> None:
        self.last_class_decisions[experience_index][target_class] = decision
        self.class_map.record(
            ClassRecord(
                experience_index=experience_index,
                class_id=target_class,
                decision=decision["decision"],
                skill=decision["skill"],
                new_score=decision.get("new_score", 0.0),
                old_accuracy=decision.get("old_accuracy", 0.0),
                new_accuracy=decision.get("new_accuracy", 0.0),
            )
        )

    # ------------------------------------------------------------------
    # TRAINING HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_calibration(previous, result: TrainingResult) -> dict:
        """Merge a call's hold-out into the per-class calibration examples.

        The hold-out is **calibration data only** (Platt scaling of the
        verifier); it was excluded from training and is never replayed.
        """
        merged = dict(previous or {})
        targets = result.validation_targets
        for class_id in torch.unique(targets).tolist():
            mask = targets == int(class_id)
            merged[int(class_id)] = (
                result.validation_inputs[mask].clone(),
                targets[mask].clone(),
            )
        return merged

    def _record_training(self, experience_index: int, result: TrainingResult) -> None:
        entry = result.provenance.as_dict()
        entry.update(
            experience_index=experience_index,
            update_mode=self.update_policy.mode,
            replay_samples_per_class=self.update_policy.per_class,
        )
        self.training_log.append(entry)

    def _train_class(
        self,
        strategy,
        experience,
        experience_index: int,
        target_class: int,
        historical_memory,
    ) -> TrainingResult:
        """Train one class under the replay policy and log its provenance."""
        policy = self.update_policy
        with self.timing.track(self.TIMING_CLASS_TRAINING):
            result = train_on_class(
                strategy,
                experience,
                target_class,
                self.class_train_epochs,
                self.batch_size,
                validation_fraction=self.validation_fraction,
                split_seed=self.seed,
                retained_memory=historical_memory,
                samples_per_class=self.train_samples_per_class,
                historical_samples_per_class=policy.per_class,
                sampler_seed=derive_seed(self.seed, experience_index, target_class),
            )
        self._record_training(experience_index, result)
        return result

    def _refresh_existing_skills(self, strategy, experience, experience_index) -> None:
        r"""Retrain pre-existing binary skills on the enlarged class domain.

        Runs only when ``update_mode="refresh"``. With :math:`S`
        pre-existing skills the cost is
        :math:`S` extra training passes for this experience; skills created
        in this very experience were already trained on the observed domain
        and are skipped.  Historical data enters under the replay policy.
        """
        policy = self.update_policy
        if not policy.refreshes_existing_skills:
            return
        if not self.reuse_is_mutable:
            self._log("Skill refresh skipped: reuse_is_mutable=False")
            return

        retained_memory = policy.history_for_training(self.replay_memory)
        current_classes = set(classes_in_experience(experience))
        observed_classes = set(current_classes)
        for skill in self.memory.slots():
            observed_classes.update(self.class_map.classes_for_skill(skill))

        self._log(
            f"Skill refresh: policy={policy.describe()}; "
            f"current_classes={sorted(current_classes)}; "
            f"historical_classes={sorted(observed_classes - current_classes)}"
        )
        if len(observed_classes) < 2:
            return

        for skill in sorted(self.memory.slots()):
            owned_classes = self.class_map.classes_for_skill(skill)
            if not owned_classes:
                continue

            previous_metadata = self.memory.metadata(skill)
            if skill in self._new_skills_this_experience:
                self.memory.store(
                    skill,
                    self.memory.state(skill),
                    metadata={
                        **previous_metadata,
                        "domain_classes": sorted(observed_classes),
                        "domain_update_experience": experience_index,
                    },
                )
                self._log(
                    f"Skill refresh: skill {skill} already trained on the "
                    "current observed domain; skipping"
                )
                continue

            if set(previous_metadata.get("domain_classes", [])) == observed_classes:
                self._log(
                    f"Skill refresh: skill {skill} already covers the "
                    "observed domain; skipping"
                )
                continue

            group_by_name = self._capture_optimizer_groups(strategy)
            apply_skill_state_exact(strategy.model, self.memory.state(skill))
            prepare_for_classes(strategy.model, observed_classes)
            self._reset_optimizer(strategy, group_by_name)

            with self.timing.track(self.TIMING_DOMAIN_REFRESH):
                result = train_skill_on_domain(
                    strategy,
                    experience,
                    owned_classes,
                    observed_classes,
                    self.class_train_epochs,
                    self.batch_size,
                    validation_fraction=self.validation_fraction,
                    split_seed=self.seed,
                    retained_memory=retained_memory,
                    samples_per_class=self.train_samples_per_class,
                    historical_samples_per_class=policy.per_class,
                    sampler_seed=derive_seed(self.seed, experience_index, skill, 7919),
                )
            self._record_training(experience_index, result)

            self.memory.store(
                skill,
                strategy.model.state_dict(),
                metadata={
                    **previous_metadata,
                    "domain_update_experience": experience_index,
                    "domain_classes": sorted(observed_classes),
                    CALIBRATION_EXAMPLES_KEY: self._merge_calibration(
                        previous_metadata.get(CALIBRATION_EXAMPLES_KEY), result
                    ),
                },
            )
            self._probe_cache.invalidate_skill(skill)
            self._log(
                f"Skill refresh: skill {skill} (owns {sorted(owned_classes)}) "
                f"retrained on domain {sorted(observed_classes)}; "
                f"counts={result.provenance.retained}"
            )

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Close the logical experience (last sub-experience) and retain examples."""
        experience = strategy.experience
        if self._is_last_subexp(experience):
            self._close_logical_experience(strategy, experience)
        # Retention runs after the lifecycle step on purpose: a refresh must
        # only ever see classes of *earlier* experiences.
        self._retain_examples(experience)

    def _close_logical_experience(self, strategy, experience) -> None:
        """Log assignments, refresh existing skills and reset per-experience state."""
        experience_index = self._current_training_experience_index
        if experience_index is None:
            raise RuntimeError("Missing current training experience index")

        grouped = self.class_map.skills_for_experience(experience_index)
        for skill, classes in grouped:
            self._log(
                f"Experience {experience_index}: skill {skill} covers "
                f"classes {sorted(classes)}"
            )

        # Explicit class -> skill view.  This is intentionally printed for
        # every class in the logical experience, including classes that were
        # attached to an already-existing skill.  It makes it impossible to
        # mistake a skill index for a class index.
        assignments = self.class_map.class_skill_for_experience(experience_index)
        self._log(
            f"Experience {experience_index}: class->skill "
            + ", ".join(
                f"{class_id}->{skill}"
                for class_id, skill in sorted(assignments.items())
            )
        )

        self._refresh_existing_skills(strategy, experience, experience_index)

        self._new_skills_this_experience = set()

        if self._original_train_epochs is not None:
            strategy.train_epochs = self._original_train_epochs
        self._original_train_epochs = None
        self._seen_experiences.append(origin_experience(experience))
        # Skill states change as training proceeds; old-class probes stay
        # valid (their cache keys encode the exact pool they came from).
        self._probe_cache.end_of_experience()
        self._current_training_experience_index = None
        self._task_active = False

    def _retain_examples(self, experience) -> None:
        """Append the replay memory of this (sub-)experience's classes."""
        memories = self._build_replay_memory(experience)
        if self.strict_protocol:
            # The memory may only hold classes of THIS (already trained)
            # experience -- never another, in particular future, one.
            assert_memory_classes_match(
                [memory.class_id for memory in memories], experience
            )
        self.replay_memory.extend(memories)
        self._log(
            f"Replay memory +{sum(m.size for m in memories)} samples, "
            f"classes={[m.class_id for m in memories]}"
        )

    def _build_replay_memory(
        self,
        experience,
    ) -> list[ReplayMemory]:
        """Pick a deterministic bounded sample (``memory_per_class``) per class.

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
        generator.manual_seed(self.memory_seed + experience_index)
        memories: list[ReplayMemory] = []

        for class_id in sorted(samples_by_class):
            indices = samples_by_class[class_id]

            if len(indices) > self.memory_per_class:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()

                indices = [
                    indices[position]
                    for position in permutation[: self.memory_per_class]
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
                raise RuntimeError(f"Class {class_id} produced an empty replay memory.")

            memories.append(
                ReplayMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.tensor(
                        targets,
                        dtype=torch.long,
                    ),
                    class_id=class_id,
                )
            )

        return memories

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def before_eval(self, strategy, **kwargs) -> None:
        """Snapshot model state before the evaluation phase."""
        self._pre_eval_state = self._snapshot(strategy.model)
        self._eval_active = True

    def after_eval(self, strategy, **kwargs) -> None:
        """Restore the model state that existed before evaluation."""
        if not self._eval_active:
            return
        try:
            if self._pre_eval_state is not None:
                group_by_name = self._capture_optimizer_groups(strategy)
                restore_initial_state(strategy.model, self._pre_eval_state)
                self._reset_optimizer(strategy, group_by_name)
        finally:
            self._pre_eval_state = None
            self._eval_active = False
