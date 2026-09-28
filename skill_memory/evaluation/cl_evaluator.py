# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Standalone Skill Memory evaluator.

This evaluator uses only the stored Skill Memory snapshots at prediction time.
Each canonical class is scored by its own YES/NO verifier. Held-out training
examples stored with the skill are used only to calibrate that verifier's raw
logit into a comparable one-vs-rest score. Test labels never participate in
routing or prediction.

The evaluator is intentionally separate from the independent ML evaluator:
the two can be selected independently by the experiment configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from avalanche.training.plugins import SupervisedPlugin
from torch import nn

from ..utils.probing import incremental_active_units, predict_logits
from ..utils.protocol_guard import assert_evaluation_experiences
from .memory import EvaluationMemoryPlugin


def _class_logit(
    logits: torch.Tensor,
    class_id: int,
    owned_classes: Sequence[int],
    negative_classes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Extract the verifier logit for one owned class."""
    if logits.ndim != 2:
        raise RuntimeError("Skill logits must have shape [batch, classes].")
    if logits.shape[1] > len(owned_classes):
        if class_id >= logits.shape[1]:
            raise RuntimeError(
                f"class {class_id} is outside skill classifier width {logits.shape[1]}"
            )
        target_logit = logits[:, class_id]
    else:
        try:
            position = list(owned_classes).index(class_id)
        except ValueError as exc:
            raise RuntimeError(
                f"class {class_id} is not owned by the selected skill"
            ) from exc
        target_logit = logits[:, position]
    negatives = (
        sorted(int(value) for value in negative_classes)
        if negative_classes is not None
        else []
    )
    negatives = [value for value in negatives if value != class_id]
    if not negatives:
        return target_logit
    if max(negatives) >= logits.shape[1]:
        raise RuntimeError(
            f"negative class {max(negatives)} is outside skill classifier "
            f"width {logits.shape[1]}"
        )
    return target_logit - torch.logsumexp(logits[:, negatives], dim=1)


def _active_classes(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    owned_classes: Sequence[int],
) -> list[int]:
    active_units = incremental_active_units(model, state)
    if active_units is None:
        return sorted({int(class_id) for class_id in owned_classes})
    return [
        class_id
        for class_id, active in enumerate(active_units.tolist())
        if int(active) != 0
    ]


def _fit_platt_calibrator(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Fit monotonic sigmoid calibration on held-out training data.

    The returned values satisfy:

        calibrated_logit = scale * raw_logit + bias

    Only the held-out validation examples stored by Skill Memory are used.
    """
    scores = scores.detach().float().flatten()
    targets = targets.detach().float().flatten()

    positive = targets > 0.5
    negative = ~positive
    if scores.numel() < 2 or not positive.any() or not negative.any():
        return 1.0, 0.0

    scale = torch.nn.Parameter(torch.tensor(1.0, device=scores.device))
    bias = torch.nn.Parameter(torch.tensor(0.0, device=scores.device))
    optimizer = torch.optim.LBFGS(
        [scale, bias],
        lr=0.5,
        max_iter=25,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        calibrated = scale * scores + bias
        loss = criterion(calibrated, targets)
        loss = loss + 1e-4 * (scale.square() + bias.square())
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)

    # A verifier should remain monotonic in its raw YES/NO evidence.
    fitted_scale = max(float(scale.detach().item()), 1e-3)
    fitted_bias = float(bias.detach().item())
    return fitted_scale, fitted_bias


class CLEvaluationPlugin(SupervisedPlugin):
    """Anonymous evaluator driven entirely by stored Skill Memory skills.

    Prediction is:

        x -> canonical Skill Memory verifiers -> calibrated class scores -> y

    No evaluator model is trained and no ML evaluator output is consulted.
    """

    def __init__(
        self,
        *,
        memory_plugin: EvaluationMemoryPlugin,
        verbose: bool = True,
        strict_protocol: bool = True,
        unseen_logit: float = -20.0,
    ) -> None:
        super().__init__()
        self.memory_plugin = memory_plugin
        self.verbose = bool(verbose)
        self.strict_protocol = bool(strict_protocol)
        self.unseen_logit = float(unseen_logit)

        self._active = False
        self._class_to_experience: dict[int, int] = {}
        self._accuracy_history: list[dict[int, float]] = []
        self._loss_history: list[dict[int, float]] = []
        self._diagonal_accuracy_history: list[float] = []
        self._diagonal_loss_history: list[float] = []
        self._current_class_loss: dict[int, float] = {}
        self._current_class_correct: dict[int, int] = {}
        self._current_class_total: dict[int, int] = {}
        self._raw_current_class_correct: dict[int, int] = {}
        self._calibrators: dict[int, tuple[int, float, float]] = {}
        self._num_classes = 0
        self._eval_skill_groups: list[list[tuple[int, dict[str, torch.Tensor]]]] = []
        self._score_debug_printed = False

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Record class introduction for forgetting bookkeeping."""
        experience = strategy.experience
        experience_index = int(
            getattr(
                experience,
                "current_experience",
                len(self._class_to_experience),
            )
        )
        for class_id in experience.classes_in_this_experience:
            self._class_to_experience.setdefault(int(class_id), experience_index)

    def _build_calibrators(self, strategy) -> None:
        """Calibrate every canonical class from its skill's held-out data."""
        memory = self.memory_plugin.memory
        class_map = self.memory_plugin.class_map
        device = strategy.device
        self._calibrators = {}

        for skill in sorted(memory.slots()):
            owned_classes = sorted(class_map.classes_for_skill(skill))
            if not owned_classes:
                continue

            metadata = memory.metadata(skill)
            examples_by_class = metadata.get("verification_examples_by_class", {})
            state = memory.state(skill)

            validation_inputs = []
            validation_targets = []
            for examples in examples_by_class.values():
                if examples is None or len(examples) != 2:
                    continue
                inputs, targets = examples
                if inputs.numel() == 0:
                    continue
                validation_inputs.append(inputs)
                validation_targets.append(targets)

            if not validation_inputs:
                for class_id in owned_classes:
                    self._calibrators[class_id] = (skill, 1.0, 0.0)
                continue

            inputs = torch.cat(validation_inputs, dim=0).to(device)
            targets = torch.cat(validation_targets, dim=0).to(device)

            with torch.no_grad():
                logits = predict_logits(strategy.model, state, inputs)

            for class_id in owned_classes:
                domain_classes = _active_classes(strategy.model, state, owned_classes)
                raw_score = _class_logit(
                    logits,
                    class_id,
                    owned_classes,
                    [value for value in domain_classes if value != class_id],
                )
                binary_target = targets.eq(class_id).to(dtype=torch.float32)
                scale, bias = _fit_platt_calibrator(raw_score, binary_target)

                if self.verbose:
                    positive = binary_target > 0.5
                    negative = ~positive
                    positive_scores = raw_score[positive]
                    negative_scores = raw_score[negative]
                    if positive_scores.numel() and negative_scores.numel():
                        pairwise = (
                            (positive_scores[:, None] > negative_scores[None, :])
                            .float()
                            .mean()
                        )
                        ties = (
                            (positive_scores[:, None] == negative_scores[None, :])
                            .float()
                            .mean()
                        )
                        roc_auc = pairwise + 0.5 * ties
                    else:
                        roc_auc = torch.tensor(float("nan"), device=raw_score.device)
                    calibrated = scale * raw_score + bias
                    probabilities = torch.sigmoid(calibrated)
                    print(
                        f"CL calibration debug: class {class_id} skill={skill} "
                        f"scale={scale:+.6f} bias={bias:+.6f}"
                    )
                    print(
                        "  validation positive: "
                        f"n={int(positive.sum().item())} "
                        f"mean={raw_score[positive].mean().item():+.4f} "
                        f"min={raw_score[positive].min().item():+.4f} "
                        f"max={raw_score[positive].max().item():+.4f}"
                    )
                    print(
                        "  validation negative: "
                        f"n={int(negative.sum().item())} "
                        f"mean={raw_score[negative].mean().item():+.4f} "
                        f"min={raw_score[negative].min().item():+.4f} "
                        f"max={raw_score[negative].max().item():+.4f}"
                    )
                    print(
                        f"  validation separation: "
                        f"pairwise_accuracy={pairwise.item():.4f} "
                        f"roc_auc={roc_auc.item():.4f}"
                    )
                    print(
                        "  calibrated probability: "
                        f"positive_mean={probabilities[positive].mean().item():.4f} "
                        f"negative_mean={probabilities[negative].mean().item():.4f}"
                    )

                self._calibrators[class_id] = (skill, scale, bias)

        if not self._calibrators:
            raise RuntimeError("No canonical Skill Memory classes are available.")

        self._num_classes = max(self._calibrators) + 1

    def _prepare_eval_skill_batches(self, strategy) -> None:
        """Cache device-resident skill states and batch compatible states."""
        memory = self.memory_plugin.memory
        device = strategy.device
        calibrated_skills = {skill for skill, _, _ in self._calibrators.values()}
        groups: dict[tuple, list[tuple[int, dict[str, torch.Tensor]]]] = {}

        for skill in sorted(memory.slots()):
            if skill not in calibrated_skills:
                continue
            state = memory.state(skill)
            params = {name: value.to(device) for name, value in state.items()}
            signature = tuple(
                (name, tuple(value.shape), value.dtype)
                for name, value in sorted(params.items())
            )
            groups.setdefault(signature, []).append((skill, params))

        self._eval_skill_groups = list(groups.values())

    def before_eval(self, strategy, **kwargs) -> None:
        """Prepare the CL-only evaluator before normal Avalanche evaluation."""
        if self.strict_protocol:
            assert_evaluation_experiences(
                getattr(strategy, "current_eval_stream", None) or (),
                self.memory_plugin._seen_experiences,
            )

        self._active = bool(self.memory_plugin.memory.slots())
        if not self._active:
            return

        self._build_calibrators(strategy)
        self._prepare_eval_skill_batches(strategy)
        self._current_class_loss = {}
        self._current_class_correct = {}
        self._current_class_total = {}
        self._raw_current_class_correct = {}
        self._raw_mb_output = None
        self._score_debug_printed = False

        if self.verbose:
            print(
                "CL evaluation: using stored Skill Memory verifiers only "
                f"({len(self._calibrators)} calibrated classes)"
            )

    def before_eval_exp(self, strategy, **kwargs) -> None:
        return

    @torch.no_grad()
    def after_eval_forward(self, strategy, **kwargs) -> None:
        """Produce CL decisions and retain raw verifier scores for diagnostics."""
        if not self._active:
            return

        inputs = strategy.mbatch[0]
        raw_scores = torch.full(
            (inputs.shape[0], self._num_classes),
            self.unseen_logit,
            device=inputs.device,
            dtype=torch.float32,
        )
        calibrated_scores = torch.full_like(raw_scores, self.unseen_logit)

        class_map = self.memory_plugin.class_map

        for members in self._eval_skill_groups:
            for start in range(0, len(members), 8):
                chunk = members[start : start + 8]
                skills = [skill for skill, _ in chunk]
                names = chunk[0][1].keys()
                stacked_params = {
                    name: torch.stack([params[name] for _, params in chunk])
                    for name in names
                }

                def call(one_skill_params: dict[str, torch.Tensor]) -> torch.Tensor:
                    return torch.func.functional_call(
                        strategy.model,
                        one_skill_params,
                        (inputs,),
                    )

                batched_logits = torch.vmap(call)(stacked_params)
                for index, skill in enumerate(skills):
                    _, one_skill_params = chunk[index]
                    owned_classes = sorted(class_map.classes_for_skill(skill))
                    logits = batched_logits[index]
                    for class_id, (
                        calibrated_skill,
                        scale,
                        bias,
                    ) in self._calibrators.items():
                        if calibrated_skill != skill:
                            continue
                        domain_classes = _active_classes(
                            strategy.model,
                            one_skill_params,
                            owned_classes,
                        )
                        raw_score = _class_logit(
                            logits,
                            class_id,
                            owned_classes,
                            [value for value in domain_classes if value != class_id],
                        )
                        raw_scores[:, class_id] = raw_score
                        calibrated_scores[:, class_id] = scale * raw_score + bias

        self._raw_mb_output = raw_scores

        if self.verbose and not self._score_debug_printed:
            targets = strategy.mbatch[1]
            print("CL score election debug:")
            for sample_index in range(min(10, inputs.shape[0])):
                true_class = int(targets[sample_index].item())
                raw_row = raw_scores[sample_index]
                calibrated_row = calibrated_scores[sample_index]
                print(
                    f"  sample {sample_index}: true={true_class} "
                    f"raw_winner={int(raw_row.argmax().item())} "
                    f"calibrated_winner={int(calibrated_row.argmax().item())}"
                )
                for class_id in sorted(self._calibrators):
                    print(
                        f"    class {class_id}: "
                        f"raw={raw_row[class_id].item():+.4f} "
                        f"calibrated={calibrated_row[class_id].item():+.4f}"
                    )
            self._score_debug_printed = True

        strategy.mb_output = calibrated_scores

    def after_eval_iteration(self, strategy, **kwargs) -> None:
        if not self._active:
            return

        outputs = strategy.mb_output
        targets = strategy.mbatch[1]
        predictions = outputs.argmax(dim=1)
        if self._raw_mb_output is None:
            raise RuntimeError("Raw CL evaluation scores are missing.")
        raw_predictions = self._raw_mb_output.argmax(dim=1)

        for class_id in torch.unique(targets).tolist():
            class_id = int(class_id)
            mask = targets == class_id
            self._raw_current_class_correct[class_id] = (
                self._raw_current_class_correct.get(class_id, 0)
                + int((raw_predictions[mask] == targets[mask]).sum().item())
            )
        per_sample_loss = nn.functional.cross_entropy(
            outputs,
            targets,
            reduction="none",
        )

        for class_id in torch.unique(targets).tolist():
            class_id = int(class_id)
            mask = targets == class_id
            self._current_class_loss[class_id] = self._current_class_loss.get(
                class_id, 0.0
            ) + float(per_sample_loss[mask].sum().item())
            self._current_class_correct[class_id] = self._current_class_correct.get(
                class_id, 0
            ) + int((predictions[mask] == targets[mask]).sum().item())
            self._current_class_total[class_id] = self._current_class_total.get(
                class_id, 0
            ) + int(mask.sum().item())

    def after_eval_exp(self, strategy, **kwargs) -> None:
        return

    def after_eval(self, strategy, **kwargs) -> None:
        if not self._active:
            return

        current_accuracy: dict[int, float] = {}
        current_loss: dict[int, float] = {}
        for class_id in sorted(self._current_class_total):
            total = self._current_class_total[class_id]
            if total:
                current_accuracy[class_id] = (
                    self._current_class_correct[class_id] / total
                )
                current_loss[class_id] = self._current_class_loss[class_id] / total

        if not current_accuracy:
            self._active = False
            return

        self._accuracy_history.append(current_accuracy)
        self._loss_history.append(current_loss)

        diagonal_classes = [
            class_id
            for class_id, introduction in self._class_to_experience.items()
            if introduction == len(self._accuracy_history) - 1
            and class_id in current_accuracy
        ]
        if diagonal_classes:
            self._diagonal_accuracy_history.append(
                float(np.mean([current_accuracy[c] for c in diagonal_classes]))
            )
            self._diagonal_loss_history.append(
                float(np.mean([current_loss[c] for c in diagonal_classes]))
            )

        raw_total = sum(self._current_class_total.values())
        raw_correct = sum(self._raw_current_class_correct.values())
        raw_accuracy = raw_correct / raw_total if raw_total else 0.0

        if self.verbose:
            print(
                "CL evaluation: "
                f"calibrated_accuracy={np.mean(list(current_accuracy.values())):.4f}, "
                f"raw_accuracy={raw_accuracy:.4f}"
            )
        self._eval_skill_groups = []
        self._active = False

    @property
    def evaluator_model(self):
        """CL evaluation has no separately trained evaluator model."""
        return None

    @property
    def current_accuracy(self) -> dict[int, float]:
        if not self._accuracy_history:
            return {}
        return dict(self._accuracy_history[-1])

    @property
    def current_loss(self) -> dict[int, float]:
        if not self._loss_history:
            return {}
        return dict(self._loss_history[-1])

    def results(self) -> dict[str, Any]:
        if not self._accuracy_history:
            raise RuntimeError(
                "No CL evaluation results are available. "
                "Call strategy.eval() after training."
            )

        final_accuracy = self._accuracy_history[-1]
        final_loss = self._loss_history[-1]
        raw_total = sum(self._current_class_total.values())
        raw_correct = sum(self._raw_current_class_correct.values())
        raw_accuracy = raw_correct / raw_total if raw_total else 0.0
        return {
            "final_class_accuracy": dict(final_accuracy),
            "final_class_loss": dict(final_loss),
            "mean_final_accuracy": float(np.mean(list(final_accuracy.values()))),
            "mean_final_loss": float(np.mean(list(final_loss.values()))),
            "raw_mean_final_accuracy": float(raw_accuracy),
            "diagonal_accuracy": np.asarray(
                self._diagonal_accuracy_history,
                dtype=np.float64,
            ),
            "diagonal_loss": np.asarray(
                self._diagonal_loss_history,
                dtype=np.float64,
            ),
            "peak_forgetting": self._peak_forgetting(),
        }

    def _peak_forgetting(self) -> np.ndarray:
        """Return peak-relative forgetting for CL evaluation."""
        if not self._accuracy_history:
            return np.zeros(0, dtype=np.float64)

        values = []
        num_experiences = len(self._accuracy_history)
        final_accuracy = self._accuracy_history[-1]
        for experience_index in range(num_experiences):
            class_values = []
            for class_id, introduction in self._class_to_experience.items():
                if introduction != experience_index:
                    continue
                observed = [
                    history[class_id]
                    for history in self._accuracy_history[introduction:]
                    if class_id in history
                ]
                if observed and class_id in final_accuracy:
                    class_values.append(max(observed) - final_accuracy[class_id])
            values.append(float(np.mean(class_values)) if class_values else 0.0)
        return np.asarray(values, dtype=np.float64)
