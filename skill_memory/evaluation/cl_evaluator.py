# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Skill Memory CL evaluator (the evaluator used by ``strategy.eval()``).

At prediction time only the stored Skill Memory snapshots are used; no
evaluator model is trained.  Each canonical class :math:`c` is scored by the
YES/NO verifier of the skill :math:`s(c)` that owns it:

.. math::

    r_c(x) = z^{(s(c))}_c(x), \qquad
    \tilde r_c(x) = a_c\, r_c(x) + b_c, \qquad
    \hat y(x) = \arg\max_c \tilde r_c(x),

where :math:`(a_c, b_c)` is a monotone Platt calibration (see
:func:`_fit_platt_calibrator`) fitted on the skill's **calibration hold-out**
(``CALIBRATION_EXAMPLES_KEY``) -- examples excluded from training.  Classes
that have not been trained yet receive the constant ``unseen_logit``.  Test
labels never participate in routing or prediction; they are used only *after*
prediction to count correct answers.

Two accuracies are reported: ``raw`` (argmax of :math:`r_c`) and
``calibrated`` (argmax of :math:`\tilde r_c`).  Treat raw as the primary
diagnostic until calibration is shown to help consistently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from avalanche.training.plugins import SupervisedPlugin
from torch import nn

from ..cl.skill_registry import CALIBRATION_EXAMPLES_KEY
from ..utils.probing import predict_logits
from ..utils.protocol_guard import assert_evaluation_experiences
from .memory import EvaluationMemoryPlugin

#: Number of skills evaluated together through one ``torch.vmap`` call.
#: Larger chunks use more memory (one stacked parameter copy per skill in the
#: chunk); the value changes speed only, never predictions.
DEFAULT_EVAL_CHUNK_SIZE = 8


def _class_logit(
    logits: torch.Tensor,
    class_id: int,
    owned_classes: Sequence[int],
) -> torch.Tensor:
    """Extract the verifier logit of ``class_id`` (column ``class_id``).

    Skills are trained with the *global* class index as the output column
    (see :func:`skill_memory.cl.training.train_on_class`), so the column is
    the class id itself.  ``owned_classes`` is only used for validation.
    """
    if logits.ndim != 2:
        raise RuntimeError("Skill logits must have shape [batch, classes].")
    if class_id not in owned_classes:
        raise RuntimeError(f"class {class_id} is not owned by the selected skill")
    if class_id >= logits.shape[1]:
        raise RuntimeError(
            f"class {class_id} is outside skill classifier width {logits.shape[1]}"
        )
    return logits[:, class_id]


def _fit_platt_calibrator(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    r"""Fit monotonic sigmoid (Platt) calibration on held-out data.

    Minimises, over :math:`(a, b)`,

    .. math::

        \frac1n\sum_i \operatorname{BCE}\bigl(a\,r_i + b,\ t_i\bigr)
        + 10^{-4}(a^2 + b^2),

    and returns :math:`(a, b)` with :math:`a \ge 10^{-3}` so the calibrated
    score stays monotone in the raw verifier logit.  Falls back to the
    identity :math:`(1, 0)` when the hold-out lacks positives or negatives.
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

    Prediction is ``x -> canonical Skill Memory verifiers -> calibrated class
    scores -> y``.  No evaluator model is trained.

    Parameters
    ----------
    eval_chunk_size:
        How many skills share one ``torch.vmap`` evaluation (speed/memory
        trade-off only; default :data:`DEFAULT_EVAL_CHUNK_SIZE`).
    debug_scores:
        Print per-sample raw/calibrated scores for the first evaluation
        batch.  Off by default; the printout shows true labels, so it is a
        human diagnostic that never influences predictions.
    """

    def __init__(
        self,
        *,
        memory_plugin: EvaluationMemoryPlugin,
        verbose: bool = True,
        strict_protocol: bool = True,
        unseen_logit: float = -20.0,
        eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE,
        debug_scores: bool = False,
    ) -> None:
        super().__init__()
        if eval_chunk_size < 1:
            raise ValueError("eval_chunk_size must be positive")
        self.eval_chunk_size = int(eval_chunk_size)
        self.debug_scores = bool(debug_scores)
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
        self._raw_mb_output: torch.Tensor | None = None
        self._eval_width = 0

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
            examples_by_class = metadata.get(CALIBRATION_EXAMPLES_KEY, {})
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
                raw_score = _class_logit(logits, class_id, owned_classes)
                binary_target = targets.eq(class_id).to(dtype=torch.float32)
                scale, bias = _fit_platt_calibrator(raw_score, binary_target)

                if self.debug_scores:
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

    def _score_width(self, strategy) -> int:
        """Width of the score matrix handed back to Avalanche.

        It must cover every class that can appear as a *target* (otherwise
        Avalanche's loss metric indexes out of bounds for classes that have
        not been trained yet), so it is the larger of the trained width and
        the class-id range declared by the evaluation stream's metadata.
        Only metadata is read -- never samples or labels -- and the extra
        columns hold the constant ``unseen_logit``.
        """
        width = self._num_classes
        for experience in getattr(strategy, "current_eval_stream", None) or ():
            declared = getattr(experience, "classes_in_this_experience", None) or ()
            if len(declared):
                width = max(width, max(int(c) for c in declared) + 1)
        return width

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
        self._eval_width = self._score_width(strategy)
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
            (inputs.shape[0], self._eval_width),
            self.unseen_logit,
            device=inputs.device,
            dtype=torch.float32,
        )
        calibrated_scores = torch.full_like(raw_scores, self.unseen_logit)

        class_map = self.memory_plugin.class_map

        for members in self._eval_skill_groups:
            for start in range(0, len(members), self.eval_chunk_size):
                chunk = members[start : start + self.eval_chunk_size]
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
                    owned_classes = sorted(class_map.classes_for_skill(skill))
                    logits = batched_logits[index]
                    for class_id, (
                        calibrated_skill,
                        scale,
                        bias,
                    ) in self._calibrators.items():
                        if calibrated_skill != skill:
                            continue
                        raw_score = _class_logit(logits, class_id, owned_classes)
                        raw_scores[:, class_id] = raw_score
                        calibrated_scores[:, class_id] = scale * raw_score + bias

        self._raw_mb_output = raw_scores

        if self.debug_scores and not self._score_debug_printed:
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
