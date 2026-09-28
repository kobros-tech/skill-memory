# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Cheap runtime guards for the continual-learning evaluation protocol.

These checks turn the most common *silent* protocol mistakes into loud
errors. They are constant-time (stream-name and object-identity checks; no
dataset scans), so they are enabled by default
(``strict_protocol=True``) and add no measurable runtime.

What they enforce, at experience ``t``:

* **Training only sees training-stream experiences.** Handing a
  ``test_stream`` experience to ``strategy.train`` would train Skill Memory
  *and* seed the evaluation memory with test data.
* **Evaluation never runs on training-stream experiences, and never on the
  very dataset objects that were trained on.** Otherwise reported accuracy
  measures memorisation of training data.
* **Evaluation memory only contains classes of the experience it was
  drawn from** (see :func:`assert_memory_classes_match`), so nothing from
  another experience -- in particular a future one -- can enter it.

They are *identity/name* checks, not content checks: a benchmark whose test
split literally contains copies of training samples is invisible to them.
For that, run :func:`skill_memory.diagnostics.leakage.audit_split_overlap`.
"""

from __future__ import annotations

from collections.abc import Iterable


class ProtocolViolation(RuntimeError):
    """Raised when the train/evaluation protocol is violated."""


def stream_name(experience) -> str | None:
    """Return the name of the stream `experience` came from, if it has one."""
    stream = getattr(experience, "origin_stream", None)
    name = getattr(stream, "name", None)
    return None if name is None else str(name)


def assert_training_experience(experience) -> None:
    """Reject experiences that originate from a test stream."""
    if stream_name(experience) == "test":
        raise ProtocolViolation(
            "strategy.train() received an experience from a 'test' stream. "
            "Training on test data leaks it into Skill Memory and the "
            "evaluation memory. Pass experiences from the train stream, or "
            "construct the strategy with strict_protocol=False to override."
        )


def assert_evaluation_experiences(
    experiences: Iterable,
    trained_experiences: Iterable,
) -> None:
    """Reject evaluation on training-stream data.

    `trained_experiences` are the experiences Skill Memory has already
    trained on; an evaluation experience sharing a dataset *object* with one
    of them is a direct train/test overlap.
    """
    trained_dataset_ids = {
        id(getattr(experience, "dataset", None)) for experience in trained_experiences
    }
    trained_dataset_ids.discard(id(None))

    for index, experience in enumerate(experiences):
        if stream_name(experience) == "train":
            raise ProtocolViolation(
                f"strategy.eval() received experience {index} from a 'train' "
                "stream. Evaluating on training data measures memorisation, "
                "not retention. Pass test-stream experiences, or construct "
                "the strategy with strict_protocol=False to override."
            )
        if id(getattr(experience, "dataset", None)) in trained_dataset_ids:
            raise ProtocolViolation(
                f"strategy.eval() experience {index} shares its dataset "
                "object with an experience Skill Memory already trained on."
            )


def assert_memory_classes_match(
    retained_classes: Iterable[int],
    experience,
) -> None:
    """Reject evaluation-memory classes the source experience does not declare."""
    declared = getattr(experience, "classes_in_this_experience", None)
    if declared is None:
        return
    declared_set = {int(class_id) for class_id in declared}
    extra = {int(class_id) for class_id in retained_classes} - declared_set
    if extra:
        raise ProtocolViolation(
            f"evaluation memory captured classes {sorted(extra)} that the "
            f"source experience does not declare ({sorted(declared_set)})."
        )
