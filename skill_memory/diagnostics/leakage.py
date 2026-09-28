# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Content-level train/test overlap audit.

The runtime guards in :mod:`skill_memory.utils.protocol_guard` are
constant-time *name/identity* checks. They cannot notice a benchmark whose
test split simply contains copies of training samples, because such
samples live in different dataset objects. This module closes that gap by
hashing the actual tensors:

* **evaluation memory vs. test data** -- the independent evaluator is
  trained on the retained evaluation memory, so any test sample identical to
  a retained one is answered by memorisation rather than retention;
* **training data vs. test data** -- Skill Memory itself trains on the
  training experiences, so the same reasoning applies to them.

It is a diagnostic (``diagnose=True`` required, like everything in this
package) because it scans every test sample and is therefore not free.

Limits, stated plainly: hashing detects **exact** duplicates of the tensors
the model actually sees. It cannot see near-duplicates, and random data
augmentation applied inside ``__getitem__`` defeats it (two draws of the same
image hash differently). An empty audit is evidence, not proof.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..utils.protocol_guard import ProtocolViolation
from ._gate import require_diagnose


def _row_digest(row: torch.Tensor) -> bytes:
    array = row.detach().cpu().contiguous().numpy()
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(array.dtype).encode())
    hasher.update(str(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.digest()


def _dataset_digests(dataset, batch_size: int) -> list[tuple[bytes, int]]:
    """Return ``[(digest, label), ...]`` for every sample in `dataset`."""
    out: list[tuple[bytes, int]] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        inputs, targets = batch[0], batch[1]
        for row, target in zip(inputs, targets, strict=True):
            out.append((_row_digest(row), int(target)))
    return out


def audit_split_overlap(
    *,
    eval_memory: Iterable,
    test_stream,
    train_experiences: Iterable | None = None,
    batch_size: int = 256,
    diagnose: bool,
) -> dict[str, Any]:
    """Count exact tensor overlaps between training-side data and a test stream.

    `eval_memory` is ``strategy.skill_memory_plugin.eval_memory`` (a list of
    :class:`~skill_memory.evaluation.memory.EvaluationMemory`).
    `train_experiences`, if given, are hashed in full as well (pass
    ``strategy.skill_memory_plugin._seen_experiences`` for the data Skill
    Memory actually trained on).

    Returns a dict with ``n_memory_samples``, ``n_train_samples``,
    ``n_test_samples``, ``memory_test_overlaps``, ``train_test_overlaps``
    and, for each overlap kind, a short ``*_examples`` list of
    ``(test_experience_index, test_label, source_label)`` tuples.
    """
    require_diagnose(diagnose, "audit_split_overlap")

    memory_digests: dict[bytes, list[int]] = {}
    n_memory = 0
    for item in eval_memory:
        for row, target in zip(item.inputs, item.targets, strict=True):
            memory_digests.setdefault(_row_digest(row), []).append(int(target))
            n_memory += 1

    train_digests: dict[bytes, list[int]] = {}
    n_train = 0
    for experience in train_experiences or ():
        for digest, label in _dataset_digests(experience.dataset, batch_size):
            train_digests.setdefault(digest, []).append(label)
            n_train += 1

    n_test = 0
    memory_hits = 0
    train_hits = 0
    memory_examples: list[tuple[int, int, int]] = []
    train_examples: list[tuple[int, int, int]] = []
    for experience_index, experience in enumerate(test_stream):
        for digest, label in _dataset_digests(experience.dataset, batch_size):
            n_test += 1
            if digest in memory_digests:
                memory_hits += 1
                if len(memory_examples) < 10:
                    memory_examples.append(
                        (experience_index, label, memory_digests[digest][0])
                    )
            if digest in train_digests:
                train_hits += 1
                if len(train_examples) < 10:
                    train_examples.append(
                        (experience_index, label, train_digests[digest][0])
                    )

    return {
        "n_memory_samples": n_memory,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "memory_test_overlaps": memory_hits,
        "train_test_overlaps": train_hits,
        "memory_test_examples": memory_examples,
        "train_test_examples": train_examples,
    }


def assert_no_split_overlap(
    *,
    eval_memory: Iterable,
    test_stream,
    train_experiences: Iterable | None = None,
    batch_size: int = 256,
    diagnose: bool,
) -> dict[str, Any]:
    """Run :func:`audit_split_overlap` and raise if any overlap was found."""
    report = audit_split_overlap(
        eval_memory=eval_memory,
        test_stream=test_stream,
        train_experiences=train_experiences,
        batch_size=batch_size,
        diagnose=diagnose,
    )
    if report["memory_test_overlaps"] or report["train_test_overlaps"]:
        raise ProtocolViolation(
            "train/test overlap detected: "
            f"{report['memory_test_overlaps']} test samples identical to "
            f"evaluation memory, {report['train_test_overlaps']} identical to "
            "training data. Reported accuracy is contaminated."
        )
    return report


def audit_strategy_leakage(
    strategy,
    test_stream,
    *,
    batch_size: int = 256,
    diagnose: bool,
) -> dict[str, Any]:
    """Convenience: audit a :class:`~skill_memory.SkillMemoryStrategy` in place."""
    require_diagnose(diagnose, "audit_strategy_leakage")
    plugin = strategy.skill_memory_plugin
    return audit_split_overlap(
        eval_memory=plugin.eval_memory,
        test_stream=test_stream,
        train_experiences=plugin._seen_experiences,
        batch_size=batch_size,
        diagnose=diagnose,
    )
