# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Shared runner for the dataset demos (CIFAR-100, SplitMNIST).

A demo supplies only what is dataset specific -- the benchmark, the model and
the number of classes -- and this module does the rest: the argument parser,
the sequential train -> evaluate loop, the forgetting metric and the optional
diagnostics.

**Forgetting.** With :math:`A_t(c)` the accuracy on class :math:`c` after
experience :math:`t` and :math:`t_c` the experience that introduced it,

.. math::

    \text{forgetting}_t=\frac1{|\mathcal D_t|}\sum_{c\in\mathcal D_t}
        \Bigl(\max_{t_c\le u\le t}A_u(c)-A_t(c)\Bigr).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import (
    evaluate_class_oracle,
    evaluate_skill_memory,
    replay_provenance_report,
    timing_report,
)


def build_parser(
    description: str, *, default_experiences: int, default_epochs: int = 3
) -> argparse.ArgumentParser:
    """Argument parser holding every flag the demos share."""
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("--dataset-root", default="data")
    add("--download-only", action="store_true")
    add("--n-experiences", type=int, default=default_experiences)
    add(
        "--max-experiences",
        type=int,
        default=None,
        help="Run the first N experiences only; always starts at experience 0.",
    )
    add(
        "--update-mode",
        choices=("new_class", "replay", "refresh"),
        default="replay",
        help=(
            "One complete CL policy: new_class uses current data only; replay "
            "uses retained history; refresh = replay + one refresh pass for "
            "each existing skill."
        ),
    )
    add(
        "--replay-samples-per-class",
        type=int,
        default=None,
        help="Historical examples per old class (replay/refresh). Omit for all.",
    )
    add(
        "--memory-per-class",
        type=int,
        default=20,
        help="Frozen examples retained per class (the replay memory).",
    )
    add(
        "--train-samples-per-class",
        type=int,
        default=20,
        help="Current-experience training examples used per class.",
    )
    add(
        "--class-train-epochs",
        dest="train_epochs",
        type=int,
        default=default_epochs,
        help="Epochs of each class-training pass.",
    )
    add("--batch-size", type=int, default=64, help="Class-training mini-batch size.")
    add("--eval-batch-size", type=int, default=64)
    add("--learning-rate", type=float, default=0.01)
    add("--seed", type=int, default=0, help="Seeds data order, probes and training.")
    add(
        "--skill-validation-fraction",
        type=float,
        default=0.2,
        help="Fraction held out from each class to calibrate the evaluator.",
    )
    add("--diagnose", action="store_true", help="Run opt-in diagnostics and timing.")
    return parser


def check_args(args: argparse.Namespace) -> None:
    if args.max_experiences is not None and not (
        1 <= args.max_experiences <= args.n_experiences
    ):
        raise ValueError("--max-experiences must be in [1, --n-experiences]")


def mean_forgetting(
    history: list[dict[int, float]],
    first_seen: dict[int, int],
    current: dict[int, float],
) -> float:
    """Mean over classes of (best accuracy since introduction) - (current)."""
    drops = []
    for class_id, start in first_seen.items():
        if class_id in current:
            seen = [h[class_id] for h in history[start:] if class_id in h]
            if seen:
                drops.append(max(seen) - current[class_id])
    return float(np.mean(drops)) if drops else 0.0


def run_experiment(
    args: argparse.Namespace,
    *,
    title: str,
    benchmark,
    make_model: Callable[[torch.device], nn.Module],
    num_classes: int,
) -> dict:
    """Train sequentially, evaluate after every experience and print a summary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_run = args.max_experiences or len(benchmark.train_stream)
    indices = list(range(n_run))

    print(f"=== {title} Skill Memory experiment ===")
    print(f"Update policy: {args.update_mode}")
    cap = args.replay_samples_per_class
    print("Replay samples per old class:", "all retained" if cap is None else cap)
    print(f"Diagnostics: {'enabled' if args.diagnose else 'disabled'}")
    print(f"Device: {device}")
    print(f"Experiences: {n_run}")
    print(f"Memory per class: {args.memory_per_class}")
    print(f"Train samples per class: {args.train_samples_per_class}")
    print(f"Class-training epochs: {args.train_epochs}")
    for index in indices:
        exp = benchmark.train_stream[index]
        print(
            f"  Exp {index}: classes={sorted(exp.classes_in_this_experience)} "
            f"samples={len(exp.dataset)}"
        )

    model = make_model(device)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=args.learning_rate),
        criterion=nn.CrossEntropyLoss(),
        max_skills=num_classes,
        update_mode=args.update_mode,
        replay_samples_per_class=cap,
        memory_per_class=args.memory_per_class,
        train_samples_per_class=args.train_samples_per_class,
        validation_fraction=args.skill_validation_fraction,
        class_train_epochs=args.train_epochs,
        train_mb_size=args.batch_size,
        eval_mb_size=args.eval_batch_size,
        seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
    )

    history: list[dict[int, float]] = []
    first_seen: dict[int, int] = {}
    for step, index in enumerate(indices):
        experience = benchmark.train_stream[index]
        print(f"\n========== Training experience {index} ==========")
        print("Classes:", sorted(int(c) for c in experience.classes_in_this_experience))
        strategy.train(experience)

        plugin = strategy.skill_memory_plugin
        print("Skill Memory groups after training:")
        for skill in sorted(plugin.memory.slots()):
            classes = sorted(plugin.class_map.classes_for_skill(skill))
            print(f"  skill {skill}: classes={classes}")

        print(f"========== Evaluation after experience {index} ==========")
        strategy.eval([benchmark.test_stream[i] for i in indices[: step + 1]])
        current = dict(strategy.results()["final_class_accuracy"])
        history.append(current)
        for class_id in current:
            first_seen.setdefault(class_id, step)
        print(
            f"Sequential metrics after experience {index}: "
            f"accuracy={np.mean(list(current.values())):.4f} "
            f"forgetting={mean_forgetting(history, first_seen, current):.4f}"
        )
        if args.diagnose:
            _print_diagnostics(strategy, benchmark, index, num_classes, args, device)

    if args.diagnose:
        _print_audit(strategy)
    return _print_summary(args, strategy, history, first_seen)


def _print_diagnostics(strategy, benchmark, index, num_classes, args, device) -> None:
    print("---------- Diagnostics ----------")
    plugin = strategy.skill_memory_plugin
    common = dict(
        num_classes=num_classes,
        batch_size=args.eval_batch_size,
        device=device,
        diagnose=True,
    )
    oracle = evaluate_class_oracle(
        strategy.model, plugin, benchmark.test_stream, index, **common
    )
    probe = evaluate_skill_memory(
        strategy.model, plugin, benchmark.test_stream, index, routing="probe", **common
    )
    print(
        "class_oracle_mean_accuracy=",
        f"{np.mean([r['accuracy'] for r in oracle.values()]):.4f}",
    )
    print(
        "direct_probe_mean_accuracy=",
        f"{np.mean([r['accuracy'] for r in probe.values()]):.4f}",
    )
    for bucket, stats in timing_report(strategy).items():
        print(
            f"timing[{bucket}]: total={stats['total_seconds']:.2f}s "
            f"calls={stats['calls']} mean={stats['mean_seconds']:.3f}s"
        )


def _print_audit(strategy) -> None:
    audit = replay_provenance_report(strategy, diagnose=True)
    print("replay provenance:")
    for kind in ("class_training", "refresh"):
        row = audit[kind]
        print(
            f"  {kind}: calls={row['calls']} "
            f"optimizer_steps={row['optimizer_steps']} "
            f"historical_examples={row['historical_examples']}"
        )
    print(f"  violations: {audit['violations'] or 'none'}")


def _print_summary(args, strategy, history, first_seen) -> dict:
    results = strategy.results()
    final = dict(results["final_class_accuracy"])
    forgetting = mean_forgetting(history, first_seen, final)
    print("\n=== Summary ===")
    print("update_policy=", args.update_mode)
    print("mean_forgetting=", f"{forgetting:.4f}")
    print("mean_final_accuracy=", f"{results['mean_final_accuracy']:.4f}")
    print("raw_mean_final_accuracy=", f"{results['raw_mean_final_accuracy']:.4f}")
    print("mean_final_loss=", f"{results['mean_final_loss']:.4f}")
    print("final_class_accuracy:")
    for class_id, accuracy in final.items():
        print(f"  class {class_id}: {accuracy:.4f}")
    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")
    return {"mean_forgetting": forgetting, **results}
