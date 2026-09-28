# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""CIFAR-100 Skill Memory experiment with CL evaluation.

The public SkillMemoryStrategy owns the complete experiment lifecycle:

* Skill Memory training
* frozen per-class evaluation memory
* anonymous CL evaluation through stored Skill Memory states
* accuracy/loss tracking
* forgetting metrics

The demo only configures SplitMNIST and the strategy, then reports results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.models import SlimResNet18
from torch import nn

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import (
    evaluate_class_oracle,
    evaluate_skill_memory,
    timing_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Skill Memory on CIFAR-100."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=20)
    parser.add_argument(
        "--experience-index",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "CIFAR-100 experience indices to run sequentially in one process. "
            "For example: --experience-index 1 2 3."
        ),
    )
    parser.add_argument(
        "--eval-memory-per-class",
        type=int,
        default=20,
        help="Number of retained evaluation examples per class.",
    )
    parser.add_argument(
        "--skill-train-samples-per-class",
        type=int,
        default=20,
        help="Number of training samples per class used by Skill Memory.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument(
        "--class-train-mode",
        choices=("multiclass", "binary_one_vs_rest"),
        default="binary_one_vs_rest",
        help=(
            "Skill class-training objective: multiclass positive-only or "
            "binary one-vs-rest YES/NO."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--force-decision",
        choices=("none", "reuse", "scratch"),
        default="none",
        help=(
            "Force Skill Memory decisions for new classes. Use 'reuse' in the "
            "candidate-routing experiment to create multi-class skills; "
            "'none' keeps the normal decision policy."
        ),
    )
    parser.add_argument(
        "--eval-method",
        choices=("cl",),
        default="cl",
        help="Compatibility flag; Skill Memory CL is the only production evaluator.",
    )
    parser.add_argument(
        "--cl-update-mode",
        choices=("replay", "small_replay", "new_class"),
        default="replay",
        help=(
            "CL skill update data: full retained-history replay, bounded historical "
            "replay for every existing skill, or newly exposed class only."
        ),
    )
    parser.add_argument(
        "--cl-replay-per-class",
        type=int,
        default=5,
        help=(
            "Examples per historical class during small_replay. Current "
            "classes keep --skill-train-samples-per-class."
        ),
    )
    parser.add_argument(
        "--skill-validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction held out from Skill Memory training for verification calibration."
        ),
    )
    parser.add_argument(
        "--skill-validation-seed",
        type=int,
        default=0,
        help="Seed for the disjoint Skill Memory verification holdout.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Run optional Skill Memory diagnostics (class oracle and "
            "anonymous probe). Diagnostics never affect production evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # benchmark = SplitMNIST(
    #     n_experiences=args.n_experiences,
    #     seed=args.seed,
    #     dataset_root=args.dataset_root,
    # )

    print("Preparing CIFAR-100 dataset...")
    print(f"Dataset root: {args.dataset_root}")
    print("If CIFAR-100 is not already downloaded, downloading it now...")

    benchmark = SplitCIFAR100(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print("CIFAR-100 dataset is ready.")

    experience_indices = list(dict.fromkeys(args.experience_index))
    if not experience_indices:
        raise ValueError("At least one --experience-index is required.")
    invalid = [
        index
        for index in experience_indices
        if index < 0 or index >= len(benchmark.train_stream)
    ]
    if invalid:
        raise ValueError(
            f"--experience-index values must be between 0 and "
            f"{len(benchmark.train_stream) - 1}: {invalid}"
        )

    print(
        "Selected CIFAR-100 experiences: "
        + ", ".join(str(index) for index in experience_indices)
    )

    print("=== CIFAR-100 Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print(
        f"CL update mode: {args.cl_update_mode}"
        + (
            f" (replay_per_class={args.cl_replay_per_class})"
            if args.cl_update_mode == "small_replay"
            else ""
        )
    )
    print(f"Skill class-training mode: {args.class_train_mode}")
    print("Evaluation: Skill Memory CL evaluator")
    print(f"Diagnostics: {'enabled' if args.diagnose else 'disabled'}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(
        "Evaluation memory samples per class:",
        args.eval_memory_per_class,
    )
    print(
        "Skill training samples per class:",
        args.skill_train_samples_per_class,
    )
    print(
        "Skill Memory decision policy:",
        args.force_decision,
    )

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"CIFAR-100 dataset prepared at {args.dataset_root}")
        return

    # model = SimpleMLP(num_classes=10).to(device)
    model = SlimResNet18(nclasses=100).to(device)

    force_decision = None if args.force_decision == "none" else args.force_decision

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=nn.CrossEntropyLoss(),
        max_skills=args.max_skills,
        class_train_mode=args.class_train_mode,
        skill_train_samples_per_class=args.skill_train_samples_per_class,
        validation_fraction=args.skill_validation_fraction,
        validation_seed=args.skill_validation_seed,
        force_decision=force_decision,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        eval_memory_per_class=args.eval_memory_per_class,
        probe_seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
        cl_update_mode=args.cl_update_mode,
        cl_replay_per_class=args.cl_replay_per_class,
    )

    accuracy_history: list[dict[int, float]] = []
    class_to_step: dict[int, int] = {}

    for step, experience_index in enumerate(experience_indices):
        experience = benchmark.train_stream[experience_index]
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(class_id) for class_id in experience.classes_in_this_experience),
        )

        strategy.train(experience)

        # Evaluate only classes introduced so far. This is the same cumulative
        # test population for both evaluator choices and prevents an early
        # CL evaluator from being penalized for classes that have no skill yet.
        # eval_stream = [
        #     benchmark.test_stream[index] for index in range(experience_index + 1)
        # ]

        eval_stream = [
            benchmark.test_stream[index] for index in experience_indices[: step + 1]
        ]

        class_map = strategy.skill_memory_plugin.class_map
        memory = strategy.skill_memory_plugin.memory
        print("Skill Memory groups after training:")
        for skill in sorted(memory.slots()):
            classes = sorted(class_map.classes_for_skill(skill))
            print(f"  skill {skill}: classes={classes}")

        print(f"========== Evaluation after experience {experience_index} ==========")
        strategy.eval(eval_stream)

        current_accuracy = dict(strategy.results()["final_class_accuracy"])
        accuracy_history.append(current_accuracy)
        for class_id in current_accuracy:
            class_to_step.setdefault(class_id, step)

        forgetting_values = []
        for class_id, introduction_step in class_to_step.items():
            if introduction_step > step or class_id not in current_accuracy:
                continue
            observed = [
                history[class_id]
                for history in accuracy_history[introduction_step:]
                if class_id in history
            ]
            if observed:
                forgetting_values.append(max(observed) - current_accuracy[class_id])
        mean_forgetting = (
            float(np.mean(forgetting_values)) if forgetting_values else 0.0
        )
        print(
            f"Sequential metrics after experience {experience_index}: "
            f"accuracy={np.mean(list(current_accuracy.values())):.4f} "
            f"forgetting={mean_forgetting:.4f}"
        )

        if args.diagnose:
            print("---------- Diagnostics ----------")
            class_oracle = evaluate_class_oracle(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )
            direct_probe = evaluate_skill_memory(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                routing="probe",
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )
            print(
                "class_oracle_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in class_oracle.values()]):.4f}",
            )
            print(
                "direct_probe_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in direct_probe.values()]):.4f}",
            )

            for bucket, stats in timing_report(strategy).items():
                print(
                    f"timing[{bucket}]: total={stats['total_seconds']:.2f}s "
                    f"calls={stats['calls']} mean={stats['mean_seconds']:.3f}s"
                )

    results = strategy.results()

    final_accuracy = dict(results["final_class_accuracy"])
    final_forgetting_values = []
    for class_id, introduction_step in class_to_step.items():
        if class_id not in final_accuracy:
            continue
        observed = [
            history[class_id]
            for history in accuracy_history[introduction_step:]
            if class_id in history
        ]
        if observed:
            final_forgetting_values.append(max(observed) - final_accuracy[class_id])
    mean_forgetting = (
        float(np.mean(final_forgetting_values)) if final_forgetting_values else 0.0
    )

    print()
    print("=== Summary ===")
    print("mean_forgetting=", f"{mean_forgetting:.4f}")
    print(
        "mean_final_accuracy=",
        f"{results['mean_final_accuracy']:.4f}",
    )
    print(
        "mean_final_loss=",
        f"{results['mean_final_loss']:.4f}",
    )

    print("final_class_accuracy:")
    for class_id, accuracy in final_accuracy.items():
        print(f"  class {class_id}: {accuracy:.4f}")

    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
