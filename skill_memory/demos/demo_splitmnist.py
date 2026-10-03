# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Sequential SplitMNIST Skill Memory experiment.

The demo uses the same public update-policy API as CIFAR-100. A run always
starts at experience 0 so replay and forgetting have a valid history.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from torch import nn

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import (
    evaluate_class_oracle,
    evaluate_skill_memory,
    replay_provenance_report,
    timing_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sequential Skill Memory experiment on SplitMNIST."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument(
        "--max-experiences",
        type=int,
        default=None,
        help="Run the first N experiences only; always starts at experience 0.",
    )
    parser.add_argument(
        "--memory-per-class",
        dest="memory_per_class",
        type=int,
        default=20,
        help="Frozen examples retained per class for replay and evaluation.",
    )
    parser.add_argument(
        "--train-samples-per-class",
        dest="train_samples_per_class",
        type=int,
        default=20,
        help="Current-experience training examples used per class.",
    )
    parser.add_argument(
        "--class-train-epochs",
        dest="train_epochs",
        type=int,
        default=3,
        help="Epochs for each explicit class-training pass.",
    )
    parser.add_argument(
        "--class-train-mode",
        choices=("multiclass", "binary_one_vs_rest"),
        default="binary_one_vs_rest",
        help="Training objective used by each stored skill.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--update-mode",
        dest="update_mode",
        choices=("new_class", "replay", "refresh"),
        default="replay",
        help=(
            "One complete CL policy: new_class uses current data only; "
            "replay uses retained history; refresh uses replay plus one "
            "refresh pass for each existing skill."
        ),
    )
    parser.add_argument(
        "--replay-samples-per-class",
        dest="replay_samples_per_class",
        type=int,
        default=None,
        help=(
            "Historical examples per old class used by replay/refresh. "
            "Omit for all retained examples."
        ),
    )
    parser.add_argument(
        "--skill-validation-fraction",
        type=float,
        default=0.2,
        help="Fraction held out from current skill training for calibration.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run optional non-production diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_experiences is not None and args.max_experiences < 1:
        raise ValueError("--max-experiences must be positive")
    if args.max_experiences is not None and args.max_experiences > args.n_experiences:
        raise ValueError(
            "--max-experiences cannot exceed --n-experiences"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Preparing SplitMNIST dataset...")
    print(f"Dataset root: {args.dataset_root}")
    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )
    print("SplitMNIST dataset is ready.")

    num_experiences = args.max_experiences or len(benchmark.train_stream)
    experience_indices = list(range(num_experiences))

    print(
        "Selected SplitMNIST experiences: "
        + ", ".join(str(index) for index in experience_indices)
    )
    print("=== SplitMNIST Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print(f"Update policy: {args.update_mode}")
    print(
        "Replay samples per old class:",
        "all retained" if args.replay_samples_per_class is None else args.replay_samples_per_class,
    )
    print(f"Skill class-training mode: {args.class_train_mode}")
    print("Evaluation: Skill Memory CL evaluator")
    print(f"Diagnostics: {'enabled' if args.diagnose else 'disabled'}")
    print(f"Device: {device}")
    print(f"Experiences: {len(experience_indices)}")
    print(f"Memory per class: {args.memory_per_class}")
    print(f"Train samples per class: {args.train_samples_per_class}")
    print(f"Class-training epochs: {args.train_epochs}")

    for index in experience_indices:
        experience = benchmark.train_stream[index]
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    model = SimpleMLP(num_classes=10).to(device)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=args.learning_rate),
        criterion=nn.CrossEntropyLoss(),
        max_skills=10,
        class_train_mode=args.class_train_mode,
        train_samples_per_class=args.train_samples_per_class,
        validation_fraction=args.skill_validation_fraction,
        validation_seed=args.seed,
        train_mb_size=args.batch_size,
        class_train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        memory_per_class=args.memory_per_class,
        probe_seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
        update_mode=args.update_mode,
        replay_samples_per_class=args.replay_samples_per_class,
        training_seed=args.seed,
    )

    accuracy_history: list[dict[int, float]] = []
    class_to_step: dict[int, int] = {}

    for step, experience_index in enumerate(experience_indices):
        experience = benchmark.train_stream[experience_index]
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(c) for c in experience.classes_in_this_experience),
        )

        strategy.train(experience)
        eval_stream = [
            benchmark.test_stream[index] for index in experience_indices[: step + 1]
        ]

        print("Skill Memory groups after training:")
        for skill in sorted(strategy.skill_memory_plugin.memory.slots()):
            classes = sorted(
                strategy.skill_memory_plugin.class_map.classes_for_skill(skill)
            )
            print(f"  skill {skill}: classes={classes}")

        print(f"========== Evaluation after experience {experience_index} ==========")
        strategy.eval(eval_stream)

        current_accuracy = dict(strategy.results()["final_class_accuracy"])
        accuracy_history.append(current_accuracy)
        for class_id in current_accuracy:
            class_to_step.setdefault(class_id, step)

        forgetting_values = []
        for class_id, introduction_step in class_to_step.items():
            if class_id not in current_accuracy:
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
                num_classes=10,
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )
            direct_probe = evaluate_skill_memory(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=10,
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
    print("update_policy=", args.update_mode)
    print("mean_forgetting=", f"{mean_forgetting:.4f}")
    print(
        "mean_final_accuracy (calibrated)=",
        f"{results['mean_final_accuracy']:.4f}",
    )
    print(
        "raw_mean_final_accuracy=",
        f"{results['raw_mean_final_accuracy']:.4f}",
    )
    print("mean_final_loss=", f"{results['mean_final_loss']:.4f}")

    if args.diagnose:
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

    print("final_class_accuracy:")
    for class_id, accuracy in final_accuracy.items():
        print(f"  class {class_id}: {accuracy:.4f}")
    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
