# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Controlled replay-policy ablation on synthetic data.

All configurations use identical data, initial weights and seeds. The only
algorithmic dimensions are the complete update policy and the retained-history
budget:

    new_class          no retained history
    replay (K)         retained history capped at K/class
    replay (all)       all retained history
    refresh (K)        replay K/class + existing-skill refresh
    refresh (all)      replay all retained history + existing-skill refresh

Refresh is a first-class update policy, not a second flag that can accidentally
be combined with another experiment. The report includes optimizer work so
accuracy is never interpreted without its processing cost.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from torch.utils.data import TensorDataset

from skill_memory import SkillMemoryStrategy
from skill_memory.diagnostics import replay_provenance_report

N_FEATURES = 8

CONFIGURATIONS = (
    ("new_class", "new_class", None),
    ("replay(K)", "replay", "K"),
    ("replay(all)", "replay", None),
    ("refresh(K)", "refresh", "K"),
    ("refresh(all)", "refresh", None),
)


@dataclass
class Row:
    name: str
    calibrated_accuracy: float
    raw_accuracy: float
    forgetting: float
    historical_examples: float
    class_steps: float
    refresh_steps: float
    seconds: float
    violations: int


def make_benchmark(n_classes: int, n_experiences: int, n_per_class: int, seed: int):
    """Create deterministic Gaussian class blobs."""
    def split(offset: int):
        generator = torch.Generator().manual_seed(seed * 100 + offset)
        xs, ys = [], []
        for class_id in range(n_classes):
            center = torch.zeros(N_FEATURES)
            center[class_id % N_FEATURES] = 4.0
            center[(class_id * 3 + 1) % N_FEATURES] += 2.0
            xs.append(
                center + torch.randn(n_per_class, N_FEATURES, generator=generator)
            )
            ys.append(torch.full((n_per_class,), class_id, dtype=torch.long))
        return TensorDataset(torch.cat(xs), torch.cat(ys))

    return nc_benchmark(
        split(1),
        split(2),
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        shuffle=False,
        fixed_class_order=list(range(n_classes)),
    )


def run_configuration(
    mode: str,
    replay_budget: str | None,
    *,
    seed: int,
    n_classes: int,
    n_experiences: int,
    n_per_class: int,
    replay_per_class: int,
    memory_per_class: int,
) -> dict:
    benchmark = make_benchmark(n_classes, n_experiences, n_per_class, seed)
    torch.manual_seed(seed)
    model = SimpleMLP(input_size=N_FEATURES, hidden_size=16, num_classes=n_classes)

    replay_samples = (
        replay_per_class if replay_budget == "K" else None
    )
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        class_train_mode="binary_one_vs_rest",
        update_mode=mode,
        replay_samples_per_class=replay_samples,
        memory_per_class=memory_per_class,
        train_samples_per_class=memory_per_class,
        max_skills=n_classes,
        train_mb_size=16,
        eval_mb_size=64,
        class_train_epochs=2,
        probe_seed=seed,
        training_seed=seed,
        verbose=False,
    )

    history: list[dict[int, float]] = []
    introduced: dict[int, int] = {}
    started = time.perf_counter()
    results: dict = {}
    for step, experience in enumerate(benchmark.train_stream):
        strategy.train(experience)
        results = strategy.eval(list(benchmark.test_stream)[: step + 1])
        history.append(dict(results["final_class_accuracy"]))
        for class_id in history[-1]:
            introduced.setdefault(class_id, step)
    seconds = time.perf_counter() - started

    final = history[-1]
    drops = [
        max(h[c] for h in history[introduced[c]:] if c in h) - final[c]
        for c in final
    ]
    audit = replay_provenance_report(strategy, diagnose=True)
    return {
        "calibrated_accuracy": float(results["mean_final_accuracy"]),
        "raw_accuracy": float(results["raw_mean_final_accuracy"]),
        "forgetting": float(np.mean(drops)),
        "historical_examples": audit["class_training"]["historical_examples"]
        + audit["refresh"]["historical_examples"],
        "class_steps": audit["class_training"]["optimizer_steps"],
        "refresh_steps": audit["refresh"]["optimizer_steps"],
        "seconds": seconds,
        "violations": len(audit["violations"]),
    }


def run_ablation(seeds, **settings) -> list[Row]:
    rows = []
    for name, mode, replay_budget in CONFIGURATIONS:
        runs = [
            run_configuration(
                mode,
                replay_budget,
                seed=seed,
                **settings,
            )
            for seed in seeds
        ]
        mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        mean["violations"] = int(sum(r["violations"] for r in runs))
        rows.append(Row(name=name, **mean))
    return rows


def format_table(rows: list[Row]) -> str:
    header = (
        f"{'configuration':<16}{'acc(cal)':>9}{'acc(raw)':>9}{'forget':>8}"
        f"{'hist':>8}{'cls-steps':>10}{'ref-steps':>10}{'sec':>7}{'viol':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.name:<16}{row.calibrated_accuracy:>9.3f}"
            f"{row.raw_accuracy:>9.3f}{row.forgetting:>8.3f}"
            f"{row.historical_examples:>8.0f}{row.class_steps:>10.0f}"
            f"{row.refresh_steps:>10.0f}{row.seconds:>7.2f}"
            f"{row.violations:>6d}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--n-experiences", type=int, default=4)
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--replay-per-class", type=int, default=3)
    parser.add_argument("--memory-per-class", type=int, default=12)
    parser.add_argument("--json", help="also write the rows to this JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_ablation(
        args.seeds,
        n_classes=args.n_classes,
        n_experiences=args.n_experiences,
        n_per_class=args.n_per_class,
        replay_per_class=args.replay_per_class,
        memory_per_class=args.memory_per_class,
    )
    print(
        f"classes={args.n_classes} experiences={args.n_experiences} "
        f"K={args.replay_per_class} retained/class={args.memory_per_class} "
        f"seeds={args.seeds}\n"
    )
    print(format_table(rows))
    print(
        "\nhist = historical examples consumed; cls-steps / ref-steps = "
        "optimizer steps spent on class training / existing-skill refresh."
    )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([asdict(row) for row in rows], handle, indent=2)


if __name__ == "__main__":
    main()
