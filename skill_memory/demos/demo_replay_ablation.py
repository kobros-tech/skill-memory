# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

r"""Replay-quantity ablation on synthetic data (no download, runs in seconds).

Compares, under *identical* data, model initialisation and seeds::

    new_class             historical replay = 0
    small_replay          <= K retained examples / old class
    replay                all currently retained examples / old class
    <mode> + refresh      the same, plus retraining of existing skills

Because ``cl_update_mode`` and ``refresh_existing_skills`` are independent
switches, every row differs from its neighbour in exactly **one** factor, so
an accuracy or time difference can be attributed to that factor.  Besides
accuracy, each row reports *how much* history was consumed and how many
optimiser steps were spent, taken from the replay provenance audit
(:func:`skill_memory.diagnostics.replay_provenance_report`); the ``viol``
column must always be ``0``.

Run::

    python -m skill_memory.demos.demo_replay_ablation
    python -m skill_memory.demos.demo_replay_ablation --seeds 0 1 2 --json out.json
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

#: (label, cl_update_mode, refresh_existing_skills)
CONFIGURATIONS = (
    ("new_class", "new_class", False),
    ("small_replay", "small_replay", False),
    ("replay", "replay", False),
    ("small_replay+refresh", "small_replay", True),
    ("replay+refresh", "replay", True),
)


@dataclass
class Row:
    """One configuration, averaged over seeds."""

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
    """Gaussian blobs: class ``c`` is shifted along feature ``c % N_FEATURES``."""

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
    refresh: bool,
    *,
    seed: int,
    n_classes: int,
    n_experiences: int,
    n_per_class: int,
    replay_per_class: int,
    memory_per_class: int,
) -> dict:
    """Train + evaluate one configuration; return a flat metrics dict."""
    benchmark = make_benchmark(n_classes, n_experiences, n_per_class, seed)
    torch.manual_seed(seed)  # identical initial weights for every configuration
    model = SimpleMLP(input_size=N_FEATURES, hidden_size=16, num_classes=n_classes)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        class_train_mode="binary_one_vs_rest",
        cl_update_mode=mode,
        cl_replay_per_class=replay_per_class,
        refresh_existing_skills=refresh,
        eval_memory_per_class=memory_per_class,
        skill_train_samples_per_class=memory_per_class,
        max_skills=n_classes,
        train_mb_size=16,
        eval_mb_size=64,
        train_epochs=2,
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
        max(h[c] for h in history[introduced[c] :] if c in h) - final[c] for c in final
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
    """Run every configuration for every seed and average the metrics."""
    rows = []
    for name, mode, refresh in CONFIGURATIONS:
        runs = [
            run_configuration(mode, refresh, seed=seed, **settings) for seed in seeds
        ]
        mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        mean["violations"] = int(sum(r["violations"] for r in runs))
        rows.append(Row(name=name, **mean))
    return rows


def format_table(rows: list[Row]) -> str:
    header = (
        f"{'configuration':<22}{'acc(cal)':>9}{'acc(raw)':>9}{'forget':>8}"
        f"{'hist':>8}{'cls-steps':>10}{'ref-steps':>10}{'sec':>7}{'viol':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<22}{r.calibrated_accuracy:>9.3f}{r.raw_accuracy:>9.3f}"
            f"{r.forgetting:>8.3f}{r.historical_examples:>8.0f}"
            f"{r.class_steps:>10.0f}{r.refresh_steps:>10.0f}"
            f"{r.seconds:>7.2f}{r.violations:>6d}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--n-experiences", type=int, default=4)
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--replay-per-class", type=int, default=3, help="K")
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
        "\nhist = historical examples consumed; cls-steps / ref-steps = optimiser "
        "steps spent on class training / on refreshing existing skills."
    )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([asdict(r) for r in rows], handle, indent=2)


if __name__ == "__main__":
    main()
