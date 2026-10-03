#!/usr/bin/env python3
# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Benchmark: sequential vs. batched (`batch_stage1=True`) stage-1 scoring.

Stage 1 of the decision (`score_class_against_skills` in
`skill_memory.cl.decision`) scores *every* stored skill against a new class
and has no bound (unlike stage 2, which is capped by
`max_safety_candidates`). As the number of stored skills grows, this is the
part of decision time that keeps growing.

This script isolates exactly that cost -- it does not run a full
SkillMemoryStrategy training loop (see `skill_memory/demos/demo_splitmnist.py`
for that) -- so the two conditions being compared differ in nothing except
`batch_stage1`. Run it on the hardware you actually care about; see the
module docstring of `evaluate_states_batch`
(`skill_memory/utils/probing.py`) for why this is not free/GPU-vs-CPU
neutral: functorch's `vmap` batching rules do not always map to a single
fused kernel, so measured speedup (or slowdown) is model- and
device-dependent, not something to assume.

Usage::

    python -m skill_memory.benchmarks.stage1_batching
    python -m skill_memory.benchmarks.stage1_batching --n-skills 20 50 100 200
    python -m skill_memory.benchmarks.stage1_batching --model resnet --device cuda
    python -m skill_memory.benchmarks.stage1_batching --chunk-size 16
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import torch
from avalanche.models.dynamic_modules import IncrementalClassifier
from torch import nn

from skill_memory.utils.probing import (
    FunctionalStateCache,
    evaluate_state,
    evaluate_states_batch,
)


class MLPBackbone(nn.Module):
    """Cheap stand-in for SplitMNIST-scale experiments (no BatchNorm)."""

    def __init__(self, in_features: int = 28 * 28, hidden: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.body = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.classifier = IncrementalClassifier(hidden, initial_out_features=2)

    def forward(self, x):
        return self.classifier(self.body(self.flatten(x)))


class SmallResNetBackbone(nn.Module):
    """A small conv + BatchNorm backbone: what actually matters here.

    Not the real `SlimResNet18` from the OCL Survey harness (not a
    dependency of this package), but exercises the same ingredients that
    make stage 1 expensive there: convolutions and BatchNorm running stats
    that differ per stored skill, evaluated on 32x32x3 images.
    """

    def __init__(self, width: int = 32):
        super().__init__()

        def block(c_in, c_out, stride):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.stem = block(3, width, 1)
        self.layer1 = block(width, width, 1)
        self.layer2 = block(width, width * 2, 2)
        self.layer3 = block(width * 2, width * 4, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = IncrementalClassifier(width * 4, initial_out_features=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


MODELS = {"mlp": MLPBackbone, "resnet": SmallResNetBackbone}


def make_states(model_fn, n_skills: int, device: str, n_prior_classes: int = 6):
    """`n_skills` independently-perturbed snapshots of `model_fn()`, all
    already grown to `n_prior_classes` columns (as stored skills would be
    partway through a real run)."""
    torch.manual_seed(0)
    template = model_fn().to(device)
    template.classifier.adaptation(
        SimpleNamespace(classes_in_this_experience=list(range(n_prior_classes)))
    )
    states = []
    for slot in range(n_skills):
        torch.manual_seed(slot)
        state = {k: v.clone() for k, v in template.state_dict().items()}
        for key, value in state.items():
            if value.is_floating_point():
                state[key] = value + torch.randn_like(value) * 0.02
        states.append((slot, state))
    return template, states


def run_once(
    model,
    states,
    x,
    y,
    experience,
    *,
    batch_stage1: bool,
    chunk_size: int | None,
    cache: FunctionalStateCache | None,
) -> float:
    start = time.perf_counter()
    if batch_stage1:
        evaluate_states_batch(
            model, states, x, y, experience, seed=0, cache=cache, chunk_size=chunk_size
        )
    else:
        for slot, state in states:
            evaluate_state(
                model,
                state,
                x,
                y,
                nn.functional.cross_entropy,
                experience,
                seed=0,
                cache=cache,
                slot=slot,
            )
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def benchmark(
    model_name: str,
    n_skills_list: list[int],
    device: str,
    probe_batch_size: int,
    n_prior_classes: int,
    n_new_classes: int,
    chunk_size: int | None,
    repeats: int,
    use_cache: bool,
):
    model_fn = MODELS[model_name]
    image_size = 32 if model_name == "resnet" else 28
    channels = 3 if model_name == "resnet" else 1

    print(
        f"model={model_name} device={device} probe_batch_size={probe_batch_size} "
        f"chunk_size={chunk_size} cache={'on' if use_cache else 'off'}"
    )
    print(f"{'n_skills':>9} {'sequential(s)':>15} {'batched(s)':>12} {'speedup':>9}")

    for n_skills in n_skills_list:
        model, states = make_states(model_fn, n_skills, device, n_prior_classes)
        experience = SimpleNamespace(
            classes_in_this_experience=list(
                range(n_prior_classes, n_prior_classes + n_new_classes)
            )
        )
        x = torch.randn(
            probe_batch_size, channels, image_size, image_size, device=device
        )
        y = torch.randint(
            n_prior_classes,
            n_prior_classes + n_new_classes,
            (probe_batch_size,),
            device=device,
        )

        seq_times, batch_times = [], []
        for _ in range(repeats):
            cache = FunctionalStateCache() if use_cache else None
            seq_times.append(
                run_once(
                    model,
                    states,
                    x,
                    y,
                    experience,
                    batch_stage1=False,
                    chunk_size=chunk_size,
                    cache=cache,
                )
            )
            cache = FunctionalStateCache() if use_cache else None
            batch_times.append(
                run_once(
                    model,
                    states,
                    x,
                    y,
                    experience,
                    batch_stage1=True,
                    chunk_size=chunk_size,
                    cache=cache,
                )
            )

        t_seq = min(seq_times)
        t_batch = min(batch_times)
        print(f"{n_skills:>9} {t_seq:>15.3f} {t_batch:>12.3f} {t_seq / t_batch:>8.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="resnet")
    parser.add_argument("--n-skills", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--probe-batch-size", type=int, default=320)
    parser.add_argument("--n-prior-classes", type=int, default=6)
    parser.add_argument("--n-new-classes", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    benchmark(
        args.model,
        args.n_skills,
        args.device,
        args.probe_batch_size,
        args.n_prior_classes,
        args.n_new_classes,
        args.chunk_size,
        args.repeats,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
