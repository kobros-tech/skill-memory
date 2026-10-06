# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Sequential Skill Memory experiment on Split CIFAR-100 (downloads the data).

Example (the configuration used for the published CIFAR-100 results)::

    python -m skill_memory.demos.demo_cifar100 --n-experiences 20 \
        --max-experiences 3 --update-mode replay --memory-per-class 50 \
        --train-samples-per-class 50 --class-train-epochs 10 --seed 3 --diagnose

Switch ``--update-mode`` between ``new_class``, ``replay`` and ``refresh`` to
compare the policies; everything else (data, init, seeds) stays identical.
"""

from __future__ import annotations

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.models import SlimResNet18

from skill_memory.demos._common import build_parser, check_args, run_experiment


def main() -> None:
    args = build_parser(
        "Sequential Skill Memory experiment on CIFAR-100.", default_experiences=20
    ).parse_args()
    check_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Preparing CIFAR-100 dataset... (root: {args.dataset_root})")
    benchmark = SplitCIFAR100(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )
    if args.download_only:
        print(f"CIFAR-100 dataset prepared at {args.dataset_root}")
        return
    run_experiment(
        args,
        title="CIFAR-100",
        benchmark=benchmark,
        make_model=lambda device: SlimResNet18(nclasses=100).to(device),
        num_classes=100,
    )


if __name__ == "__main__":
    main()
