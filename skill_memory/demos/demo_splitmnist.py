# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Sequential Skill Memory experiment on SplitMNIST (downloads the data).

Example::

    python -m skill_memory.demos.demo_splitmnist --update-mode refresh \
        --class-train-epochs 1 --seed 0 --diagnose

Same flags and output format as ``demo_cifar100`` (see its docstring).
"""

from __future__ import annotations

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP

from skill_memory.demos._common import build_parser, check_args, run_experiment


def main() -> None:
    args = build_parser(
        "Sequential Skill Memory experiment on SplitMNIST.",
        default_experiences=5,
        default_epochs=1,
    ).parse_args()
    check_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Preparing SplitMNIST dataset... (root: {args.dataset_root})")
    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )
    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return
    run_experiment(
        args,
        title="SplitMNIST",
        benchmark=benchmark,
        make_model=lambda device: SimpleMLP(num_classes=10).to(device),
        num_classes=10,
    )


if __name__ == "__main__":
    main()
