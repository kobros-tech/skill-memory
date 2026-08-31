"""
config.py

One small, explicit config object so notebooks can tweak experiment
parameters without hunting through the code. Mirrors the defaults already
used by skill-cloning's experiments (200 train / 300 eval examples, 1500-step
budget, 0.02 lr, 0.5 accuracy tolerance, 0.85 solve target) so results stay
comparable to that project's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # capacity: soft budget on number of registered skills (not neuron ranges --
    # each skill here is its own independent network, see registry.py)
    max_skills: int = 8

    # training
    hidden_dim: int = 32
    max_steps: int = 1500
    learning_rate: float = 0.02
    n_train: int = 200
    n_eval: int = 300

    # solve / accuracy criteria (matches skill-cloning's ACC_TOL / ACC_SOLVE_TARGET)
    acc_tol: float = 0.5
    solve_threshold: float = 0.85

    # skill identification (this project's addition -- see identify.py)
    identification_tolerance: float = 0.5
    identification_match_threshold: float = 0.9
    identification_ambiguity_margin: float = 0.05

    # task domain, forwarded to skill-cloning's tasks.sample_task(..., domain=...)
    domain: str = "nonnegative"

    random_seed: int = 0
