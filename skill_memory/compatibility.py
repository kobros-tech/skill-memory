"""
Compatibility scoring and the reuse/clone/scratch decision gate.
"""
from __future__ import annotations
import numpy as np

SCALE = 60.0
TAU_SOLVE = 0.90
TAU_CLONE = 0.15
PROBE_N = 64
PROBE_SEED_OFFSET = 10_000
SOLVE_PROBE_SEED_OFFSET = 20_000
ACC_TOL = 0.5
ACC_SOLVE_TARGET = 0.85

def _probe(skill_net, task, base_seed, domain="nonnegative"):
    from .tasks import sample_task
    X, y = sample_task(task, PROBE_N, seed=base_seed + PROBE_SEED_OFFSET, domain=domain)
    mse = skill_net.mse(X, y)
    score = float(np.exp(-mse / SCALE))
    return score, X, y

def compatibility_score(skill_net, task, base_seed, domain="nonnegative"):
    score, _, _ = _probe(skill_net, task, base_seed, domain=domain)
    return score

def solve_probe_accuracy(skill_net, task, base_seed, domain="nonnegative"):
    from .tasks import sample_task
    X, y = sample_task(task, PROBE_N, seed=base_seed + SOLVE_PROBE_SEED_OFFSET, domain=domain)
    return float(skill_net.accuracy(X, y, tol=ACC_TOL))

def rank_skills(skills, task, base_seed, domain="nonnegative"):
    scored = [(name, compatibility_score(s.net, task, base_seed, domain=domain))
              for name, s in skills.items()]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

def decide(skills, task, base_seed, domain="nonnegative"):
    if not skills:
        return {"action": "scratch", "parent": None, "score": 0.0,
                "solve_accuracy": 0.0, "ranking": []}
    ranking = rank_skills(skills, task, base_seed, domain=domain)
    best_name, best_score = ranking[0]
    solve_accuracy = solve_probe_accuracy(skills[best_name].net, task, base_seed, domain=domain)
    if best_score >= TAU_SOLVE and solve_accuracy >= ACC_SOLVE_TARGET:
        return {"action": "reuse", "parent": best_name, "score": best_score,
                "solve_accuracy": solve_accuracy, "ranking": ranking}
    if best_score >= TAU_CLONE:
        return {"action": "clone", "parent": best_name, "score": best_score,
                "solve_accuracy": solve_accuracy, "ranking": ranking}
    return {"action": "scratch", "parent": None, "score": best_score,
            "solve_accuracy": solve_accuracy, "ranking": ranking}
