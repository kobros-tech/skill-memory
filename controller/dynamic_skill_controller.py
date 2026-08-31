"""
dynamic_skill_controller.py

Thin orchestrator. It does not implement any acquisition decision logic
itself -- that already exists in skill-cloning's `compatibility.decide()`.
This class just sequences: identify -> (reuse existing | try clone
candidates in ranked order | fall back to scratch) -> validate -> register.

Deliberately not a general "many small classes" design (per the project's
own scope goal): one controller, delegating to skill-cloning's Skill /
TinyMLP / compatibility for anything that's actual ML, and to
skill_memory.registry / skill_memory.identify for the bookkeeping this
project adds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from skill_memory.skill import TinyMLP, Skill
from skill_memory import compatibility as comp
from skill_memory.tasks import sample_task

from skill_memory.config import Config
from skill_memory.registry import SkillRegistry, CapacityExhaustedError
from skill_memory import identify as identify_mod


class DynamicSkillController:
    def __init__(self, config: Optional[Config] = None, registry: Optional[SkillRegistry] = None):
        self.config = config or Config()
        self.registry = registry or SkillRegistry(max_skills=self.config.max_skills)

    # ---- identification / execution -------------------------------------
    def identify(self, examples):
        return identify_mod.identify(
            examples,
            self.registry,
            tol=self.config.identification_tolerance,
            match_threshold=self.config.identification_match_threshold,
            ambiguity_margin=self.config.identification_ambiguity_margin,
        )

    def execute(self, skill_name: str, X):
        return identify_mod.execute(self.registry, skill_name, X)

    # ---- acquisition ------------------------------------------------------
    def acquire(self, task_name: str, seed: Optional[int] = None) -> dict:
        """
        Acquire an unknown skill for `task_name` (must be a name skill-cloning's
        tasks.sample_task recognizes). Tries clone candidates in compatibility-
        ranked order, falls back to scratch, validates independently, and
        registers on success.

        NOTE (known v1 simplification): Skill.train() only early-stops against
        an MSE target (target_mse), not the accuracy criterion used elsewhere
        in this project. We don't pass one here, so `steps` in the result
        below is always the full max_steps budget rather than the step at
        which the accuracy target was actually reached. Tracking accuracy-
        based early stopping (like skill-cloning's own strategies.py does
        internally) is a follow-up, not implemented in this milestone.
        """
        if self.registry.contains(task_name):
            raise ValueError(f"'{task_name}' is already registered")

        cfg = self.config
        seed = cfg.random_seed if seed is None else seed

        X_train, y_train = sample_task(task_name, cfg.n_train, seed=seed, domain=cfg.domain)
        X_val, y_val = sample_task(task_name, cfg.n_eval, seed=seed + 999, domain=cfg.domain)

        existing = self.registry.skills_dict()
        if existing:
            decision = comp.decide(existing, task_name, base_seed=seed, domain=cfg.domain)
        else:
            decision = {"action": "scratch", "parent": None, "score": 0.0,
                        "solve_accuracy": 0.0, "ranking": []}

        attempts = []

        # Reuse is zero-training acquisition: share the already-solved
        # immutable network under the new task name.
        if decision.get("action") == "reuse":
            parent_name = decision["parent"]
            parent_skill = self.registry.get(parent_name).skill
            parent_skill.tasks_covered.append(task_name)
            attempt = {
                "parent": parent_name,
                "origin": "reuse",
                "steps": 0,
                "final_accuracy": decision.get("solve_accuracy", 0.0),
                "solved": True,
                "skill": parent_skill,
            }
            attempts.append(attempt)
            return self._finalize(task_name, attempt, attempts, decision, seed)

        for parent_name, _score in decision.get("ranking", []):
            attempt = self._attempt_clone(parent_name, task_name, X_train, y_train, X_val, y_val, seed)
            attempts.append(attempt)
            if attempt["solved"]:
                return self._finalize(task_name, attempt, attempts, decision, seed)

        attempt = self._attempt_scratch(task_name, X_train, y_train, X_val, y_val, seed)
        attempts.append(attempt)
        if attempt["solved"]:
            return self._finalize(task_name, attempt, attempts, decision, seed)

        return {
            "success": False,
            "skill_name": task_name,
            "attempts": attempts,
            "decision": decision,
        }

    # ---- internal -----------------------------------------------------
    def _attempt_clone(self, parent_name, task_name, X_train, y_train, X_val, y_val, seed):
        parent_skill = self.registry.get(parent_name).skill
        parent_params_before = {k: v.copy() for k, v in parent_skill.net.params.items()}

        cloned_net = parent_skill.net.clone()
        candidate = Skill(task_name, cloned_net, origin="clone", parent=parent_name)
        history = candidate.train(
            X_train, y_train, epochs=self.config.max_steps, lr=self.config.learning_rate,
        )
        val_accuracy = candidate.net.accuracy(X_val, y_val, tol=self.config.acc_tol)
        solved = val_accuracy >= self.config.solve_threshold

        # hard invariant: cloning + training the target must never mutate the parent
        for k, v in parent_params_before.items():
            if not np.allclose(v, parent_skill.net.params[k]):
                raise AssertionError(
                    f"parent skill '{parent_name}' was mutated while training clone '{task_name}'"
                )

        return {
            "parent": parent_name, "origin": "clone", "steps": len(history),
            "final_accuracy": val_accuracy, "solved": solved, "skill": candidate,
        }

    def _attempt_scratch(self, task_name, X_train, y_train, X_val, y_val, seed):
        net = TinyMLP(hidden_dim=self.config.hidden_dim, seed=seed)
        candidate = Skill(task_name, net, origin="scratch", parent=None)
        history = candidate.train(
            X_train, y_train, epochs=self.config.max_steps, lr=self.config.learning_rate,
        )
        val_accuracy = candidate.net.accuracy(X_val, y_val, tol=self.config.acc_tol)
        solved = val_accuracy >= self.config.solve_threshold
        return {
            "parent": None, "origin": "scratch", "steps": len(history),
            "final_accuracy": val_accuracy, "solved": solved, "skill": candidate,
        }

    def _finalize(self, task_name, attempt, attempts, decision, seed):
        try:
            record = self.registry.register(
                task_name,
                attempt["skill"],
                parent=attempt["parent"],
                origin=attempt["origin"],
                acquisition_steps=attempt["steps"],
                solved=attempt["solved"],
                final_accuracy=attempt["final_accuracy"],
                compatibility_score=decision.get("score", 0.0),
                seed=seed,
            )
        except CapacityExhaustedError as e:
            return {"success": False, "skill_name": task_name, "attempts": attempts,
                    "decision": decision, "error": str(e)}
        return {"success": True, "skill_name": task_name, "record": record,
                "attempts": attempts, "decision": decision}
