"""
identify.py

The genuinely new interface this project adds on top of skill-cloning:
figuring out *which* registered skill explains a batch of incoming
(x, y, observed_result) examples, as distinct from *executing* a skill on
(x, y) once it's known which one to use.

A single example does not necessarily identify a unique skill -- e.g.
(2, 2, 4) is explained by both addition and multiplication. `identify()`
therefore takes a batch and can report AMBIGUOUS as well as KNOWN/UNKNOWN,
rather than forcing a decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

KNOWN = "known"
UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"

Example = tuple[float, float, float]  # (x, y, observed_result), raw arithmetic operands

# skill-cloning trains its TinyMLPs on inputs pre-scaled by /10 (see tasks.py:
# "inputs are scaled by /10 to keep the tanh layer out of saturation"). The
# (x, y, observed_result) interface this project adds takes *raw* operands
# (e.g. (4, 4, 16), matching the plan's examples), so every call into a
# skill's net.predict() has to apply the same scaling skill-cloning used
# during training, or predictions are silently garbage (out-of-distribution
# inputs). Keep this constant in sync with tasks.py if that ever changes.
INPUT_SCALE = 10.0


@dataclass
class IdentificationResult:
    status: str                       # KNOWN | UNKNOWN | AMBIGUOUS
    skill_name: Optional[str] = None  # set only when status == KNOWN
    scores: dict[str, float] = None   # skill name -> match rate in [0, 1]


def _match_rate(net, examples: Iterable[Example], tol: float) -> float:
    """Fraction of examples where net.predict(x, y) is within tol of observed_result."""
    examples = list(examples)
    X = np.array([[x, y] for x, y, _ in examples], dtype=float) / INPUT_SCALE
    observed = np.array([r for _, _, r in examples], dtype=float)
    pred = net.predict(X)
    return float(np.mean(np.abs(pred - observed) <= tol))


def identify(
    examples: Iterable[Example],
    registry,
    tol: float = 0.5,
    match_threshold: float = 0.9,
    ambiguity_margin: float = 0.05,
) -> IdentificationResult:
    """
    Score every registered skill against a batch of (x, y, observed_result)
    examples and classify the batch as KNOWN (one skill matches well and is
    clearly best), UNKNOWN (no skill matches well enough), or AMBIGUOUS
    (two or more skills match comparably well).
    """
    examples = list(examples)
    if not examples:
        raise ValueError("identify() needs at least one (x, y, observed_result) example")

    names = registry.list_skills()
    scores = {
        name: _match_rate(registry.get(name).skill.net, examples, tol)
        for name in names
    }

    if not scores:
        return IdentificationResult(status=UNKNOWN, scores=scores)

    qualifying = {n: s for n, s in scores.items() if s >= match_threshold}
    if not qualifying:
        return IdentificationResult(status=UNKNOWN, scores=scores)

    best_name = max(qualifying, key=qualifying.get)
    best_score = qualifying[best_name]
    contenders = [n for n, s in qualifying.items() if best_score - s <= ambiguity_margin]

    if len(contenders) > 1:
        return IdentificationResult(status=AMBIGUOUS, scores=scores)

    return IdentificationResult(status=KNOWN, skill_name=best_name, scores=scores)


def execute(registry, skill_name: str, X) -> np.ndarray:
    """Run a *known* skill on raw (x, y) pairs only -- no observed_result involved."""
    net = registry.get(skill_name).skill.net
    X = np.asarray(X, dtype=float) / INPUT_SCALE
    return net.predict(X)
