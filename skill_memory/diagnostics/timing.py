# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Wall-clock timing: a tiny accumulator, plus the report/reset entry points.

`TimingAccumulator` itself lives in this package (not next to the
production code that holds one) so every piece of diagnostic
infrastructure is in one place -- but a `SkillMemoryStrategy` only ever
*records* time through it when built with ``diagnose=True``
(``enabled=False`` makes `track` a true no-op: it doesn't even call
`time.perf_counter()`), and `timing_report`/`reset_timing` refuse to run
against a strategy that wasn't. That way importing this module costs
nothing, and turning timing on is always a decision the caller made, not
a side effect of importing the package.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingAccumulator:
    """Cumulative wall-clock time and call counts, grouped by bucket name.

    A bucket is any string a caller chooses (e.g. ``"decision"``); the same
    name can be tracked many times (once per class, once per experience,
    ...) and the accumulator sums the elapsed time and counts the calls.
    When ``enabled=False``, `track` still works as a context manager but
    records nothing -- no `time.perf_counter()` call, no dict writes --
    so a production run built with ``diagnose=False`` pays zero cost for
    every ``self.timing.track(...)`` call site in the codebase.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    @contextmanager
    def track(self, bucket: str):
        """Time one block of code, adding it to `bucket`'s running total."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[bucket] = self._seconds.get(bucket, 0.0) + elapsed
            self._calls[bucket] = self._calls.get(bucket, 0) + 1

    def report(self) -> dict[str, dict[str, float]]:
        """Return ``{bucket: {total_seconds, calls, mean_seconds}}``, per bucket."""
        return {
            bucket: {
                "total_seconds": total,
                "calls": self._calls[bucket],
                "mean_seconds": total / self._calls[bucket],
            }
            for bucket, total in self._seconds.items()
        }

    def reset(self) -> None:
        """Clear every recorded bucket, e.g. between independent experiments."""
        self._seconds.clear()
        self._calls.clear()


def _require_diagnosing_strategy(strategy, function_name: str) -> None:
    if not getattr(strategy, "diagnose", False):
        raise RuntimeError(
            f"skill_memory.diagnostics.{function_name} needs a strategy "
            "built with SkillMemoryStrategy(..., diagnose=True): timing "
            "is never recorded otherwise, so there would be nothing to "
            "report. Rebuild the strategy with diagnose=True if you want "
            "to measure where time is going."
        )


def timing_report(strategy) -> dict[str, dict[str, float]]:
    """Return cumulative wall-clock time for each stage of `strategy`'s lifecycle.

    Requires `strategy` to have been built with
    ``SkillMemoryStrategy(..., diagnose=True)`` -- timing is never
    recorded otherwise, by design, so there is nothing to report from a
    plain production strategy. Buckets, each accumulated since `strategy`
    was created (or since the last :func:`reset_timing`); a bucket only
    appears once it has been used:

    - ``"skill_memory_decision_probing"`` -- every call to `decide_class`
      (the REUSE-vs-SCRATCH probing loop in
      :mod:`skill_memory.cl.decision`), timed once per class.
    - ``"skill_memory_class_training"`` -- every call to `train_on_class`
      (:mod:`skill_memory.cl.training`), timed once per class, whether it
      trained a brand-new skill from scratch or updated a reused one.
    - ``"skill_memory_domain_refresh"`` -- every `train_skill_on_domain`
      call made by ``refresh_existing_skills=True``; timed separately so the
      cost of refreshing existing skills is never confused with the cost of
      the replay policy itself.
    - ``"cl_evaluation"`` -- every call to `strategy.eval(...)`, timed once
      per call: the Avalanche evaluation loop with the stored-skill
      :class:`~skill_memory.evaluation.cl_evaluator.CLEvaluationPlugin`
      (calibration fitting included).

    Each bucket reports ``total_seconds``, ``calls``, and
    ``mean_seconds`` -- comparing `total_seconds` across buckets tells you
    which stage of a slow run to optimize next, instead of guessing.
    """
    _require_diagnosing_strategy(strategy, "timing_report")
    report = strategy.skill_memory_plugin.timing.report()
    report.update(strategy.timing.report())
    return report


def reset_timing(strategy) -> None:
    """Clear every bucket :func:`timing_report` would otherwise accumulate.

    Also requires ``diagnose=True`` (see :func:`timing_report`). Useful
    for timing one specific experience or `eval()` call in isolation, e.g.
    immediately before the training/eval step you want to measure.
    """
    _require_diagnosing_strategy(strategy, "reset_timing")
    strategy.skill_memory_plugin.timing.reset()
    strategy.timing.reset()
