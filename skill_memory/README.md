# `skill_memory` — implementation notes

This is a contributor-facing companion to the [root README](../README.md),
which covers the concepts, the math, and how to use the package. This
document covers the internal contracts each module relies on, for anyone
changing the code rather than just calling it.

## Module dependency order

Lower modules never import from higher ones — breaking this ordering
reintroduces a real circular import, not just a lint warning:

```
utils/probing.py            (no dependency on cl/ or evaluation/)
        |
        v
cl/skill_registry.py  -->  cl/decision.py  -->  cl/skill_memory_plugin.py
        |                                              |
        v                                              v
cl/training.py                          cl/persistent_skill_memory_plugin.py
        |
        v
evaluation/routing.py
        |
        v
evaluation/behavior.py --> evaluation/reverse_engineering.py
        |
        v
evaluation/fingerprint_routing.py, evaluation/global_fingerprint_refresh.py
        |
        v
evaluation/independent_evaluator.py   (subclasses cl.skill_memory_plugin.SkillMemoryPlugin)
        |
        v
strategy.py                             (top-level; imports everything above)

diagnostics/   (depends only on utils/ and evaluation/routing.py; imported by
                production code in exactly three places -- see below)
```

`evaluation/independent_evaluator.py` is deliberately **not** re-exported
from `evaluation/__init__.py` — only from the top-level `skill_memory`
package — for exactly this reason (see the comment at the top of
`evaluation/__init__.py`).

## Bookkeeping invariants (`cl/skill_registry.py`)

- `SkillMemory` stores `state_dict` snapshots by integer slot;
  `ExperienceClassMap` stores which slot owns which class. These are kept
  as two separate objects on purpose: a skill can master more than one
  class, and one experience can therefore be associated with more than one
  `(skill, classes)` group.
- Once `ExperienceClassMap` records a class → skill mapping, it is never
  overwritten. `find_skill_for_class_anywhere` is the one lookup every
  other module should use rather than re-deriving it.
- `SkillMemory.allocate()` reserves the *lowest free* slot and raises
  `RuntimeError` once `max_skills` is reached — callers (`decision.py`,
  `skill_memory_plugin.py`) are expected to handle that as "memory full,"
  not as a bug.

## State application (`utils/probing.py`)

Two distinct contracts live side by side here; picking the wrong one for
a new call site either corrupts the live model or silently reintroduces
the O(skills) cost this module exists to avoid.

**Mutating** (a real, persistent state change): `apply_skill_state_exact`
first calls `resize_incremental_classifiers_for_state` so
`nn.Module.load_state_dict` never fails on a shape mismatch between the
model's current `IncrementalClassifier` width and the snapshot's recorded
width, then loads it for real. Use this (via `restore_initial_state`,
its own name for the same operation used to undo scratch-training
adaptation) wherever a skill's weights need to actually become the live
model's weights going forward — `SkillMemoryPlugin`'s REUSE/SCRATCH
training paths, and its before/after-eval snapshot restore.

**Functional** (a disposable probe): `predict_logits` and
`evaluate_state` apply a stored snapshot with
`torch.func.functional_call` instead, so the model they're given is
*never mutated* — no resize, no restore, and (critically) no per-skill
`load_state_dict` copy of every parameter tensor. This is what makes
`score_class_against_skills`' "for every stored skill, forward a probe
batch" loop, and `MLEvaluationPlugin.after_eval_forward`'s per-batch
routing, cheap: trying skill `k+1` costs one more forward pass, not one
more full parameter copy. `evaluate_state`'s classifier-growth rule for a
genuinely new class (`_functional_growth_for_experience`) deliberately
duplicates `IncrementalClassifier.adaptation`'s math rather than calling
`prepare_for_experience` (the mutating version), for the same reason.
When adding a new read-only probe, prefer this contract; reach for the
mutating one only when the caller genuinely needs the model itself to
keep the new state afterwards.

`classes_in_experience`/`class_indices` cache each dataset's full label
list, keyed by the dataset *object* (a `weakref.WeakKeyDictionary`, not
`id()`), so `decision.py`'s per-`(skill, class)` probing doesn't rescan
the same dataset once per pair. See
[`tests/test_probing_cache.py`](tests/test_probing_cache.py) for the
exact scanning-cost guarantee this cache makes.

## Decision policy (`cl/decision.py`)

See the [root README's math section](../README.md#how-it-decides-reuse-vs-scratch)
for the formulas. Implementation notes that don't belong in that
higher-level explanation:

- `score_class_against_skills` is two-staged on purpose: stage 1
  (`evaluate_state` against the new class) runs for *every* stored skill,
  cheaply — one functional forward pass each, no model copies (see
  above); stage 2 verifies old classes for the `max_safety_candidates`
  strongest candidates (**default 5**; `None` verifies every skill
  exactly). A finite cap is an explicit performance approximation: it can
  miss a reusable lower-ranked skill.
- Stage 2 is judged on old-class **accuracy only**, so it uses the
  accuracy-only `evaluate_state_accuracy` (no loss, no softmax) and
  **short-circuits**: a candidate's old classes are checked one at a time and
  checking stops at the first class with `acc <= chance + forgetting_margin`.
  That skill is already unsafe under `find_best_skill`, so the REUSE/SCRATCH
  outcome is unchanged; only its `old_metrics` list is partial
  (`safety_complete=False`).
- The old-class probability *score* (`old_score`) is not part of the decision
  path any more. Use `skill_memory.diagnostics.measure_old_class_scores(
  strategy, skill, diagnose=True)` to compute it on demand.
- `DecisionProbeCache` (one per plugin) makes repeated probing cheaper
  without changing any result: expanded functional skill states are cached
  per experience (keyed by the identity of the stored snapshot, so a
  re-stored skill is never served stale), and seeded old-class probe batches
  are drawn once per *pool of experiences that contain the class*. Unseeded
  runs (`probe_seed=None`) are never cached, because their probes are random
  by design.
- `_strongest_candidates` finds the largest gap in a sorted metric
  ranking rather than a fixed threshold, so the "how much better than the
  runner-up does a candidate need to be" question doesn't need its own
  magic number.
- `evaluate_state` never takes a gradient step — "imagination" means
  measuring a frozen skill's *existing* representation on a class it may
  never have trained on, not training it further.

## Timing instrumentation (`diagnostics/timing.py`)

`SkillMemoryPlugin.timing` and `SkillMemoryStrategy.timing` are each a
`TimingAccumulator`; `diagnostics.timing_report(strategy)` merges both
into the three buckets described in the
[root README](../README.md#performance-functional-probing-and-where-the-time-goes).
If you add a new expensive stage to the lifecycle, wrap it with
`self.timing.track("some_bucket_name")` on whichever plugin/strategy owns
it, rather than adding another ad hoc `time.perf_counter()` call — the
existing three buckets are read together specifically so the report
stays comparable across runs.

## Anonymous routing (`evaluation/routing.py`, `diagnostics/routing.py`)

`evaluation/routing.py` holds the shared primitives
(`score_skill_compatibility`, `select_skill_from_scores`,
`_normalize_routing_scores`) used by both the evaluator-based probe router
(`evaluation/independent_evaluator.py`) and the evaluator-free anonymous
router (`find_best_routing_skill` in `diagnostics/routing.py`). If
you change the temperature/normalization rule in one, check whether the
other's tests
([`tests/test_routing.py`](tests/test_routing.py),
[`tests/test_continuous_fingerprint_routing.py`](tests/test_continuous_fingerprint_routing.py))
still hold — they intentionally share the same math.

## Running the tests

```bash
pytest skill_memory/tests -q
```

96 tests, no network access and no GPU required; the slowest ones
(`test_strategy.py`) build tiny synthetic Avalanche benchmarks rather than
downloading a real dataset, so the whole suite runs in well under two
minutes on CPU.

## Adding a new diagnostic

1. Put it in `skill_memory/diagnostics/`, never next to production code.
2. If it can read a true label or costs real forward passes, make
   `diagnose` a **required keyword-only argument with no default** and call
   `require_diagnose(diagnose, "your_function_name")` first (see
   `diagnostics/_gate.py`), then add it to
   `GATED_FUNCTIONS` in `tests/test_diagnostics_gate.py` — that test
   asserts the no-default rule mechanically.
3. Do **not** import it from production modules. If production code truly
   must (as `fingerprint_routing.py` does for its own already-flag-gated
   reports), add the import to the `allowed` set in
   `test_production_modules_do_not_import_diagnostic_functions` and say why
   in the docstring — the point of that test is that this list stays tiny
   and deliberate.

## Protocol guards and leakage audit

Two layers, deliberately separate:

- **Runtime guards** (`utils/protocol_guard.py`, on by default via
  `strict_protocol=True`, constant time): training on a `test`-stream
  experience, evaluating on a `train`-stream experience, evaluating on a
  dataset object Skill Memory already trained on, or capturing evaluation
  memory for classes the source experience doesn't declare all raise
  `ProtocolViolation`. These are name/identity checks; they cannot see a
  test split that contains *copies* of training samples.
- **Content audit** (`diagnostics/leakage.py`, needs `diagnose=True`):
  `audit_split_overlap` / `audit_strategy_leakage` / `assert_no_split_overlap`
  hash the actual tensors and count exact duplicates between the test stream
  and the evaluation memory / training data. Near-duplicates and random
  augmentation inside `__getitem__` defeat it — an empty audit is evidence,
  not proof.

`tests/test_leakage.py` exercises both by *trying to break the protocol*:
poisoned datasets that raise on any premature read of future or wrong-split
data, bit-identical model outputs under swapped test labels, probing side
effects (stored skills, live model, global RNG), and end-to-end equality
between the optimized and reference paths.

## Stage-1 batching (`batch_stage1`)

Stage 1 (new-class compatibility) scores *every* stored skill and, unlike
stage 2, has no cap -- it is the part of decision time that keeps growing as
skills accumulate. `batch_stage1=True` (default `False`) evaluates it via
`skill_memory.utils.probing.evaluate_states_batch` instead of one
`evaluate_state` call per skill:

- Skills are grouped by `_param_shape_signature` -- the shape *and* dtype of
  every tensor in their expanded (post-growth) state. Skills captured at
  different points of classifier growth can genuinely need different
  `IncrementalClassifier` widths for the same probe (see that function's
  docstring), so groups are computed, never assumed uniform.
- Each group of 2+ same-shaped skills is stacked and run through one
  `torch.vmap(functional_call)` call; a group of exactly 1 falls back to the
  ordinary sequential call.
- This is exact, not an approximation: same candidate skills, same expanded
  states (the same `FunctionalStateCache` is used), same forward-pass math,
  differing from the sequential path only by ordinary float32
  matmul-reassociation noise.
- **It is not a safe default.** Measured on CPU with both an MLP and a small
  conv+BatchNorm backbone (`skill_memory/benchmarks/stage1_batching.py`), it
  was consistently *slower* than the sequential loop (~0.4x-0.8x). GPU
  benefit is plausible (this is the standard `torch.func` pattern for
  evaluating many same-architecture models with different weights) but
  unverified in this repository -- benchmark your own hardware before
  enabling it, and use `stage1_chunk_size` if a run has memory headroom
  concerns.
