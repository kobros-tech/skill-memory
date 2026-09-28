# `skill_memory` — implementation notes

Contributor companion to the [root README](../README.md) (usage, parameters) and
[`docs/MATHEMATICS.md`](../docs/MATHEMATICS.md) (formulas). This file lists the
internal contracts each module relies on, for anyone changing the code.

## Module dependency order

Lower modules never import from higher ones; breaking this order creates a real
circular import.

```
utils/probing.py, utils/protocol_guard.py     (depend on neither cl/ nor evaluation/)
        |
        v
cl/skill_registry.py   cl/replay.py   -->  cl/training.py
        |                                       |
        v                                       v
cl/decision.py  -------------------->  cl/skill_memory_plugin.py
                                                |
                                                v
                      evaluation/cl_evaluator.py  -->  strategy.py
diagnostics/*   (import from everything above; production code imports only
                 `TimingAccumulator` from it, enforced by test_diagnostics_gate)
```

`evaluation/__init__.py` re-exports nothing on purpose (it would import `cl/`
back before `cl/` has finished loading).

## Update-policy contracts (`cl/replay.py`, `cl/training.py`)

- `UpdatePolicy` is the **only** place that decides what "historical data"
  means. `history_for_training(...)` returns `None` for `new_class`, so the
  replay memory is not even passed down; never add a second path that reads
  `plugin.replay_memory` directly.
- Every `train_on_class` / `train_skill_on_domain` call returns a
  `TrainingResult` whose `provenance` is appended to `plugin.training_log`. A
  new data source must be added to `TrainingProvenance` *and* to the invariant
  checks of `diagnostics/replay.py`, or the audit goes blind.
- All randomness in training flows from explicit seeds
  (`seed` -> `derive_seed` -> `_seeded_loader` + `_isolated_rng`). Never call
  `torch.manual_seed` or draw from the global RNG inside the training path;
  `tests/test_reproducibility.py` fails if you do.
- The `validation_fraction` hold-out is stored under `CALIBRATION_EXAMPLES_KEY`
  and is **calibration data only**: never trained on, never replayed.
- Replay-memory retention runs *after* the lifecycle step of
  `after_training_exp` on purpose: a refresh must only see classes of earlier
  experiences.
- **Do not change what is computed without updating the golden file.**
  `tests/test_golden_regression.py` compares provenance, decisions, stored
  weights and accuracies of every update policy against the validated
  reference in `tests/data/golden_v3.json`.

## Bookkeeping invariants (`cl/skill_registry.py`)

- `ExperienceClassMap` is write-once per class: a class's skill never changes
  after it is recorded.
- A skill's `domain_classes` metadata records the observed domain it was last
  trained on; `refresh` skips a skill whose domain is already current.

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
batch" loop, and `CLEvaluationPlugin`'s per-batch skill scoring, cheap: trying skill `k+1` costs one more forward pass, not one
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

See [`docs/MATHEMATICS.md` §2](../docs/MATHEMATICS.md) for the formulas. Implementation notes that don't belong in that
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
- The old-class probability *score* is not part of the decision path (accuracy
  is all that stage 2 needs).
- `DecisionProbeCache` (one per plugin) makes repeated probing cheaper
  without changing any result: expanded functional skill states are cached
  per experience (keyed by the identity of the stored snapshot, so a
  re-stored skill is never served stale), and seeded old-class probe batches
  are drawn once per *pool of experiences that contain the class*. Unseeded
  probing (`probe_seed=None`, only reachable by calling `decide_class`
  directly) is never cached, because its probes are random by design.
- `_strongest_candidates` finds the largest gap in a sorted metric
  ranking rather than a fixed threshold, so the "how much better than the
  runner-up does a candidate need to be" question doesn't need its own
  magic number.
- `evaluate_state` never takes a gradient step — "imagination" means
  measuring a frozen skill's *existing* representation on a class it may
  never have trained on, not training it further.

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
  conv+BatchNorm backbone (`python -m skill_memory.demos.demo_stage1_timing`), it
  was consistently *slower* than the sequential loop (~0.4x-0.8x). GPU
  benefit is plausible (this is the standard `torch.func` pattern for
  evaluating many same-architecture models with different weights) but
  unverified in this repository -- run that demo on your hardware (it prints a per-skill-count
  recommendation) before enabling it, and use `stage1_chunk_size` if a run has memory headroom
  concerns.

## Running the tests

```bash
pytest -q                                   # whole suite, offline
pytest -q skill_memory/tests/test_golden_regression.py   # "is the engine unchanged?"
```

Fixtures live in `tests/_helpers.py` (`make_benchmark`, `make_strategy`,
`train_all`): a six-class Gaussian benchmark and a tiny dropout MLP, so every
test is a real end-to-end run in well under a second. `test_ci_workflows.py`
renders every workflow `run:` script per matrix entry and checks shell quoting.

## Adding a diagnostic

Put it in `diagnostics/`, give it `*, diagnose: bool` (no default) and call
`require_diagnose(diagnose, name)` first; add it to `diagnostics/__init__.py`
and to `GATED_FUNCTIONS` in `tests/test_diagnostics_gate.py`. Production code
must not import it (the gate test fails otherwise).

## Protocol guards (`utils/protocol_guard.py`)

With `strict_protocol=True` the plugins call `assert_training_experience`,
`assert_evaluation_experiences` and `assert_memory_classes_match`, which raise
`ProtocolViolation` if training touches a test-stream experience, evaluation
touches a train-stream one, evaluation receives a single experience where a
stream is required, or the replay memory holds a class that is not one of the
experience just trained. `diagnostics.audit_split_overlap` /
`audit_strategy_leakage` additionally report exact-content overlap between
splits (a diagnostic, not a proof of no leakage).
