# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.3.0] - 2026

### Added

- `reverse_warm_start` / `reverse_warm_start_epochs` on
  `PersistentFingerprintSkillMemoryPlugin` (default `False` / `None`,
  preserving exact prior from-scratch behavior). When enabled, each
  experience after the first seeds its listwise reverse router
  (`NormalMLReverseEngineer`) with the previous experience's compatible
  weights (`_warm_start_model`: the Transformer encoder and output head are
  always shape-compatible across experiences since `hidden_size`/
  `num_heads`/`num_layers` are fixed; only `input_projection` grows when new
  classes widen the candidate feature vector, and its existing columns are
  preserved) and trains for `reverse_warm_start_epochs` steps instead of
  refitting the whole candidate history from scratch every time.
- This is **not** a numerically-exact optimization like the
  `SkillMemoryPlugin` ones from 2.1.0/2.2.0 -- it is a genuine accuracy/speed
  tradeoff, measured on a 5-experience SplitMNIST run
  (`reverse_epochs=10`, `reverse_warm_start_epochs=3`):

  | experience | skills | router fit, from-scratch | router fit, warm-start | accuracy, from-scratch | accuracy, warm-start |
  |---|---|---|---|---|---|
  | 0 | 2 | 1.3s | 1.1s | 99.4% | 99.4% |
  | 1 | 4 | 4.0s | 1.3s | 96.4% | 91.3% |
  | 2 | 6 | 8.0s | 2.8s | 92.3% | 89.0% |
  | 3 | 8 | 15.1s | 5.1s | 90.1% | 89.5% |
  | 4 | 10 | ~24s | 8.0s | 89.6% | 88.4% |

  Router-fit time roughly triples at the final experience; accuracy drops by
  up to ~5 points (most visibly at experience 1). Mean forgetting was
  measured *lower* with warm-starting (0.7%-2.4% vs. 1.3%-5.5%) in this run,
  despite lower absolute accuracy -- forgetting is peak-minus-current per
  class, and warm-starting's accuracy was more stable rather than peaking
  high early, so this is not a second accuracy advantage to read into it.
  `reverse_warm_start_epochs` was not tuned beyond the single value above;
  a larger value would likely narrow the accuracy gap at some cost to the
  speedup. Default remains `False` pending that tuning.
- `skill_memory/tests/test_reverse_router_warm_start.py`: a from-scratch
  reference implementation checked bit-for-bit against the current default
  call, direct weight-level checks that warm-starting reuses compatible
  weights (encoder/output exactly, `input_projection` columns preserved
  under both growing and shrinking feature width), and plugin-level wiring
  checks (epoch selection on the first vs. later fits, validation).

## [2.2.0] - 2026

### Added

- **`batch_stage1` (default `False`) on `SkillMemoryStrategy` /
  `SkillMemoryPlugin`.** Stage 1 of the decision (scoring a new class against
  *every* stored skill -- unlike stage 2, it has no cap) can now run through
  `skill_memory.utils.probing.evaluate_states_batch`: skills whose expanded
  parameter states share an identical shape (see that function's docstring
  for the grouping rule) are stacked and evaluated in one `torch.vmap` call
  instead of one Python-level call per skill. This is an **exact**
  optimization -- same candidate skills, same expanded skill states (the
  existing `FunctionalStateCache` is used and filled identically), same
  forward-pass math -- verified against the sequential path within ordinary
  float32 tolerance (`skill_memory/tests/test_stage1_batching.py`).
  **Default is `False` because it is not reliably faster**: measured on CPU
  (`skill_memory/benchmarks/stage1_batching.py`, both an MLP and a small
  conv+BatchNorm backbone) it was consistently **slower** than the
  sequential loop (roughly 0.4x-0.8x, i.e. 25-60% slower), because
  functorch's `vmap` batching rules do not always lower to a fused kernel on
  CPU. Its benefit on GPU is plausible (this is the standard `torch.func`
  ensembling pattern for many same-architecture, different-weight models)
  but has not been measured in this repository -- benchmark on your own
  target hardware before enabling it. `stage1_chunk_size` bounds memory when
  a shape group is large enough to risk exhausting device memory (forwarded
  to `torch.vmap(..., chunk_size=...)`).
- `skill_memory/benchmarks/stage1_batching.py`: sequential-vs-batched
  benchmark isolating stage-1 cost, with an MLP and a small conv+BatchNorm
  backbone, configurable skill count / batch size / device / chunk size.
- `skill_memory/tests/test_stage1_batching.py`: correctness of
  `evaluate_states_batch` against the sequential reference (including a
  conv+BatchNorm model, heterogeneous classifier-width grouping, chunking,
  and cache reuse), plus end-to-end tests that `batch_stage1=True` produces
  identical routing decisions and stored skill weights to `batch_stage1=False`
  on a full seeded strategy run, and that the flag actually selects the
  batched code path (not just a placebo).

### Fixed

- Diagnosed the real bottleneck at scale using a live CIFAR-100 CI run (20
  experiences, SlimResNet18): decision time was NOT dominated by the
  already-bounded/optimized safety stage (2.1.0), but by stage 1, which is
  unbounded by design and had never been addressed. See the PR discussion
  for the profiling trail; `evaluate_states_batch` above is the fix this
  produced, shipped as opt-in pending a GPU benchmark.

## [2.1.0] - 2026

### Breaking

- **`max_safety_candidates` now defaults to `5`** (was `None`, i.e. verify
  every skill). `None` is still available and means exact/full safety. A
  finite cap is an approximation that can miss a reusable lower-ranked skill.
- **`old_score` is no longer measured during training.** Safety is judged on
  old-class accuracy alone. Decision dicts and `score_class_against_skills`
  results no longer contain `old_score`, and `ClassRecord.old_score` was
  removed. Per-class `old_metrics` now hold `{"class", "accuracy"}` only.
  Migration: call `skill_memory.diagnostics.measure_old_class_scores(
  strategy, skill, diagnose=True)` when you want the score.
- **`strict_protocol=True` is a new default** on `SkillMemoryStrategy`,
  `SkillMemoryPlugin` and `MLEvaluationPlugin`. It raises
  `ProtocolViolation` when training on a test-stream experience or
  evaluating on a train-stream experience. Pass `strict_protocol=False` to
  opt out.

### Changed

- **Performance (results unchanged, verified by tests and by comparing
  every decision on 10-experience SplitMNIST):**
  - old-class safety checks short-circuit at the first failed class
    (exact: the skill is already unsafe);
  - old-class safety uses the accuracy-only `evaluate_state_accuracy`;
  - `FunctionalStateCache` caches expanded functional skill states per
    experience (identity-keyed, seeded probes only);
  - `DecisionProbeCache` draws each seeded old-class probe once per pool of
    experiences containing the class instead of once per new-class decision;
  - class membership checks (`experience_has_class`) are O(1) after the
    first call per dataset instead of rebuilding index lists;
  - `EvaluationMemoryPlugin._build_evaluation_memory` reads labels from the
    dataset's cached `.targets` instead of decoding every sample, so only the
    retained samples are decoded (bit-identical memory; ~2.8s -> ~0.1s over
    5 SplitMNIST experiences).
  Measured on SplitMNIST (10 experiences, CPU): decision probing
  1.38s -> 0.64s (exact) / 0.55s (default cap); whole training run
  ~18.5s -> ~14.3s wall clock. Class training (per-sample decode in the
  DataLoader, ~5.5s) is unchanged.

### Added

- `skill_memory.utils.protocol_guard`: `ProtocolViolation` and cheap
  train/test guards.
- `skill_memory.diagnostics.leakage` (`audit_split_overlap`,
  `assert_no_split_overlap`, `audit_strategy_leakage`): exact-duplicate
  content audit between the test stream and evaluation memory / training
  data. Requires `diagnose=True`.
- `skill_memory.diagnostics.measure_old_class_scores`: on-demand old-class
  loss/score/accuracy for a stored skill. Requires `diagnose=True`.
- `tests/test_leakage.py` (poisoned-dataset, label-independence, probing
  side-effect and optimization-exactness tests) and
  `tests/test_safety_optimizations.py`.

## [2.0.0] - 2026

### Breaking

- **All diagnostic code now lives in one package,
  `skill_memory/diagnostics/`**, replacing the top-level
  `skill_memory/diagnostics.py`, `skill_memory/evaluation/diagnostics.py`
  and `skill_memory/utils/timing.py`.
- **Every diagnostic entry point now requires `diagnose=True`** as a
  keyword-only argument with no default (`find_best_routing_skill`,
  `route_probe_logits`, `evaluate_skill_memory`, `evaluate_class_oracle`,
  `diagnose_evaluator_probe`, `routing_rank_diagnostics`,
  `class_index_alignment_report`). Omitting it is a `TypeError`;
  `diagnose=False` is a `RuntimeError`. Migration: add `diagnose=True` at
  each call site.
- `class_index_alignment_report` is no longer exported from the top-level
  `skill_memory` namespace; import it from `skill_memory.diagnostics`.
- `timing_report` / `reset_timing` now require a strategy built with
  `SkillMemoryStrategy(..., diagnose=True)` (new argument, default
  `False`). Previously timing was always recorded.

### Changed

- With `diagnose=False`, every internal `self.timing.track(...)` is a true
  no-op (no `time.perf_counter()` call), so production runs pay nothing
  for instrumentation. `PersistentFingerprintSkillMemoryPlugin`'s existing
  `diagnose` flag now also controls this.

### Added

- `tests/test_diagnostics_gate.py`: enforces the required-keyword
  signatures, the refusals, that no diagnostic name leaks into the
  top-level package, and (by parsing imports) that production modules
  import from `skill_memory.diagnostics` only in three known, gated places.
- The demo's `--diagnose` flag now drives the strategy's `diagnose=` and
  prints the timing report.

### Fixed

- README no longer claims probing operates on a `deepcopy` (it has been
  functional since 1.1.0).

## [1.1.0] - 2026

### Changed

- **Performance:** the "for every new class, for every skill: load skill
  state, forward probe batch" loop
  (`decision.score_class_against_skills`), and the equivalent per-batch
  routing in `MLEvaluationPlugin.after_eval_forward`, no longer mutate a
  model to switch between skills. `evaluate_state` and `predict_logits`
  now apply each skill's frozen weights with `torch.func.functional_call`
  instead of `load_state_dict`, removing an O(skills) full-parameter-copy
  cost from both hot loops (and the `deepcopy`s that existed only to make
  that mutation safe to undo). The full test suite's wall-clock time fell
  from ~111s to ~17s as a direct result.

### Added

- `skill_memory.diagnostics.timing_report(strategy)` /
  `reset_timing(strategy)`: cumulative wall-clock time and call counts for
  skill-memory decision/probing, skill-memory class training, and the
  independent evaluator + test-evaluation loop, so a slow run can be
  measured instead of guessed at.

## [1.0.0] - 2026

First public/production release.

### Fixed

- Restored `skill_memory/utils/` (dataset probing, exact state
  application, and `IncrementalClassifier` introspection) and the
  top-level `skill_memory/diagnostics.py` (anonymous routing and direct
  Skill Memory evaluation), both of which were missing from the previous
  internal snapshot. Without them, `pip install -e .` failed outright and
  roughly a third of the test suite could not even be collected.

### Changed

- Renamed `skill_memory/evaluation/ml_cl_evaluator.py` to
  `skill_memory/evaluation/independent_evaluator.py` to match the name
  used throughout its own documentation and the rest of the codebase.

### Added

- Copyright/SPDX headers on every source file, inserted and kept in sync
  automatically by a `pre-commit` hook (`insert-license`) rather than by
  hand.
- Mathematical description of the reuse-vs-scratch decision policy and
  the anonymous routing rule in the [README](README.md).
- Packaging metadata for a public release: license/author/classifier
  fields in `pyproject.toml`, and `skill_memory.demos` as an installable
  package (previously only importable from an editable checkout).

### Verified

- Full test suite (96 tests) passes.
- The `IncrementalClassifier` growth/resize path — not exercised by any
  existing test, since all of them use a fixed-size classifier — was
  additionally smoke-tested end to end against real Avalanche machinery.
