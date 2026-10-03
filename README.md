# Skill Memory

**Skill Memory** is a class-incremental continual-learning strategy for
[Avalanche](https://avalanche.continualai.org/) that answers a simple
question directly, instead of hoping a single shared model answers it
implicitly: _when I meet a new class, should the network reuse an
existing skill, or does this class need one of its own?_

It does this by keeping one **stored weight snapshot per skill**,
probing candidates numerically before committing to either choice, and
evaluating the result **anonymously** from the stored skills alone (no
labels, no task id, no separately trained evaluator network).

Every formula the code implements is written out in
[`docs/MATHEMATICS.md`](docs/MATHEMATICS.md).

- [Why](#why)
- [How it decides: reuse vs. scratch](#how-it-decides-reuse-vs-scratch)
- [Training with history: replay, refresh and reproducibility](#training-with-history-replay-refresh-and-reproducibility)
- [How it is measured](#how-it-is-measured)
- [Anonymous routing](#anonymous-routing-no-task-id-no-label)
- [Package layout](#package-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Demos](#demos)
- [Performance: functional probing, and where the time goes](#performance-functional-probing-and-where-the-time-goes)
- [Diagnostics and the `diagnose` contract](#diagnostics-and-the-diagnose-contract)
- [Development](#development)
- [Design invariants](#design-invariants)
- [License](#license)

## Why

Class-incremental continual learning usually asks one network to keep
absorbing new classes without forgetting old ones, then measures success
after the fact with accuracy curves. Skill Memory instead makes the
reuse-or-specialize choice an explicit, numerically justified decision at
the moment a new class arrives, and keeps whichever skill a class ends up
assigned to **frozen** for the rest of training — so a later class can
never silently overwrite an earlier one's weights.

## How it decides: reuse vs. scratch

When class $c$ first appears, [`skill_memory.cl.decision`](skill_memory/cl/decision.py)
scores every existing skill $s$ against it without training anything. For a
probe batch $(x, y)$ of $c$'s own examples, it loads skill $s$'s frozen
weights, grows the classifier head just far enough to have a column for
$c$ (Avalanche's own `IncrementalClassifier.adaptation`, so a genuinely new
class always starts from a freshly initialized column), and reads off:

$$
\text{new\_score}_s = \frac{1}{N}\sum_{i=1}^{N} \operatorname{softmax}(f_{\theta_s}(x_i))_{y_i},
\qquad
\text{new\_accuracy}_s = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\arg\max_k f_{\theta_s}(x_i)_k = y_i\right]
$$

The same measurement, run on skill $s$'s own already-mastered classes
$c' \in \text{owned}(s)$, gives the _safety_ side of the decision — the
**worst** old class, not the average, so one quietly forgotten class can't
hide behind the others:

$$
\text{old\_accuracy}_s = \min_{c' \in \text{owned}(s)} \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\!\left[\arg\max_k f_{\theta_s}(x_i')_k = y_i'\right]
$$

A skill only becomes a reuse candidate once every one of its old classes
stays above a chance-relative margin, and its new-class fit clears an
absolute floor:

$$
\text{old\_accuracy}_s > \text{chance}_s + \delta
\qquad\text{and}\qquad
\text{new\_score}_s > \tau
$$

with $\text{chance}_s = 1 / |\text{classifier width of } s|$, forgetting
margin $\delta$ (default $0.05$), and score floor $\tau$ (default $0.9$).
Among the skills that pass, the strongest candidates on `new_score` _and_
on `new_accuracy` are found independently by looking for the largest gap
in each sorted ranking (`_strongest_candidates`, not just "above the
floor") — the class is only reused if the two rankings agree on the same
skill; otherwise a fresh skill is trained from scratch. See
[`find_best_skill`](skill_memory/cl/decision.py) for the exact rule and
[`skill_memory/tests/test_decision.py`](skill_memory/tests/test_decision.py)
for worked cases.

## Training with history: replay, refresh and reproducibility

Three independent knobs control what a class is trained on. They are
deliberately separate so that an experiment changes **one factor at a time**
(formal definitions: [`docs/MATHEMATICS.md` §4–§6](docs/MATHEMATICS.md)).

| Knob | Values | Meaning |
|---|---|---|
| `cl_update_mode` | `new_class` | current-class data only — **historical replay = 0** |
| | `small_replay` | current class + at most `cl_replay_per_class` ($K$) retained examples per old class |
| | `replay` (default) | current class + **all currently retained** examples per old class |
| `refresh_existing_skills` | `False` (default) / `True` | additionally retrain every pre-existing skill on the enlarged domain after each experience (binary mode; needs history, so not with `new_class`) |
| `binary_negative_pool` | off (default) | offline *oracle* negatives for ablations; needs `allow_offline_negative_pool=True`, obeys the same cap, never combines with `new_class` |

`replay` means *all retained history*, not *all historical training data*:
the retained memory is bounded by `eval_memory_per_class`.

The `validation_fraction` hold-out is **calibration data** (it fits the
evaluator's Platt scaling). It is excluded from training and never replayed.

**Auditable, not just documented.** Every training call records how many
examples came from each source. After a run:

```python
from skill_memory.diagnostics import replay_provenance_report

audit = replay_provenance_report(strategy, diagnose=True)
audit["class_training"]  # {"calls", "optimizer_steps", "historical_examples", ...}
audit["refresh"]  # same, for refreshes of existing skills
audit["violations"]  # [] iff the invariants of the chosen mode all hold
```

**Reproducible.** `training_seed` seeds mini-batch order, balanced
re-sampling and dropout through explicit generators; the same seeds give
bit-identical stored skills whatever the global RNG state.

**Cost.** Refreshing is quadratic in the number of experiences (each of the
$\mathcal O(t)$ skills is retrained at each experience), so it has its own
timing bucket (`skill_memory_domain_refresh`) and its own step count in the
audit — it is *not* part of "replay quantity".

## How it is measured

**`strategy.eval()` uses [`CLEvaluationPlugin`](skill_memory/evaluation/cl_evaluator.py)**
and nothing else. At prediction time only $x$ is available: each trained
class $c$ is scored by the YES/NO verifier of the skill that owns it, then

$$
\hat y_{\rm raw}(x)=\arg\max_c r_c(x),\qquad
\hat y_{\rm cal}(x)=\arg\max_c\,(a_c\,r_c(x)+b_c),
$$

where $(a_c,b_c)$ is a monotone Platt calibration fitted on the skill's
held-out calibration examples. No evaluator model is trained and test labels
are used only after prediction, to count correct answers. Two accuracies are
reported: `raw_mean_final_accuracy` (**the primary diagnostic**) and
`mean_final_accuracy` (calibrated). Classes with no skill yet score a
constant, so evaluating on a stream that contains future classes is safe.

**Everything in the [`skill_memory.diagnostics`](skill_memory/diagnostics/)
package is opt-in, never runs inside `strategy.eval()`, and refuses to run
at all unless you pass `diagnose=True`** (see
[Diagnostics and the `diagnose` contract](#diagnostics-and-the-diagnose-contract)).
It answers different questions:

- `evaluate_class_oracle` routes each sample using its **true label** —
  an upper bound that presupposes knowing the answer already.
- `evaluate_skill_memory(..., routing="probe")` routes anonymously with
  `find_best_routing_skill` (below) — no label, no task id.
- `replay_provenance_report` proves which historical data each training
  call consumed (see above).
- `audit_strategy_leakage` / `audit_split_overlap` check content-level
  train/test overlap. Exact-duplicate detection is a *diagnostic*, not a proof
  of no leakage: near-duplicates and transformed copies are not detected. The
  protocol itself is enforced at runtime by `strict_protocol=True`.

Keeping these separate means a strong number from `strategy.eval()` can
never be quietly explained by peeking at ground truth.

## Anonymous routing (no task id, no label)

`find_best_routing_skill` (in [`skill_memory/diagnostics/routing.py`](skill_memory/diagnostics/routing.py))
picks one skill per sample using only _that skill's own_ raw response at
_its own_ owned class columns — no shared evaluator, no learned router.
For skill $s$'s raw logits $z_s \in \mathbb{R}^{N \times C_s}$ on owned
classes $\text{owned}(s)$:

$$
\text{score}_s =
\begin{cases}
0 & \text{owned}(s) = \varnothing \\[4pt]
\sigma(z_{s,0}) & C_s = 1 \text{ (a genuine single-class head — softmax over one column is always 1 and uninformative)} \\[4pt]
\displaystyle\sum_{c \in \text{owned}(s)} \operatorname{softmax}(z_s)_c & \text{otherwise}
\end{cases}
$$

Scores are stacked across all $S$ skills and turned into a routing
distribution with a temperature $T$ (default $1$, sharper as $T \to 0$,
uniform fallback if every skill scores $0$):

$$
p(s \mid x) = \frac{\text{score}_s^{1/T}}{\sum_{s'=1}^{S} \text{score}_{s'}^{1/T}}
$$

The routed skill is $\arg\max_s p(s \mid x)$; `confidence_gap` is the
margin between the top two entries of $p(\cdot \mid x)$. The persistent,
long-running version of this same idea —
[`PersistentFingerprintSkillMemoryPlugin`](skill_memory/evaluation/fingerprint_routing.py)
— caches each skill's fingerprint and only recomputes it when
[`global_fingerprint_refresh`](skill_memory/evaluation/global_fingerprint_refresh.py)
detects drift, and can optionally reconstruct a class's decision boundary
directly from a skill's stored _weights_ rather than from probe forward
passes — see
[`reverse_engineer_scores_from_weights`](skill_memory/evaluation/behavior.py)
and [`NormalMLReverseEngineer`](skill_memory/evaluation/reverse_engineering.py).

## Package layout

```
skill_memory/
├── strategy.py                  # SkillMemoryStrategy: the public Avalanche-facing API
├── diagnostics/                   # ALL diagnostic code; every function needs diagnose=True
│   ├── routing.py                 # find_best_routing_skill, route_probe_logits
│   ├── evaluation.py              # evaluate_class_oracle, evaluate_skill_memory
│   ├── replay.py                  # replay_provenance_report: proves what history was trained on
│   ├── alignment.py               # routing_rank_diagnostics, class_index_alignment_report
│   ├── timing.py                  # TimingAccumulator, timing_report, reset_timing
│   ├── leakage.py                 # audit_split_overlap: content-level train/test overlap audit
│   ├── old_scores.py              # measure_old_class_scores: on-demand old-class probability scores
│   └── _gate.py                   # require_diagnose: the one shared enforcement point
├── cl/                            # Skill Memory itself: what to freeze, when, and why
│   ├── skill_registry.py         # SkillMemory (frozen state store) + ExperienceClassMap (bookkeeping)
│   ├── decision.py                # The reuse-vs-scratch probing/decision policy (math above)
│   ├── replay.py                  # ReplayPolicy / RefreshPolicy: the one definition of "historical data"
│   ├── training.py                # Seeded class training + skill-domain refresh, with provenance
│   ├── skill_memory_plugin.py    # Avalanche plugin wiring decision.py into the training loop
│   └── persistent_skill_memory_plugin.py  # Long-running variant with cached anonymous routing
├── evaluation/                    # Everything about *measuring* the strategy
│   ├── memory.py                  # EvaluationMemoryPlugin: bounded per-class retained memory
│   ├── cl_evaluator.py            # CLEvaluationPlugin: stored-skill evaluator + Platt calibration (production eval)
│   ├── routing.py                 # Shared routing math: score_skill_compatibility, select_skill_from_scores
│   ├── behavior.py                 # Per-class behavior fingerprints; reverse-engineering from weights
│   ├── reverse_engineering.py     # NormalMLReverseEngineer: learned weight -> decision reconstruction
│   ├── fingerprint_routing.py     # PersistentFingerprintSkillMemoryPlugin (cached anonymous routing)
│   └── global_fingerprint_refresh.py  # Drift detection that triggers a fingerprint recompute
├── utils/
│   ├── probing.py                 # Dataset probing, exact state application, FunctionalStateCache, evaluate_states_batch
│   └── protocol_guard.py          # Cheap runtime train/test protocol guards (strict_protocol)
├── demos/
│   ├── demo_replay_ablation.py  # Offline, one-factor-at-a-time replay/refresh comparison
│   ├── demo_splitmnist.py       # End-to-end SplitMNIST example (downloads data)
│   └── demo_cifar100.py         # End-to-end CIFAR-100 example (downloads data)
├── benchmarks/
│   └── stage1_batching.py  # Sequential vs. batch_stage1 runtime comparison
└── tests/                          # offline test-suite; see "Development"
docs/
└── MATHEMATICS.md                  # every formula the code implements
```

## Installation

Requires Python ≥3.10. From a clone of this repository:

```bash
pip install -e .
pip install -r requirements-dev.txt   # for tests, ruff, pre-commit
```

`pyproject.toml` declares `torch`, `avalanche-lib`, and `numpy` as runtime
dependencies; nothing else is required to import `skill_memory`.

## Quickstart

```python
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP

from skill_memory import SkillMemoryStrategy

# Any classification dataset works; this sketch omits loading one.
benchmark = nc_benchmark(
    train_dataset,
    test_dataset,
    n_experiences=5,
    task_labels=False,
    seed=0,
)

model = SimpleMLP(input_size=28 * 28, num_classes=10)
strategy = SkillMemoryStrategy(
    model=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    criterion=torch.nn.CrossEntropyLoss(),
    class_train_mode="binary_one_vs_rest",
    cl_update_mode="small_replay",  # new_class | small_replay | replay
    cl_replay_per_class=5,  # K, used by small_replay
    refresh_existing_skills=False,  # independent of the replay mode
    eval_memory_per_class=20,
    training_seed=0,  # reproducible training
    train_mb_size=64,
    train_epochs=1,
    eval_mb_size=64,
)

for experience in benchmark.train_stream:
    strategy.train(experience)
    results = strategy.eval(benchmark.test_stream)
    print(results["mean_final_accuracy"])
```

`strategy.eval()` reports the stored-skill CL evaluator's accuracy (raw and
calibrated). To additionally check the stored skills with oracle or probe
routing, or to audit the replay:

```python
from skill_memory.diagnostics import evaluate_class_oracle, evaluate_skill_memory

oracle = evaluate_class_oracle(
    strategy.model,
    strategy.skill_memory_plugin,
    benchmark.test_stream,
    up_to_index=len(benchmark.test_stream) - 1,
    num_classes=10,
    batch_size=64,
    device=strategy.device,
    diagnose=True,  # required: this uses true labels to route
)
probe = evaluate_skill_memory(
    strategy.model,
    strategy.skill_memory_plugin,
    benchmark.test_stream,
    up_to_index=len(benchmark.test_stream) - 1,
    num_classes=10,
    routing="probe",
    batch_size=64,
    device=strategy.device,
    diagnose=True,
)
```

## Demos

**Replay / refresh ablation — offline, seconds, no download:**

```bash
python -m skill_memory.demos.demo_replay_ablation
python -m skill_memory.demos.demo_replay_ablation --seeds 0 1 2 --json out.json
```

Compares `new_class`, `small_replay`, `replay`, and the two replay modes with
`refresh_existing_skills`, on identical data, initial weights and seeds, and
prints calibrated/raw accuracy, forgetting, the historical examples consumed,
the optimiser steps spent on class training vs. refresh, wall time, and the
number of audit violations (must be `0`). Every row differs from its
neighbour in exactly one factor. A single seed on toy data is a smoke test
for the *mechanics*, not evidence about which mode is better — use several
seeds and a real benchmark for that.

**SplitMNIST / CIFAR-100 (download the dataset):**

```bash
python -m skill_memory.demos.demo_splitmnist --n-experiences 5 --train-epochs 1 \
    --cl-update-mode small_replay --cl-replay-per-class 5 --diagnose
python -m skill_memory.demos.demo_cifar100 --help
```

Key flags: `--cl-update-mode {new_class,small_replay,replay}`,
`--cl-replay-per-class`, `--refresh-existing-skills`, `--training-seed`,
`--class-train-mode`, `--diagnose` (also runs the opt-in oracle/probe
diagnostics and the replay audit; never affects the reported accuracy),
`--eval-memory-per-class`, `--skill-train-samples-per-class`, `--max-skills`.

## Development

```bash
pre-commit install        # run automatically on every commit
pre-commit run --all-files
pytest -q                    # offline; no dataset downloads
```

`pre-commit` runs `ruff` (lint + format) and `insert-license`, which adds
the two-line copyright/SPDX header from `LICENSE-HEADER.txt` to the top of
any Python file that doesn't already have one — new files get it
automatically on first commit, so it never needs to be added by hand.

## Performance: functional probing, and where the time goes

Probing many stored skills against the same class batch (the loop inside
`score_class_against_skills`, and the equivalent per-eval-batch routing
in `CLEvaluationPlugin`'s per-batch skill scoring) never mutates a model to
switch skills. `evaluate_state` and `predict_logits`
([`utils/probing.py`](skill_memory/utils/probing.py)) apply each skill's
frozen weights with `torch.func.functional_call` instead of
`load_state_dict`, so trying $S$ skills costs one forward pass each, not
$S$ full parameter copies plus a restore. (`SkillMemoryPlugin`'s own
apply-and-train paths still use the mutating
`apply_skill_state_exact` — that's a real, persistent state change, not a
disposable probe.)

To see where wall-clock time is actually going in your own run, rather
than guessing, build the strategy with `diagnose=True` (timing is never
recorded otherwise — see below):

```python
from skill_memory.diagnostics import timing_report

strategy = SkillMemoryStrategy(..., diagnose=True)
# after some strategy.train(...) / strategy.eval(...) calls:
print(timing_report(strategy))
# {
#   "skill_memory_decision_probing": {"total_seconds": ..., "calls": ..., "mean_seconds": ...},
#   "skill_memory_class_training": {"total_seconds": ..., "calls": ..., "mean_seconds": ...},
#   "skill_memory_domain_refresh": {...},   # only with refresh_existing_skills=True
#   "cl_evaluation": {"total_seconds": ..., "calls": ..., "mean_seconds": ...},
# }
```

`reset_timing(strategy)` clears every bucket, e.g. to isolate one
experience's timing from the run as a whole.

## Diagnostics and the `diagnose` contract

Production code and diagnostic code are kept structurally apart, so it is
auditable — not just promised — that nothing diagnostic can reach a
production number.

- **One package.** Every diagnostic lives in
  [`skill_memory/diagnostics/`](skill_memory/diagnostics/). Nothing in it
  is importable from the bare `skill_memory` namespace — you have to
  write `from skill_memory.diagnostics import ...` on purpose.
- **A required flag, not a default.** `find_best_routing_skill`,
  `route_probe_logits`, `evaluate_skill_memory`, `evaluate_class_oracle`,
  `replay_provenance_report`, `routing_rank_diagnostics` and
  `class_index_alignment_report` all take `diagnose` as a required
  keyword-only argument with **no default**. Leaving it out is a
  `TypeError`; passing `diagnose=False` is a `RuntimeError`. Anything that
  could use a true label (oracle routing) or costs real forward passes
  therefore can't run by accident, whatever the strategy was configured
  with.
- **Zero cost when off.** `SkillMemoryStrategy(diagnose=False)` (the
  default) makes every internal `self.timing.track(...)` a true no-op — it
  doesn't even call `time.perf_counter()`. `timing_report`/`reset_timing`
  need a strategy built with `diagnose=True` and say so clearly otherwise.
- **Enforced by tests.**
  [`tests/test_diagnostics_gate.py`](skill_memory/tests/test_diagnostics_gate.py)
  checks the signatures, the refusals, that no diagnostic name leaks into
  the top-level package, and — by parsing every production module's
  imports — that only the two files that own a `TimingAccumulator` and the
  one plugin that already gated its own reports on `diagnose` import from
  `skill_memory.diagnostics` at all.

To audit a codebase built on this package: grep for `diagnose=True`. Every
match is a place where ground truth or diagnostic cost could enter.

## Design invariants

- **One canonical skill per class, for the strategy's whole lifetime.**
  Once `ExperienceClassMap` records which skill owns a class, that mapping
  never changes; a class is never silently reassigned to a different
  skill later (see [`skill_registry.py`](skill_memory/cl/skill_registry.py)).
- **A skill's weights are frozen the moment it stops being trained.**
  Nothing outside `cl/training.py`'s two training passes (class training
  and the optional refresh) and the probing in `decision.py` (which applies stored weights functionally via
  `torch.func.functional_call`, so it never mutates — or needs a copy of —
  the live model) ever calls `.backward()` using a stored skill's weights.
- **Replay is isolated and auditable.** `new_class` consumes zero
  historical examples *by construction* (the retained memory is not even
  passed down), `small_replay` at most $K$ per old class, `replay` the
  bounded retained memory; the refresh of existing skills and the offline
  oracle pool are separate, explicit opt-ins. `replay_provenance_report`
  verifies this after every run.
- **Training, evaluation memory and test data stay separate.** Evaluation
  memory is drawn only from the experience just trained; the safety stage
  probes only experiences already trained on; test labels are used only
  after predictions are made. `strict_protocol=True` (default) turns the
  common misuses into `ProtocolViolation`, and
  [`tests/test_leakage.py`](skill_memory/tests/test_leakage.py) tries to break
  each rule. For a content-level check of your own split, run
  `skill_memory.diagnostics.audit_strategy_leakage(strategy, test_stream,
  diagnose=True)`.
- **Diagnostics never leak into production metrics.** Everything in
  `skill_memory.diagnostics` requires an explicit `diagnose=True` at the
  call site and is absent from `strategy.eval()`'s return value.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Kobros-Tech Ltd.
