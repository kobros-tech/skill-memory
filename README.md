# Skill Memory

Continual learning on [Avalanche](https://avalanche.continualai.org/) with **one
YES/NO verifier ("skill") per group of classes**. For every new class the
strategy decides, by numerically probing the stored skills, whether an existing
skill can absorb it (**REUSE**) or a new one is needed (**SCRATCH**), trains a
binary verifier under a chosen `update_mode`, and evaluates anonymously from the
stored skills alone -- no labels, no task id, no extra evaluator network.

Every formula the code implements is written out in
[`docs/MATHEMATICS.md`](docs/MATHEMATICS.md).

- [Install](#install) · [Quick start](#quick-start) · [Update modes](#update-modes)
- [Parameters and what works with what](#parameters-and-what-works-with-what)
- [How it is evaluated](#how-it-is-evaluated) · [Diagnostics](#diagnostics)
- [Demos and reference results](#demos-and-reference-results)
- [Layout](#layout) · [Development](#development)

## Install

```bash
pip install -e ".[dev]"      # Python >= 3.10, torch >= 2.0, avalanche-lib
```

## Quick start

```python
import torch
from avalanche.models import SimpleMLP
from skill_memory import SkillMemoryStrategy

model = SimpleMLP(num_classes=10)
strategy = SkillMemoryStrategy(
    model=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    criterion=torch.nn.CrossEntropyLoss(),
    update_mode="replay",  # new_class | replay | refresh
    memory_per_class=20,
    seed=0,
)
for experience in benchmark.train_stream:  # any Avalanche class-incremental benchmark
    strategy.train(experience)
results = strategy.eval(benchmark.test_stream)
print(results["raw_mean_final_accuracy"], results["mean_final_accuracy"])
```

Binary one-vs-rest training learns "this class vs. the rest", so **the first
experience needs at least two classes** (or later ones must replay history).

## Update modes

`update_mode` is one complete training policy. Skills are verifiers trained on
the *current* data plus, depending on the mode, *retained* history (a bounded
per-class replay memory of `memory_per_class` frozen examples).

| `update_mode` | history used for each new class | existing skills |
|---|---|---|
| `new_class` | none | frozen |
| `replay` | all retained examples of every old class, or `replay_samples_per_class` of them | frozen |
| `refresh` | same as `replay` | each retrained once per experience on the enlarged domain |

`replay` means *all currently retained history*, not *all historical training
data* -- the memory is bounded. `refresh` is the most accurate and by far the
most expensive: it adds one training pass per existing skill per experience
(quadratic in the number of experiences, see
[`MATHEMATICS.md` §6](docs/MATHEMATICS.md)).

The 3-experience Split-CIFAR-100 run below (seed 3, 50 examples/class, 10 epochs)
shows the trade-off; times are from one laptop CPU and only their ratios matter:

| `update_mode` | calibrated acc. | raw acc. | forgetting | class training | refresh |
|---|---|---|---|---|---|
| `new_class` | 0.267 | 0.225 | 0.107 | 453 s | -- |
| `replay` | 0.291 | 0.233 | 0.083 | 706 s | -- |
| `refresh` | 0.329 | 0.340 | 0.176 | 1615 s | 1389 s |

## Parameters and what works with what

All parameters are keyword-only. Groups:

| Group | Parameters |
|---|---|
| **policy** | `update_mode`, `replay_samples_per_class` |
| **data** | `memory_per_class` (replay memory), `train_samples_per_class` (current examples/class, default `memory_per_class`), `validation_fraction` (calibration hold-out), `class_train_epochs`, `train_mb_size` |
| **decision** | `max_skills`, `forgetting_margin`, `score_floor`, `probe_batch_size`, `probe_batches`, `max_safety_candidates`, `force_decision`, `reuse_is_mutable`, `batch_stage1`, `stage1_chunk_size` |
| **run** | `seed`, `memory_seed`, `device`, `verbose`, `diagnose`, `strict_protocol`, `eval_mb_size`, `eval_chunk_size` |

**Rules checked at construction** (a violation raises `ValueError`):

| Combination | Why |
|---|---|
| `replay_samples_per_class` with `update_mode="new_class"` | no history is replayed |
| `replay_samples_per_class > memory_per_class` | cannot replay more than is retained |
| `update_mode="refresh"` with `reuse_is_mutable=False` | refresh edits stored skills |
| `stage1_chunk_size` without `batch_stage1=True` | the chunk only exists in the batched path |
| `validation_fraction` outside `[0, 1)` | `0` is allowed and means "no calibration" |

**Silently inert** (documented, not errors): with `force_decision` set, probing
never runs, so `forgetting_margin`, `score_floor`, `probe_batch_size`,
`probe_batches`, `max_safety_candidates` and `batch_stage1` have no effect.
`diagnose` and `verbose` never change results. `batch_stage1` / `eval_chunk_size`
change speed only, never decisions or predictions.

**Seeds.** `seed` drives everything stochastic (probe batches, calibration
hold-out, replay selection, mini-batch order, balanced re-sampling, dropout)
through explicit generators, so equal seeds give bit-identical stored skills
whatever the global RNG state. `memory_seed` alone decides *which* examples the
replay memory retains and defaults to `0`, keeping that memory identical across
experiment seeds.

## How it is evaluated

`strategy.eval()` uses [`CLEvaluationPlugin`](skill_memory/evaluation/cl_evaluator.py)
and nothing else. At prediction time only `x` is available: each trained class
$c$ is scored by the verifier of the skill that owns it and

$$
\hat y_{\rm raw}(x)=\arg\max_c r_c(x),\qquad
\hat y_{\rm cal}(x)=\arg\max_c\,(a_c\,r_c(x)+b_c),
$$

where $(a_c, b_c)$ is a monotone Platt calibration fitted on the skill's
held-out calibration examples. Test labels are used only *after* prediction, to
count correct answers. Two accuracies are reported: `raw_mean_final_accuracy`
(**the primary diagnostic**) and `mean_final_accuracy` (calibrated). Classes with
no skill yet score a constant, so evaluating a stream that contains future
classes is safe. The leakage guards (`strict_protocol=True`) raise if a training
step touches a test-stream experience or evaluation touches a train-stream one.

## Diagnostics

Opt-in tools in [`skill_memory.diagnostics`](skill_memory/diagnostics/), never part
of `strategy.eval()`; each takes a required `diagnose=True` so they cannot run by
accident (`timing_report` needs a strategy built with `diagnose=True`).

| Function | Answers |
|---|---|
| `evaluate_class_oracle` | upper bound, routing every sample with its TRUE label |
| `evaluate_skill_memory(routing="probe")` | anonymous probe routing over the stored skills |
| `replay_provenance_report` | which historical data did every training call use? Lists any violation of the update-mode invariants |
| `timing_report` | where did the time go: decision probing / class training / refresh / evaluation |
| `audit_split_overlap`, `audit_strategy_leakage` | exact-content train/test overlap (a diagnostic, not a proof of no leakage) |

## Demos and reference results

All demos live in `skill_memory/demos/` and print the same report.

```bash
# Real data (download the dataset): the published CIFAR-100 configuration
python -m skill_memory.demos.demo_cifar100 --n-experiences 20 --max-experiences 3 \
    --update-mode replay --memory-per-class 50 --train-samples-per-class 50 \
    --class-train-epochs 10 --seed 3 --diagnose
python -m skill_memory.demos.demo_splitmnist --update-mode refresh --diagnose

# Offline, seconds: one-factor-at-a-time comparison of the update policies
python -m skill_memory.demos.demo_replay_ablation --seeds 0 1 2 --json out.json

# Where can time be saved? Measures stage 1 of the decision (see below)
python -m skill_memory.demos.demo_stage1_timing --model resnet --n-skills 25 50 100
```

`--diagnose` adds the oracle/probe accuracies, per-stage timing and the replay
audit (`violations: none` is expected). To compare policies, vary **only**
`--update-mode` (and `--replay-samples-per-class`); data, initial weights and
seeds then stay identical.

### Finding the time-optimal configuration

`timing_report` splits a run into decision probing, class training, refresh and
evaluation. Class training and refresh dominate with `refresh`; the decision
cost is the only part that grows with the number of *stored skills* (it probes
every skill for every new class), and `batch_stage1=True` can reduce it by
batching those probes with `torch.vmap`. Whether it helps depends on the
hardware, so measure it:

```bash
python -m skill_memory.demos.demo_stage1_timing --model resnet --device cuda \
    --n-skills 25 50 100 --chunk-size 8 16 32
```

The table lists the speedup per skill count and chunk size, and the last block
recommends `batch_stage1=True, stage1_chunk_size=N` only where the speedup
reaches 1.1x. On CPU it usually does not, which is why the default is `False`.
Other levers, all result-preserving: lower `max_safety_candidates` (default 5,
approximate), `replay_samples_per_class` (fewer replayed examples), and
`eval_chunk_size`.

## Layout

```
skill_memory/
  strategy.py              SkillMemoryStrategy (Avalanche strategy; wires the two plugins)
  cl/
    skill_memory_plugin.py SkillMemoryPlugin: REUSE/SCRATCH, training, replay memory, refresh
    decision.py            functional probing: stage 1 (fit) + stage 2 (safety) + selection
    training.py            binary class training, skill refresh, provenance, seeding
    replay.py              update_mode policy, replay memory, deterministic selection
    skill_registry.py      SkillMemory (stored states), class -> skill bookkeeping
  evaluation/cl_evaluator.py   CLEvaluationPlugin: stored-skill scoring + Platt calibration
  diagnostics/             opt-in oracle / probe / audit / timing tools
  utils/                   probing.py (functional_call, caches, batching), protocol_guard.py
  demos/                   cifar100, splitmnist, replay_ablation, stage1_timing (+ _common)
  tests/                   offline tests (no dataset download)
docs/MATHEMATICS.md        formulas, cost model, reproducibility, compatibility rules
```

Contributor notes on invariants are in [`skill_memory/README.md`](skill_memory/README.md).

## Development

```bash
pytest -q          # offline; ~20 s
ruff check . && ruff format --check .
```

`tests/test_golden_regression.py` pins the engine to the validated reference
(provenance, decisions, stored weights, accuracies for all update policies): if
it fails, a change altered *what is computed*, not just how it is organised.
`pytest.yml` runs lint, the suite on Python 3.10-3.12 plus an older-torch job
(torch 2.3, installed together with the package so the resolver sees all pins),
and the offline ablation demo.

## License

MIT, see `LICENSE`.
