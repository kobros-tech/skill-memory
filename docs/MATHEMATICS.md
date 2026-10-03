# Mathematics of Skill Memory

This document is the formal specification of what the code computes. Each
section names the module that implements it, and the tests that pin it down.
Nothing here is aspirational: if the code and this file disagree, that is a bug.

- [0. Notation](#0-notation)
- [1. Skills, ownership and the observed domain](#1-skills-ownership-and-the-observed-domain)
- [2. Reuse-or-scratch decision](#2-reuse-or-scratch-decision)
- [3. Per-class training objectives](#3-per-class-training-objectives)
- [4. Replay policy](#4-replay-policy)
- [5. Refreshing existing skills](#5-refreshing-existing-skills)
- [6. Cost model](#6-cost-model)
- [7. Evaluation and calibration](#7-evaluation-and-calibration)
- [8. Reproducibility](#8-reproducibility)
- [9. What an experiment can and cannot attribute](#9-what-an-experiment-can-and-cannot-attribute)

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $t = 1,\dots,T$ | experience index (train stream) |
| $\mathcal C_t$ | classes first presented in experience $t$ |
| $\mathcal D_t = \bigcup_{u\le t}\mathcal C_u$ | **observed domain** after experience $t$ |
| $\theta_s$ | frozen weights of skill $s$; $S$ = number of stored skills |
| $z^{(s)}_c(x)$ | raw logit of class $c$ (output column $c$) of skill $s$ at input $x$ |
| $\operatorname{own}(s)\subseteq\mathcal D_t$ | classes owned by skill $s$ (see §1) |
| $s(c)$ | the unique skill owning class $c$ |
| $R_c$ | retained memory of class $c$: at most $m$ frozen examples (`eval_memory_per_class`) |
| $K$ | `cl_replay_per_class` |
| $n$ | `skill_train_samples_per_class` (per-class cap on *current* data) |
| $f$ | `validation_fraction` |
| $\sigma(u)=1/(1+e^{-u})$ | logistic function |
| $\mathbb 1[\cdot]$ | indicator |

---

## 1. Skills, ownership and the observed domain

Ownership is a function $s:\mathcal D_T\to\{1,\dots,S\}$ that is **write-once**:
once $s(c)$ is recorded it is never changed (`ExperienceClassMap`,
`cl/skill_registry.py`). Hence $\operatorname{own}(s)=\{c: s(c)=s\}$ partitions
the observed domain,

$$
\mathcal D_t=\biguplus_{s=1}^{S}\operatorname{own}(s).
$$

Skill weights $\theta_s$ may change only through the two training paths of
§3 and §5; probing (§2) and evaluation (§7) apply $\theta_s$ functionally
(`torch.func.functional_call`) and never modify a model or a stored state.

---

## 2. Reuse-or-scratch decision

*Implemented in `cl/decision.py`. Tests: `test_decision.py`,
`test_safety_optimizations.py`, `test_probing_cache.py`, `test_stage1_batching.py`.*

When a new class $c$ appears, a probe batch of $N$ examples $(x_i,y_i=c)$ of
that class is drawn with a fixed seed.

### 2.1 Stage 1 — fit of the new class (every stored skill)

Skill $s$ is expanded by Avalanche's `IncrementalClassifier` just far enough to
own a column for $c$, then

$$
\text{new\_score}_s=\frac1N\sum_{i=1}^N\operatorname{softmax}\!\big(f_{\theta_s}(x_i)\big)_{c},\qquad
\text{new\_acc}_s=\frac1N\sum_{i=1}^N\mathbb 1\Big[\arg\max_k f_{\theta_s}(x_i)_k=c\Big].
$$

### 2.2 Stage 2 — safety on the skill's own old classes

For a candidate $s$ and each $c'\in\operatorname{own}(s)$ with probe batch
$(x'_j,c')$, define the per-class accuracy

$$
a_s(c')=\frac1N\sum_{j=1}^N\mathbb 1\Big[\arg\max_k f_{\theta_s}(x'_j)_k=c'\Big],
\qquad
\text{old\_acc}_s=\min_{c'\in\operatorname{own}(s)}a_s(c').
$$

The **worst** class is used, not the mean, so one forgotten class cannot hide
behind the others. With $\text{chance}_s=1/W_s$ ($W_s$ = classifier width of
$s$) and forgetting margin $\delta$, skill $s$ is **safe** iff

$$
\text{old\_acc}_s>\text{chance}_s+\delta .
$$

**Exact short-circuit.** Because $\text{old\_acc}_s$ is a minimum, the event
"$s$ is unsafe" is the event "$\exists\,c'$: $a_s(c')\le\text{chance}_s+\delta$".
Evaluation of $s$'s old classes may therefore stop at the first $c'$ that
satisfies it: the verdict (unsafe) is identical to the exhaustive one. Only the
recorded list of per-class accuracies is partial (`safety_complete=False`).
This is an *exact* optimisation.

**Approximate cap.** Stage 2 is run only for the $M$ best candidates ordered
by $(\text{new\_score}_s,\text{new\_acc}_s)$, with $M$ = `max_safety_candidates`
(default $5$; $M=\infty$ is exhaustive). This *can* change the outcome if the
$(M{+}1)$-th candidate would have been the only safe one, so it is documented
as approximate; the cost of stage 2 is bounded by
$\mathcal O\!\big(M\cdot\max_s|\operatorname{own}(s)|\big)$ probe evaluations
instead of $\mathcal O(\sum_s|\operatorname{own}(s)|)$.

### 2.3 Candidate selection

Among safe skills, let $\mathcal R$ be the safe set. For a statistic
$q\in\{\text{new\_score},\text{new\_acc}\}$ sort $\mathcal R$ by $q$ descending,
$q_{(1)}\ge\dots\ge q_{(r)}$, take the largest consecutive gap

$$
j^\star=\arg\max_{1\le j<r}\big(q_{(j)}-q_{(j+1)}\big),
$$

and keep the top $j^\star$ skills that also exceed a floor $\phi_q$
($\phi_{\text{score}}=\tau$, default $0.9$; $\phi_{\text{acc}}=\max_{s\in\mathcal R}\text{chance}_s$).
If the gap is $0$ the set is empty; if $r=1$ the single skill is kept iff it
exceeds the floor. With $\mathcal G_q$ the sets so obtained,

$$
\text{REUSE}\ \hat s=\arg\max_{s\in\mathcal G_{\text{score}}\cap\mathcal G_{\text{acc}}}(\text{new\_score}_s,\text{new\_acc}_s)
$$

and the decision is **SCRATCH** (train a new skill) when the intersection is
empty. Both rankings must therefore agree.

---

## 3. Per-class training objectives

*Implemented in `cl/training.py`. Tests: `test_cl_replay_modes.py`,
`test_replay_isolation.py`, `test_reproducibility.py`.*

Let $z_c(x)$ be the logit of the **global** class column $c$ (a skill's output
column for a class is the class id itself).

### 3.1 `multiclass`

Only samples of the target class are loaded; ordinary cross-entropy is used:
$\mathcal L(x,c)=-\log\operatorname{softmax}(z(x))_c$.

### 3.2 `binary_one_vs_rest` (YES/NO verifier)

With $y_c=\mathbb 1[y=c]$,

$$
\mathcal L_c(x,y)=-\,y_c\log\sigma\big(z_c(x)\big)-(1-y_c)\log\big(1-\sigma(z_c(x))\big),
$$

and **only column $c$** enters the loss.

### 3.3 Balanced sampling

The assembled training set (current + historical data, §4) contains $P$
positives and $N^-$ negatives. Each example is drawn **with replacement** with
weight

$$
w_i=\begin{cases}\dfrac1{2P}&\text{positive}\\[6pt]\dfrac1{2N^-}&\text{negative}\end{cases}
$$

so a draw is positive with probability exactly $\tfrac12$, **independently of
how much history is replayed** (replaying more negatives does not dilute the
positive signal). `num_samples` equals the dataset size.

### 3.4 Calibration hold-out

For each class $k$ present in the experience, with $n_k$ samples, a seeded
stratified split reserves

$$
v_k=\min\!\Big(\max\big(\lfloor f\,n_k\rfloor,\ \mathbb 1[f>0,\,n_k>1]\big),\ n_k-1\Big)
$$

examples for the hold-out and trains on at most $n$ of the remaining
$n_k-v_k$. Hold-out examples are used **only** to fit the Platt calibration
(§7). They are never trained on and never replayed; they are stored under the
metadata key `calibration_examples_by_class`. Training uses
$\min(n_k-v_k,\,n)$ examples of class $k$.

---

## 4. Replay policy

*Implemented in `cl/replay.py`. Tests: `test_replay_policy.py`,
`test_replay_isolation.py`.*

Let $\mathcal R_t=\{R_c: c\in\mathcal D_{t-1}\}$ be the retained memory **before**
experience $t$ (so it never contains the current classes). The historical
data $H_c$ replayed while training any class of experience $t$ is, per old
class $c\in\mathcal D_{t-1}$,

$$
H_c=\begin{cases}
\varnothing & \texttt{new\_class}\\
\operatorname{Sample}\big(R_c,\ \min(K,|R_c|)\big) & \texttt{small\_replay}\\
R_c & \texttt{replay}
\end{cases}
\qquad
|H_c|=\begin{cases}0\\\min(K,|R_c|)\\|R_c|\le m.\end{cases}
$$

`replay` therefore means *all currently retained history*, **not** all
historical training data, because $|R_c|\le m$ is bounded.
$\operatorname{Sample}$ is a draw without replacement from
`torch.Generator().manual_seed(seed + c)`, i.e. a pure function of
$(\text{seed},c,|R_c|)$ and independent of the global RNG.

The data sources of one class-training call are exactly

$$
\mathcal T=\underbrace{\textstyle\bigcup_{k\in\mathcal C_t}\text{current}_k}_{\text{current}}\ \cup\
\underbrace{\textstyle\bigcup_{c\in\mathcal D_{t-1}}H_c}_{\text{retained}}\ \cup\
\underbrace{\mathcal O}_{\text{offline pool}},
$$

where the **offline pool** $\mathcal O$ is an *oracle* source for ablations
only: it requires `allow_offline_negative_pool=True`, obeys the same cap as
$H_c$, is rejected with `new_class`, and a class already supplied by $H_c$ is
never supplied twice. Every call records $|{\rm current}|,|H|,|\mathcal O|$
per class (`TrainingProvenance`) and the invariants

$$
\texttt{new\_class}:\ \textstyle\sum_c|H_c|+|\mathcal O|=0,\qquad
\texttt{small\_replay}:\ |H_c|\le K,\qquad
\texttt{replay}:\ |H_c|\le|R_c|
$$

are checked by `diagnostics.replay_provenance_report`, which lists any
violation.

---

## 5. Refreshing existing skills

*Implemented in `SkillMemoryPlugin._refresh_existing_skills` and
`train_skill_on_domain`. Switch: `refresh_existing_skills` (default off),
**independent of the replay mode**.*

After experience $t$, each pre-existing skill $s$ (created before $t$, owning
$O=\operatorname{own}(s)$) is retrained once on the enlarged domain
$\mathcal D_t$ with a multi-label binary loss over its owned columns,

$$
\mathcal L_s(x,y)=\frac1{|O|}\sum_{c\in O}\Big[-\mathbb 1[y{=}c]\log\sigma(z_c(x))-\mathbb 1[y{\ne}c]\log\big(1-\sigma(z_c(x))\big)\Big].
$$

A positive example of owned class $c$ (count $n_c$ in the assembled set) has
weight $\tfrac1{2n_c}$ and each of the $N^-$ non-owned examples has weight
$\tfrac1{2N^-}$. The probability that a draw is a positive is therefore

$$
\frac{|O|/2}{|O|/2+1/2}=\frac{|O|}{|O|+1},\qquad\text{negative: }\frac1{|O|+1},
$$

and for $|O|=1$ this is the balanced $\tfrac12:\tfrac12$ of §3.3. Historical
examples enter under the replay policy of §4 (the refresh is rejected with
`new_class`, because no positives of old classes would exist). Skills created
in experience $t$ already saw $\mathcal D_t$ and are skipped; a skill whose
`domain_classes` already equals $\mathcal D_t$ is skipped too.

---

## 6. Cost model

One training call on an assembled set of $N$ examples, $E$ epochs and batch
size $B$ performs

$$
\text{steps}(N)=E\Big\lceil \frac NB\Big\rceil
$$

optimiser steps (the sampler draws exactly $N$ examples per epoch). Let
$u_t=|\mathcal C_t|$, and let $N^{\rm cur}_t$ be the current-experience
examples used.

**Class training**, for each of the $u_t$ new classes:

$$
N^{\rm cls}_t=N^{\rm cur}_t+\sum_{c\in\mathcal D_{t-1}}|H_c|,\qquad
\text{steps}^{\rm cls}_t=u_t\cdot E\Big\lceil N^{\rm cls}_t/B\Big\rceil .
$$

With `replay` $\sum_c|H_c|\approx m\,|\mathcal D_{t-1}|$ grows linearly with
the number of observed classes; with `small_replay` it grows as
$K\,|\mathcal D_{t-1}|$; with `new_class` it is $0$.

**Refresh**, once per pre-existing skill $S_{t-1}$:

$$
\text{steps}^{\rm ref}_t=S_{t-1}\cdot E\Big\lceil N^{\rm ref}_t/B\Big\rceil,\qquad
N^{\rm ref}_t=N^{\rm cur}_t+\sum_{c\in\mathcal D_{t-1}}|H_c| .
$$

Over a run, the refresh adds $\sum_t S_{t-1}\,E\lceil N^{\rm ref}_t/B\rceil$,
which is **quadratic** in the number of experiences when skills accumulate
(each of $\mathcal O(t)$ skills is retrained at every one of $T$
experiences). This, not the replay quantity, is what dominates run time when
the refresh is on, and is why it has its own switch and its own timing bucket
(`skill_memory_domain_refresh`). `replay_provenance_report` returns
$\text{steps}^{\rm cls}$ and $\text{steps}^{\rm ref}$ separately so the claim
can be verified on any run.

---

## 7. Evaluation and calibration

*Implemented in `evaluation/cl_evaluator.py`. Tests: `test_cl_evaluator.py`,
`test_leakage.py`.*

At prediction time only $x$ is available. Each trained class $c$ is scored by
the verifier of its owning skill,

$$
r_c(x)=z^{(s(c))}_c(x),\qquad
\tilde r_c(x)=a_c\,r_c(x)+b_c,
$$

and classes with no skill yet receive the constant $r_c=\tilde r_c=u$
(`unseen_logit`, default $-20$). The score matrix has width
$\max(\text{trained width},\ 1+\max\text{ declared class id of the stream})$,
read from stream **metadata** only, so Avalanche's loss metric can index
targets of classes that have no skill yet (those classes score $0$ accuracy
rather than raising).

**Platt calibration.** For each class $c$, with hold-out scores $r_i$ and
labels $t_i=\mathbb 1[y_i=c]$,

$$
(a_c,b_c)=\arg\min_{a,b}\ \frac1n\sum_i\operatorname{BCE}\big(a\,r_i+b,\ t_i\big)+10^{-4}\big(a^2+b^2\big),
$$

fitted by L-BFGS from $(1,0)$ and then clamped to $a_c\ge10^{-3}$ so
$\tilde r_c$ is monotone in $r_c$. If the hold-out has fewer than two examples
or lacks positives or negatives, $(a_c,b_c)=(1,0)$.

**Predictions and metrics.**

$$
\hat y_{\rm raw}(x)=\arg\max_c r_c(x),\qquad
\hat y_{\rm cal}(x)=\arg\max_c\tilde r_c(x).
$$

`raw_mean_final_accuracy` uses $\hat y_{\rm raw}$ and is the **primary
diagnostic**; `mean_final_accuracy` uses $\hat y_{\rm cal}$. With
$A_t(c)$ the accuracy on class $c$ after experience $t$ and $t_c$ the
experience that introduced $c$, the demo reports

$$
\text{forgetting}=\frac1{|\mathcal D_T|}\sum_{c\in\mathcal D_T}\Big(\max_{t_c\le t\le T}A_t(c)-A_T(c)\Big).
$$

`eval_chunk_size` $G$ only chooses how many skills share one `torch.vmap` call;
it changes memory and speed, never $r_c$ (tested for $G\in\{1,2,8,64\}$).

---

## 8. Reproducibility

Every stochastic ingredient is a pure function of explicit integers:

| Ingredient | Seed |
|---|---|
| hold-out split of class $c$ | `validation_seed + c` |
| replay selection for class $c$ | `validation_seed + 1543 + c` |
| mini-batch order, balanced re-sampling **and dropout** | $\operatorname{derive}(\texttt{training\_seed},\,t,\,c)$ for classes, $\operatorname{derive}(\texttt{training\_seed},\,t,\,s,\,7919)$ for refreshes |

with $\operatorname{derive}(b,p_1,\dots,p_k)$ the polynomial hash in
`cl/training.py`. The optimisation loop runs inside `torch.random.fork_rng`
seeded by this value, so (i) the result does not depend on the global RNG
stream and (ii) the global stream is restored afterwards. The test
`test_reproducibility.py` asserts bit-identical stored skills across runs whose
global RNG was advanced differently, for all three modes and with refresh on.

---

## 9. What an experiment can and cannot attribute

Four **independent factors** act on a run:

| Factor | Switch | Differs between rows of |
|---|---|---|
| objective | `class_train_mode` | multiclass vs. one-vs-rest |
| replay quantity | `cl_update_mode` ($H_c$, §4) | `new_class` / `small_replay` / `replay` |
| refresh | `refresh_existing_skills` (§5) | off / on |
| evaluation | raw vs. calibrated (§7) | reported side by side |

A fair comparison changes **one** factor and holds the others, the data, the
initial weights and all seeds fixed;
`python -m skill_memory.demos.demo_replay_ablation` does exactly this and
prints, per row, the historical examples consumed and the steps spent on class
training vs. refresh, so a difference can be traced to its cause. Differences
against runs made *before* the CL evaluator replaced the independent ML
evaluator change the measured system as well as the training and are **not**
attributable to replay.
