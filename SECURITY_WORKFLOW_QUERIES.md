# Security Workflow Test Queries

Paste-ready chat prompts for the multi-step security workflows.

- **Comparative sweep** — up to 5 independent attack/defense combinations against
  the same unmodified model. Report section **8.3**.
- **Cumulative hardening loop** — up to 5 rounds of attack → adversarially
  retrain → attack the new model. Report section **9**.

`N` counts combinations (sweep) or rounds (loop). `N` is user-supplied, defaults
to **2**, hard cap **5**. It is an upper bound: combinations are never duplicated
or invented to reach it.

---

## Framework support

Pick the project that matches before running, or the request is refused up front.

| Framework | Attacks | Retraining methods |
|---|---|---|
| `pytorch` | boundary_attack, hopskipjump, zoo, **fgsm**, **pgd**, **cw** | fgsm, pgd |
| `sklearn` | boundary_attack, hopskipjump, zoo | boundary_attack, hopskipjump |
| `onnx` | boundary_attack, hopskipjump, zoo | boundary_attack, hopskipjump |

Bold entries are the ones that only exist on PyTorch. The attack you **measure**
and the method you **retrain with** are independent — you can measure `fgsm` and
retrain with `pgd`.

---

## 1. Comparative sweep

### 1.1 Three combinations — PyTorch only

```
compare fgsm, pgd and cw with feature_squeezing and generate a report
```

Expect **3 combinations**. `cw` requires a PyTorch model.

### 1.2 Two combinations — works on any framework

```
run 3 combinations of boundary attack and zoo with feature_squeezing
```

Expect **2 combinations** run, `requested_steps: 3`. Only the pairs you named
are built; the third is never a duplicate of one of these.

### 1.3 Cap at five

```
compare boundary attack, hopskipjump, zoo, fgsm and pgd with feature_squeezing
```

Expect **5 combinations**, `max_steps: 5`. Six named still yields five.

### 1.4 Sweep with explicit hyperparameters

```
compare fgsm, pgd and cw with feature_squeezing using eps=0.1 max_iter=10 and generate a report
```

### 1.5 Sweep against the undefended model

```
compare boundary attack and zoo against my model and generate a report
```

Each attack is measured with **no defense** applied.

### 1.6 Sweep over defenses only

```
compare feature_squeezing, clip_values and gaussian_augmentation on my model
```

Each defense applied with **no attack** — reports clean accuracy only.

---

## 2. Cumulative hardening loop

The trigger is **a hardening verb plus round wording**. "rounds" on its own is
not enough; it routes to the sweep instead.

### 2.1 Four explicit rounds

```
harden my model over 4 rounds using pgd and generate a report
```

Expect **4 rounds**, `requested_rounds: 4`. Registers **one** new model.

### 2.2 Default two rounds

```
iteratively harden my model and generate a report
```

Expect **2 rounds** — N unstated defaults to 2.

### 2.3 Three rounds against a named threat

```
harden over 3 rounds against fgsm using pgd
```

Expect **3 rounds**, all measuring `fgsm`. Robustness accumulates because each
round attacks the model the previous round produced.

### 2.4 Measure one attack, retrain with another

```
harden my model over 3 rounds against cw using pgd and generate a report
```

Measures `cw`, retrains with `pgd`.

### 2.5 Loop without a report

```
harden my model over 2 rounds using boundary_attack
```

### 2.6 Cap at five

```
harden my model over 5 rounds using pgd
```

Expect **5 rounds** — 5 attacks plus 5 retrains. Slowest run available.

### 2.7 Mixed-threat curriculum — a different attack each round

```
harden my model over 5 rounds against fgsm, pgd, hopskipjump and zoo with feature_squeezing using pgd
```

Rounds: **fgsm → pgd → hopskipjump → zoo → fgsm**. Four named threats across
five rounds, so the fifth cycles back to the first. The defense stays
`feature_squeezing` throughout, keeping it a controlled variable. PyTorch only,
because `fgsm` and `pgd` are unavailable on sklearn.

### 2.8 Mixed-threat across frameworks

```
harden over 4 rounds with boundary attack and hopskipjump using hopskipjump
```

Rounds: **boundary_attack → hopskipjump → boundary_attack → hopskipjump**.
Works on any framework.

### 2.9 Measure one threat, retrain with another

```
harden over 3 rounds against fgsm using pgd
```

Rounds measure `fgsm`; retraining uses `pgd`. "using" names the retraining
method, not a second threat to measure.

### 2.10 Longer mixed curriculum, capped

```
harden my model over 99 rounds against fgsm, pgd, cw, zoo, hopskipjump
```

Expect **5 rounds** — the cap, regardless of the requested 99.

---

### Reading a mixed-threat run

When each round measures a different attack, the first and final numbers are
**not comparable** — a drop from round 1 to round 5 may just be that the last
round used a harsher attack. The report detects this, sets `mixed_threat: true`,
suppresses the single first→final delta, and instead prints a per-attack table
comparing only the rounds that used the same attack. In 2.7 above that means
`fgsm` round 1 → round 5 is a genuine robustness comparison, while
`hopskipjump` round 3 has no later counterpart and is reported on its own.

Read it as a curriculum, not a leaderboard: it tells you how the hardened model
holds up across several threats, not that it got monotonically better.

---

## 3. Regression checks — behaviour must be unchanged

### 3.1 Single pair, the original failing prompt

```
run fgsm attack with feature_squeezing defense filter using eps=0.1 max_iter=10 and generate a report
```

Must produce **one** combination, no `mode` key in the result, and sections
8.1/8.2 exactly as before. This is the prompt that originally failed.

### 3.2 Single retrain, unchanged

```
harden my model using pgd
```

Must produce **one** new model and **no** rounds.

### 3.3 C&W with clip values

```
run cw attack on my model with clip_values defense filter using eps=0.4 max_iter=15 and generate a report
```

---

## 4. Refusals

Each of these should refuse **before** running anything, naming valid options.

### 4.1 Unsupported attack for the framework

```
compare fgsm, pgd and cw with feature_squeezing
```

On a **sklearn** project. Should refuse and list Boundary Attack, HopSkipJump
and ZOO.

### 4.2 Unsupported retraining method

```
harden my model over 3 rounds using pgd
```

On a **sklearn** project. `pgd` retraining is PyTorch-only.

### 4.3 Multi-step with nothing named

```
run 3 attack and defense combinations
```

Refuses asking you to name at least one attack and one defense, rather than
silently running a single default pair three times.

### 4.4 Spatially impossible defense

```
compare boundary attack with spatial smoothing and generate a report
```

On a **tabular** dataset. Refuses: spatial smoothing needs image or video data.

### 4.5 Incompatible model and dataset

Ask for any attack on a project whose model and dataset feature counts disagree.

---
 ### 4.6 

harden my model over 5 rounds against fgsm, pgd, hopskipjump and zoo   with feature_squeezing using pgd


## 5. What to check in the report

### Section 8.3 — Attack/Defense Comparison

| Column | Meaning |
|---|---|
| Step | 1-based position |
| Clean Acc | accuracy with no defense, no attack |
| Defended Acc | accuracy with the defense, no attack |
| Adv Acc | accuracy after the attack, through the defense |
| Drop % | defended → adversarial, clamped to 0–100% |
| Adv P / R / F1 | macro precision, recall, F1 under attack |
| Masking | `suspected` when a decision-based cross-check contradicts the attack |

Header states the combinations are **independent measurements against the same
model** and do **not** stack — the defense is an input preprocessor that is never
written back.

### Section 9 — Adversarial Hardening Loop

| Column | Meaning |
|---|---|
| Round | 1-based position |
| Clean Acc | accuracy of the model this round received |
| Adv Acc | accuracy after this round's attack |
| Drop % | clamped 0–100% |
| Adv P / R / F1 | macro precision, recall, F1 under attack |
| Masking | `suspected` when the cross-check disagrees |

Also reports first-round → final-round adversarial accuracy with a signed delta,
the final model id, and a note that intermediate rounds were not registered as
separate assets.

### Reading the tradeoff

Clean accuracy normally **falls** as adversarial accuracy **rises**. Both columns
are shown side by side precisely so a robustness gain can't be read as a free win.

---

## 6. Persistence expectations

| Run | New models in DB |
|---|---|
| Sweep (any N) | **0** |
| Loop, N rounds | **1** — only the final hardened model |

Confirm with:

```sql
SELECT id, name, framework FROM models ORDER BY rowid DESC LIMIT 5;
```

Intermediate loop rounds retrain into a temp scratch directory and are
discarded, so a 5-round loop adds one model, not five.

---

## 7. Determinism

Every step and round scores the **same 30 rows** (seeded stratified subsample,
`n=30, seed=42`), so columns are directly comparable without extra work.

The flip side: 30 samples means each percentage moves in steps of about **3.3%**.
On a 5-row table that is coarse enough to mislead on close calls. Treat
differences under roughly 10 points as noise.
