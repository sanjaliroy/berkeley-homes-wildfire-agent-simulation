# Evaluation Results — Claude Family

*Simulation models: Claude Sonnet 4.6 (base) / Claude Haiku 4.5 (budget). Judge: `openai/gpt-5.4`, temperature 0. Based on 5 replicate runs per condition, 320 judge calls.*

## Key takeaway

**The full cognitive architecture — memory plus reflection — adds little to simulation quality on this task. A rich seed narrative already appears to determine how each resident behaves.**

Every condition scored within a hair of the others. Baseline (the full system) scored **4.48** out of 5 overall. Removing reflection changed nothing (**4.48**, a gap of exactly 0.00). Removing both memory and reflection dropped it only to **4.41** (−0.07). The cheap model (Haiku instead of Sonnet) scored **4.44** (−0.04) at **63% lower cost**. When we correct for the fact that we tested three variants against Baseline at once, **none of these gaps is larger than the run-to-run randomness of the simulation** — statistically, they are all indistinguishable from Baseline.

In plain terms: turning off the "cognition" machinery did not measurably worsen the residents' behaviour, and the budget model was nearly as good for a third of the price.

One important caveat before the details: the judge only ever used three scores (4.0, 4.5, 5.0) and never scored anything below 4.0. So these results show the conditions are **hard to tell apart**, but they cannot by themselves prove the architecture adds *nothing* — the measuring instrument may be too blunt to see small differences. See [Limitations](#limitations).

---

## How to read this document

- **The four conditions.** *Baseline* is the full system (memory + reflection). *Ablation 1* removes reflection. *Ablation 2* removes both memory and reflection. *Budget* is the full system but run on the cheaper Haiku model.
- **The three scores.** An LLM judge reads each resident's complete 60-day trajectory and rates it 1–5 on **Behavioral Plausibility (BP)** — is this behaviour believable for this person; **Persona Consistency (PC)** — does it sound like this specific resident; **Intervention Responsiveness (IR)** — did they engage with what actually happened to them. **Overall** is the average of the three.
- **Replicates.** Each condition was run 5 times, because the model's answers vary a little each time. We average across the 5 runs so no single lucky or unlucky run drives the result.

---

## 6.2.1 Overall Results

| Condition | Behavioral Plausibility | Persona Consistency | Intervention Responsiveness | Mean |
|---|---|---|---|---|
| **Baseline** | 4.56 | 4.42 | 4.46 | **4.48** |
| Ablation 1 (No Reflection) | 4.56 (+0.00) | 4.42 (+0.00) | 4.46 (−0.00) | 4.48 (+0.00) |
| Ablation 2 (No Memory or Reflection) | 4.56 (+0.00) | 4.30 (−0.12) | 4.38 (−0.08) | 4.41 (−0.07) |
| Budget | 4.54 (−0.02) | 4.42 (+0.00) | 4.36 (−0.10) | 4.44 (−0.04) |

*Mean across 5 agents × 5 runs (25 judge calls per condition). Deltas vs Baseline in brackets.*

**What it says:** all four conditions land between 4.41 and 4.48. Removing reflection (Ablation 1) makes no difference at all. Removing memory too (Ablation 2) costs a little, and only on Persona Consistency (−0.12) and Intervention Responsiveness (−0.08) — the two scores you'd expect memory to help. Behavioral Plausibility never moves.

---

## 6.2.3 Model Comparison (Baseline vs Budget)

| Condition | Behavioral Plausibility | Persona Consistency | Intervention Responsiveness | Mean |
|---|---|---|---|---|
| Baseline (Claude Sonnet 4.6) | 4.56 | 4.42 | 4.46 | 4.48 |
| Budget (Claude Haiku 4.5) | 4.54 | 4.42 | 4.36 | 4.44 |

*Same scores, comparing the expensive and cheap models directly.*

**What it says:** the cheaper Haiku model matches Sonnet on two of three dimensions. The only visible drop is Intervention Responsiveness (4.46 → 4.36) — the budget model engaged slightly less specifically with each event.

---

## 6.2.4 Cost and Latency

| Condition | Quality (overall, 1–5) | Cost / run (USD) | Latency / run (sec) |
|---|---|---|---|
| Baseline (Sonnet 4.6) | 4.48 | $0.59 | 397 |
| Ablation 1 (No Reflection, Sonnet) | 4.48 | $0.36 | 217 |
| Ablation 2 (No Memory or Reflection, Sonnet) | 4.41 | $0.25 | 201 |
| Budget (Haiku 4.5) | 4.44 | $0.22 | 230 |

*Cost and time cover the simulation's decision and reflection calls only, averaged over 5 runs.*

**What it says:** this is where the differences are real. Turning off reflection cuts cost by ~40% ($0.59 → $0.36) and time nearly in half, for the same quality score. The budget model is cheapest of all at $0.22 — **63% less than Baseline** — while scoring 4.44. In short, the expensive parts of the system (reflection, the larger model) cost a lot and barely moved the quality score.

---

## Simulation Variance — how stable is each condition?

| Condition | Behavioral Plausibility | Persona Consistency | Intervention Responsiveness | Mean |
|---|---|---|---|---|
| Baseline | 4.56 ± 0.024 | 4.42 ± 0.049 | 4.46 ± 0.051 | 4.48 ± 0.020 |
| Ablation 1 (No Reflection) | 4.56 ± 0.024 | 4.42 ± 0.073 | 4.46 ± 0.024 | 4.48 ± 0.036 |
| Ablation 2 (No Memory or Reflection) | 4.56 ± 0.040 | 4.30 ± 0.000 | 4.38 ± 0.020 | 4.41 ± 0.017 |
| Budget | 4.54 ± 0.024 | 4.42 ± 0.020 | 4.36 ± 0.024 | 4.44 ± 0.012 |

*Mean ± SE across the 5 runs of each condition.*

**What "± SE" means:** SE (standard error) is how tightly the 5 runs cluster around their average — a small ± means the runs agreed closely, so the average is trustworthy. Here every ± is tiny (around 0.02–0.07 on a 5-point scale), so each condition's score is well pinned down. This matters because the gaps between conditions (up to 0.07) are about the same size as this run-to-run wobble — which is the first hint that the gaps aren't meaningful.

---

## Judge Variance — how stable is the judge?

| Condition | Behavioral Plausibility | Persona Consistency | Intervention Responsiveness | Mean |
|---|---|---|---|---|
| Baseline | 4.56 ± 0.024 | 4.52 ± 0.037 | 4.44 ± 0.051 | 4.51 ± 0.036 |
| Ablation 1 (No Reflection) | 4.58 ± 0.020 | 4.52 ± 0.020 | 4.46 ± 0.024 | 4.52 ± 0.017 |
| Ablation 2 (No Memory or Reflection) | 4.56 ± 0.024 | 4.38 ± 0.066 | 4.40 ± 0.032 | 4.45 ± 0.027 |
| Budget | 4.50 ± 0.000 | 4.26 ± 0.024 | 4.38 ± 0.020 | 4.38 ± 0.008 |

*We took one run from each condition and had the judge score the exact same trajectory 5 times. Mean ± SE across those 5 re-scorings.*

**What it says:** even scoring the *identical* text, the judge's answer wobbles a little each time (the ± values here). This is a check on the judge itself. The takeaway: the judge is fairly consistent, but its own wobble is real and roughly the same size as the differences between conditions — set up in the next table.

---

## Variance Comparison — is the wobble from the simulation or the judge?

| Condition | Simulation SE (5 runs) | Judge SE (5 re-scores) | Judge / Sim ratio |
|---|---|---|---|
| Baseline | 0.020 | 0.036 | 1.78 |
| Ablation 1 (No Reflection) | 0.036 | 0.017 | 0.47 |
| Ablation 2 (No Memory or Reflection) | 0.017 | 0.027 | 1.59 |
| Budget | 0.012 | 0.008 | 0.65 |

*Two sources of wobble side by side. Ratio under 1 = most wobble is real behaviour differences between runs; ratio over 1 = the judge's own noise dominates.*

**What it says:** the two sources of noise are roughly equal in size (ratios near 1). For Baseline and Ablation 2 the judge is actually the *bigger* source of wobble (ratio 1.78 and 1.59). That's a warning sign: when the judge's noise is as large as the thing you're trying to measure, small real differences are easily buried.

---

## Is any difference statistically real? (CI build-up ①–③)

These three tables show, step by step, whether the small gaps in Table 6.2.1 are real effects or just noise. Nothing here is hidden — you could redo the arithmetic by hand.

### ① The raw inputs — each run's overall score

| Condition | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 |
|---|---|---|---|---|---|
| Baseline | 4.433 | 4.533 | 4.500 | 4.433 | 4.500 |
| Ablation 1 (No Reflection) | 4.533 | 4.567 | 4.433 | 4.367 | 4.500 |
| Ablation 2 (No Memory or Reflection) | 4.400 | 4.433 | 4.367 | 4.467 | 4.400 |
| Budget | 4.467 | 4.467 | 4.433 | 4.400 | 4.433 |

*The overall score for each of the 5 runs. Everything below is computed from these 20 numbers.*

### ② Summary per condition

| Condition | Runs | Mean | SD | SE |
|---|---|---|---|---|
| Baseline | 5 | 4.480 | 0.045 | 0.020 |
| Ablation 1 (No Reflection) | 5 | 4.480 | 0.080 | 0.036 |
| Ablation 2 (No Memory or Reflection) | 5 | 4.413 | 0.038 | 0.017 |
| Budget | 5 | 4.440 | 0.028 | 0.012 |

*SD = how spread out the 5 runs are; SE = SD ÷ √5, the precision of the average.*

### ③ Each gap vs Baseline, with confidence intervals

| Comparison | Gap (Δ) | 95% CI (on its own) | Real on its own? | Family-wise 95% CI | Real overall? |
|---|---|---|---|---|---|
| Ablation 1 − Baseline | +0.000 | [−0.100, +0.100] | no | [−0.085, +0.085] | no |
| Ablation 2 − Baseline | −0.067 | [−0.127, −0.006] | **yes** | [−0.151, +0.018] | **no** |
| Budget − Baseline | −0.040 | [−0.096, +0.016] | no | [−0.125, +0.045] | no |

*A "confidence interval" (CI) is the range the true gap could plausibly be. If the range includes 0, the gap could really be nothing.*

**What it says:** looked at one at a time, the Ablation 2 gap (−0.067) is just big enough to look real — its interval `[−0.127, −0.006]` sits below zero. But once you correct for having made three comparisons at once (see method note below), that same gap's interval becomes `[−0.151, +0.018]`, which now includes 0. **So after the honest correction, no condition differs significantly from Baseline.**

---

## A note on the methods (in plain language)

- **Why 5 runs and not 25 scores?** Each run scores 5 residents, but those 5 aren't independent — they share the same simulation. So we first average each run into a single number, then treat the 5 *runs* as our data. Counting all 25 resident-scores as separate would fake a bigger sample and make weak results look strong.

- **Welch's t-test — used to compare two conditions.** A t-test asks: "is the gap between two averages bigger than their wobble?" The plain version assumes both conditions wobble by the same amount. Ours don't — scores bunch up near the 5.0 ceiling by different amounts. **Welch** is the version that allows the two conditions to have different wobble, so it's the safer, more honest default. We used it for each ablation-vs-Baseline gap.

- **Dunnett correction — used because we made three comparisons.** Every time you run a test at "95% confidence," there's a 5% chance of a false alarm. Run three tests and the chance of at least one false alarm climbs to ~14%. **Dunnett** is the standard fix when comparing several treatments against one shared control (here, three ablations vs Baseline): it widens each interval just enough to keep the *overall* confidence at 95%. That's why the "family-wise" column is the one to trust for any significance claim — and by that column, nothing is significant.

- **Why no confidence interval on the judge?** We re-scored identical trajectories only to *check* the judge is stable (it mostly is). That's a quality check, not a hypothesis test, so it doesn't get its own interval. The judge's noise from normal scoring is already baked into each run's score.

---

## Limitations

Two things keep these results modest, and both point the same way:

1. **The judge barely used its scale.** Across all 320 scorings it only ever gave 4.0, 4.5, or 5.0 — three values out of nine possible, and never anything below 4.0. A judge crowded up against the top of its scale can't show small differences even if they exist.

2. **The judge's noise is as big as the effects.** As the variance tables show, re-scoring the same text wobbles by about as much as the gaps between conditions.

Together these mean the safe conclusion is the narrow one: **the conditions are hard to tell apart, and the cheap setups (no reflection; budget model) are nearly as good as the full system for far less cost.** What we *cannot* firmly claim from these numbers alone is that the architecture adds *nothing* — a more discriminating evaluation (for example, comparing trajectories head-to-head, or checking them against the real interview transcripts) would be needed to separate "the architecture doesn't help" from "our judge couldn't see the help."
