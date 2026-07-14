# Evaluation Results: Claude Family

*Simulation models: Claude Sonnet 4.6 (base) and Claude Haiku 4.5 (budget). Judge: `openai/gpt-5.4`, temperature 0. Based on 5 replicate runs per condition and 320 judge calls.*

## Key takeaway

The full cognitive architecture (memory plus reflection) contributes little to simulation quality on this task. The results are consistent with the seed narrative alone largely determining each resident's behaviour.

Overall scores differ only marginally across conditions. Baseline (the full system) scored 4.48 out of 5. Removing reflection left the score unchanged at 4.48. Removing both memory and reflection reduced it to 4.41, a gap of 0.07. The budget model scored 4.44, a gap of 0.04, at 63% lower cost. After correcting for the fact that three conditions were each tested against Baseline, none of these gaps is larger than the run-to-run variation of the simulation, so all three are statistically indistinguishable from Baseline.

One caveat qualifies these findings. The judge used only three scores (4.0, 4.5, and 5.0) and never scored below 4.0. The results therefore establish that the conditions are difficult to distinguish, but they cannot on their own establish that the architecture contributes nothing, because the measure may be too coarse to detect small differences. See [Limitations](#limitations).

## How to read this document

**The four conditions.** *Baseline* is the full system (memory and reflection). *Ablation 1* removes reflection. *Ablation 2* removes both memory and reflection. *Budget* is the full system run on the cheaper Haiku model.

**The three scores.** An LLM judge reads each resident's complete 60-day trajectory and rates it from 1 to 5 on three dimensions: Behavioral Plausibility (BP), whether the behaviour is believable for that person; Persona Consistency (PC), whether it reflects the specific resident; and Intervention Responsiveness (IR), whether the resident engaged with the events they experienced. Overall is the mean of the three.

**Replicates.** Each condition was run 5 times because model outputs vary between runs. Scores are averaged across the 5 runs so that no single run drives the result.

## 6.2.1 Overall Results

| Condition | BP | PC | IR | Mean |
|---|---|---|---|---|
| Baseline | 4.56 | 4.42 | 4.46 | 4.48 |
| Ablation 1 (No Reflection) | 4.56 (+0.00) | 4.42 (+0.00) | 4.46 (-0.00) | 4.48 (+0.00) |
| Ablation 2 (No Memory or Reflection) | 4.56 (+0.00) | 4.30 (-0.12) | 4.38 (-0.08) | 4.41 (-0.07) |
| Budget | 4.54 (-0.02) | 4.42 (+0.00) | 4.36 (-0.10) | 4.44 (-0.04) |

*Mean across 5 agents by 5 runs (25 judge calls per condition). Deltas versus Baseline in parentheses.*

All four conditions fall between 4.41 and 4.48. Removing reflection produces no change. Removing memory as well (Ablation 2) reduces only Persona Consistency and Intervention Responsiveness, the two dimensions memory would be expected to support, while Behavioral Plausibility is unaffected.

## 6.2.3 Model Comparison (Baseline vs Budget)

| Condition | BP | PC | IR | Mean |
|---|---|---|---|---|
| Baseline (Claude Sonnet 4.6) | 4.56 | 4.42 | 4.46 | 4.48 |
| Budget (Claude Haiku 4.5) | 4.54 | 4.42 | 4.36 | 4.44 |

*The same scores, comparing the expensive and cheap models directly.*

The cheaper Haiku model matches Sonnet on two of three dimensions. The only appreciable difference is on Intervention Responsiveness (4.46 against 4.36), indicating that the budget model engaged slightly less specifically with each event.

## 6.2.4 Cost and Latency

| Condition | Quality (overall, 1 to 5) | Cost / run (USD) | Latency / run (sec) |
|---|---|---|---|
| Baseline (Sonnet 4.6) | 4.48 | $0.59 | 397 |
| Ablation 1 (No Reflection, Sonnet) | 4.48 | $0.36 | 217 |
| Ablation 2 (No Memory or Reflection, Sonnet) | 4.41 | $0.25 | 201 |
| Budget (Haiku 4.5) | 4.44 | $0.22 | 230 |

*Cost and time cover the simulation's decision and reflection calls only, averaged over 5 runs.*

The differences in cost are substantial. Disabling reflection reduces cost by roughly 40% ($0.59 to $0.36) and nearly halves latency at equal quality. The budget model is the cheapest at $0.22, 63% below Baseline, while scoring 4.44. The most expensive components of the system, reflection and the larger model, add considerable cost for a negligible change in quality.

## Simulation Variance

| Condition | BP | PC | IR | Mean |
|---|---|---|---|---|
| Baseline | 4.56 ± 0.024 | 4.42 ± 0.049 | 4.46 ± 0.051 | 4.48 ± 0.020 |
| Ablation 1 (No Reflection) | 4.56 ± 0.024 | 4.42 ± 0.073 | 4.46 ± 0.024 | 4.48 ± 0.036 |
| Ablation 2 (No Memory or Reflection) | 4.56 ± 0.040 | 4.30 ± 0.000 | 4.38 ± 0.020 | 4.41 ± 0.017 |
| Budget | 4.54 ± 0.024 | 4.42 ± 0.020 | 4.36 ± 0.024 | 4.44 ± 0.012 |

*Mean ± standard error (SE) across the 5 runs of each condition.*

The standard error measures how tightly the 5 runs cluster around their average; a small value indicates close agreement and a reliable average. Every SE here is small (approximately 0.02 to 0.07 on a 5-point scale), so each condition's score is well determined. This is relevant because the gaps between conditions (up to 0.07) are comparable in size to this run-to-run variation, a first indication that the gaps may not be meaningful.

## Judge Variance

| Condition | BP | PC | IR | Mean |
|---|---|---|---|---|
| Baseline | 4.56 ± 0.024 | 4.52 ± 0.037 | 4.44 ± 0.051 | 4.51 ± 0.036 |
| Ablation 1 (No Reflection) | 4.58 ± 0.020 | 4.52 ± 0.020 | 4.46 ± 0.024 | 4.52 ± 0.017 |
| Ablation 2 (No Memory or Reflection) | 4.56 ± 0.024 | 4.38 ± 0.066 | 4.40 ± 0.032 | 4.45 ± 0.027 |
| Budget | 4.50 ± 0.000 | 4.26 ± 0.024 | 4.38 ± 0.020 | 4.38 ± 0.008 |

*One run from each condition was scored 5 times by the judge on the identical trajectory. Mean ± SE across those 5 scorings.*

Even when scoring identical text, the judge's output varies slightly between calls. This table isolates that variation. The judge is reasonably consistent, but its variation is real and comparable in size to the differences between conditions, which the following table quantifies.

## Variance Comparison: Simulation vs Judge

| Condition | Simulation SE (5 runs) | Judge SE (5 re-scores) | Judge / Sim ratio |
|---|---|---|---|
| Baseline | 0.020 | 0.036 | 1.78 |
| Ablation 1 (No Reflection) | 0.036 | 0.017 | 0.47 |
| Ablation 2 (No Memory or Reflection) | 0.017 | 0.027 | 1.59 |
| Budget | 0.012 | 0.008 | 0.65 |

*Two sources of variation compared per condition. A ratio below 1 indicates that most variation reflects genuine behavioural differences between runs; a ratio above 1 indicates that judge noise dominates.*

The two sources of variation are similar in magnitude. For Baseline and Ablation 2 the judge is the larger source (ratios of 1.78 and 1.59). When judge noise is as large as the quantity being measured, small genuine differences are difficult to detect.

## Statistical Significance (CI build-up)

The following three tables show, in sequence, whether the small gaps in Table 6.2.1 are genuine effects or noise. The calculation is fully transparent and can be reproduced by hand.

### 1. Raw inputs: each run's overall score

| Condition | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 |
|---|---|---|---|---|---|
| Baseline | 4.433 | 4.533 | 4.500 | 4.433 | 4.500 |
| Ablation 1 (No Reflection) | 4.533 | 4.567 | 4.433 | 4.367 | 4.500 |
| Ablation 2 (No Memory or Reflection) | 4.400 | 4.433 | 4.367 | 4.467 | 4.400 |
| Budget | 4.467 | 4.467 | 4.433 | 4.400 | 4.433 |

*The overall score for each of the 5 runs. All statistics below derive from these 20 values.*

### 2. Summary per condition

| Condition | Runs | Mean | SD | SE |
|---|---|---|---|---|
| Baseline | 5 | 4.480 | 0.045 | 0.020 |
| Ablation 1 (No Reflection) | 5 | 4.480 | 0.080 | 0.036 |
| Ablation 2 (No Memory or Reflection) | 5 | 4.413 | 0.038 | 0.017 |
| Budget | 5 | 4.440 | 0.028 | 0.012 |

*SD is the spread of the 5 runs; SE is SD divided by the square root of 5, the precision of the average.*

### 3. Each gap versus Baseline, with confidence intervals

| Comparison | Gap | 95% CI (individual) | Significant individually | Family-wise 95% CI | Significant overall |
|---|---|---|---|---|---|
| Ablation 1 minus Baseline | +0.000 | [-0.100, +0.100] | No | [-0.085, +0.085] | No |
| Ablation 2 minus Baseline | -0.067 | [-0.127, -0.006] | Yes | [-0.151, +0.018] | No |
| Budget minus Baseline | -0.040 | [-0.096, +0.016] | No | [-0.125, +0.045] | No |

*A confidence interval (CI) is the range within which the true gap plausibly lies. If the range includes 0, the gap is consistent with no difference.*

Considered individually, the Ablation 2 gap of 0.067 appears significant, since its interval [-0.127, -0.006] lies below zero. After correcting for the three simultaneous comparisons (see the method note below), the interval widens to [-0.151, +0.018], which includes 0. Once this correction is applied, no condition differs significantly from Baseline. Testing each dimension separately, or treating the 25 individual agent-scores as separate observations, yields the same result, so the overall mean remains the appropriate summary.

## Method notes

**Welch's t-test, used to compare two conditions.** A t-test asks whether the gap between two averages is larger than their variation. The standard version assumes both conditions vary by the same amount, which does not hold here because scores cluster near the 5.0 ceiling to differing degrees. Welch's version allows the two conditions to have different variation and is the appropriate default. It was applied to each ablation-versus-Baseline gap.

**Dunnett correction, used because three comparisons were made.** Each test at 95% confidence carries a 5% chance of a false positive. Across three tests, the chance of at least one false positive rises to approximately 14%. The Dunnett correction is the standard adjustment when several treatments are compared against a single control, here three ablations against Baseline. It widens each interval so that the combined confidence across all three comparisons remains 95%. The family-wise column is therefore the one to use for any claim of significance, and by that column no difference is significant.

## Limitations

Two factors constrain these results, and both point in the same direction.

First, the judge made limited use of its scale. Across all 320 scorings it produced only 4.0, 4.5, and 5.0, three of nine possible values, and never scored below 4.0. A judge operating near the top of its range cannot reveal small differences even where they exist.

Second, the judge's noise is comparable in size to the effects. As the variance tables show, re-scoring identical text varies by roughly as much as the gaps between conditions.

Together these mean the safe conclusion is the narrow one: the conditions are hard to tell apart, and the cheap setups (no reflection; budget model) are nearly as good as the full system for far less cost. What we cannot firmly claim from these numbers alone is that the architecture adds nothing, human validation will help us check this more thoroughly.
