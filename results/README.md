# Results

This section holds the raw results of our experiments. Each folder is a separate experiment, and this is what each one is and what it does.

| Folder | The question it answers | The answer |
|---|---|---|
| `ablation/` | Does the memory and reflection machinery make the agents behave better? | Barely. Scores 4.80–4.93 out of 5 across all four conditions. |
| `ablation_replicated/` | Is that small difference real, or run-to-run noise? | Noise. No condition is significantly different from Baseline. |
| `persona_attribution/` | Are the five agents actually different people, or do they all sound the same? | Genuinely different. The judge picks the right author 88% of the time against 25% chance. |

---

## `ablation/` — the evaluation in the paper

Four versions of the simulation were run and scored, to see how much each part of the architecture matters:

| Condition | What was removed |
|---|---|
| Baseline | Nothing. The full system. |
| Ablation 1 | Reflection |
| Ablation 2 | Memory and reflection |
| Budget | Nothing, but run on a cheaper model (Haiku instead of Sonnet) |

An LLM judge read each resident's 60-day story and scored it 1–5 on three things: whether the behaviour was believable, whether it sounded like that specific resident, and whether they actually engaged with the events. Every condition scored between 4.80 and 4.93, so stripping out memory and reflection cost only 0.13.

The run was made on 2026-04-29 and judged by Claude Opus 4.6. **These are the numbers in the paper's results tables.**

- `eval/evaluation_20260429_025149.xlsx` — the scores. Three sheets: per intervention, per resident trajectory, and cost/latency.
- `runs/` — the raw decision logs, one file per condition.
- `simulation_results/` — the same runs exported as a readable workbook.

## `ablation_replicated/` — the same test, done more carefully

The limitation of the run above is that it happened once, and running the simulation twice gives slightly different scores, so a gap of 0.13 could easily be chance rather than a real effect.

The whole thing was therefore run again, five times per condition (20 runs), judged by a model from a different company (`openai/gpt-5.4` rather than Claude) so the judge could not be accused of favouring output from its own family.

Once you account for run-to-run variation, and for the fact that three comparisons were being made at once, **none of the conditions is significantly different from Baseline**, and the cheap setups are as good as the full system.

The summary records one caveat: the judge only ever used the scores 4.0, 4.5 and 5.0. A measure that never drops below 4.0 may be too blunt to detect small differences, so "we could not find a difference" is a safer reading than "there is no difference".

- `eval/RESULTS_SUMMARY.md` — the full write-up, with the statistics explained step by step.
- `eval/*.csv` — the raw judge scores.
- `runs/` — 20 raw decision logs.

These numbers are **not** cited in the paper.

## `persona_attribution/` — are the agents actually distinguishable?

A high "persona consistency" score does not prove much on its own, because a bland, generic response can score well against anybody's profile when nothing in it contradicts anybody.

This test asks the question directly instead: show the judge one response plus all five residents' profiles, and ask which of the five wrote it.

- Guessing at random would be right 25% of the time (four homeowners).
- The judge was right **88%** of the time.
- Removing memory and reflection barely changed this (a 4% drop, well within noise).

This tells us the agents are genuinely distinct from one another, and that the distinctness comes from the interview-grounded profile each agent starts with rather than from memories accumulated during the simulation, which is why removing memory changed so little.

It is also the basis for the paper's argument that scoring an agent against its own profile is a weak test, and that asking the judge to pick the author is a better one.

- `results_*.txt` — plain-English summary of all three findings.
- `results_*.json` — the same numbers, machine-readable.
- `swap_raw_*.csv` — scores of each response against the right and wrong profiles.
- `attrib_*_raw_*.csv` — every individual "who wrote this?" judgment.

---

## Where new runs go

Running the notebooks writes to `outputs/`, which is ignored by git. Results worth keeping are copied into this folder. That is why `outputs/` is empty in a fresh checkout, and why the notebooks fall back to reading from `results/` when it is.
