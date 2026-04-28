# Evaluation Plan — LLM-as-Judge Framework

## Overview

This document describes the design and rationale for the simulation evaluation framework. It covers the run matrix, judge design, output format, and implementation plan.

---

## Run Matrix

Four simulation configurations are run against five agents, producing 20 agent trajectories total.

| Run Label | Agent Model | Ablation | Judge Model |
|---|---|---|---|
| `Premium_Full` | claude-sonnet-4-6 | Full (memory + retrieval + reflection) | claude-opus-4-6 |
| `Baseline_Full` | moonshotai/kimi-k2.6 | Full (memory + retrieval + reflection) | claude-opus-4-6 |
| `Baseline_No_Reflection` | moonshotai/kimi-k2.6 | Memory + retrieval, no reflection | claude-opus-4-6 |
| `Baseline_No_Memory_No_Retrieval_No_Reflection` | moonshotai/kimi-k2.6 | Seed personality only | claude-opus-4-6 |

**Rationale for run labels:** The ablation label explicitly names every disabled component so there is no ambiguity when reading results. `Baseline_No_Memory_No_Retrieval_No_Reflection` is verbose but unambiguous — all three components are off, not just memory.

**Agents selected** (copied to `config/agents/selected/`):
- `jennifer.yaml` — partially compliant, pragmatic, does work herself
- `beth.yaml` — partially compliant, working lawyer, values time over money
- `edward.yaml` — fully compliant, used RAP program, skeptical of fire science
- `lola.yaml` — fully compliant, $60K roof, frustrated by non-compliant neighbours
- `synthetic_non_compliant.yaml` — non-compliant renter, ecological framing, runs Keep the Hills Green campaign

This set covers the full compliance spectrum (non-compliant → partially → fully) and includes both real-interview-grounded agents and a synthetic persona.

---

## Ablation Variants

Following Park et al. (2023), three variants test the contribution of each architectural component:

| Variant | Memory Stream + Retrieval | Reflection | Description |
|---|---|---|---|
| **Variant 1 — Full** | Yes | Yes | Complete system — memories accumulate, top-K retrieval, reflections fire when importance threshold met |
| **Variant 2 — No Reflection** | Yes | No | Agents retrieve memories but never synthesise higher-level beliefs |
| **Variant 3 — No Memory, No Retrieval, No Reflection** | No | No | Seed personality only — no memory stream, no retrieval, no reflection |

Note: removing the memory stream (Variant 3) implicitly removes retrieval and reflection, since both depend on an accumulated memory stream. The label makes this explicit.

---

## Judge Design

### Criteria

All judge calls score on three dimensions (1–5 scale), matching the generation evaluation in `stage2_validation_beth_v2.ipynb`:

| Criterion | 1 | 3 | 5 |
|---|---|---|---|
| **Behavioral Plausibility** | Not at all plausible | Reasonable but generic | Reflects nuanced situational understanding |
| **Persona Consistency** | Very inconsistent with seed personality | Neutral | Strongly reflects specific personality traits |
| **Intervention Responsiveness** | Did not engage with the event | Engaged broadly but not with specifics | Meaningfully integrated event with prior context |

Each criterion also has a **1–2 sentence note** explaining the score (replacing the combined critique from Stage 2, where per-criterion reasoning is more useful for comparison).

### Judge Context — Why seed_narrative AND memory seeds

Park et al. (2023) used human crowdworkers as evaluators, not an LLM judge. Critically, those raters were given access to the **full agent memory stream** when assessing believability — not just a summary description. This means their evaluation had richer context than what a judge prompt built from seed_narrative alone would provide.

The `seed_narrative` in our system is the equivalent of Park et al.'s `[Agent's Summary Description]` — a high-level distillation of who the agent is. It captures personality, values, and history in 800–1000 words but necessarily elides specific details. Memory seeds contain those specifics: the exact dollar amounts, the particular frustrations, the specific incidents that define the agent.

**For persona consistency judging in particular**, the specific details matter. A judge with only the seed_narrative might give a high consistency score to a response that contradicts a key memory (e.g. an agent who "hates getting multiple estimates" suddenly comparing five contractors). A judge with the memory seeds can catch that.

**What the judge receives:**

*Per-intervention judge:*
- Full `seed_narrative`
- All memory seeds (description only, no metadata)
- Intervention text
- Agent decision
- Agent reasoning

*Per-simulation (full trajectory) judge:*
- Full `seed_narrative`
- All memory seeds (description only, no metadata)
- All interventions in chronological sequence, each with decision + reasoning

**Held-out responses are not used** in simulation evaluation. By the time a matching event fires, the agent has accumulated new memories and may have reflected — divergence from the held-out response is expected and not a failure signal. This is consistent with the design principle in `systemdesign.md`: held-outs are Stage 2 pre-simulation validation only.

### Judge Model

`claude-opus-4-6` at `temperature=0.0` for all judge calls. The judge is always independent of the agent model being evaluated — same judge across all four run configurations.

---

## Two-Level Evaluation

### Level 1 — Per Intervention

One judge call per agent per event (6 events × 5 agents × 4 runs = 120 calls). Scores each individual decision in context.

### Level 2 — Full Simulation

One judge call per agent per run (5 agents × 4 runs = 20 calls). The judge sees the complete trajectory — all 6 interventions with decisions and reasoning in chronological order — and gives a holistic score. This is not an average of per-intervention scores; it is a separate holistic assessment that can capture trajectory-level patterns (e.g. an agent that becomes more resistant over time, or one that is consistent across all events).

---

## Concise Output Toggle

Agent decisions and reasoning can get verbose, which makes the results export harder to read. A `CONCISE_OUTPUT` flag in `Config` (default `False`) injects a conciseness instruction into the decision prompt when `True`:

> "Keep both the decision and reasoning fields concise — 1-2 sentences each."

This is set to `True` for all simulation runs in the evaluation matrix. The instruction is appended to the user prompt inside `decide()` in `client.py`, so it applies transparently without changing prompt templates elsewhere.

---

## Cost and Latency Tracking

### What is tracked

- **Agent model cost**: tokens in/out for all agent LLM calls (decisions, importance scoring, reflection questions, reflection synthesis) — tracked per simulation run
- **Judge model cost**: tokens in/out for all judge calls — tracked per evaluation run
- **Simulation latency**: wall-clock time in seconds for the full simulation run (from first agent seed to logger close)
- Costs are calculated from token counts × per-model pricing rates

### Model pricing

| Model | Input (per M tokens) | Output (per M tokens) | Source |
|---|---|---|---|
| moonshotai/kimi-k2.6 | $0.7448 | $4.655 | OpenRouter (verified Apr 2026) |
| claude-sonnet-4-6 | $3.00 | $15.00 | Anthropic (verified Apr 2026) |
| claude-opus-4-6 | $5.00 | $25.00 | Anthropic (verified Apr 2026) |

### Pre-run cost estimates

Estimates assume: 6 events × 5 agents = 30 decision calls per run; ~90 importance-scoring calls per run (Full and No-Reflection variants, ~3 new memories per event × 6 events × 5 agents); ~12 reflection cycles per Full run (2–3 per agent); concise output mode throughout. Token estimates: decision input ~1,700 tokens (Full) / ~1,250 tokens (no-memory), decision output ~150 tokens; importance input ~600, output ~30; reflection input ~1,900, output ~280; judge-intervention input ~3,500, output ~150; judge-full-sim input ~5,500, output ~200.

**Agent runs**

| Run | Agent model | Agent tokens in | Agent tokens out | Agent cost |
|---|---|---|---|---|
| Premium_Full | claude-sonnet-4-6 | ~128,000 | ~10,600 | ~$0.54 |
| Baseline_Full | moonshotai/kimi-k2.6 | ~128,000 | ~10,600 | ~$0.14 |
| Baseline_No_Reflection | moonshotai/kimi-k2.6 | ~105,000 | ~7,200 | ~$0.11 |
| Baseline_No_Memory_No_Retrieval_No_Reflection | moonshotai/kimi-k2.6 | ~37,500 | ~4,500 | ~$0.05 |
| **Total agent** | | | | **~$0.84** |

**Judge (claude-opus-4-6, shared across all runs)**

| Judge type | Calls | Tokens in | Tokens out | Cost |
|---|---|---|---|---|
| Per-intervention (6 × 5 × 4) | 120 | ~420,000 | ~18,000 | ~$2.55 |
| Full simulation (5 × 4) | 20 | ~110,000 | ~4,000 | ~$0.65 |
| **Total judge** | 140 | ~530,000 | ~22,000 | **~$3.20** |

**Grand total: ~$4.05** — judge cost dominates (~79%) due to Opus 4.6 pricing. Actual costs will be logged per run in the JSONL and reported in Tab 3 of the Excel export.

### Implementation

A `UsageTracker` class in `client.py` accumulates token counts via a module-level instance. `_call_llm()` records usage after each call, tagged as `"agent"` or `"judge"`. The tracker is reset at the start of each simulation run and read at the end to produce the cost summary logged to the JSONL.

---

## Notebook Architecture

Two notebooks, decoupled so simulations and evaluation can be run independently.

### `run_simulation.ipynb` (updated)

Runs all four simulation configurations. Defines a `RUN_MATRIX` config list at the top — tick the runs you want to execute. For each run:
1. Reset usage tracker + start timer
2. Run simulation (all 5 agents, baseline scenario)
3. Stop timer, read usage tracker
4. Log run summary (cost + latency) as final JSONL entry
5. Print cost/latency summary

Output: one `.jsonl` per run in `outputs/runs/`, named `{run_label}_{timestamp}.jsonl`.

### `run_evaluation.ipynb` (new)

Reads saved JSONLs and judges all runs in batch. Sections:
1. **Config** — point at runs dir, set judge model, set output path
2. **Load** — scan `outputs/runs/`, group by run_label, load matching agent YAMLs from `config/agents/selected/`
3. **Per-intervention judging** — for each decision entry: call judge, store scores + notes
4. **Full simulation judging** — for each (run, agent): collect all decisions in order, call judge, store holistic scores + notes
5. **Track judge cost** — accumulate judge token usage during evaluation
6. **Export to Excel** — write three-tab workbook to `outputs/eval/`

---

## Excel Output Format

### Tab 1 — `Interventions`

One row per intervention per agent per run.

| Column | Description |
|---|---|
| run_label | e.g. `Premium_Full` |
| agent_id | e.g. `jennifer` |
| agent_display_name | e.g. `Laura` |
| day | Simulation day number |
| event_type | e.g. `firewise_outreach` |
| intervention | Full intervention text |
| decision | Agent decision (concise) |
| reasoning | Agent reasoning (concise) |
| behavioral_plausibility | Score 1–5 |
| bp_note | 1–2 sentence explanation |
| persona_consistency | Score 1–5 |
| pc_note | 1–2 sentence explanation |
| intervention_responsiveness | Score 1–5 |
| ir_note | 1–2 sentence explanation |

### Tab 2 — `Full Simulation`

One row per agent per run. Holistic scores from the full trajectory judge.

| Column | Description |
|---|---|
| run_label | |
| agent_id | |
| agent_display_name | |
| behavioral_plausibility | Holistic score 1–5 |
| bp_note | |
| persona_consistency | Holistic score 1–5 |
| pc_note | |
| intervention_responsiveness | Holistic score 1–5 |
| ir_note | |

### Tab 3 — `Cost & Latency`

One row per simulation run.

| Column | Description |
|---|---|
| run_label | |
| agent_model | Model used for agent decisions |
| agent_tokens_in | Total input tokens for all agent calls |
| agent_tokens_out | Total output tokens for all agent calls |
| agent_cost_usd | Cost of agent model calls |
| judge_tokens_in | Total input tokens for all judge calls |
| judge_tokens_out | Total output tokens for all judge calls |
| judge_cost_usd | Cost of judge calls |
| total_cost_usd | Combined |
| run_latency_seconds | Wall-clock time for simulation only |

---

## Implementation Checklist

- [ ] Copy 5 agent YAMLs to `config/agents/selected/` (fix `agent_id` → `id` in synthetic)
- [ ] `src/llm/client.py` — add `CONCISE_OUTPUT` to `Config`, add `MODEL_PRICING`, add `UsageTracker`, modify `_call_llm()` to track usage, modify `decide()` for concise toggle, update `judge_simulation()` to accept memory seeds and return per-criteria notes, add `judge_full_simulation()`
- [ ] `src/output/logger.py` — add `log_run_summary()` for cost + latency
- [ ] `src/engine/simulation.py` — add `run_label` to `log_run_config`, add `run_label` to `SimulationConfig`, update `describe()` for new ablation labels
- [ ] `notebooks/run_simulation.ipynb` — update for multi-run matrix with `RUN_MATRIX` config block
- [ ] `notebooks/run_evaluation.ipynb` — create batch judge + Excel export notebook
