# Project Plan: April 15–30

## Guiding principle
Write the report as you build — not after. Sections that need no results (Intro, Related Work, Data Sources, Architecture) are written this week alongside development. The Experiments section gets a skeleton with placeholder tables on Day 1, filled in as runs complete. The last few days are dropping in numbers and polishing, not writing from scratch.

**Judge model:** Sonnet at temp=0 is the default judge (cheaper than Opus, consistency matters more than raw capability). Kimi (`moonshotai/kimi-k2` via OpenRouter) is a valid cost-saving swap — one config field change.

---

## Current State

**Built (Stage 1 + Stage 2):**
- `src/agents/memory.py`, `retrieval.py`, `reflection.py`, `prompts.py`
- `src/llm/client.py` (Config, decide, reflect, judge functions, OpenRouter support)
- `config/agents/jennifer.yaml`, `synthetic_agents.yaml`
- `notebooks/prototype-1.ipynb`, `prototype-2.ipynb`, `stage2_validation.ipynb`

**Not yet built:**
- `src/agents/agent.py`, `src/engine/simulation.py`, `src/engine/scheduler.py`
- `src/environment/channels.py`, `src/output/logger.py`
- `config/agents/lola.yaml`, `config/scenarios/baseline.yaml` (correct path)
- `notebooks/run_simulation.ipynb`, `ablation_analysis.ipynb`

---

## Week 1: April 15–22 — Stage 2 Tuning + Stage 3 Build + Report foundations

### Days 1–2 (Apr 15–16): Stage 2 — Lola + report skeleton

**Dev:**
- Build `lola.yaml` (seed narrative, core beliefs, memory seeds, held-out responses — same format as `jennifer.yaml`)

**Report (write in parallel):**
- Create the report document (Overleaf / Google Docs)
- Write **Introduction** in full — problem definition, why LLM agents, I/O statement, paper outline
- Write **Data Sources** section — interview transcripts, preprocessing pipeline, dev/val/test agent split (Jennifer=dev, Lola=val, Beth=test), synthetic agent design rationale
- Drop in **Experiments section skeleton** — section headers, table shells with TBD cells for all planned results (retrieval grid, reflection grid, ablation, model comparison)

---

### Days 2–3 (Apr 16–17): Stage 2 — Hyperparameter tuning runs

**Dev:**
Run `stage2_validation.ipynb` on Jennifer → Lola → Tester1 + Tester4 + Tester5. Export CSV after each agent.

**Retrieval grid (Section 2 of notebook):**

| Parameter | Values |
|---|---|
| `top_k` | 8, 12, 15 |
| `relevance_weight` | 1.0, 2.0, 3.0 (hold others at 1.0) |
| `retrieval_mode` | `dense`, `hybrid` |
| `recency_decay` | 0.95, 0.99 |

~12 configs. Lock winner that scores ≥7.0 across Jennifer AND Lola.

**Reflection grid (Section 3 of notebook):**

| Parameter | Values |
|---|---|
| `threshold` | 50, 100, 150 |
| `num_questions` | 2, 3, 5 |

9 configs. Lock winner on insight quality judge score.

**Lock params in `stage2_changes.md`:**
```
Best RetrievalConfig:  top_k=__ recency_weight=__ importance_weight=__ relevance_weight=__ mode=__
Best ReflectionConfig: threshold=__ num_questions=__
Jennifer overall score: __/10
Lola overall score:     __/10
```

**Report (fill in as results come in):**
- Fill retrieval grid results table in Experiments section
- Fill reflection grid results table
- Write **Related Work** section — Park et al. 2023 (Generative Agents) + one other (AgentBench / prior evacuation/disaster behavior modeling)
- Start **System Architecture** section — describe the 5-layer architecture, each component's role

---

### Days 4–7 (Apr 18–22): Stage 3 — Simulation loop build

Build in this order (each unblocks the next):

1. **`src/output/logger.py`** — JSONL writer; everything logs through it
2. **`src/agents/agent.py`** — `Agent` class: `perceive()` → `retrieve()` → `decide()` → `act()` → `store()`; holds MemoryStream + config YAML + state (compliance_status, attitude, actions_taken)
3. **`src/engine/scheduler.py`** — `EventScheduler` with heapq; `get_events(tick)` from `baseline.yaml`
4. **`src/environment/channels.py`** — framing wrappers for `official_mail`, `news_media`, `social`, `direct_experience`
5. **`config/scenarios/baseline.yaml`** — 30-day timeline, 3 agents (Jennifer + Tester1 + Tester4 for contrasting compliance profiles)
6. **`src/engine/simulation.py`** — tick loop: get events → frame → cognition cycle → reflection check → log
7. **`notebooks/run_simulation.ipynb`** — runner with `inspect_agent()`, `inspect_memory()`, `compare_agents()`, `show_tick_summary()`

**Test milestone (Apr 22):** `run_simulation.ipynb` produces clean JSONL for a 3-agent, 30-day run. Verify agents respond differently to the same event. Check token counts per tick.

**Report (write alongside build):**
- Finish **System Architecture** section — add variants table (full / no-memory / no-retrieval-scoring / no-reflection / model swap), architecture diagram
- Write the methodology subsection of **Experiments** (what conditions are being tested and why, how judge scoring works, evaluation metrics)

---

## Week 2: April 23–30 — Experiments, Results + Report completion

### Days 8–10 (Apr 23–25): Run experiments

**Ablation study** — 4 conditions, run via `ablation_analysis.ipynb` or inline in `run_simulation.ipynb`:

| Condition | What's disabled |
|---|---|
| `full` | Nothing — best configs from Stage 2 |
| `no_memory` | Agent gets no retrieved memories, just persona |
| `no_retrieval_scoring` | All memories passed flat (no top-K scoring) |
| `no_reflection` | Reflection step skipped entirely |

Judge (Sonnet, temp=0) scores each condition: character fidelity + decision alignment + reasoning alignment. 4 conditions × 3 agents = 12 data points. This is the core comparison table.

**Model comparison** (run if budget allows):

| Model | Config field |
|---|---|
| `claude-sonnet-4-6` | baseline |
| `openai/gpt-4o` | via OpenRouter |
| `moonshotai/kimi-k2` | via OpenRouter |

Use `judge_generation()` scores to compare. If budget is tight, run on Jennifer's held-out scenarios only.

**Sensitivity analysis** — vary one dimension at a time:
- Intervention ordering: `law_first.yaml` vs `insurance_first.yaml` vs `baseline.yaml`
- Agent population: all-compliant vs mixed vs resistant-heavy

**Report (fill in results as they come in):**
- Drop ablation numbers into the results table
- Drop model comparison numbers
- Add one qualitative agent output example (a memory + reflection + decision showing the pipeline working end-to-end)

---

### Days 10–12 (Apr 25–27): Report completion

Write the remaining sections (all results should be in hand by now):

- **Abstract** — finalise with actual key finding from ablation
- **Discussion** (~0.25 pages):
  - Which variant had the largest impact (expected: reflection for emotionally complex scenarios)
  - Which failed most noticeably (expected: no-memory on trust/history-dependent decisions)
  - Systematic failure modes or biases observed
  - Cost/quality tradeoff of Sonnet vs Opus for reflection; Kimi as low-cost alternative
- **Conclusions and Future Work** (~0.25 pages):
  - Key contributions and insights
  - Limitations (single-city real data, held-out ground truth only for 3 agents)
  - What would be extended with more time (Stage 4 social network, more real agents, Beth evaluation)
- **Contributions** — breakdown of each member's specific notebooks and report sections

**Appendix (optional, if space/time allows):**
- Full hyperparameter search tables
- Example prompts used in `prompts.py`
- Cost + latency analysis per condition
- Additional agent transcript excerpts

---

### Days 12–14 (Apr 27–29): Poster + final polish

**Poster layout (one PDF, Letter or A0):**
- Title + team names + emails
- System architecture diagram
- Key results: ablation table + one qualitative agent output example
- 3-bullet takeaways

**April 30:** Submit report PDF + poster PDF to Gradescope. Tag all team members.

---

## What to test where — summary

| What | Where | Primary metric |
|---|---|---|
| Retrieval hyperparams (top-K, weights, mode) | `stage2_validation.ipynb` Section 2 | Judge relevance score ≥7/10 on Jennifer + Lola |
| Reflection hyperparams (threshold, num_questions) | `stage2_validation.ipynb` Section 3 | Judge insight quality score |
| Generation quality (locked config) | `stage2_validation.ipynb` Section 4 | Overall judge score (fidelity + alignment) |
| Ablation (full vs 3 degraded conditions) | `ablation_analysis.ipynb` | Delta in generation judge scores |
| Model comparison (Sonnet vs GPT-4o vs Kimi) | `stage2_validation.ipynb` Section 4, swap `DECISION_MODEL` | Judge scores + cost per decision |
| Simulation coherence | `run_simulation.ipynb` | Agent divergence in JSONL; token counts |

---

## Report writing schedule — at a glance

| Section | When | Status |
|---|---|---|
| Experiments skeleton (table shells) | Day 1 (Apr 15) | |
| Introduction | Day 1–2 (Apr 15–16) | |
| Data Sources | Day 1–2 (Apr 15–16) | |
| Related Work | Day 3–4 (Apr 17–18) | |
| System Architecture | Day 4–7 (Apr 18–22, alongside Stage 3 build) | |
| Experiments methodology | Day 4–7 (Apr 18–22) | |
| Retrieval/reflection results | Day 3 (Apr 17, after tuning runs) | |
| Ablation + model comparison results | Day 8–10 (Apr 23–25, fill as runs complete) | |
| Discussion | Day 10–12 (Apr 25–27) | |
| Conclusions | Day 10–12 (Apr 25–27) | |
| Abstract (finalise) | Day 12 (Apr 27) | |
| Contributions breakdown | Day 12 (Apr 27) | |
| Poster | Day 12–14 (Apr 27–29) | |

---

## Suggested team contribution split

| Member | Primary notebook | Report sections |
|---|---|---|
| Sam | `run_simulation.ipynb` | System Architecture, Experiments (simulation results) |
| Madeleine | `stage2_validation.ipynb` | Data Sources, agent YAML refinement |
| Amrita | `ablation_analysis.ipynb` | Experiments (ablation + model comparison), Discussion |
| All | — | Each contributes to Architecture + Experiments sections |

---

## Risk flags

1. **Lola YAML quality** — Lola's generation score is the paper's key validation number. If her held-out responses aren't detailed, scores will look noisy. Prioritize this first.
2. **Stage 3 scope** — if `simulation.py` takes longer than expected, cut sensitivity analysis and rely on Stage 2 ablations only. Paper is still strong without a full simulation run.
3. **API costs** — Sonnet-as-judge across 12 ablation conditions × 3 scenarios is manageable; Kimi is cheaper if budget tightens. Run a cost estimate before scaling.
4. **Beth (test agent)** — do not build or evaluate until all tuning is locked. Use Lola's score as the reported validation result if Beth isn't ready in time.
5. **Stage 4 (social network)** — `network.py` + `conversation.py` are stretch goals. Attempt only if simulation loop is clean by Apr 22. Not required for a strong paper.
