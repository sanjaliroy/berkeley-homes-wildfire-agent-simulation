# Project Report Structure
*At most 5 pages (excluding appendix) · Letter size · 10–12pt · ≥0.5in margins*

---

## Title + Team

- Project title
- Full names and Berkeley.edu emails for all team members

---

## Abstract
*1 paragraph*

- Problem: homeowners in Berkeley Hills face wildfire hardening decisions under institutional pressure (insurance non-renewal, city ordinances) — their responses vary by personality, trust, and financial capacity
- System type: agentic workflow using LLM-driven agents grounded in real interview transcripts
- Key finding: [fill in after ablation runs — expected: retrieval + reflection together drive the largest gain in character fidelity over a no-memory baseline]

---

## 1. Introduction
*~0.5 pages*

- Define the problem: predicting and understanding how homeowners respond to wildfire mitigation interventions is hard to study at scale through interviews alone
- Why GenAI: LLM agents can be grounded in real interview data to simulate psychologically realistic behaviour across many agents and intervention sequences
- I/O statement: *"Our system takes homeowner interview transcripts and an intervention schedule as input, and uses a retrieval-augmented agentic workflow with reflection to produce per-tick decisions, reasoning, and evolving belief states for each simulated homeowner."*
- Briefly outline: data → architecture → experiments → findings

---

## 2. Related Work
*~0.5 pages*

- **Park et al. 2023 — Generative Agents:** our retrieval (recency × importance × relevance), reflection (two-step: questions → synthesis), and memory stream design follow this paper directly; we extend it by grounding agents in real interview data rather than fictional personas
- **[Second source — TBD]:** options include AgentBench (LLM agent evaluation), prior agent-based models of disaster/evacuation behaviour, or LLM simulation of social attitudes — pick whichever the team has read or can quickly read
- Key differentiator from both: our agents are initialised from real interview transcripts with held-out ground-truth responses, enabling quantitative fidelity evaluation against actual human behaviour

---

## 3. Data Sources and Inputs
*~0.5 pages*

- **Real interview transcripts:** 12 Berkeley Hills homeowners interviewed about wildfire hardening experience, insurance, and institutional trust; preprocessed into structured YAML (seed narrative, core beliefs, memory seeds, held-out responses)
- **Data split:** Jennifer (dev — all Stage 2 tuning), Lola (validation — cross-agent tuning signal), Beth (test — evaluated last, no peeking during tuning)
- **Synthetic agents:** 5 profiles generated to fill gaps in the real dataset (financially constrained, passive complier, uninformed newcomer, active denier, disillusioned former complier); used for retrieval/reflection stress-testing only, not final evaluation
- **Intervention schedule:** structured YAML defining a 30-day sequence of events (insurance letter, city ordinance, neighbour conversation, direct fire experience) delivered via four channels: `official_mail`, `news_media`, `social`, `direct_experience`
- **Embeddings:** sentence-transformer (`all-MiniLM-L6-v2`, local, no API cost) used for dense retrieval

---

## 4. System Architecture and Variants
*~1 page*

### 4.1 Architecture overview
- Five-layer pipeline: Config → Engine → Environment → Agent Cognition → Output
- Per-tick flow: scheduler pulls due events → channels frame each event → each targeted agent runs perceive → retrieve → decide → act → store → reflection check → logger writes JSONL
- Three separate config objects with no cross-dependencies: `Config` (LLM model choices), `RetrievalConfig` (top-K, weights, mode), `ReflectionConfig` (threshold, num_questions)

### 4.2 Agent cognition detail
- **Memory stream:** append-only list of typed memories (observation / decision / reflection / conversation); each memory carries importance score (LLM-assigned 1–10), embedding, timestamp, and `last_accessed`
- **Retrieval:** scores every memory as `recency × w_r + importance × w_i + relevance × w_rel` (all min-max normalised); recency decays from `last_accessed` so frequently retrieved memories stay fresh; supports dense, sparse, and hybrid modes
- **Reflection:** fires when cumulative importance of unprocessed memories exceeds threshold; two-step (generate N questions from recent memories → retrieve per question → synthesise first-person insight); insights stored as high-importance reflection memories that re-enter the retrieval stream

### 4.3 System variants tested
| Variant | Description |
|---|---|
| `full` | Complete pipeline — best-tuned RetrievalConfig + ReflectionConfig |
| `no_memory` | Agent receives only persona prompt; no retrieved memories |
| `no_retrieval_scoring` | All memories passed flat (no top-K scoring or weighting) |
| `no_reflection` | Reflection step disabled; no synthesised belief memories |
| Model swap | `DECISION_MODEL` swapped to GPT-4o / Kimi while keeping everything else fixed |

---

## 5. Experiments and Evaluation
*~2 pages*

### 5.1 Evaluation approach
- **LLM-as-judge:** Sonnet (`claude-sonnet-4-6`, temp=0) scores all outputs for reproducibility; Kimi (`moonshotai/kimi-k2` via OpenRouter) used as cost-saving cross-check
- **Judge functions:** `judge_retrieval()` (relevance score 1–10, missed memories, critique) and `judge_generation()` (character fidelity, decision alignment, reasoning alignment, overall score 1–10)
- Ground truth for generation evaluation: held-out interview responses (real human answers to the same scenario prompts)

### 5.2 Stage 2: Retrieval hyperparameter search
- Grid over top-K (8/12/15), relevance_weight (1.0/2.0/3.0), retrieval_mode (dense/hybrid), recency_decay (0.95/0.99)
- Evaluated on Jennifer (dev) and Lola (validation); winner = config scoring ≥7.0 on both
- [Results table — fill after tuning runs]

### 5.3 Stage 2: Reflection hyperparameter search
- Grid over threshold (50/100/150) × num_questions (2/3/5)
- Judge scores insight quality and accuracy of synthesised beliefs against memory stream
- [Results table — fill after tuning runs]

### 5.4 Ablation study
- Four conditions run on 3-agent, 30-day simulation using locked best configs
- Judge scores each condition on character fidelity + decision alignment + reasoning alignment
- [Ablation results table — fill after simulation runs]

| Condition | Character Fidelity | Decision Alignment | Reasoning Alignment | Overall |
|---|---|---|---|---|
| full | TBD | TBD | TBD | TBD |
| no_memory | TBD | TBD | TBD | TBD |
| no_retrieval_scoring | TBD | TBD | TBD | TBD |
| no_reflection | TBD | TBD | TBD | TBD |

### 5.5 Model comparison
- `DECISION_MODEL` swapped across Sonnet, GPT-4o (OpenRouter), Kimi (OpenRouter); all other config held fixed
- Compared on judge generation scores + cost per decision call
- [Results table — fill after model comparison runs]

### 5.6 Qualitative example
- One end-to-end trace showing: memory seeds → retrieval output → reflection firing → decision + reasoning for a representative scenario (e.g. insurance non-renewal letter)
- Illustrates how retrieved memories and a reflection insight shape the agent's reasoning in a way that matches the real interviewee's response

---

## 6. Discussion
*~0.25 pages*

- **Largest impact variant:** [expected: reflection — emotionally loaded scenarios require synthesised beliefs to respond authentically; evidence: delta between `full` and `no_reflection` conditions]
- **Worst failure mode:** [expected: `no_memory` on trust/history-dependent decisions — agent defaults to generic risk-averse response with no personal history]
- **Systematic biases:** [fill after seeing results — watch for over-compliance in no-reflection condition, or under-retrieval of cost/grant memories for financially constrained agents]
- **Trade-offs:** Opus reflection model produces richer insights but ~5× cost vs Sonnet; Kimi comparable to Sonnet at lower cost; reflection adds latency per tick but measurably improves fidelity
- **Robustness:** Lola's generation score (real validation data) vs synthetic agent scores — if gap is small, system generalises; if large, synthetic seeds are too different from real interview grounding

---

## 7. Conclusions and Future Work
*~0.25 pages*

- **Key contributions:** agentic simulation framework grounded in real interview transcripts; LLM-as-judge evaluation pipeline with held-out ground truth; evidence that retrieval + reflection together are necessary (not just sufficient) for character fidelity
- **Limitations:** real interview ground truth only for 3 agents (dev/val/test split); single-city, single-hazard context; Beth (test agent) not evaluated if time doesn't allow
- **Future work:** Stage 4 social network (information propagation between agents), more real agent transcripts, Beth final evaluation, sensitivity analysis over intervention orderings

---

## 8. Contributions

*Detailed breakdown — to be filled by team*

| Member | Coding contributions | Writing contributions |
|---|---|---|
| Sam | `run_simulation.ipynb` (primary), Stage 3 build (agent.py, simulation.py, scheduler.py, channels.py, logger.py) | System Architecture section, Experiments simulation results |
| Madeleine | `stage2_validation.ipynb` (primary), agent YAML refinement (jennifer.yaml, lola.yaml) | Data Sources section, agent YAML documentation |
| Amrita | `ablation_analysis.ipynb` (primary), reflection evaluation runs | Experiments ablation + model comparison, Discussion |
| [4th member if any] | | |

---

## Appendix (optional)

- Full hyperparameter search result tables (all retrieval + reflection grid configs with scores)
- Prompt templates used in `prompts.py` and judge prompts
- Cost and latency analysis per condition (tokens in/out, API cost, time per tick)
- Additional agent transcript excerpts showing divergent decision-making across agent profiles
