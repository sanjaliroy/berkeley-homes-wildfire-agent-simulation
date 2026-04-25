# Detailed Design & Build Plan

## Staged Build Plan

### Stage 1: Single Agent, Single Event (prototype.ipynb)
**Build:** `client.py`, `memory.py`, `prompts.py`, one `maria.yaml`. Hardcode one event, call the LLM, store the decision.

**Test:** Does Maria sound like Maria? Iterate on prompt template and seed narrative.

### Stage 2: Retrieval + Reflection + Validation (stage2_validation.ipynb)
**Add:** `retrieval.py` (RetrievalConfig + scored retrieval), `reflection.py` (ReflectionConfig + two-step reflection), extend `client.py` with `judge_retrieval()` and `judge_generation()`, add `last_accessed` to `Memory`.

**Single agent focus:** All Stage 2 work uses Jennifer only. Lola and Beth transcripts are set aside as validation and test agents respectively — not built until needed.

**Test sequence (run in order):**
1. Memory seeding — LLM-as-judge: do seeds give Jennifer enough context for each scenario?
2. Retrieval — grid search over RetrievalConfig variants (top-K, weights, retrieval_mode); judge scores each
3. Reflection — test ReflectionConfig variants (threshold, num_questions); judge scores insight quality
4. Generation — full pipeline vs. held-out interview responses; judge scores fidelity + alignment
5. Ablations — compare full / no-memory / no-retrieval-scoring / no-reflection conditions

**Why reflection is in Stage 2, not Stage 5:** Reflections become high-importance memories that re-enter the retrieval stream. Retrieval cannot be properly tuned without understanding how reflections interact with it.

**Deferred:** Synthetic agent generation (using a different LLM to create fictional homeowner profiles in YAML format) — to be added after Stage 2 validation is stable, to expand the evaluation dataset.

### Stage 3: Simulation Loop + Multi-Agent (new files)
**Add:** `scheduler.py`, `simulation.py`, `channels.py`, `logger.py`, `baseline.yaml`, 3-5 agent YAMLs, `run_simulation.ipynb` with observability helpers.

**Test:** Do different agents respond differently to the same events? Check JSONL output for consistency.

### Stage 4: Social Network + Conversations
**Add:** `network.py`, `conversation.py`, relationship fields in YAMLs, social trigger detection.

**Test:** Does information propagate between agents? Are API costs manageable? Do transcripts read naturally?

### Stage 5: Scale + Full Evaluation (full architecture)
**Add:** Scale to 10 agents (build remaining YAMLs, synthetic agents if needed), `ablation_analysis.ipynb`, `sensitivity_analysis.ipynb`, `interview_validation.ipynb`.

**Test:** Sensitivity analysis (vary intervention orderings, social structures, populations), full interview validation across all agents, cost analysis at scale.

---

### Note: Role of held_out_responses in evaluation

Each agent YAML contains `held_out_responses` — verbatim quotes from real interview transcripts, each tied to a specific scenario (e.g. `insurance_non_renewal`, `defensible_space_inspection`). These serve different purposes at different stages and must not be conflated:

**Stage 2 (pre-simulation validation):** The primary use. Held-out responses are ground truth for checking whether the agent accurately represents the real person *before* the simulation runs. The stage2_validation notebooks use them to score response fidelity via LLM-as-judge.

**Main simulation evaluation (Stage 3–5):** Held-out responses are *not* appropriate as direct ground truth for intervention responses. By the time a matching event fires (e.g. `insurance_non_renewal` on day 42), the agent has accumulated new memories and may have reflected — so its response will legitimately diverge from the interview quote even if the agent is working correctly. That divergence is the point of the simulation.

**How to use them in simulation evaluation:**
- **Consistency check:** An agent's response to a matching event should not *contradict* the held-out response, even if it doesn't match it. Contradiction is a failure signal; divergence is expected.
- **Baseline anchoring:** Compare agent responses at day 0 (before any events) vs. after, using the held-out response as the pre-simulation reference point to measure position shift.
- **Do not use for direct similarity scoring** in the main simulation — use the LLM-as-judge dimensions (behavioral plausibility, persona consistency, intervention responsiveness) instead.

**Scenario overlaps to be aware of:**
- Day 42 `insurance_non_renewal` → both Beth and Eleanor have `held_out_responses.insurance_non_renewal`
- Day 10 `defensible_space_law` → Beth has `held_out_responses.defensible_space_inspection`
- Day 21 `fire_service_doorknock` (Margaret) → Margaret has `held_out_responses.official_regulatory_notice`

---

## Detailed System Architecture Outline

### Architecture Diagram

The system flows top-to-bottom through six layers:

- The **Config Layer** files define scenarios and agent personalities, feeding into the **Engine Layer** which orchestrates the tick loop.
- Each tick, `simulation.py` pulls due events from `scheduler.py`, routes them through `channels.py` for channel-specific framing, then passes framed events into the **Agent Cognition Layer**.
- Within the **Agent Cognition Layer**, each agent runs a perceive, retrieve, decide, act, store cycle per tick.
  - `retrieval.py` reads from the agent's memory stream in `memory.py`
  - Embeds the current perception via `client.py`
  - Scores all memories by recency × importance × relevance
  - Returns the top-K most relevant
- `prompts.py` assembles the full LLM prompt from the seed personality, retrieved memories, current situation, and a decision question
- `client.py` returns a decision, which is parsed into an action and stored as new memories
- If an agent's action targets another agent (e.g., "Maria calls Tom"), `network.py` triggers an interrupt tick
- `conversation.py` runs a multi-turn dialogue, with each turn hitting the full cognition cycle for both agents and storing each utterance to both memory streams via `memory.py`
- After each tick, `reflection.py` checks whether cumulative importance of recent memories exceeds a threshold
  - If so, it calls `client.py` with a stronger model (potentially Claude Opus) to synthesise higher-level beliefs
  - Beliefs are stored back as first-class memories in `memory.py`
  - This creates the abstraction layers that allow agents to form coherent opinions over time
- All decisions, memories, reflections, conversations, and state snapshots are logged to JSONL files by `logger.py`
- The evaluation notebooks read these outputs for interview validation, ablation studies, and sensitivity analysis

---

## Detailed File Breakdown by Layer

### Potential File Structure

```
wildfire-sim/
├── config/
│   ├── scenarios/
│   │   ├── baseline.yaml
│   │   ├── law_first.yaml
│   │   └── insurance_first.yaml
│   └── agents/
│       ├── jennifer.yaml          ← dev agent (Stage 2)
│       ├── lola.yaml              ← validation agent (build when needed)
│       ├── beth.yaml              ← test agent (build last, don't peek)
│       └── ... ×10
├── src/
│   ├── engine/
│   │   ├── simulation.py
│   │   └── scheduler.py
│   ├── environment/
│   │   ├── channels.py
│   │   └── network.py
│   ├── agents/
│   │   ├── agent.py
│   │   ├── memory.py              ← Memory (+ last_accessed), MemoryStream
│   │   ├── retrieval.py           ← RetrievalConfig, retrieve_memories()   [Stage 2]
│   │   ├── reflection.py          ← ReflectionConfig, maybe_reflect()      [Stage 2]
│   │   ├── prompts.py
│   │   └── conversation.py
│   ├── llm/
│   │   └── client.py              ← Config (all LLM choices), decide(), reflect(),
│   │                                 judge_retrieval(), judge_generation(), embed()
│   └── output/
│       └── logger.py
├── notebooks/
│   ├── prototype-2.ipynb          ← Stage 1 prototype (complete)
│   ├── stage2_validation.ipynb    ← Stage 2 eval harness             [Stage 2]
│   ├── run_simulation.ipynb
│   ├── interview_validation.ipynb
│   ├── ablation_analysis.ipynb
│   └── sensitivity_analysis.ipynb
├── data/
│   ├── interview_transcripts/
│   ├── interview_notes/
│   └── embeddings_cache/
└── outputs/
    ├── runs/
    └── eval/                      ← JSONL eval logs from stage2_validation.ipynb
```

---

## LLM Configuration Strategy

All model choices live in the `Config` dataclass in `src/llm/client.py`. This is the single place to change which LLM is used for each role. The notebook creates a `Config` object and can override any field for an experiment without touching source files.

| Role | Config field | Default (Exp 2 Variant 1 baseline) | Notes |
|---|---|---|---|
| Agent decisions | `DECISION_MODEL` | `claude-haiku-4-5` | Routine decisions — smaller/cheaper model |
| Reflection synthesis | `REFLECTION_MODEL` | `claude-sonnet-4-6` | Larger model for higher-quality belief synthesis |
| LLM-as-judge | `JUDGE_MODEL` | `openai/gpt-4o` | Evaluation only — external model; deterministic (temp=0) |
| Importance scoring | uses `DECISION_MODEL` | — | Short call, same model as decisions |
| Embeddings | `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embeddings via API |

The judge model is kept separate from agent models intentionally — a consistent, independent evaluator regardless of which model the agent uses. Judge calls always use `temperature=0` for reproducible scores. The default split (large model for reflection, small model for decisions) is the Experiment 2 Variant 1 baseline; swapping both fields to the same model enacts Variants 2 and 3.

---

## Config Layer (Controls what gets simulated)

| File | Purpose | Key Contents | Hyperparameters |
|------|---------|--------------|-----------------|
| `config/scenarios/baseline.yaml` | One simulation run definition | Duration, tick interval, intervention schedule (list of {day, event_type, target_agents, channel, content}), social structure parameters | **Duration:** days to simulate. **Tick_interval:** daily vs multi-day. **Max_interactions_per_tick:** caps social costs. **Intervention_ordering:** varied across scenario files |
| `config/agents/agent.yaml` | Each agent's full identity | Agent persona prompt including things like beliefs, risk_zone, property_type, institutional_trust, years_at_property | **Persona / Pre-Seeded Beliefs / Memories** |

---

## Engine Layer (Controls when things happen and orchestrates the tick loop)

| File | Purpose | Hyperparameters |
|------|---------|-----------------|
| `notebooks/run_simulation.ipynb` | Main file for running experiments, allowing for analysis and monitoring. Loads config, runs sim loop. After each tick displays agent states, new memories, reflections, conversations. Can pause mid-run to inspect any agent | N/A |
| `src/engine/simulation.py` | Runs the simulation loop. Per tick will include: get events, channel framing, agent cognition cycle, parse social triggers, resolve interactions, check reflections, log. | N/A |
| `src/engine/scheduler.py` | EventScheduler class. Loads intervention schedule from YAML into heapq priority queue. Method: `get_events(tick)` → `list[Event]` | N/A |

---

## Environment Layer (Creates the environment for how agents will experience the world)

| File | Purpose | Hyperparameters |
|------|---------|-----------------|
| `src/environment/channels.py` | Create channels for how agents will receive information and wrap raw events in channel-specific framing. Channels could be: official_mail, news_media, social, direct_experience | How we frame each channel - different framing may produce different agent responses |
| `src/environment/network.py` | Manages who knows who, pre-seeded from agent YAMLs. Could also enforce interaction caps to ensure token efficiency | Number of interactions per simulation tick. How strong relationships are |

---

## Agent Cognition Layer (controls how agent prompts are built and interact)

| File | Purpose | Hyperparameters |
|------|---------|-----------------|
| `src/agents/agent.py` | Agent class which holds their identity, memory_stream reference, current state. Methods: `perceive()`, `decide()`, `act()`. May also include a `generate_summary()` method that periodically synthesises the agent's current goals, disposition, recent reflections, and key experiences into a dynamically updated summary paragraph. This summary replaces the static seed_narrative in prompt assembly, so the agent's self-description evolves as they accumulate experiences. | What to track per agent (compliance_status, actions_taken, attitude) |
| `src/agents/memory.py` | MemoryStream class which is an append-only list of Memory objects such as timestamp, description, importance, embedding, type. Methods: `add()`, `get_all()`, `get_recent()`, `get_cumulative_importance()`. Each Memory object also stores a last_accessed timestamp, updated whenever the memory is returned in a top-K retrieval. Recency decay in `retrieval.py` is calculated from last_accessed, not creation timestamp, meaning frequently retrieved memories stay "fresh" even if old. | **Importance_prompt:** wording affects scoring distribution. **Memory_type_taxonomy:** observation / decision / reflection / conversation |
| `src/agents/retrieval.py` | Function to retrieve_memories. Will score every memory by weighted combination of recency, importance, and relevance (aligned with Park et al's approach). Before applying the weights, all three scores (recency, importance, relevance) will need to be normalised to [0, 1] using min-max scaling across the full memory stream. | **recency weight:** higher = recent events dominate. **importance weight:** higher = dramatic events dominate. **relevance weight:** higher = semantic similarity matters most. **k (top-K):** memories to retrieve (10-15). **recency_decay_rate** |
| `src/agents/reflection.py` | `ReflectionConfig` + `maybe_reflect()`. Two-step process aligned with Park et al. Step 1: generate N high-level questions from recent memories ("What is Jennifer most concerned about?"). Step 2: for each question, retrieve relevant memories and synthesise an insight ("I am increasingly worried that insurance feels arbitrary — memories 2, 5, 8 show..."). Insights are stored as first-class 'reflection' memories. Fires when cumulative importance of unprocessed memories since last reflection exceeds threshold. | **threshold:** cumulative importance trigger. **num_questions:** how many to generate (Park et al. used 3). **top_k_per_question:** memories retrieved per question. **reflection_importance:** importance score assigned to stored reflections. **REFLECTION_MODEL** in Config: Opus vs Sonnet. |
| `src/agents/prompts.py` | Assemble the prompt using seed + memories + situation + question. | **Prompt structure:** what goes first vs last |
| `src/agents/conversation.py` | Runs conversations when there are multiple agents - multi-turn dialogue, each turn hits full cognition cycle, stores to both memory streams | **Number of conversation turns** (i.e. 4-8). **Conversation Model:** likely a cheaper model (Haiku/LLama) for casual chatter. |

---

## LLM Layer (Controls model API interactions)

| File | Purpose | Hyperparameters |
|------|---------|-----------------|
| `src/llm/client.py` | Control API calls for each method / behaviour such as `decide()` - Sonnet 4.6 for routine decisions. `reflect()` - potentially Opus for more complex belief synthesis. | **Decision model choice:** Sonnet / GPT-4o / DeepSeek-R1. **Reflection model choice:** Opus / GPT-4o/5. **Embedding model choice**. **Temperature**. **Max_tokens** |

---

## Output Layer (shows the output of the model for analysis and inspection)

| File | Purpose |
|------|---------|
| `src/output/logger.py` | Logs each simulation writing structured JSONL, one JSON object per event. Each decision log entry **must** contain four fields required for judge calls: `seed_personality` (the agent's full seed paragraph), `intervention` (the exact intervention text delivered, including channel), `decision` (what the agent does), and `reasoning` (why, given personality and memories). Without all four, the judge call cannot be made. Also logs memory additions, reflections, conversations, and state snapshots. |
| `outputs/runs/` | One .jsonl per run + config copy for reproducibility |

---

## Evaluation & Observation Layer (to help with analysis and simulation observation)

| File | Purpose |
|------|---------|
| `notebooks/run_simulation.ipynb` | Live simulation runner showing tick-by-tick rendering of agent decisions, new memories, reflections, conversations. Include helper functions such as `inspect_agent()`, `inspect_memory()`, `compare_agents()`, `show_tick_summary()` |
| `notebooks/prototype.ipynb` | Dev sandbox for initial build to test one agent, one event, one retrieval cycle. Tune retrieval weights interactively. Inspect prompts and raw LLM output. |
| `notebooks/ablation_analysis.ipynb` | Experiment 1 (ablation study). Runs 3 variants and scores each with LLM-as-a-judge. |
| `notebooks/model_comparison.ipynb` | Experiment 2 (model comparison). Runs 3 model configuration variants and scores each with LLM-as-a-judge plus cost/runtime tracking. |

---

## Evaluation Design

### LLM-as-a-Judge

Every time an intervention is executed, the agent produces a decision and its reasoning, logged as a structured JSONL entry. An LLM judge (e.g., GPT-4o, `temperature=0`) scores each entry independently on three dimensions, each on a 1–5 scale (Liu et al., 2023):

| Dimension | Description | 1 | 3 | 5 |
|---|---|---|---|---|
| **Behavioral Plausibility** | How plausible is the agent's reasoning for a homeowner in this situation? | Not at all plausible | Reasonable but generic | Reflects nuanced situational understanding |
| **Persona Consistency** | How closely does the agent's decision align with its seed personality? | Very inconsistent with seed personality | Neutral | Strongly reflects specific traits |
| **Intervention Responsiveness** | How meaningfully did the agent engage with the intervention? | Did not engage with the event | Engaged broadly but not with the details | Responded meaningfully, integrated event with prior memories and context |

All three dimensions are scored in a **single judge call**. The judge is provided:
1. The agent's seed personality paragraph
2. The intervention delivered (text + channel)
3. The agent's decision and reasoning

The judge produces **chain-of-thought reasoning before each score**. This requires each JSONL decision entry to include `seed_personality`, `intervention`, `decision`, and `reasoning` (see Output Layer above).

---

### Experiment 1: Ablation Study

**Goal:** Determine which components of the agent cognition architecture contribute to behavioral realism, using 3 variants (following Park et al., 2023).

| Variant | Memory stream + retrieval | Reflection | Description |
|---|---|---|---|
| **Variant 1** (full) | Yes | Yes | Complete system — memories accumulate with importance scores and embeddings, top-K retrieval, reflections fire when cumulative importance exceeds threshold |
| **Variant 2** (no reflection) | Yes | No | Agents retrieve memories but never synthesise higher-level beliefs |
| **Variant 3** (no memory) | No | No | Agents operate from seed personality only; no memory stream, no retrieval, no reflection |

Note: reflection depends on accumulated memories, so removing the memory stream (Variant 3) implicitly removes reflection. Variants are implemented as flags in the simulation loop.

**Constants:** seed personality paragraphs (from interview data), scenario YAML (intervention sequence, timing, channels), agent count, social structure, LLM model, prompt templates, embedding model (`text-embedding-3-small`), retrieval parameters (top-K count).

**Metrics:** LLM-as-a-judge scores (Behavioral Plausibility, Persona Consistency, Intervention Responsiveness) for each agent's decision at each scenario event, averaged across agents and events to compare the three variants.

---

### Experiment 2: Model Comparison

**Goal:** Understand the cost–quality tradeoff between larger and smaller models for agent cognition. Hou et al. (2025) found simulation performance varied substantially across models, motivating a controlled comparison.

| Variant | Reflection model | Decision model | Notes |
|---|---|---|---|
| **Variant 1** (baseline) | Larger (e.g., Claude Sonnet 4.6) | Smaller (e.g., Claude Haiku, DeepSeek-R1) | Mixed configuration — larger model for belief synthesis, smaller for routine calls |
| **Variant 2** | Smaller | Smaller | All calls use the smaller model |
| **Variant 3** (budget permitting) | Larger | Larger | All calls use the larger model |

Variant 1 is the baseline. Variant 2 tests degradation under a small model only. Variant 3 (if compute budget allows) tests whether upgrading routine decisions to the larger model meaningfully improves quality beyond Variant 1.

**Constants:** Full cognitive architecture (memory, retrieval, reflection), seed personality paragraphs, scenario YAML, agent count, social structure, prompt templates, embedding model (`text-embedding-3-small`), retrieval parameters (top-K count).

**Metrics:** LLM-as-a-judge scores averaged per agent per intervention (quality metric). API cost and runtime (per agent, per tick, across simulation) quantify efficiency. Each agent makes multiple LLM calls per tick, so costs scale quickly across variants.

---
