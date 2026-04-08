# Stage 2 Changes & Usage Notes

## What changed from Stage 1

Stage 1 built a single working prototype: load Jennifer, seed memories, run one event, get a decision, run an LLM-as-judge comparison against her held-out interview response.

Stage 2 adds three things on top of that:
1. **Retrieval** — instead of passing all memories to every LLM call, memories are scored and the most relevant ones are selected
2. **Reflection** — after enough important events accumulate, the agent synthesises higher-level beliefs which become memories themselves
3. **Validation framework** — a structured notebook for testing and tuning each component, with results exportable to CSV and Google Sheets

Reflection was moved from Stage 5 (original plan) to Stage 2 because reflections add high-importance memories back into the stream, which directly affects retrieval. You cannot tune retrieval properly without knowing how reflections interact with it.

---

## Files added or changed

### `src/agents/retrieval.py` — NEW

Parameterised memory retrieval. The core function is `retrieve_memories()`, which scores every memory in the stream on three components (following Park et al. 2023) and returns the top-K:

```
score = (recency_weight × recency) + (importance_weight × importance) + (relevance_weight × relevance)
```

All three components are min-max normalised to [0, 1] before weighting, so no single component dominates by scale.

**`RetrievalConfig`** is the single place to change all retrieval parameters:

| Parameter | Default | What it does |
|---|---|---|
| `top_k` | 12 | How many memories to return |
| `recency_weight` | 1.0 | How much recent memories are favoured |
| `importance_weight` | 1.0 | How much LLM-scored importance matters |
| `relevance_weight` | 1.0 | How much semantic similarity to the query matters |
| `recency_decay` | 0.99 | Exponential decay rate per simulation day |
| `retrieval_mode` | `"dense"` | `"dense"` (embeddings), `"sparse"` (keyword overlap), or `"hybrid"` |
| `sparse_weight` | 0.5 | In hybrid mode: weight given to sparse score (dense gets `1 - sparse_weight`) |

**Dense vs sparse:**
- Dense uses cosine similarity on sentence-transformer embeddings — good for semantic matches ("insurance" ↔ "coverage")
- Sparse uses Jaccard token overlap — good for literal keyword matches, no extra dependencies
- Hybrid combines both

**`last_accessed` on Memory objects:**
When `retrieve_memories()` returns a memory, it updates `last_accessed` to the current simulation day. Recency decay is calculated from `last_accessed`, not `timestamp` — so frequently retrieved memories stay fresh even if they were originally created a long time ago.

**How to use:**
```python
from src.agents.retrieval import RetrievalConfig, retrieve_memories

rc = RetrievalConfig(top_k=8, relevance_weight=2.0, retrieval_mode="dense")
query_embed = embed("insurance non-renewal letter")
retrieved = retrieve_memories(stream, query_embed, query_text, rc, current_day=42)
```

---

### `src/agents/reflection.py` — NEW

Two-step reflection following Park et al. (2023). Fires when the cumulative importance of unprocessed memories exceeds a threshold.

**Step 1:** Generate N high-level questions from recent memories
> *"What is Jennifer most concerned about right now?"*

**Step 2:** For each question, retrieve relevant memories and synthesise a first-person insight
> *"I am increasingly worried that insurance decisions feel arbitrary — the non-renewal letter and my neighbor's experience (memories 3, 7) suggest compliance alone is not enough."*

Each insight is stored as a `reflection` type memory with `importance = reflection_importance` (default 8). These re-enter the stream and can be retrieved like any other memory.

**`ReflectionConfig`** controls all parameters:

| Parameter | Default | What it does |
|---|---|---|
| `threshold` | 100.0 | Cumulative importance of unprocessed memories needed to trigger reflection |
| `num_questions` | 3 | How many introspective questions to generate (Park et al. used 3) |
| `memories_for_questions` | 10 | How many recent memories are shown to the question generator |
| `top_k_per_question` | 8 | Memories retrieved per question for synthesis |
| `reflection_importance` | 8 | Importance score assigned to stored reflection memories |

**`maybe_reflect()`** returns `(new_reflections, new_last_reflection_index)`. Pass `new_last_reflection_index` back on the next call so the cumulative importance check starts from where it left off.

**How to use:**
```python
from src.agents.reflection import ReflectionConfig, maybe_reflect

rc = ReflectionConfig(threshold=80.0, num_questions=3)
new_reflections, last_idx = maybe_reflect(
    stream=jennifer_memory,
    client_anthropic=client_anthropic,
    config=config,
    reflection_config=rc,
    agent_seed=jennifer_config['seed_narrative'],
    current_day=42,
    last_reflection_index=0,  # update this each call
)
```

---

### `src/llm/client.py` — UPDATED

Three changes:

**1. Config now has four clearly separated roles:**

```python
config = Config(
    DECISION_MODEL    = "claude-sonnet-4-6",   # agent decisions
    REFLECTION_MODEL  = "claude-sonnet-4-6",   # belief synthesis (swap to Opus for quality)
    JUDGE_MODEL       = "claude-opus-4-6",     # evaluation — keep strong, always temp=0
)
```

**2. Two judge functions added:**

- `judge_retrieval(client, config, all_memories, intervention, retrieved_memories)` — given the full memory stream, the intervention, and the top-K retrieved, returns `relevance_score` (1–10), `missed_memories` (list), and `critique`
- `judge_generation(client, config, simulated, real_response, context)` — compares simulated response to real held-out response, returns `character_fidelity`, `decision_alignment`, `reasoning_alignment`, `overall_score` (all 1–10), and `critique`

Both return structured dicts. Judge calls always use `JUDGE_TEMPERATURE = 0.0` for reproducible scores.

**3. OpenRouter support added:**

Any model name not starting with `claude-` is automatically routed to OpenRouter (OpenAI-compatible API). This lets us test GPT-4o, DeepSeek-R1, Llama, etc. without changing any function signatures.

```python
# In Colab secrets: add OPENROUTER_API_KEY
client_openrouter = init_openrouter_client()

# Then pass it alongside client_anthropic
config.DECISION_MODEL = "openai/gpt-4o"
decide(client_anthropic, system_prompt, user_prompt, client_openrouter=client_openrouter)
```

All functions (`decide`, `reflect`, `score_importance`, `judge_retrieval`, `judge_generation`) accept `client_openrouter=None` as an optional argument. If not passed, routing still works for Claude models. `prototype-2.ipynb` is unchanged.

---

### `src/agents/memory.py` — UPDATED

`Memory` dataclass now has a `last_accessed` field (defaults to `timestamp` on creation). Used by `retrieval.py` for recency decay. Included in `to_dict()` output.

---

### `notebooks/stage2_validation.ipynb` — NEW

The experiment harness for Stage 2. Five sections run in sequence — each section builds on the last.

**Section 1: Memory Seeding**
LLM-as-judge reviews the seeded memory stream against each held-out scenario. Tells you if Jennifer's seeds give her enough context to respond authentically before any events happen.

**Section 2: Retrieval Evaluation**
Define a list of `RetrievalConfig` variants, run them against each held-out scenario, and compare judge scores. A summary table shows which config got the highest relevance scores. This is where you tune top-K, weights, and retrieval mode.

**Section 3: Reflection Evaluation**
Feeds Jennifer the full baseline scenario (all events), then tests different `ReflectionConfig` variants. The judge scores the resulting reflections on insight quality and accuracy.

**Section 4: Generation Evaluation**
Full pipeline (seed → retrieve → decide) for each held-out scenario. Judge compares simulated response to real interview response on character fidelity, decision alignment, and reasoning alignment.

**Section 5: Ablations**
Runs four conditions — `full`, `no_memory`, `no_retrieval_scoring`, `no_reflection` — and compares generation scores. Shows the contribution of each component.

**Results export:**
After running, cell 33 saves results as JSONL and CSV (timestamped, in `outputs/eval/`), and displays an inline DataFrame table. Cell 35 (optional) pushes to a new Google Sheet with one tab per result type and prints the shareable URL.

---

## How the three config objects relate

Each config object has a completely separate responsibility:

| Config | File | Controls | Used by |
|---|---|---|---|
| `Config` | `client.py` | All LLM model choices and call settings | `decide()`, `reflect()`, `judge_*()`, `score_importance()` |
| `RetrievalConfig` | `retrieval.py` | Retrieval scoring (top-K, weights, mode) | `retrieve_memories()` |
| `ReflectionConfig` | `reflection.py` | Reflection behaviour (threshold, questions) | `maybe_reflect()` |

They do not interact. Each function only reads from its own config. To change which LLM the agent uses, edit `Config`. To change how memories are scored and retrieved, edit `RetrievalConfig`. To change when and how reflection fires, edit `ReflectionConfig`.

---

## Data split

| Role | Agent | Status |
|---|---|---|
| Dev | Jennifer | Built — used for all Stage 2 work |
| Validation | Lola | Transcript set aside — build YAML when needed for hyperparameter validation |
| Test | Beth | Transcript set aside — build YAML last, do not evaluate until final |

Synthetic agents (generated from a different LLM using the same YAML format) are deferred until after Stage 2 validation is stable. They will expand the evaluation dataset for retrieval tuning.

---

## Tuning order

Run `stage2_validation.ipynb` sections in order:

1. Check memory seeding scores — if adequacy is low, add seeds to `jennifer.yaml`
2. Tune retrieval — adjust `RetrievalConfig` variants until relevance scores are consistently high
3. Set `best_retrieval_cfg` in Section 3 to the winner from Section 2
4. Tune reflection — adjust `ReflectionConfig` variants; higher `num_questions` gives richer beliefs but costs more
5. Check generation scores — if overall score is low, consider tuning `DECISION_MODEL` or the prompt in `prompts.py`
6. Run ablations — if removing reflection has no impact on generation score, reflection may need further tuning before it adds value

---

## Next steps

**Status:** Initial Jennifer validation complete. Now refining params across Lola + synthetic agents before locking and moving to the simulation loop.

---

### Step A: Build Lola's agent YAML ✅ (do next)

Lola's transcript is set aside as the validation agent. Build `config/agents/lola.yaml` using the same structure as `jennifer.yaml` (seed narrative, core beliefs, memory seeds, held-out responses). This is the ground-truth agent for cross-validation.

---

### Step B: Cross-agent retrieval and reflection tuning

Run `stage2_validation.ipynb` on each of the following agents using the candidate configs from the Jennifer run. The goal is to find configs that score consistently well across all agents — not just Jennifer.

**Agents to test (in order of priority):**
1. **Lola** (`lola.yaml`) — ground-truth validation agent; scores here are the primary signal
2. **Synthetic_Tester1** — financially constrained, partial compliance; stresses retrieval of cost/grant memories
3. **Synthetic_Tester4** — active denier; tests whether reflection correctly surfaces resistant beliefs rather than compliant ones
4. **Synthetic_Tester5** — compliant but insurance dropped; the most complex memory stream (pride → betrayal → withdrawal); stress-tests reflection accuracy
5. **Synthetic_Tester3** — uninformed newcomer; few seeds, low-importance memories; tests retrieval when the stream is sparse

Testers 2 (passive complier) is lower priority — their memory stream is simple and unlikely to surface edge cases.

**What to look for across agents:**
- Retrieval relevance scores should stay above ~7/10 across all agent types, not just Jennifer
- Reflection accuracy score (currently ~7/10 on Jennifer), identify any ways to improve this
- Generation scores on Lola are the key number — this is real interview data
- Continue to refine with agent seed changes by Madeleine and Amrita

**How to run:** Swap `AGENT_CONFIG_PATH` in `stage2_validation.ipynb` to point at each agent YAML in turn. Export results to CSV after each run so scores are comparable across agents.

---

### Step C: Lock in parameters

Once scores are stable across Lola and at least 2–3 synthetic agents, record the winning configs:

- **`RetrievalConfig`:** top-K, weights, retrieval_mode — hardcode as the default in the notebook
- **`ReflectionConfig`:** threshold, num_questions
- **Generation baseline:** Lola's overall score — this is the number Stage 3 must not regress on

Write the locked values here (fill in after running):

```
Best RetrievalConfig:  top_k=__ recency_weight=__ importance_weight=__ relevance_weight=__ mode=__
Best ReflectionConfig: threshold=__ num_questions=__
Jennifer overall score: __/10
Lola overall score:     __/10
```

---

### Step D: Stage 3 — simulation loop

Once params are locked, Stage 3 adds the infrastructure to run a full multi-agent simulation. Files to build (in order):

1. **`src/agents/agent.py`** — `Agent` class wrapping the cognition cycle: `perceive()` → `retrieve()` → `decide()` → `act()` → `store()`. Each agent holds a reference to its `MemoryStream`, its config YAML, and current state (compliance status, attitude, actions taken).

2. **`src/engine/scheduler.py`** — `EventScheduler` class. Loads the intervention schedule from `baseline.yaml` into a heapq priority queue. Method: `get_events(tick)` returns all events due on that day.

3. **`src/environment/channels.py`** — Channel framing. Takes a raw event and wraps it in channel-specific language. The four channels: `official_mail`, `news_media`, `social`, `direct_experience`.

4. **`src/engine/simulation.py`** — The tick loop. Per tick: get events → frame via channels → run cognition cycle for each targeted agent → check reflection thresholds → log. Keep it thin; heavy logic lives in `agent.py`.

5. **`src/output/logger.py`** — Structured JSONL logging. One entry per event: `tick`, `agent`, `event_type`, `channel`, `retrieved_memories`, `decision`, `reasoning`, `new_memories`. The `decision` / `reasoning` split lets the judge score them independently.

6. **`config/scenarios/baseline.yaml`** — Scenario definition: duration, tick interval, intervention schedule (`{day, type, channel, target_agents, content}`). Start with 3 agents and a 30-day timeline.

7. **`notebooks/run_simulation.ipynb`** — Runner notebook. Tick-by-tick inline summaries (decisions, new memories, reflections fired). Helper functions: `inspect_agent()`, `inspect_memory()`, `compare_agents()`.

Build and test with Jennifer + Synthetic_Tester1 first (contrasting compliance profiles). Check token counts per tick before scaling to more agents.

---

### Step E: After simulation loop works — social network

`src/environment/network.py` and `src/agents/conversation.py` are Stage 4. Don't build until `run_simulation.ipynb` produces clean JSONL for at least a 3-agent, 30-day run.
