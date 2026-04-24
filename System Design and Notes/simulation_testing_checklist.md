# Simulation Testing Checklist

Run these checks in order when testing `run_simulation.ipynb` for the first time,
or after any changes to agents, scenarios, or source files.
Don't move to the next section until the current one passes.

---

## 1. Import and setup

Run the setup cells in Section 1.

- [ ] No `ModuleNotFoundError` — if you see one, the working directory isn't set to the project root
- [ ] Anthropic client initialises and prints confirmation
- [ ] Embedding model downloads and prints "Embedding model loaded" (first run downloads ~90 MB, subsequent runs are instant)

---

## 2. Agent YAML loading (Section 3 preview — no LLM calls)

Run the agent preview cell.

- [ ] Both agents display `id`, `display_name`, seed narrative, and key concerns
- [ ] Margaret shows **52 memory seeds**, Eleanor shows **10**
- [ ] No `KeyError: 'seed_narrative'` — if you see that, the YAML is in the wrong format (must have `agents: [...]` wrapper with `id`, `display_name`, `seed_narrative`, `memory_seeds`)

---

## 3. Scenario preview (Section 3 — no LLM calls)

Run the event timeline cell.

- [ ] **7 events** displayed across 60 days
- [ ] Day 21 targets `Margaret` only, day 24 targets `Eleanor` only, the rest target `all`
- [ ] Content previews look correct — not `None` or truncated mid-word

---

## 4. Memory seeding (Section 4 initialise)

First cell that uses the embedding model. Watch the output carefully.

- [ ] Prints "Seeding 52 memories..." for Margaret, "Seeding 10 memories..." for Eleanor
- [ ] No `ValueError: Memory type 'X' must be one of...` — if you see this, a seed has an invalid type
- [ ] Final counts match: Margaret 52, Eleanor 10
- [ ] Spot check: call `inspect_memory('Margaret', n=3)` — descriptions should be real seed text, not empty strings
- [ ] All importance values are 1–10 (seeds have hardcoded importances, so no LLM calls needed here)

---

## 5. Scheduler behaviour (step through early days)

Step through days 1–9 (no events until day 10).

- [ ] Days 1–9 print "Day X: no events scheduled" — confirms the loop runs cleanly on empty ticks
- [ ] Day 1 fires the `wildfire_news` event — confirm it appears when you step to day 1

---

## 6. First real event — Day 1 (wildfire_news, channel: news_media, target: all)

Step to day 1. This is the first LLM call.

- [ ] Both Margaret and Eleanor respond — both names appear in the output
- [ ] Each response shows "Memories retrieved: N" where N > 0 (expect 8–12)
- [ ] **Decision field is not empty** — if it's blank every time, the model isn't returning valid JSON; add `print(raw)` temporarily in `agent._parse_response()` to see the raw output
- [ ] Reasoning references something specific to their personality (hills location, wildfire risk, cost concerns) — not just "I will review my defensible space plan"

---

## 7. Official mail event — Day 10 (defensible_space_law, channel: official_mail, target: all)

- [ ] Framing in the prompt starts "You received an official letter in the mail" — confirms channel framing is working
- [ ] Margaret and Eleanor give **different** reasoning — Margaret tends toward frustration with the system; Eleanor toward pragmatic compliance
- [ ] Memory count per agent: seeds + 2 new per event (1 observation + 1 decision) → Margaret ~56, Eleanor ~14

---

## 8. Retrieval quality (critical check)

After day 10, call `show_tick_summary(10)` and inspect the retrieved memories section.

- [ ] Memories are **topically relevant** — for a defensible space law letter, expect memories about compliance, inspections, and costs; not unrelated ones like "I have a parrot"
- [ ] At least some retrieved memories have importance ≥ 7 — high-importance memories should surface preferentially
- [ ] If retrieval looks random, check that `retrieval_config.relevance_weight` is set (default 2.0) and the embedding model loaded correctly

---

## 9. Social channel event — Day 21 (fire_service_doorknock, target: Margaret only)

- [ ] **Only Margaret responds** — Eleanor should not appear in this tick
- [ ] Social framing starts "While outside, you notice or hear from a neighbour"
- [ ] Margaret's reasoning references community or social context rather than official authority

---

## 10. Eleanor-only event — Day 24 (firewise_outreach, target: Eleanor only)

- [ ] **Only Eleanor responds** — confirms `target_agents: Eleanor` routing works
- [ ] If you see "Warning: target 'Eleanor' not loaded", the `display_name` in `eleanor_v2.yaml` doesn't match — it must be `display_name: Eleanor` exactly

---

## 11. Reflection firing check

After running through all events (at least to day 55), call `show_all_reflections()`.

- [ ] If nothing has fired by day 55, the threshold (100) may be too high for the memories accumulated — lower it to 70 in `reflection_config` and re-run from Section 4
- [ ] If reflection fires, each insight should be a coherent first-person belief (not a JSON fragment or broken sentence)
- [ ] Each reflection should add 3 new memories with `type: reflection` — check with `inspect_memory('Margaret', memory_type='reflection')`
- [ ] Reflection memories have importance 8 (hardcoded in `ReflectionConfig.reflection_importance`)

---

## 12. Ablation flags (Experiment 1 variants)

Re-run from Section 4 with each variant. Change only the flags in Section 2.

**Variant 3 — no memory** (`USE_MEMORY=False, USE_REFLECTION=False`):
- [ ] "Memories retrieved: 0" for every event
- [ ] Decisions are shorter and more generic — no references to past experiences
- [ ] `show_all_reflections()` returns "No reflections have fired"

**Variant 2 — no reflection** (`USE_MEMORY=True, USE_REFLECTION=False`):
- [ ] Memories are retrieved normally (N > 0 per event)
- [ ] `show_all_reflections()` returns "No reflections have fired" — reflection is disabled

**Variant 1 — full** (`USE_MEMORY=True, USE_REFLECTION=True`):
- [ ] Memories retrieved and reflection fires if threshold met
- [ ] Variant 1 decisions should be the most grounded and personality-specific of the three

---

## 13. JSONL output integrity

After running at least a few events, verify the log file.

- [ ] File exists at `outputs/runs/<run_id>.jsonl`
- [ ] First line is a `run_config` entry with scenario path, agent names, and ablation flags
- [ ] Every `decision` entry has all four judge-required fields: `seed_personality`, `intervention`, `decision`, `reasoning` — none should be `null`
- [ ] `retrieved_memories` is a list of dicts (not raw Memory objects)
- [ ] File is valid JSONL — each line parses independently:

  ```bash
  python -c "import json; [json.loads(l) for l in open('outputs/runs/YOURRUNID.jsonl')]; print('valid')"
  ```

---

## 14. Agent divergence (main qualitative check)

After running all 7 events, call `compare_agents('insurance_non_renewal')` (day 42).

- [ ] Margaret and Eleanor give **meaningfully different** responses — Margaret's reasoning should reference resentment at system inconsistency and cost burden; Eleanor's should lean toward documentation and pragmatic action
- [ ] If both responses are nearly identical generic text, retrieval isn't surfacing personality-specific memories — check the top retrieved memories for each agent with `show_tick_summary(42)`
- [ ] Decision fields are not empty for either agent on this event

---

## 15. Cost and token estimate

After a full run, do a quick check before scaling up.

- [ ] Count decision entries in the log:
  ```bash
  grep -c '"entry_type": "decision"' outputs/runs/<run_id>.jsonl
  ```
  Should be 10–11 total (7 events × 1–2 agents each)
- [ ] Each decision call is roughly 1,000–2,000 tokens in + 200–500 tokens out
- [ ] Full run should cost under $0.10 on Sonnet 4.6 — if significantly higher, check that importance scoring isn't being called unnecessarily (seeds should all have hardcoded scores)
