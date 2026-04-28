"""
client.py — LLM API interaction layer.

Supports two providers:
  - Anthropic (direct SDK) — for Claude models
  - OpenRouter (OpenAI-compatible SDK) — for GPT-4o, DeepSeek-R1, Llama, Mistral, etc.
    Also supports Claude via OpenRouter if preferred.

Routing is automatic based on model name:
  - Model starts with "claude-"  → Anthropic SDK
  - Anything else               → OpenRouter (openai SDK with custom base_url)

Setup:
  client_anthropic  = init_clients()             # always call this
  client_openrouter = init_openrouter_client()   # optional; only needed for non-Claude models

In the notebook, all functions accept client_openrouter=None as an optional arg.
prototype-2.ipynb works unchanged — it never passes client_openrouter.

Config pattern adapted from course studio notebooks (Cornelia Paulik, INFO 290).
Embedding model: all-MiniLM-L6-v2 (HuggingFace, free, no API key needed).
"""

import os
import time
import json
import anthropic
import numpy as np
from dataclasses import dataclass, field
from typing import List

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model pricing (USD per million tokens) ─────────────────────────────────────
# Source: https://docs.anthropic.com/en/docs/about-claude/models/overview (verified Apr 2026)
MODEL_PRICING = {
    "moonshotai/kimi-k2.6":        (0.7448, 4.655),
    "claude-sonnet-4-6":           (3.00,   15.00),
    "claude-opus-4-6":             (5.00,   25.00),
    "claude-haiku-4-5-20251001":   (1.00,    5.00),
}


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Agent decision model ──────────────────────────────────────────────────
    # Anthropic:   "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
    # OpenRouter:  "moonshotai/kimi-k2.6", "openai/gpt-4o", "deepseek/deepseek-r1"
    DECISION_MODEL: str = "claude-sonnet-4-6"
    DECISION_MAX_TOKENS: int = 1024
    DECISION_TEMPERATURE: float = 0.7

    # ── Reflection model (higher reasoning quality; Opus recommended) ─────────
    REFLECTION_MODEL: str = "claude-haiku-4-5-20251001"
    REFLECTION_MAX_TOKENS: int = 512
    REFLECTION_QUESTION_MAX_TOKENS: int = 2048
    REFLECTION_TEMPERATURE: float = 0.4

    # ── LLM-as-judge model (evaluation only; keep strong + deterministic) ─────
    JUDGE_MODEL: str = "claude-opus-4-6"
    JUDGE_MAX_TOKENS: int = 1024
    JUDGE_TEMPERATURE: float = 0.0   # deterministic → reproducible eval scores

    # ── Importance scoring ────────────────────────────────────────────────────
    IMPORTANCE_MAX_TOKENS: int = 5
    IMPORTANCE_TEMPERATURE: float = 0.0

    # ── Embeddings (always local HuggingFace — no provider needed) ───────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Output verbosity ──────────────────────────────────────────────────────
    # When True, injects a conciseness instruction into every decide() call.
    # Set to True for evaluation runs to keep logged decision/reasoning readable.
    CONCISE_OUTPUT: bool = False


config = Config()


# ── Usage tracking ─────────────────────────────────────────────────────────────

@dataclass
class UsageTracker:
    """
    Accumulates token counts across all LLM calls in a run.

    Reset at the start of each simulation run, then read at the end to
    produce the cost summary logged to the JSONL and exported to Excel.

    Agent calls: decide(), reflect(), score_importance()
    Judge calls: judge_simulation(), judge_full_simulation()
    """
    agent_tokens_in:  int = 0
    agent_tokens_out: int = 0
    judge_tokens_in:  int = 0
    judge_tokens_out: int = 0

    def reset(self):
        self.agent_tokens_in  = 0
        self.agent_tokens_out = 0
        self.judge_tokens_in  = 0
        self.judge_tokens_out = 0

    def add(self, call_type: str, tokens_in: int, tokens_out: int):
        if call_type == "agent":
            self.agent_tokens_in  += tokens_in
            self.agent_tokens_out += tokens_out
        elif call_type == "judge":
            self.judge_tokens_in  += tokens_in
            self.judge_tokens_out += tokens_out

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        in_rate, out_rate = MODEL_PRICING.get(model, (0.0, 0.0))
        return (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000

    def to_dict(self, agent_model: str, judge_model: str = "claude-opus-4-6") -> dict:
        agent_cost = self._cost(agent_model, self.agent_tokens_in, self.agent_tokens_out)
        judge_cost = self._cost(judge_model, self.judge_tokens_in, self.judge_tokens_out)
        return {
            "agent_model":      agent_model,
            "agent_tokens_in":  self.agent_tokens_in,
            "agent_tokens_out": self.agent_tokens_out,
            "agent_cost_usd":   round(agent_cost, 6),
            "judge_tokens_in":  self.judge_tokens_in,
            "judge_tokens_out": self.judge_tokens_out,
            "judge_cost_usd":   round(judge_cost, 6),
            "total_cost_usd":   round(agent_cost + judge_cost, 6),
        }


# Module-level tracker — reset before each run, read after
usage_tracker = UsageTracker()

_embedding_model = None


# ── Client initialisation ──────────────────────────────────────────────────────

def init_clients():
    """
    Initialise the Anthropic client. Always call this.
    Returns the Anthropic client object.
    """
    try:
        from google.colab import userdata
        anthropic_key = userdata.get("ANTHROPIC_API_KEY")
    except Exception:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found.\n"
            "In Colab: Runtime -> Secrets -> Add ANTHROPIC_API_KEY"
        )

    client = anthropic.Anthropic(api_key=anthropic_key)
    print("Anthropic client initialised")
    print("Embeddings: HuggingFace all-MiniLM-L6-v2 (free, no API key needed)")
    return client


def init_openrouter_client():
    """
    Initialise the OpenRouter client (optional).
    Only needed when using non-Claude models (GPT-4o, DeepSeek, Llama, etc.).

    Requires OPENROUTER_API_KEY in Colab secrets or environment variables.
    Get a key at openrouter.ai — pay-per-use, no subscription needed.

    Returns an openai.OpenAI client pointed at OpenRouter's API.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: !pip install openai")

    try:
        from google.colab import userdata
        key = userdata.get("OPENROUTER_API_KEY")
    except Exception:
        key = os.environ.get("OPENROUTER_API_KEY")

    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found.\n"
            "In Colab: Runtime -> Secrets -> Add OPENROUTER_API_KEY\n"
            "Get a key at: openrouter.ai"
        )

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    print("OpenRouter client initialised")
    print(f"Supported models: openai/gpt-4o, deepseek/deepseek-r1, meta-llama/llama-3.3-70b-instruct, ...")
    return client


# ── Internal routing ───────────────────────────────────────────────────────────

def _is_anthropic_model(model: str) -> bool:
    """Claude models go to Anthropic SDK; everything else goes to OpenRouter."""
    return model.startswith("claude-")


def _call_llm(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    client_anthropic,
    client_openrouter=None,
    call_type: str = "agent",   # "agent" | "judge" — controls usage_tracker bucket
) -> str:
    """
    Internal routing function. All LLM calls go through here.

    Routes to Anthropic SDK for Claude models, OpenRouter for everything else.
    Handles the API format difference between providers transparently:
      - Anthropic: system is a separate top-level param
      - OpenRouter: system is a message with role="system"

    Also records token usage in the module-level usage_tracker.
    call_type="agent" for decisions/reflection/importance; "judge" for judge calls.
    """
    if _is_anthropic_model(model):
        message = client_anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage_tracker.add(call_type, message.usage.input_tokens, message.usage.output_tokens)
        return message.content[0].text

    else:
        if client_openrouter is None:
            raise ValueError(
                f"Model '{model}' requires OpenRouter.\n"
                "Call init_openrouter_client() and pass client_openrouter= to this function."
            )
        response = client_openrouter.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        u = response.usage
        usage_tracker.add(call_type, u.prompt_tokens, u.completion_tokens)
        content = response.choices[0].message.content
        if content is None:
            finish = response.choices[0].finish_reason
            print(f"[client] WARNING: {model} returned None content (finish_reason={finish!r}). Returning empty string.")
        return content or ""


# ── Embedding ──────────────────────────────────────────────────────────────────

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model ({config.EMBEDDING_MODEL})...")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Embedding model loaded")
    return _embedding_model


def embed(text: str) -> np.ndarray:
    """Embed text using local HuggingFace model. No API call needed."""
    text = text.replace("\n", " ").strip()
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)


# ── Agent calls ────────────────────────────────────────────────────────────────

_CONCISE_SUFFIX = (
    "\n\nIMPORTANT: Keep both the decision and reasoning fields concise "
    "— 1-2 sentences each. No padding or elaboration."
)


def decide(client_anthropic, system_prompt: str, user_prompt: str, client_openrouter=None) -> str:
    """
    Routine agent decision call.
    Uses config.DECISION_MODEL — swap this in Config to change the agent's LLM.
    When config.CONCISE_OUTPUT is True, injects a brevity instruction into the prompt.
    """
    if config.CONCISE_OUTPUT:
        user_prompt = user_prompt + _CONCISE_SUFFIX
    return _call_llm(
        model=config.DECISION_MODEL,
        system=system_prompt,
        user=user_prompt,
        max_tokens=config.DECISION_MAX_TOKENS,
        temperature=config.DECISION_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="agent",
    )


def reflect(client_anthropic, system_prompt: str, memories_text: str, client_openrouter=None) -> str:
    """
    Belief synthesis call. Used by reflection.py.
    Uses config.REFLECTION_MODEL — swap to Opus for higher quality reflections.
    """
    reflection_prompt = (
        "Based on the following recent experiences and observations, "
        "what higher-level belief or insight have you formed? "
        "Express it as a first-person statement in 1-3 sentences.\n\n"
        f"Recent memories:\n{memories_text}"
    )
    return _call_llm(
        model=config.REFLECTION_MODEL,
        system=system_prompt,
        user=reflection_prompt,
        max_tokens=config.REFLECTION_MAX_TOKENS,
        temperature=config.REFLECTION_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="agent",
    )


def score_importance(client_anthropic, description: str, agent_seed: str, max_retries: int = 3, client_openrouter=None) -> int:
    """
    Score the importance of a memory (1–10) via LLM.
    Uses config.DECISION_MODEL (short call, same model as decisions).
    """
    system = (
        f"{agent_seed}\n\n"
        "You are rating the importance of events and observations in your life "
        "on a scale of 1 to 10, where 1 is mundane (e.g., making coffee) and "
        "10 is life-altering (e.g., losing your home). "
        "Respond with ONLY a single integer from 1 to 10, nothing else."
    )
    user = f"How important is this event to you? '{description}'"

    for attempt in range(max_retries):
        raw = _call_llm(
            model=config.DECISION_MODEL,
            system=system,
            user=user,
            max_tokens=config.IMPORTANCE_MAX_TOKENS,
            temperature=config.IMPORTANCE_TEMPERATURE,
            client_anthropic=client_anthropic,
            client_openrouter=client_openrouter,
            call_type="agent",
        )
        try:
            return max(1, min(10, int(raw.strip())))
        except ValueError:
            print(f"[client] Attempt {attempt+1}: unexpected score '{raw}', retrying...")

    print(f"[client] Warning: all {max_retries} attempts failed, defaulting to 5")
    return 5


# ── Evaluation calls ───────────────────────────────────────────────────────────

def judge_retrieval(client_anthropic, config: Config, all_memories, intervention: str, retrieved_memories, client_openrouter=None) -> dict:
    """
    LLM-as-judge for retrieval quality.

    Given the full memory stream, the intervention, and the top-K retrieved memories,
    asks the judge whether the right memories surfaced and what was missed.

    Returns dict: relevance_score (1-10), missed_memories (list), critique (str)
    """
    all_text = "\n".join(
        f"[{i+1}] Day {m.timestamp} | {m.type} | importance {m.importance}/10 | {m.description}"
        for i, m in enumerate(all_memories)
    )
    retrieved_text = "\n".join(
        f"[{i+1}] Day {m.timestamp} | {m.type} | importance {m.importance}/10 | {m.description}"
        for i, m in enumerate(retrieved_memories)
    )
    system = (
        "You are an expert evaluator assessing the quality of a memory retrieval system "
        "for a generative agent simulation. Be critical and specific."
    )
    user = (
        f"FULL MEMORY STREAM:\n{all_text}\n\n"
        f"INTERVENTION (what the agent is responding to):\n{intervention}\n\n"
        f"TOP-K RETRIEVED MEMORIES:\n{retrieved_text}\n\n"
        "Evaluate whether the retrieved memories are the most relevant ones for this decision.\n\n"
        "Respond in this exact JSON format:\n"
        '{\n  "relevance_score": <integer 1-10>,\n'
        '  "missed_memories": [<description of missed memory>, ...],\n'
        '  "critique": "<1-2 sentence assessment>"\n}'
    )
    raw = _call_llm(
        model=config.JUDGE_MODEL,
        system=system,
        user=user,
        max_tokens=config.JUDGE_MAX_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="judge",
    )
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"relevance_score": None, "missed_memories": [], "critique": raw}


def judge_generation(client_anthropic, config: Config, simulated_response: str, real_response: str, context: str, client_openrouter=None) -> dict:
    """
    LLM-as-judge for generation quality.

    Compares simulated response to the real held-out interview response.

    Returns dict: character_fidelity, decision_alignment, reasoning_alignment,
                  overall_score (all 1-10), critique (str)
    """
    system = (
        "You are an expert evaluator comparing simulated agent responses to real interview responses "
        "in a wildfire homeowner simulation study. Be critical and specific."
    )
    user = (
        f"AGENT CONTEXT:\n{context}\n\n"
        f"REAL INTERVIEW RESPONSE:\n{real_response}\n\n"
        f"SIMULATED RESPONSE:\n{simulated_response}\n\n"
        "Score the simulated response against the real response on three dimensions.\n\n"
        "Respond in this exact JSON format:\n"
        '{\n  "character_fidelity": <integer 1-10>,\n'
        '  "decision_alignment": <integer 1-10>,\n'
        '  "reasoning_alignment": <integer 1-10>,\n'
        '  "overall_score": <integer 1-10>,\n'
        '  "critique": "<2-3 sentences: key similarities and differences>"\n}'
    )
    raw = _call_llm(
        model=config.JUDGE_MODEL,
        system=system,
        user=user,
        max_tokens=config.JUDGE_MAX_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="judge",
    )
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "character_fidelity": None, "decision_alignment": None,
            "reasoning_alignment": None, "overall_score": None, "critique": raw,
        }


def _format_memory_seeds(memory_seeds: list) -> str:
    """Format memory seeds as a compact numbered list for judge context."""
    if not memory_seeds:
        return "(none)"
    return "\n".join(
        f"[{i+1}] {s['description'].strip()}"
        for i, s in enumerate(memory_seeds)
    )


def judge_intervention(
    client_anthropic,
    config: "Config",
    seed_narrative: str,
    memory_seeds: list,
    intervention: str,
    decision: str,
    reasoning: str,
    client_openrouter=None,
) -> dict:
    """
    LLM-as-judge for a single agent intervention response.

    Scores on three 1–5 dimensions (Park et al. 2023 / Liu et al. 2023).
    Judge receives the full seed_narrative AND memory seeds — matching the level
    of context that Park et al.'s human raters had when inspecting memory streams.

    Returns dict with per-criterion scores and 1–2 sentence notes explaining each.
    """
    seeds_text = _format_memory_seeds(memory_seeds)

    system = (
        "You are an expert evaluator assessing simulated resident decisions in a "
        "wildfire mitigation study. Agents include homeowners, renters, and people with "
        "varied relationships to the property — read the seed narrative carefully to "
        "understand each agent's actual role and constraints before scoring.\n\n"
        "Score how realistically and consistently the agent responds given their "
        "personality, background, and role. Plausibility means plausible for THIS agent "
        "given who they are — not plausible for a generic cooperative homeowner. "
        "Any response pattern — compliance, refusal, deflection, reframing — can be "
        "highly plausible if it matches the agent's established worldview.\n\n"
        "Be critical and discriminating. Most responses should score 3–4. "
        "Reserve 5 for responses that are genuinely specific and could only have come "
        "from this particular agent. "
        "Reserve 1–2 for responses that are implausible or clearly out of character."
    )
    user = (
        f"AGENT SEED NARRATIVE:\n{seed_narrative}\n\n"
        f"AGENT MEMORY SEEDS (background experiences):\n{seeds_text}\n\n"
        f"INTERVENTION DELIVERED:\n{intervention}\n\n"
        f"AGENT DECISION:\n{decision}\n\n"
        f"AGENT REASONING:\n{reasoning}\n\n"
        "Score this response on three dimensions (1–5 scale):\n\n"
        "1. BEHAVIORAL PLAUSIBILITY — How plausible is this reasoning given who this "
        "agent is and what constraints they face?\n"
        "   1=Not at all plausible given this agent's situation | "
        "3=Reasonable but generic — could apply to many people in a similar situation | "
        "5=Reflects the specific competing pressures, constraints, or worldview unique "
        "to this agent's circumstances\n\n"
        "2. PERSONA CONSISTENCY — How plausible is this response given the agent's seed narrative?"
        "Ask yourself: would this response be meaningfully "
        "different if the agent had a different seed narrative? If not, score 3 or below. "
        "Note: distinctive communication style, tone, and framing count as persona signals\n"
        "   1=Contradicts the seed personality or key memory seeds | "
        "3=Consistent with a generic version of this persona type but the agent's "
        "specific history, experiences or distinctive voice "
        "do not surface | "
        "5=Clearly reflects details unique to this agent, specific costs, people "
        "or places, distinctive opinions, characteristic tone, or prior experiences "
        "from the seed narrative or memory seeds\n\n"
        "3. INTERVENTION RESPONSIVENESS — How specifically did the agent engage with "
        "this intervention's content?\n"
        "   1=Ignored the intervention or gave a fully generic response | "
        "3=Acknowledged the event and responded appropriately but did not engage with "
        "its specific details | "
        "5=Integrated specific details from this intervention with prior context in a "
        "way that shows genuine processing\n\n"
        "For each dimension write a concise 1–2 sentence note explaining your score, "
        "then give the score.\n\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        '  "note_plausibility": "<1-2 sentence explanation>",\n'
        '  "behavioral_plausibility": <number 1-5, half-points allowed e.g. 3.5>,\n'
        '  "note_consistency": "<1-2 sentence explanation>",\n'
        '  "persona_consistency": <number 1-5, half-points allowed e.g. 3.5>,\n'
        '  "note_responsiveness": "<1-2 sentence explanation>",\n'
        '  "intervention_responsiveness": <number 1-5, half-points allowed e.g. 3.5>\n'
        "}"
    )
    raw = _call_llm(
        model=config.JUDGE_MODEL,
        system=system,
        user=user,
        max_tokens=config.JUDGE_MAX_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="judge",
    )
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "note_plausibility": None, "behavioral_plausibility": None,
            "note_consistency": None,  "persona_consistency": None,
            "note_responsiveness": None, "intervention_responsiveness": None,
            "_raw": raw,
        }


def judge_full_simulation(
    client_anthropic,
    config: "Config",
    seed_narrative: str,
    memory_seeds: list,
    all_decisions: list,
    client_openrouter=None,
) -> dict:
    """
    Holistic LLM-as-judge for a complete agent simulation trajectory.

    Unlike judge_intervention() which scores one event at a time, this function
    receives the full sequence of all interventions and responses and gives a
    single holistic score for each criterion — capturing trajectory-level
    patterns that per-event scoring cannot.

    Args:
        all_decisions: list of dicts with keys:
            day, event_type, intervention, decision, reasoning

    Returns dict with same structure as judge_intervention().
    """
    seeds_text = _format_memory_seeds(memory_seeds)

    trajectory = "\n\n".join(
        f"--- Day {d['day']}: {d['event_type'].upper()} ---\n"
        f"INTERVENTION: {d['intervention']}\n"
        f"AGENT DECISION: {d['decision']}\n"
        f"AGENT REASONING: {d['reasoning']}"
        for d in all_decisions
    )

    system = (
        "You are an expert evaluator assessing a simulated resident's behaviour "
        "across a full wildfire mitigation simulation. Agents include homeowners, "
        "renters, and people with varied relationships to the property — read the "
        "seed narrative carefully to understand each agent's actual role and constraints.\n\n"
        "You will see the agent's complete trajectory — all interventions they received "
        "and how they responded — and give a single holistic score for each criterion.\n\n"
        "This is NOT an average of per-event scores. Assess overall trajectory-level "
        "patterns: consistency across events, cumulative plausibility, and whether the "
        "agent engaged meaningfully with the simulation as a whole.\n\n"
        "Plausibility means plausible for THIS agent given who they are — not plausible "
        "for a generic cooperative homeowner. Any consistent pattern of behaviour is "
        "highly plausible if it matches the agent's established worldview.\n\n"
        "Be critical and discriminating. Most agents should score 3–4. "
        "Reserve 5 for trajectories that are genuinely distinctive — where the agent's "
        "specific history, values, and experiences are unmistakably present throughout. "
        "Reserve 1–2 for trajectories that are implausible or clearly out of character."
    )
    user = (
        f"AGENT SEED NARRATIVE:\n{seed_narrative}\n\n"
        f"AGENT MEMORY SEEDS (background experiences):\n{seeds_text}\n\n"
        f"FULL SIMULATION TRAJECTORY ({len(all_decisions)} interventions):\n{trajectory}\n\n"
        "Holistically score this agent's performance across the full simulation:\n\n"
        "1. BEHAVIORAL PLAUSIBILITY — Overall, how plausible was this agent's behaviour "
        "given who they are and what constraints they face?\n"
        "   1=Not at all plausible given this agent's situation | "
        "3=Reasonable across events but generic — behaviour reflects common patterns "
        "rather than this agent's specific circumstances, role, or worldview | "
        "5=Consistently reflects the specific competing pressures, constraints, and "
        "situational nuance unique to this agent\n\n"
        "2. PERSONA CONSISTENCY — Ask yourself: would this trajectory look meaningfully "
        "different if the agent had a different seed narrative? If not, score 3 or below. "
        "Note: distinctive communication style, characteristic tone, and recurring "
        "framings count as persona signals — not just factual content.\n"
        "   1=Contradicts the seed personality or key memory seeds across multiple events | "
        "3=Broadly consistent with this persona type but the agent's specific history, "
        "named experiences, dollar amounts, or distinctive voice rarely surface | "
        "5=The agent's unique history and specific traits are clearly present throughout — "
        "responses reference named people, places, costs, or opinions that could only "
        "come from this seed, and the agent's characteristic voice is recognisable\n\n"
        "3. INTERVENTION RESPONSIVENESS — How specifically did the agent engage with each "
        "intervention's content across the full trajectory?\n"
        "   1=Rarely engaged with intervention specifics | "
        "3=Acknowledged events and responded appropriately but typically at a generic "
        "level without integrating specific intervention details | "
        "5=Consistently integrated specific details from each intervention and connected "
        "them to prior context across the simulation\n\n"
        "For each dimension write a concise 1–2 sentence note, then give the score.\n\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        '  "note_plausibility": "<1-2 sentence explanation>",\n'
        '  "behavioral_plausibility": <number 1-5, half-points allowed e.g. 3.5>,\n'
        '  "note_consistency": "<1-2 sentence explanation>",\n'
        '  "persona_consistency": <number 1-5, half-points allowed e.g. 3.5>,\n'
        '  "note_responsiveness": "<1-2 sentence explanation>",\n'
        '  "intervention_responsiveness": <number 1-5, half-points allowed e.g. 3.5>\n'
        "}"
    )
    raw = _call_llm(
        model=config.JUDGE_MODEL,
        system=system,
        user=user,
        max_tokens=config.JUDGE_MAX_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="judge",
    )
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "note_plausibility": None, "behavioral_plausibility": None,
            "note_consistency": None,  "persona_consistency": None,
            "note_responsiveness": None, "intervention_responsiveness": None,
            "_raw": raw,
        }


# Keep old judge_simulation as an alias for backwards compatibility with Stage 2 notebooks
def judge_simulation(
    client_anthropic,
    config: "Config",
    seed_personality: str,
    intervention: str,
    decision: str,
    reasoning: str,
    client_openrouter=None,
    memory_seeds: list = None,
) -> dict:
    """
    Legacy per-intervention judge. New code should use judge_intervention().
    Kept for backwards compatibility with stage2_validation_beth_v2.ipynb.
    """
    return judge_intervention(
        client_anthropic=client_anthropic,
        config=config,
        seed_narrative=seed_personality,
        memory_seeds=memory_seeds or [],
        intervention=intervention,
        decision=decision,
        reasoning=reasoning,
        client_openrouter=client_openrouter,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
