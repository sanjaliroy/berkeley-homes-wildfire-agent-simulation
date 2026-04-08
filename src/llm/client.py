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
import json
import anthropic
import numpy as np
from dataclasses import dataclass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Agent decision model ──────────────────────────────────────────────────
    # Anthropic:   "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
    # OpenRouter:  "openai/gpt-4o", "deepseek/deepseek-r1", "meta-llama/llama-3.3-70b-instruct"
    DECISION_MODEL: str = "claude-sonnet-4-6"
    DECISION_MAX_TOKENS: int = 1024
    DECISION_TEMPERATURE: float = 0.7

    # ── Reflection model (higher reasoning quality; Opus recommended) ─────────
    REFLECTION_MODEL: str = "claude-sonnet-4-6"
    REFLECTION_MAX_TOKENS: int = 512
    REFLECTION_TEMPERATURE: float = 0.4

    # ── LLM-as-judge model (evaluation only; keep strong + deterministic) ─────
    JUDGE_MODEL: str = "claude-sonnet-4-6"
    JUDGE_MAX_TOKENS: int = 1024
    JUDGE_TEMPERATURE: float = 0.0   # deterministic → reproducible eval scores

    # ── Importance scoring ────────────────────────────────────────────────────
    IMPORTANCE_MAX_TOKENS: int = 5
    IMPORTANCE_TEMPERATURE: float = 0.0

    # ── Embeddings (always local HuggingFace — no provider needed) ───────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384


config = Config()

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
) -> str:
    """
    Internal routing function. All LLM calls go through here.

    Routes to Anthropic SDK for Claude models, OpenRouter for everything else.
    Handles the API format difference between providers transparently:
      - Anthropic: system is a separate top-level param
      - OpenRouter: system is a message with role="system"
    """
    if _is_anthropic_model(model):
        message = client_anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
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
        return response.choices[0].message.content


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

def decide(client_anthropic, system_prompt: str, user_prompt: str, client_openrouter=None) -> str:
    """
    Routine agent decision call.
    Uses config.DECISION_MODEL — swap this in Config to change the agent's LLM.
    """
    return _call_llm(
        model=config.DECISION_MODEL,
        system=system_prompt,
        user=user_prompt,
        max_tokens=config.DECISION_MAX_TOKENS,
        temperature=config.DECISION_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
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
    )
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "character_fidelity": None, "decision_alignment": None,
            "reasoning_alignment": None, "overall_score": None, "critique": raw,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
