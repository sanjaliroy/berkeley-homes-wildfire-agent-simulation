"""
client.py — LLM API interaction layer.

Handles all calls to the Anthropic API. Provides three methods:
  - decide(): routine agent decisions (Sonnet)
  - reflect(): belief synthesis (Opus — stronger reasoning)
  - embed(): text embedding via OpenAI (text-embedding-3-small)

In Stage 1, only decide() and embed() are used.
reflect() is wired up but won't be called until Stage 5.

Config pattern adapted from course studio notebooks (Cornelia Paulik, INFO 290).
"""

import os
import anthropic
import openai
import numpy as np
from dataclasses import dataclass
from google.colab import userdata


# ── Config (adapted from course studio @dataclass pattern) ────────────────────
@dataclass
class Config:
    # --- Decision model (Sonnet: fast, cheap, good for routine decisions) ---
    DECISION_MODEL: str = "claude-sonnet-4-5-20251001"
    DECISION_MAX_TOKENS: int = 1024
    DECISION_TEMPERATURE: float = 0.7   # some variation in decisions

    # --- Reflection model (Opus: stronger reasoning for belief synthesis) ---
    # Used in Stage 5 only
    REFLECTION_MODEL: str = "claude-opus-4-5-20251001"
    REFLECTION_MAX_TOKENS: int = 512
    REFLECTION_TEMPERATURE: float = 0.4  # more deterministic for synthesis

    # --- Importance scoring (deterministic — always 0.0) ---
    IMPORTANCE_MAX_TOKENS: int = 5
    IMPORTANCE_TEMPERATURE: float = 0.0

    # --- Embedding model ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI
    EMBEDDING_DIM: int = 1536

    # --- Retrieval hyperparameters (used from Stage 2 onward) ---
    TOP_K_MEMORIES: int = 12        # how many memories to pass to LLM
    RECENCY_WEIGHT: float = 1.0     # α — recency decay weight
    IMPORTANCE_WEIGHT: float = 1.0  # β — importance score weight
    RELEVANCE_WEIGHT: float = 1.0   # γ — cosine similarity weight
    RECENCY_DECAY: float = 0.99     # λ — exponential decay per tick

config = Config()


# ── Client initialisation (initialise once, reuse — course notebook pattern) ──
def init_clients():
    """
    Initialise Anthropic and OpenAI clients from Colab secrets.
    Call once at the top of the notebook. Returns (anthropic_client, openai_client).

    Usage in notebook:
        client_anthropic, client_openai = init_clients()
    """
    try:
        anthropic_key = userdata.get("ANTHROPIC_API_KEY")
        openai_key = userdata.get("OPENAI_API_KEY")
    except Exception:
        # Fallback to environment variables (local dev)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in Colab secrets or environment.")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not found in Colab secrets or environment.")

    client_anthropic = anthropic.Anthropic(api_key=anthropic_key)
    client_openai = openai.OpenAI(api_key=openai_key)

    print("✓ Anthropic client initialised")
    print("✓ OpenAI client initialised")
    return client_anthropic, client_openai


# ── LLM calls ─────────────────────────────────────────────────────────────────

def decide(client_anthropic: anthropic.Anthropic, system_prompt: str, user_prompt: str) -> str:
    """
    Call the decision model (Sonnet) for routine agent decisions.

    Args:
        client_anthropic: Initialised Anthropic client (from init_clients()).
        system_prompt:    Full assembled system prompt (seed + memories + situation).
        user_prompt:      Decision question posed to the agent.

    Returns:
        The model's response text (agent decision + reasoning).
    """
    message = client_anthropic.messages.create(
        model=config.DECISION_MODEL,
        max_tokens=config.DECISION_MAX_TOKENS,
        temperature=config.DECISION_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return message.content[0].text


def reflect(client_anthropic: anthropic.Anthropic, system_prompt: str, memories_text: str) -> str:
    """
    Call the reflection model (Opus) to synthesise higher-level beliefs.
    Used in Stage 5. Not called in Stage 1.

    Args:
        client_anthropic: Initialised Anthropic client (from init_clients()).
        system_prompt:    The agent's seed personality.
        memories_text:    Formatted string of recent high-importance memories.

    Returns:
        A synthesised belief statement stored back as a reflection memory.
    """
    reflection_prompt = (
        "Based on the following recent experiences and observations, "
        "what higher-level belief or insight have you formed? "
        "Express it as a first-person statement in 1-3 sentences.\n\n"
        f"Recent memories:\n{memories_text}"
    )

    message = client_anthropic.messages.create(
        model=config.REFLECTION_MODEL,
        max_tokens=config.REFLECTION_MAX_TOKENS,
        temperature=config.REFLECTION_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": reflection_prompt}]
    )
    return message.content[0].text


def embed(client_openai: openai.OpenAI, text: str) -> np.ndarray:
    """
    Embed a string using OpenAI text-embedding-3-small.

    Args:
        client_openai: Initialised OpenAI client (from init_clients()).
        text:          Text to embed (memory description or perception string).

    Returns:
        A numpy array of shape (EMBEDDING_DIM,).
    """
    # Embedding models don't handle newlines well — clean first
    text = text.replace("\n", " ").strip()

    response = client_openai.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def score_importance(
    client_anthropic: anthropic.Anthropic,
    description: str,
    agent_seed: str,
    max_retries: int = 3
) -> int:
    """
    Ask the LLM to score a memory's importance on a scale of 1-10.
    Uses the decision model (cheaper) since this is simple classification.
    Includes retry logic adapted from the LLM-as-judge pattern in course notebooks.

    Args:
        client_anthropic: Initialised Anthropic client (from init_clients()).
        description:      The memory description to score.
        agent_seed:       The agent's seed narrative (for persona context).
        max_retries:      Number of retries on unexpected output (default 3).

    Returns:
        An integer from 1 to 10.
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
        message = client_anthropic.messages.create(
            model=config.DECISION_MODEL,
            max_tokens=config.IMPORTANCE_MAX_TOKENS,
            temperature=config.IMPORTANCE_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        raw = message.content[0].text.strip()
        try:
            score = int(raw)
            return max(1, min(10, score))  # clamp to [1, 10]
        except ValueError:
            print(f"[client] Attempt {attempt+1}: unexpected importance score '{raw}', retrying...")

    print(f"[client] Warning: all {max_retries} attempts failed, defaulting to 5")
    return 5