"""
client.py — LLM API interaction layer.

Handles all calls to the Anthropic API. Provides three methods:
  - decide(): routine agent decisions (Sonnet)
  - reflect(): belief synthesis (Opus — stronger reasoning)
  - embed(): text embedding via HuggingFace sentence-transformers (free, no API key)

In Stage 1, only decide() and embed() are used.
reflect() is wired up but won't be called until Stage 5.

Config pattern adapted from course studio notebooks (Cornelia Paulik, INFO 290).
Embedding model: all-MiniLM-L6-v2 (same model used in course RAG studios).
"""

import os
import anthropic
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    DECISION_MODEL: str = "claude-sonnet-4-5"
    DECISION_MAX_TOKENS: int = 1024
    DECISION_TEMPERATURE: float = 0.7
    REFLECTION_MODEL: str = "claude-opus-4-5"
    REFLECTION_MAX_TOKENS: int = 512
    REFLECTION_TEMPERATURE: float = 0.4
    IMPORTANCE_MAX_TOKENS: int = 5
    IMPORTANCE_TEMPERATURE: float = 0.0
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    TOP_K_MEMORIES: int = 12
    RECENCY_WEIGHT: float = 1.0
    IMPORTANCE_WEIGHT: float = 1.0
    RELEVANCE_WEIGHT: float = 1.0
    RECENCY_DECAY: float = 0.99

config = Config()

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model ({config.EMBEDDING_MODEL})...")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Embedding model loaded")
    return _embedding_model

def init_clients():
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

    client_anthropic = anthropic.Anthropic(api_key=anthropic_key)
    print("Anthropic client initialised")
    print("Embeddings: HuggingFace all-MiniLM-L6-v2 (free, no API key needed)")
    return client_anthropic

def decide(client_anthropic, system_prompt, user_prompt):
    message = client_anthropic.messages.create(
        model=config.DECISION_MODEL,
        max_tokens=config.DECISION_MAX_TOKENS,
        temperature=config.DECISION_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return message.content[0].text

def reflect(client_anthropic, system_prompt, memories_text):
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

def embed(text):
    text = text.replace("\n", " ").strip()
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)

def score_importance(client_anthropic, description, agent_seed, max_retries=3):
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
            return max(1, min(10, score))
        except ValueError:
            print(f"[client] Attempt {attempt+1}: unexpected score '{raw}', retrying...")
    print(f"[client] Warning: all {max_retries} attempts failed, defaulting to 5")
    return 5
