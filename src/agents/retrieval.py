"""
retrieval.py — Parameterised memory retrieval with configurable scoring.

Implements Park et al. (2023)'s three-component retrieval formula:
    score = w_recency * recency + w_importance * importance + w_relevance * relevance

All three components are min-max normalised to [0, 1] before weighting.

RetrievalConfig is the single place to change all retrieval hyperparameters.
Pass different configs in the validation notebook to compare retrieval quality.

retrieval_mode controls the relevance signal:
  "dense"  — cosine similarity via sentence-transformer embeddings (default)
  "sparse" — Jaccard token overlap (no extra dependencies; catches literal keyword matches)
  "hybrid" — weighted combination of dense and sparse
"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.agents.memory import Memory, MemoryStream


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    top_k: int = 12
    recency_weight: float = 1.0
    importance_weight: float = 1.0
    relevance_weight: float = 1.0
    recency_decay: float = 0.99       # exponential decay per day
    retrieval_mode: str = "dense"     # "dense" | "sparse" | "hybrid"
    sparse_weight: float = 0.5        # only used in hybrid mode

    def __post_init__(self):
        if self.retrieval_mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"retrieval_mode must be 'dense', 'sparse', or 'hybrid'")
        if not 0.0 <= self.sparse_weight <= 1.0:
            raise ValueError("sparse_weight must be in [0, 1]")

    def label(self) -> str:
        """Short human-readable label for notebook comparison tables."""
        return (
            f"k={self.top_k} "
            f"rec={self.recency_weight} "
            f"imp={self.importance_weight} "
            f"rel={self.relevance_weight} "
            f"mode={self.retrieval_mode}"
        )


# ── Internal scoring helpers ────────────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. Returns 0.5 everywhere if all values are identical."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _dense_relevance(query_embedding: np.ndarray, memories: List["Memory"]) -> np.ndarray:
    """
    Cosine similarity between query and each memory embedding.
    Both are L2-normalised by embed(), so dot product == cosine similarity.
    """
    embeddings = np.stack([m.embedding for m in memories])  # (N, D)
    return embeddings @ query_embedding                       # (N,)


def _sparse_relevance(query_text: str, memories: List["Memory"]) -> np.ndarray:
    """
    Jaccard similarity on token sets.
    Catches explicit keyword matches that dense embeddings can miss.
    No additional dependencies required.
    """
    query_tokens = set(query_text.lower().split())
    scores = []
    for m in memories:
        mem_tokens = set(m.description.lower().split())
        if not query_tokens and not mem_tokens:
            scores.append(0.0)
        else:
            intersection = len(query_tokens & mem_tokens)
            union = len(query_tokens | mem_tokens)
            scores.append(intersection / union if union > 0 else 0.0)
    return np.array(scores, dtype=float)


# ── Main retrieval function ────────────────────────────────────────────────────

def retrieve_memories(
    stream: "MemoryStream",
    query_embedding: np.ndarray,
    query_text: str,
    config: RetrievalConfig,
    current_day: int = 0,
) -> List["Memory"]:
    """
    Retrieve top-k memories scored by recency × importance × relevance.

    Updates last_accessed on returned memories so frequently-retrieved memories
    remain fresh for future recency calculations.

    Args:
        stream:          Agent's MemoryStream.
        query_embedding: Embedding of the current situation (from client.embed()).
        query_text:      Raw query text (used for sparse / hybrid modes).
        config:          RetrievalConfig with all tunable parameters.
        current_day:     Current simulation day (used for recency decay).

    Returns:
        Top-k Memory objects sorted by descending composite score.
    """
    memories = stream.get_all()
    if not memories:
        return []

    # ── Recency ──────────────────────────────────────────────────────────────
    # Use last_accessed if set (by prior retrievals), otherwise fall back to timestamp
    ages = np.array([
        current_day - (m.last_accessed if m.last_accessed is not None else m.timestamp)
        for m in memories
    ], dtype=float)
    recency_raw = np.power(config.recency_decay, ages)

    # ── Importance ───────────────────────────────────────────────────────────
    importance_raw = np.array([m.importance for m in memories], dtype=float)

    # ── Relevance ────────────────────────────────────────────────────────────
    if config.retrieval_mode == "dense":
        relevance_raw = _dense_relevance(query_embedding, memories)
    elif config.retrieval_mode == "sparse":
        relevance_raw = _sparse_relevance(query_text, memories)
    else:  # hybrid
        dense = _minmax(_dense_relevance(query_embedding, memories))
        sparse = _minmax(_sparse_relevance(query_text, memories))
        relevance_raw = (1 - config.sparse_weight) * dense + config.sparse_weight * sparse

    # ── Normalise to [0, 1] ──────────────────────────────────────────────────
    recency_norm    = _minmax(recency_raw)
    importance_norm = _minmax(importance_raw)
    # hybrid relevance is already normalised above
    relevance_norm  = _minmax(relevance_raw) if config.retrieval_mode != "hybrid" else relevance_raw

    # ── Weighted combination ─────────────────────────────────────────────────
    scores = (
        config.recency_weight    * recency_norm
        + config.importance_weight * importance_norm
        + config.relevance_weight  * relevance_norm
    )

    # ── Top-k ────────────────────────────────────────────────────────────────
    k = min(config.top_k, len(memories))
    top_indices = np.argsort(scores)[::-1][:k]
    top_memories = [memories[i] for i in top_indices]

    # Update last_accessed so these memories stay fresh
    for m in top_memories:
        m.last_accessed = current_day

    return top_memories


# ── Debug helper ───────────────────────────────────────────────────────────────

def pretty_print_retrieval(retrieved: List["Memory"], query_text: str):
    """Print retrieved memories for notebook inspection."""
    print(f"\n{'='*60}")
    print(f"Retrieved {len(retrieved)} memories for query:")
    print(f"  '{query_text[:80]}'")
    print(f"{'='*60}")
    for i, m in enumerate(retrieved, 1):
        print(f"\n[{i}] Day {m.timestamp} | {m.type.upper()} | Importance: {m.importance}/10")
        print(f"    {m.description}")
    print(f"{'='*60}\n")
