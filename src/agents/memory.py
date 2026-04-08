"""
memory.py — Agent memory stream.

Implements the MemoryStream class following Park et al. (2023)'s Generative Agents architecture.
Each Memory object stores:
  - timestamp:    simulation day (int)
  - description:  natural language description of the event/observation/decision
  - importance:   LLM-scored 1-10 integer
  - embedding:    numpy vector for cosine similarity retrieval
  - type:         'observation' | 'decision' | 'reflection' | 'conversation'

The stream is append-only. Retrieval is handled by retrieval.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ── Memory types ──────────────────────────────────────────────────────────────
MEMORY_TYPES = {"observation", "decision", "reflection", "conversation"}


@dataclass
class Memory:
    """A single entry in an agent's memory stream."""

    timestamp: int              # simulation day this memory was created
    description: str            # natural language description
    importance: int             # LLM-scored 1–10
    embedding: np.ndarray       # vector embedding (shape: [EMBEDDING_DIM])
    type: str                   # 'observation' | 'decision' | 'reflection' | 'conversation'
    last_accessed: Optional[int] = None  # updated by retrieval.py; None means use timestamp

    def __post_init__(self):
        if self.type not in MEMORY_TYPES:
            raise ValueError(f"Memory type '{self.type}' must be one of {MEMORY_TYPES}")
        if not (1 <= self.importance <= 10):
            raise ValueError(f"Importance must be 1–10, got {self.importance}")
        # default last_accessed to creation time so recency decay starts from the right point
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

    def to_dict(self) -> dict:
        """Serialise to dict for JSONL logging (embedding stored as list)."""
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "importance": self.importance,
            "embedding": self.embedding.tolist(),
            "type": self.type,
            "last_accessed": self.last_accessed,
        }

    def __repr__(self) -> str:
        return (
            f"Memory(day={self.timestamp}, type={self.type}, "
            f"importance={self.importance}, "
            f"desc='{self.description[:60]}...')"
        )


class MemoryStream:
    """
    Append-only memory stream for a single agent.

    Agents never forget — retrieval controls what surfaces at decision time.
    The stream grows throughout the simulation.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._memories: List[Memory] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        description: str,
        importance: int,
        embedding: np.ndarray,
        memory_type: str,
        timestamp: int = 0,
    ) -> Memory:
        """
        Add a new memory to the stream.

        Args:
            description:  Natural language description of the event.
            importance:   LLM-scored importance (1–10).
            embedding:    Embedding vector for the description.
            memory_type:  One of 'observation', 'decision', 'reflection', 'conversation'.
            timestamp:    Simulation day (default 0 for Stage 1 hardcoded events).

        Returns:
            The created Memory object.
        """
        memory = Memory(
            timestamp=timestamp,
            description=description,
            importance=importance,
            embedding=embedding,
            type=memory_type,
        )
        self._memories.append(memory)
        return memory

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_all(self) -> List[Memory]:
        """Return all memories in chronological order."""
        return list(self._memories)

    def get_recent(self, n: int) -> List[Memory]:
        """Return the n most recent memories."""
        return self._memories[-n:]

    def get_by_type(self, memory_type: str) -> List[Memory]:
        """Return all memories of a given type."""
        return [m for m in self._memories if m.type == memory_type]

    def get_cumulative_importance(self, since_index: int = 0) -> int:
        """
        Sum of importance scores for memories from since_index onwards.
        Used by reflection.py to decide when to trigger reflection.
        """
        return sum(m.importance for m in self._memories[since_index:])

    def count(self) -> int:
        """Total number of memories in the stream."""
        return len(self._memories)

    # ── Seed ───────────────────────────────────────────────────────────────────

    def load_seeds(
        self,
        seeds: List[dict],
        client_anthropic,
        agent_seed_narrative: str,
    ):
        """
        Pre-load memory seeds from the agent YAML into the stream.

        Each seed dict has: description, importance (optional), type.
        If importance is not set, calls score_importance() to get it via LLM.
        Calls embed() to generate the embedding (free HuggingFace model, no API key needed).

        Args:
            seeds:                 List of seed dicts from agent YAML.
            client_anthropic:      Anthropic client (from client.init_clients()).
            agent_seed_narrative:  The agent's seed narrative (for importance scoring context).
        """
        from src.llm.client import embed, score_importance

        print(f"[memory] Loading {len(seeds)} seed memories for {self.agent_name}...")
        for i, seed in enumerate(seeds):
            description = seed["description"]
            memory_type = seed.get("type", "observation")
            importance = seed.get("importance")

            # Score importance via LLM if not hardcoded in YAML
            if importance is None:
                importance = score_importance(client_anthropic, description, agent_seed_narrative)

            embedding = embed(description)

            self.add(
                description=description,
                importance=importance,
                embedding=embedding,
                memory_type=memory_type,
                timestamp=0,  # seeds exist before simulation starts
            )
            print(f"  [{i+1}/{len(seeds)}] '{description[:60]}...' (importance={importance})")

        print(f"[memory] Done. Stream has {self.count()} memories.\n")

    # ── Inspect ────────────────────────────────────────────────────────────────

    def pretty_print(self, n: Optional[int] = None):
        """Print memories in a readable format for notebook inspection."""
        memories = self._memories if n is None else self._memories[-n:]
        print(f"\n{'='*60}")
        print(f"Memory stream: {self.agent_name} ({self.count()} total memories)")
        print(f"{'='*60}")
        for i, m in enumerate(memories):
            print(f"\n[{i+1}] Day {m.timestamp} | {m.type.upper()} | Importance: {m.importance}/10")
            print(f"    {m.description}")
        print(f"{'='*60}\n")