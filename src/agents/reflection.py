"""
reflection.py — Periodic belief synthesis via two-step reflection.

Implements Park et al. (2023)'s reflection process:
  Step 1 — Generate N high-level questions from recent memories.
            ("What is Margaret most concerned about right now?")
  Step 2 — For each question, retrieve relevant memories and synthesise an insight.
            ("I am increasingly worried that insurance feels arbitrary — memories 2, 5, 8 show...")

Reflection fires when cumulative importance of unprocessed memories exceeds a threshold.
Each insight is stored as a first-class 'reflection' memory in the agent's stream.

ReflectionConfig is the single place to change all reflection hyperparameters.
Pass different configs in the validation notebook to compare reflection quality.
"""

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.memory import Memory, MemoryStream
    from src.llm.client import Config


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionConfig:
    threshold: float = 100.0           # cumulative importance that triggers reflection
    num_questions: int = 3             # questions to generate (Park et al. used 3)
    memories_for_questions: int = 10   # recent memories fed to question generator
    top_k_per_question: int = 8        # memories retrieved per question for synthesis
    reflection_importance: int = 8     # importance assigned to stored reflection memories


# ── Step 1: question generation ────────────────────────────────────────────────

def _generate_questions(
    client_anthropic,
    config: "Config",
    reflection_config: ReflectionConfig,
    recent_memories: List["Memory"],
    agent_seed: str,
    client_openrouter=None,
) -> List[str]:
    """
    Ask the reflection model to generate high-level introspective questions
    from recent memories.
    """
    from src.llm.client import _call_llm

    memories_text = "\n".join(
        f"[Day {m.timestamp} | {m.type} | importance {m.importance}/10] {m.description}"
        for m in recent_memories
    )
    system = (
        f"{agent_seed}\n\n"
        "You are reflecting on your recent experiences to form deeper beliefs."
    )
    user = (
        f"Here are your most recent memories:\n\n{memories_text}\n\n"
        f"Generate exactly {reflection_config.num_questions} insightful questions about yourself "
        "that these memories raise — questions about your beliefs, attitudes, fears, or motivations. "
        "Each question must be answerable using only the memories listed above. "
        "Do not ask questions that require experiences or context beyond what is described. "
        "Output ONLY the questions, one per line, no numbering or extra text."
    )
    text = _call_llm(
        model=config.REFLECTION_MODEL,
        system=system,
        user=user,
        max_tokens=1024,
        temperature=config.REFLECTION_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="agent",
    )
    lines = [q.strip() for q in text.strip().split("\n") if q.strip()]
    return lines[:reflection_config.num_questions]


# ── Step 2: insight synthesis ──────────────────────────────────────────────────

def _synthesise_insight(
    client_anthropic,
    config: "Config",
    question: str,
    relevant_memories: List["Memory"],
    agent_seed: str,
    client_openrouter=None,
) -> str:
    """
    Synthesise a first-person belief that answers the question,
    citing the specific experiences that led to it.
    """
    from src.llm.client import _call_llm

    memories_text = "\n".join(
        f"[Memory #{m.id} | Day {m.timestamp} | {m.type} | importance {m.importance}/10] {m.description}"
        for m in relevant_memories
    )
    system = (
        f"{agent_seed}\n\n"
        "You are forming higher-level beliefs based on your experiences."
    )
    user = (
        f"Question: {question}\n\n"
        f"Relevant memories:\n{memories_text}\n\n"
        "In 1-3 sentences, write a first-person belief or insight that answers this question.\n\n"
        "IMPORTANT: Only draw insights from the specific memories listed above. "
        "Do not reference memories by number — instead, describe the experience directly "
        "(e.g., 'when I saw my neighbor's fence catch fire' not 'Memory #12'). "
        "Do not infer or invent experiences that are not explicitly described in the list above. "
        "If an insight cannot be grounded in the provided memories, do not include it."
    )
    return _call_llm(
        model=config.REFLECTION_MODEL,
        system=system,
        user=user,
        max_tokens=config.REFLECTION_MAX_TOKENS,
        temperature=config.REFLECTION_TEMPERATURE,
        client_anthropic=client_anthropic,
        client_openrouter=client_openrouter,
        call_type="agent",
    )


# ── Main reflection function ────────────────────────────────────────────────────

def maybe_reflect(
    stream: "MemoryStream",
    client_anthropic,
    config: "Config",
    reflection_config: ReflectionConfig,
    agent_seed: str,
    current_day: int = 0,
    last_reflection_index: int = 0,
    client_openrouter=None,
) -> Tuple[List["Memory"], int]:
    """
    Check whether cumulative importance since last reflection exceeds the threshold.
    If so, run the two-step reflection process and store new reflection memories.

    Args:
        stream:                 Agent's MemoryStream.
        client_anthropic:       Anthropic client.
        config:                 LLM Config (controls which model is used).
        reflection_config:      ReflectionConfig with all tunable parameters.
        agent_seed:             Agent's seed narrative (anchors identity in reflection prompts).
        current_day:            Current simulation day.
        last_reflection_index:  Stream index where the previous reflection ended.
                                Pass the returned new_index on subsequent calls.

    Returns:
        (new_reflections, new_last_reflection_index)
        new_reflections is an empty list if the threshold was not met.
    """
    from src.agents.retrieval import retrieve_memories, RetrievalConfig
    from src.llm.client import embed

    cumulative = stream.get_cumulative_importance(since_index=last_reflection_index)

    if cumulative < reflection_config.threshold:
        return [], last_reflection_index

    print(f"\n[reflection] Threshold met (cumulative importance = {cumulative:.0f}). Reflecting...")

    recent_memories = stream.get_recent(reflection_config.memories_for_questions)

    # Step 1: generate questions
    questions = _generate_questions(
        client_anthropic, config, reflection_config, recent_memories, agent_seed,
        client_openrouter=client_openrouter,
    )
    print(f"[reflection] Generated {len(questions)} questions:")
    for q in questions:
        print(f"  → {q}")

    # Step 2: for each question, retrieve + synthesise
    retrieval_cfg = RetrievalConfig(
        top_k=reflection_config.top_k_per_question,
        relevance_weight=2.0,   # bias toward semantic relevance for reflection
    )
    new_reflections = []

    for question in questions:
        query_embed = embed(question)
        relevant = retrieve_memories(stream, query_embed, question, retrieval_cfg, current_day)
        insight = _synthesise_insight(client_anthropic, config, question, relevant, agent_seed,
                                       client_openrouter=client_openrouter)
        reflection_memory = stream.add(
            description=insight,
            importance=reflection_config.reflection_importance,
            embedding=embed(insight),
            memory_type="reflection",
            timestamp=current_day,
        )
        new_reflections.append(reflection_memory)
        print(f"\n[reflection] Insight: {insight[:100]}...")

    new_index = stream.count()
    print(f"\n[reflection] Done. {len(new_reflections)} new reflections stored.")
    return new_reflections, new_index
