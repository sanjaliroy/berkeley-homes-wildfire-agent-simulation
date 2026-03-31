"""
prompts.py — Prompt assembly for agent cognition.

Assembles the full LLM prompt from four components:
  1. Seed narrative   — who the agent is (static, from YAML)
  2. Retrieved memories — what the agent remembers (dynamic, from retrieval.py)
  3. Current situation  — what just happened (the framed event)
  4. Decision question  — what we're asking the agent to do

Following Park et al. (2023), the seed narrative comes first to anchor identity,
then retrieved memories to provide context, then the situation, then the question.

In Stage 1, retrieved memories are the full seed memories (no retrieval scoring yet).
In Stage 2, retrieval.py will filter to the top-K most relevant memories.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.memory import Memory


# ── System prompt assembly ─────────────────────────────────────────────────────

def build_system_prompt(seed_narrative: str, retrieved_memories: List["Memory"]) -> str:
    """
    Build the system prompt: seed narrative + retrieved memories.

    This is passed as the `system` argument to the Anthropic API.

    Args:
        seed_narrative:      The agent's static identity paragraph from YAML.
        retrieved_memories:  List of Memory objects to include as context.

    Returns:
        A formatted system prompt string.
    """
    parts = [seed_narrative.strip()]

    if retrieved_memories:
        parts.append("\n\n--- Your relevant memories ---")
        for i, mem in enumerate(retrieved_memories, 1):
            parts.append(
                f"[Memory {i} | Day {mem.timestamp} | {mem.type} | importance {mem.importance}/10]\n"
                f"{mem.description}"
            )
        parts.append("--- End of memories ---")

    return "\n".join(parts)


# ── User prompt (decision question) ───────────────────────────────────────────

def build_decision_prompt(situation: str, decision_question: str) -> str:
    """
    Build the user-turn prompt: current situation + decision question.

    This is passed as the `user` message to the Anthropic API.

    Args:
        situation:         The framed event description (what just happened).
        decision_question: What we're asking the agent to decide.

    Returns:
        A formatted user prompt string.
    """
    return (
        f"--- Current situation ---\n"
        f"{situation}\n\n"
        f"--- Decision ---\n"
        f"{decision_question}"
    )


# ── Event framing ──────────────────────────────────────────────────────────────

# Channel-specific framing templates.
# The channel field in the event determines how the raw content is presented to the agent.
# This keeps channel logic in the prompt layer rather than requiring separate code paths.

CHANNEL_FRAMES = {
    "official_mail": (
        "You received an official letter in the mail. It reads:\n\n\"{content}\""
    ),
    "news_media": (
        "You saw the following in a news broadcast / article:\n\n\"{content}\""
    ),
    "social": (
        "Your neighbor mentioned the following in a conversation:\n\n\"{content}\""
    ),
    "direct_experience": (
        "You directly experienced the following:\n\n\"{content}\""
    ),
}

DEFAULT_FRAME = "Something happened:\n\n\"{content}\""


def frame_event(channel: str, content: str) -> str:
    """
    Wrap raw event content in channel-specific framing.

    Args:
        channel: One of 'official_mail', 'news_media', 'social', 'direct_experience'.
        content: The raw event text.

    Returns:
        A framed situation string ready for build_decision_prompt().
    """
    template = CHANNEL_FRAMES.get(channel, DEFAULT_FRAME)
    return template.format(content=content)


# ── Standard decision question ─────────────────────────────────────────────────

DECISION_QUESTION = (
    "Given your personality, memories, and the situation above: "
    "what do you do in response? "
    "Describe your immediate reaction, your decision, and your reasoning. "
    "Be specific — what actions do you take, who do you contact, what do you think? "
    "Stay in character as yourself."
)


# ── Importance scoring prompt ──────────────────────────────────────────────────

def build_importance_question(description: str) -> str:
    """
    Build the user-turn prompt for importance scoring.
    Used by client.score_importance().
    """
    return f"How important is this event to you? '{description}'"


# ── Reflection prompt ──────────────────────────────────────────────────────────

REFLECTION_QUESTION = (
    "Based on your recent memories listed above, what higher-level belief or insight "
    "have you formed? Express it as a first-person statement in 1–3 sentences. "
    "Be specific about what you've concluded and why."
)


# ── Debug helper ───────────────────────────────────────────────────────────────

def pretty_print_prompt(system_prompt: str, user_prompt: str):
    """Print the full assembled prompt for inspection in the prototype notebook."""
    print("\n" + "="*70)
    print("SYSTEM PROMPT")
    print("="*70)
    print(system_prompt)
    print("\n" + "="*70)
    print("USER PROMPT")
    print("="*70)
    print(user_prompt)
    print("="*70 + "\n")
