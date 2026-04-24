"""
channels.py — Channel-specific framing for simulation events.

The channel field on each event in the scenario YAML determines HOW the agent
receives the information. The framing language matters because it shapes which
memories the agent retrieves when processing the event:

  official_mail   → formal, institutional language  → pulls institutional trust memories
  news_media      → broadcast/third-person language → pulls general risk-awareness memories
  social          → peer/neighbour observation      → pulls community and relationship memories
  direct_experience → first-person sensory          → pulls personal history and emotional memories

Channel framing is an environment concern, not a prompt-assembly concern — it lives here
rather than in prompts.py.
"""


# ── Channel framing templates ─────────────────────────────────────────────────
# {content} is replaced with the raw event text from the scenario YAML.

CHANNEL_FRAMES = {
    "official_mail": (
        "You received an official letter in the mail:\n\n{content}"
    ),
    "news_media": (
        "You saw the following in a news report:\n\n{content}"
    ),
    "social": (
        "While outside, you notice or hear from a neighbour:\n\n{content}"
    ),
    "direct_experience": (
        "You are directly experiencing the following right now:\n\n{content}"
    ),
}

# Fallback if an unknown channel is used
DEFAULT_FRAME = "Something has happened:\n\n{content}"


def frame_event(channel: str, content: str) -> str:
    """
    Wrap raw event content in channel-specific framing.

    The framing language signals to the agent how the information reached them,
    which shapes which memories surface during retrieval.

    Args:
        channel: One of 'official_mail', 'news_media', 'social', 'direct_experience'.
        content: The raw event text from the scenario YAML.

    Returns:
        Framed situation string, ready to pass into build_decision_prompt().
    """
    template = CHANNEL_FRAMES.get(channel, DEFAULT_FRAME)
    return template.format(content=content)


def channel_label(channel: str) -> str:
    """Short human-readable label for notebook display tables."""
    labels = {
        "official_mail":    "Official letter",
        "news_media":       "News media",
        "social":           "Social / neighbour",
        "direct_experience": "Direct experience",
    }
    return labels.get(channel, channel)
