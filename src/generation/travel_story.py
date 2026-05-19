"""
Travel story generation using the Anthropic Claude API.

Given a sequence of identified landmarks (one per uploaded photo), this module
constructs a structured prompt that lists each landmark's name, region, and
retrieved Wikipedia description, then asks Claude to write a first-person
travel narrative connecting them.

Responsible-AI design choices:
  - The prompt explicitly instructs Claude to ground every factual claim in
    the provided descriptions and not invent history or dates.  This limits
    hallucination to stylistic prose rather than factual errors about the landmarks.
  - Low-confidence landmark identifications are surfaced to the user in the UI
    before the story is generated, so they can judge whether the retrieved
    landmark is correct.
  - A disclaimer is shown beneath the story reminding users to verify facts
    before sharing.

The default model is claude-haiku-4-5-20251001 (fast, low cost).  Swap to
claude-sonnet-4-6 in the generate_story call for richer prose.
"""

from __future__ import annotations

import anthropic

_SYSTEM = """\
You are a travel influencer writing Instagram posts. The user visited a series of \
landmarks and you have their factual descriptions. Write exactly one Instagram post.

Format:
✨ Line 1: a hook - one short punchy sentence, no "What a day", no em-dashes, \
use an emoji or two
.
Line 2-5: 2-4 short paragraphs, each 1-3 sentences. Mix personal feeling with \
1-2 real facts taken strictly from the provided descriptions. \
Short sentences. Line breaks between paragraphs (use a lone dot on its own line \
to force the Instagram line break). Emojis scattered naturally - not every sentence.
.
Last line: 10-15 hashtags, mix of popular (#travel #wanderlust) and specific \
(landmark name, city, country). No full stops after hashtags.

Rules:
- Never invent facts, dates, or details not in the provided descriptions
- Do not use the phrase "hidden gem"
- Do not start sentences with "I"
- Under 250 words total\
"""


def build_prompt(landmarks: list[dict]) -> str:
    """
    Construct the user message from a list of landmark dicts.
    Each dict must have keys: name, region (optional), description.
    Landmarks are listed in the order they were uploaded (visit order).
    """
    lines = ["Here are the landmarks from my trip (in order). Write my Instagram post:"]
    for i, lm in enumerate(landmarks, 1):
        region = f", {lm['region']}" if lm.get("region") else ""
        lines.append(f"\n{i}. {lm['name']}{region}")
        lines.append(f"   {lm['description']}")
    return "\n".join(lines)


def generate_story(
    landmarks: list[dict],
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Call the Anthropic API and return a travel narrative string.

    landmarks: list of dicts with keys name, region, description (in visit order).
    api_key:   Anthropic API key - loaded from .env by config.py, never exposed in the UI.
    model:     Claude model ID.  Haiku is used by default for speed and cost;
               swap to claude-sonnet-4-6 for longer, richer prose.
    """
    if not landmarks:
        raise ValueError("At least one landmark is required.")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": build_prompt(landmarks)}],
    )
    return message.content[0].text
