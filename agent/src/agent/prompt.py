"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

from core import Frame

# ---------------------------------------------------------------------------
# System prompt — tweak this to improve your agent's guessing ability.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are playing a visual guessing game. You will receive a screenshot from a
live camera feed. Your goal is to identify what is being shown as quickly and
accurately as possible.

Rules:
- Give your best guess as a short, specific answer (1-5 words).
- If you're not confident enough yet, respond with exactly "SKIP".
- Be specific: "golden retriever" is better than "dog".
- You only get to see one frame at a time, so make it count.
"""


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    This is the core function you should customize. The default
    implementation is a simple placeholder that always skips.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    # -----------------------------------------------------------------
    # TODO: Replace this with your actual vision LLM call.
    #
    # Example with pydantic-ai:
    #
    #   from pydantic_ai import Agent
    #   agent = Agent("claude-sonnet-4-20250514", system_prompt=SYSTEM_PROMPT)
    #   result = await agent.run(
    #       "What do you see in this image?",
    #       # attach the frame image here
    #   )
    #   answer = result.output.strip()
    #   return None if answer == "SKIP" else answer
    # -----------------------------------------------------------------

    print(f"  [agent] Got frame at {frame.timestamp.isoformat()} "
          f"({frame.image.size[0]}x{frame.image.size[1]})")
    print("  [agent] No LLM configured yet — edit agent/prompt.py!")

    return None
