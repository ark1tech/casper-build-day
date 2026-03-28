"""System prompt and analysis logic for the charades guessing agent.

=== EDIT THIS FILE ===

Strategy summary
----------------
- Model: Google Gemini 2.5 Flash via OpenRouter (LLM_API_KEY env var)
- Temporal context: accumulate per-frame observations so the model can reason
  across time, not just from a single snapshot.
- Conservative guessing: never guess in the first 5 seconds; require high
  confidence (>= 0.75) before submitting; never repeat a guess.
- Escalating fallback: after 60 s of silence, drop threshold to 0.5 so we
  don't time out with guess budget unused.

Scoring model reminder
----------------------
  score = guesses_remaining * speed_bonus
  => fewer guesses + faster correct answer = maximum points
  => every wasted guess costs one multiplier point forever, so do NOT guess randomly.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core import Frame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum seconds to observe before committing to any guess.
_HOLD_PERIOD_S: float = 5.0

# Normal confidence threshold (0–1). Below this → SKIP.
_CONFIDENCE_HIGH: float = 0.75

# After this many seconds with no correct guess, relax the threshold.
_ESCALATION_AFTER_S: float = 60.0
_CONFIDENCE_LOW: float = 0.50

# How many recent frame observations to feed back as context.
_MAX_CONTEXT_FRAMES: int = 10

# ---------------------------------------------------------------------------
# Model setup (lazy — created on first analyze() call so the module can be
# imported without LLM_API_KEY set, e.g. during uv sync or linting).
# ---------------------------------------------------------------------------
# Routes to Gemini 2.5 Flash via OpenRouter using LLM_API_KEY from .env.
# OpenRouter accepts the OpenAI wire format; we use OpenAIChatModel + a
# custom base_url so pydantic-ai's structured-output and image attachment
# support works without any extra provider-specific packages.

_gemini: Agent[Any, FrameAnalysis] | None = None


def _get_agent() -> Agent[Any, FrameAnalysis]:
    global _gemini
    if _gemini is None:
        api_key = os.environ.get("LLM_API_KEY", "")
        _gemini = Agent(
            OpenAIChatModel(
                "google/gemini-2.5-flash",
                provider=OpenAIProvider(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                ),
            ),
        )
    return _gemini

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FrameAnalysis(BaseModel):
    """Structured response from the vision model for a single charades frame."""

    observation: str = Field(
        description=(
            "A precise description of what the performer is doing in this specific "
            "frame: posture, gestures, facial expression, prop usage, body position."
        )
    )
    accumulated_reasoning: str = Field(
        description=(
            "Reasoning that synthesizes ALL observations so far (including prior "
            "frames in the context) to build toward a charades answer. Consider "
            "syllable gestures, 'sounds like' signs, category signs, and action verbs."
        )
    )
    guess: str | None = Field(
        default=None,
        description=(
            "Your best charades answer (1-6 words). Use None / null if you are not "
            "yet confident enough to commit. Do NOT include filler like 'I think' — "
            "just the answer itself, e.g. 'Titanic' or 'swimming upstream'."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in this guess (0.0 = wild guess, 1.0 = certain). "
            "Only set above 0.75 when the accumulated evidence strongly points to one answer."
        )
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert charades judge watching a LIVE CAMERA FEED of a performer playing charades.

## Your Job
Interpret the performer's body language, gestures, and mime to identify the secret word or phrase.
There is NO audio. You must rely entirely on visual cues.

## Charades Conventions to Recognize
- **Category sign**: pulling ear = "sounds like"; tugging collar = "it's a song"; book open in hands = "it's a book/movie"; fingers as quotation marks = exact phrase
- **Syllable count**: holding up N fingers near the chin = N syllables
- **Word count**: holding up N fingers = N-word phrase
- **Which word**: holding up N fingers after word-count = working on the Nth word
- **Small word**: pinching fingers together = a short/small word (the, a, and, in…)
- **Longer/shorter**: stretching hands apart or bringing together = make the guess longer or shorter
- **Action verbs**: miming a physical activity (swimming, flying, eating) → the word IS that verb or strongly related to it
- **Compound words**: performer may act out each part separately

## Answer Complexity
Answers are NOT always simple nouns. They may be:
- Multi-word phrases ("leap of faith", "swimming upstream")
- Movie or book titles ("The Dark Knight", "Gone with the Wind")
- Idioms or proverbs ("kick the bucket", "bite the bullet")
- Abstract concepts ("freedom", "nostalgia")
- Verbs or gerunds ("skydiving", "negotiating")

## Output Contract
You MUST respond with valid JSON matching the FrameAnalysis schema.
- `observation`: what you see in THIS frame (be precise about body parts and gestures)
- `accumulated_reasoning`: synthesize ALL prior observations + this frame to build toward the answer
- `guess`: your best answer right now, or null if not confident enough
- `confidence`: a float 0.0–1.0; only exceed 0.75 when the evidence is compelling

## Strategy
- Accumulate evidence across frames before guessing. One frame is rarely enough.
- If you see a syllable or word-count gesture, factor it into every future guess.
- Prefer a specific, concrete answer over a vague one (e.g. "Titanic" > "ship movie").
- If uncertain between two options, output null and wait for more frames.
"""

# ---------------------------------------------------------------------------
# Per-round state (module-level — resets when the process restarts each round)
# ---------------------------------------------------------------------------

_frame_observations: list[str] = []
_submitted_guesses: set[str] = set()
_round_start: datetime | None = None


def _scene_changed(prev: str, curr: str) -> bool:
    """Return True when the current observation diverges enough from the previous one.

    Uses Jaccard overlap on word sets: < 40 % shared → scene considered changed.
    """
    prev_words = set(prev.lower().split())
    curr_words = set(curr.lower().split())
    if not prev_words:
        return True
    union = prev_words | curr_words
    return len(prev_words & curr_words) / len(union) < 0.40


def _log_frame(
    elapsed_s: float,
    analysis: "FrameAnalysis",
    context_changed: bool,
) -> None:
    """Pretty-print a rich per-frame log entry."""
    sep = "─" * 72
    change_tag = "  ◆ SCENE CHANGE" if context_changed else ""
    print(f"\n{sep}")
    print(f"  [agent] t={elapsed_s:.1f}s{change_tag}")
    print(f"  OBSERVATION  : {analysis.observation}")
    print(f"  REASONING    : {analysis.accumulated_reasoning}")
    print(
        f"  GUESS        : {analysis.guess!r}  "
        f"[conf={analysis.confidence:.2f}]"
    )
    print(sep)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def analyze(frame: Frame) -> str | None:
    """Analyze a single charades frame and return a guess, or None to skip.

    This function is called by the main loop for every sampled frame. It:
      1. Converts the PIL image to JPEG bytes for the vision model.
      2. Builds a context string from prior frame observations.
      3. Calls Gemini 2.5 Flash with the image + context.
      4. Saves the observation to the rolling history.
      5. Applies the confidence + hold-period + dedup gates.
      6. Returns the guess string or None.

    Args:
        frame: A Frame with .image (PIL Image, RGB) and .timestamp (UTC datetime).

    Returns:
        A guess string to submit, or None to skip this frame.
    """
    global _round_start

    # ------------------------------------------------------------------ #
    # 1. Track round start time
    # ------------------------------------------------------------------ #
    now = frame.timestamp
    if _round_start is None:
        _round_start = now

    elapsed_s = (now - _round_start).total_seconds()

    # ------------------------------------------------------------------ #
    # 2. Convert PIL Image → JPEG bytes
    # ------------------------------------------------------------------ #
    buf = io.BytesIO()
    # Resize to 720p max to keep token count / latency reasonable
    img = frame.image
    if img.width > 1280 or img.height > 720:
        img.thumbnail((1280, 720))
    img.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    # ------------------------------------------------------------------ #
    # 3. Build context string from prior observations
    # ------------------------------------------------------------------ #
    context_parts: list[str] = []
    context_parts.append(
        f"Round elapsed: {elapsed_s:.1f}s | "
        f"Guesses submitted so far: {len(_submitted_guesses)}"
    )

    if _submitted_guesses:
        context_parts.append(
            f"Already-submitted guesses (do NOT repeat these): "
            + ", ".join(f'"{g}"' for g in sorted(_submitted_guesses))
        )

    recent_obs = _frame_observations[-_MAX_CONTEXT_FRAMES:]
    if recent_obs:
        obs_block = "\n".join(
            f"  Frame {i + 1}: {obs}" for i, obs in enumerate(recent_obs)
        )
        context_parts.append(f"Prior frame observations:\n{obs_block}")

    context_parts.append(
        "Now analyze the NEW frame below. Fill all fields in FrameAnalysis JSON."
    )

    user_context = "\n\n".join(context_parts)

    # ------------------------------------------------------------------ #
    # 4. Call Gemini with structured output
    # ------------------------------------------------------------------ #
    try:
        result = await _get_agent().run(
            [user_context, BinaryContent(data=jpeg_bytes, media_type="image/jpeg")],
            output_type=FrameAnalysis,
        )
        analysis: FrameAnalysis = result.output
    except Exception as exc:
        print(f"  [agent] LLM error: {exc}")
        return None

    # ------------------------------------------------------------------ #
    # 5. Record observation for future frames
    # ------------------------------------------------------------------ #
    prev_observation = _frame_observations[-1] if _frame_observations else ""
    context_changed = _scene_changed(prev_observation, analysis.observation)

    _frame_observations.append(analysis.observation)
    # Keep the rolling window bounded
    if len(_frame_observations) > 15:
        _frame_observations.pop(0)

    _log_frame(elapsed_s, analysis, context_changed)

    # ------------------------------------------------------------------ #
    # 6. Decision gates
    # ------------------------------------------------------------------ #

    # Gate A: hold period — observe before committing
    if elapsed_s < _HOLD_PERIOD_S:
        print(f"  [agent] SKIP — hold period ({elapsed_s:.1f}s < {_HOLD_PERIOD_S}s)")
        return None

    # Gate B: no guess from model
    if not analysis.guess:
        print("  [agent] SKIP — model returned no guess")
        return None

    # Gate C: confidence threshold (escalates after 60 s)
    threshold = _CONFIDENCE_LOW if elapsed_s >= _ESCALATION_AFTER_S else _CONFIDENCE_HIGH
    if analysis.confidence < threshold:
        print(
            f"  [agent] SKIP — confidence {analysis.confidence:.2f} < {threshold:.2f} "
            f"(threshold at {elapsed_s:.1f}s)"
        )
        return None

    # Gate D: no duplicate guesses
    normalized_guess = analysis.guess.strip().lower()
    if normalized_guess in _submitted_guesses:
        print(f"  [agent] SKIP — already submitted {analysis.guess!r}")
        return None

    # ------------------------------------------------------------------ #
    # 7. Commit the guess
    # ------------------------------------------------------------------ #
    _submitted_guesses.add(normalized_guess)
    print(f"  [agent] GUESS → {analysis.guess!r} (conf={analysis.confidence:.2f}, {elapsed_s:.1f}s)")
    return analysis.guess
