"""System prompt and analysis logic for the charades guessing agent — V3.

=== EDIT THIS FILE ===

V3 Architecture
---------------
- Single unified agent: Gemini 2.5 Flash sees each frame and guesses directly
- Time-based rate limiter: max ~0.67 LLM calls/sec, no motion-detection blind spots
- 3-phase gating: observe (0-3s) → normal (3-50s, conf >= 0.70) → aggressive (50-120s, conf >= 0.45)
- Feedback loop: set_last_result() excludes confirmed-wrong guesses from all future calls
- Semantic dedup: Jaccard similarity prevents "swim" → "swimming" waste

Scoring model reminder
----------------------
  score = guesses_remaining * speed_bonus
  => fewer wasted guesses + faster correct answer = maximum points
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core import Frame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rate limiter — prevents concurrent LLM calls and caps token spend
_MIN_CALL_INTERVAL_S: float = 1.5   # max ~0.67 LLM calls/sec
_FIRST_N_ALWAYS: int = 3            # always call LLM for first N frames
_LLM_TIMEOUT_S: float = 15.0       # hard timeout per LLM call

# Phase boundaries (seconds elapsed)
_PHASE_OBSERVE_END: float = 3.0
_PHASE_AGGRESSIVE_START: float = 50.0

# Confidence thresholds per phase
_CONF_NORMAL: float = 0.70
_CONF_AGGRESSIVE: float = 0.45

# Context window
_MAX_CONTEXT_OBS: int = 8

# Debug mode — set DEBUG_FRAMES=1 env var to save every analyzed frame to disk
_DEBUG_FRAMES: bool = os.environ.get("DEBUG_FRAMES", "").strip() in ("1", "true", "yes")
_DEBUG_DIR: Path | None = None


def _get_debug_dir() -> Path:
    global _DEBUG_DIR
    if _DEBUG_DIR is None:
        _DEBUG_DIR = Path("debug_frames")
        _DEBUG_DIR.mkdir(exist_ok=True)
        print(f"  [debug] Saving analyzed frames to {_DEBUG_DIR.resolve()}")
    return _DEBUG_DIR

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
    reasoning: str = Field(
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
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in this guess (0.0 = wild guess, 1.0 = certain). "
            "Set above 0.70 when the accumulated evidence clearly points to one answer. "
            "If the visual evidence is unambiguous — a held pose you can read immediately — "
            "commit with confidence >= 0.80."
        ),
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert charades judge watching a LIVE CAMERA FEED of a performer playing charades.

## Your Job
Interpret the performer's body language, gestures, and mime to identify the secret word or phrase.
There is NO audio. You must rely entirely on visual cues.

## Charades Conventions to Recognize
- **Category sign**: pulling ear = "sounds like"; tugging collar = "it's a song"; \
book open in hands = "it's a book/movie"; fingers as quotation marks = exact phrase
- **Syllable count**: holding up N fingers near the chin = N syllables
- **Word count**: holding up N fingers = N-word phrase
- **Which word**: holding up N fingers after word-count = working on the Nth word
- **Small word**: pinching fingers together = a short/small word (the, a, and, in…)
- **Longer/shorter**: stretching hands apart or bringing together = make the guess longer or shorter
- **Action verbs**: miming a physical activity (swimming, flying, eating) → the word IS \
that verb or strongly related to it
- **Compound words**: performer may act out each part separately

## Non-Standard Performers
Many performers do NOT use formal charades conventions. They simply act out the concept directly:
- A peace sign held up means "peace" — not a syllable count
- Clapping hands means "applause" or "clapping" — not a charades convention
- Someone waving means "hello" or "wave" — they're being literal, not symbolic
Focus on WHAT the performer is physically doing. Ask: what concept does this action literally represent?

## Held Poses & Hand Shapes Are High-Signal
When a performer freezes in a deliberate pose, that IS the clue. A held gesture means they are \
showing you the answer, not transitioning. Do not wait for more evidence — recognize and commit.

Pay special attention when the performer holds their hands still in front of the camera:
- They are deliberately forming a SHAPE with their hands for you to read. Study the exact \
finger positions, gaps between fingers, and overall silhouette.
- Common hand shapes: heart (two hands forming a heart), gun/pistol (finger gun), \
binoculars (circles over eyes), camera (framing rectangle), butterfly (interlocked thumbs \
with spread fingers), phone (thumb + pinky extended), OK sign, thumbs up/down, \
antlers/horns, shadow puppet animals, letters of the alphabet (ASL or finger spelling).
- If the hands are cupped, curved, or arranged into an outline, ask: what OBJECT or SYMBOL \
does this silhouette resemble? The answer is often the shape itself or what the shape represents.
- Stillness is emphasis — the longer they hold it, the more certain they are that the shape \
IS the answer. Treat a frozen hand shape as the strongest single-frame signal you can get.

## Answer Complexity
Answers are NOT always simple nouns. They may be:
- Multi-word phrases ("leap of faith", "swimming upstream")
- Movie or book titles ("The Dark Knight", "Gone with the Wind")
- Idioms or proverbs ("kick the bucket", "bite the bullet")
- Abstract concepts ("freedom", "nostalgia")
- Verbs or gerunds ("skydiving", "negotiating")

## Strategy
- Accumulate evidence across frames before guessing. One frame is rarely enough.
- If you see a syllable or word-count gesture, factor it into every future guess.
- Prefer a specific, concrete answer over a vague one (e.g. "Titanic" > "ship movie").
- If uncertain between two options, output null and wait for more frames.

## Handling Wrong Guesses
When you see prior wrong guesses in the context, use them as signal:
- A wrong single-word guess might mean the answer is a PHRASE containing that word \
(e.g. "swimming" was wrong → try "swimming upstream" or "keep swimming").
- A wrong phrase might mean you have the right idea but wrong wording \
(e.g. "leap of faith" was wrong → try "take a leap" or "blind faith").
- If multiple similar guesses were all wrong, PIVOT to a completely different interpretation.
- Consider that the performer may be acting out syllables or "sounds like" — the answer may \
be phonetically related, not literally what they are miming.
- Each guess costs points. Do NOT submit a variation unless you have new visual evidence.

## Scoring & Live Mode
Your score = guesses_remaining * speed_bonus. This means:
- Every wrong guess directly reduces your score — be selective, not trigger-happy.
- Faster correct answers earn a higher speed bonus — don't overthink when evidence is clear.
- You have a maximum of 10 guesses per round. If you hit the limit (HTTP 429), the round is over.
- Balance urgency against accuracy: commit quickly on strong evidence, hold back on weak evidence.

## Confidence Calibration
- If the visual evidence is unambiguous (clear held pose, obvious mime), set confidence >= 0.80.
- If you are reasonably sure but could imagine 1-2 other answers, set 0.60-0.79.
- If you are genuinely uncertain, output null guess and confidence < 0.50.
- Do NOT artificially suppress confidence. If you can read the gesture clearly, say so.

## Output Contract
You MUST respond with valid JSON matching the FrameAnalysis schema.
- `observation`: what you see in THIS frame (precise about body parts and gestures)
- `reasoning`: synthesize ALL prior observations + this frame to build toward the answer
- `guess`: your best answer right now, or null if not confident enough
- `confidence`: 0.0-1.0; exceed 0.60 when evidence reasonably supports your guess, \
commit above 0.70 when evidence is clear
"""

# ---------------------------------------------------------------------------
# Model setup (lazy — created on first analyze() call)
# ---------------------------------------------------------------------------

_agent: Agent[Any, FrameAnalysis] | None = None


def _get_agent() -> Agent[Any, FrameAnalysis]:
    global _agent
    if _agent is None:
        _agent = Agent(
            OpenAIChatModel(
                "google/gemini-2.5-flash",
                provider=OpenAIProvider(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("LLM_API_KEY", ""),
                ),
            ),
            system_prompt=SYSTEM_PROMPT,
            retries=3,
        )
    return _agent


# ---------------------------------------------------------------------------
# Module-level state (resets automatically on process restart = new round)
# ---------------------------------------------------------------------------

_round_start: datetime | None = None
_frame_observations: list[str] = []   # (timestamp_label, observation), capped at 15
_submitted_guesses: list[str] = []    # normalized lowercase, in submission order
_wrong_guesses: list[str] = []        # confirmed 409 from judge
_round_solved: bool = False
_last_llm_call_s: float = -999.0      # for rate limiter
_frame_count: int = 0
_wall_start: float | None = None      # monotonic clock at round start


# ---------------------------------------------------------------------------
# Feedback hook — called by __main__.py after each API response
# ---------------------------------------------------------------------------


def set_last_result(guess: str, correct: bool) -> None:
    """Feed judge result back into the agent state.

    Call this from __main__.py immediately after client.guess() returns:
        set_last_result(guess, result.correct)
    """
    global _round_solved
    normalized = guess.strip().lower()
    if correct:
        _round_solved = True
        print(f"  [feedback] CORRECT! Round solved: {guess!r}")
    else:
        if normalized not in _wrong_guesses:
            _wrong_guesses.append(normalized)
        print(
            f"  [feedback] WRONG — {guess!r} added to exclusion list "
            f"({len(_wrong_guesses)} wrong so far)"
        )


# ---------------------------------------------------------------------------
# Semantic deduplication
# ---------------------------------------------------------------------------

_STEM_SUFFIXES = ("ing", "tion", "er", "ed", "ness", "ment", "ly", "s")
_ARTICLES = ("the ", "a ", "an ")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    for article in _ARTICLES:
        if text.startswith(article):
            text = text[len(article):]
    return text


def _stem(word: str) -> str:
    for suffix in _STEM_SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _word_set(text: str) -> set[str]:
    return {_stem(w) for w in _normalize(text).split() if len(w) > 1}


def _jaccard(a: str, b: str) -> float:
    ws_a, ws_b = _word_set(a), _word_set(b)
    union = ws_a | ws_b
    return len(ws_a & ws_b) / len(union) if union else 0.0


def _is_semantic_dup(candidate: str, existing: list[str]) -> bool:
    return any(_jaccard(candidate, g) >= 0.70 for g in existing)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_frame(elapsed_s: float, analysis: FrameAnalysis) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  [agent] t={elapsed_s:.1f}s")
    print(f"  OBSERVATION : {analysis.observation}")
    print(f"  REASONING   : {analysis.reasoning}")
    print(f"  GUESS       : {analysis.guess!r}  [conf={analysis.confidence:.2f}]")
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def analyze(frame: Frame) -> str | None:
    """Analyze a single charades frame and return a guess, or None to skip.

    Called by the main loop for every sampled frame. Coordinates the rate
    limiter, unified LLM agent, and phase-gated arbiter.
    """
    global _round_start, _last_llm_call_s, _frame_count, _wall_start

    if _round_solved:
        return None

    now = frame.timestamp
    wall_now = time.monotonic()

    if _round_start is None:
        _round_start = now
        _wall_start = wall_now

    elapsed_s = (now - _round_start).total_seconds()
    wall_elapsed = wall_now - _wall_start if _wall_start else 0.0
    _frame_count += 1

    # Detect frame staleness (frame timestamp vs wall clock drift)
    drift = abs(wall_elapsed - elapsed_s)
    if drift > 3.0:
        print(
            f"  [agent] WARNING: frame drift={drift:.1f}s "
            f"(wall={wall_elapsed:.1f}s, frame_ts={elapsed_s:.1f}s) — "
            f"frame may be stale"
        )

    print(
        f"  [agent] Frame #{_frame_count} | "
        f"elapsed={elapsed_s:.1f}s | wall={wall_elapsed:.1f}s | "
        f"size={frame.image.size}"
    )

    # ------------------------------------------------------------------ #
    # 1. Rate limiter — skip if last LLM call was too recent
    # ------------------------------------------------------------------ #
    if _frame_count > _FIRST_N_ALWAYS and (elapsed_s - _last_llm_call_s) < _MIN_CALL_INTERVAL_S:
        return None

    # ------------------------------------------------------------------ #
    # 2. Save debug frame to disk (if DEBUG_FRAMES=1)
    # ------------------------------------------------------------------ #
    if _DEBUG_FRAMES:
        debug_dir = _get_debug_dir()
        debug_path = debug_dir / f"frame_{_frame_count:04d}_t{elapsed_s:.1f}s.jpg"
        frame.image.save(str(debug_path), format="JPEG", quality=90)
        print(f"  [debug] Saved {debug_path.name}")

    # ------------------------------------------------------------------ #
    # 3. Build context string
    # ------------------------------------------------------------------ #
    context_parts: list[str] = [
        f"Round elapsed: {elapsed_s:.1f}s | Guesses used: {len(_submitted_guesses)}/10"
    ]

    if _wrong_guesses:
        context_parts.append(
            "Wrong guesses confirmed by judge — do NOT repeat or guess anything similar: "
            + ", ".join(f'"{g}"' for g in _wrong_guesses)
        )

    if _submitted_guesses:
        context_parts.append(
            "Already submitted (do NOT repeat): "
            + ", ".join(f'"{g}"' for g in _submitted_guesses)
        )

    if _frame_observations:
        obs_block = "\n".join(
            f"  {label}" for label in _frame_observations[-_MAX_CONTEXT_OBS:]
        )
        context_parts.append(f"Prior observations:\n{obs_block}")

    context_parts.append(
        "Analyze the NEW frame below. Fill all FrameAnalysis fields."
    )

    user_context = "\n\n".join(context_parts)

    # ------------------------------------------------------------------ #
    # 4. Convert PIL Image → JPEG bytes
    # ------------------------------------------------------------------ #
    buf = io.BytesIO()
    img = frame.image
    if img.width > 1280 or img.height > 720:
        img.thumbnail((1280, 720))
    img.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    # ------------------------------------------------------------------ #
    # 5. Call unified LLM agent (with timeout)
    # ------------------------------------------------------------------ #
    _last_llm_call_s = elapsed_s
    llm_start = time.monotonic()

    try:
        result = await asyncio.wait_for(
            _get_agent().run(
                [user_context, BinaryContent(data=jpeg_bytes, media_type="image/jpeg")],
                output_type=FrameAnalysis,
            ),
            timeout=_LLM_TIMEOUT_S,
        )
        analysis: FrameAnalysis = result.output
    except asyncio.TimeoutError:
        llm_dur = time.monotonic() - llm_start
        print(f"  [agent] LLM TIMEOUT after {llm_dur:.1f}s — skipping frame")
        return None
    except Exception as exc:
        llm_dur = time.monotonic() - llm_start
        print(f"  [agent] LLM error after {llm_dur:.1f}s: {exc}")
        return None

    llm_dur = time.monotonic() - llm_start
    print(f"  [agent] LLM responded in {llm_dur:.1f}s")

    # ------------------------------------------------------------------ #
    # 6. Record observation
    # ------------------------------------------------------------------ #
    _frame_observations.append(f"t={elapsed_s:.1f}s: {analysis.observation}")
    if len(_frame_observations) > 15:
        _frame_observations.pop(0)

    _log_frame(elapsed_s, analysis)

    # ------------------------------------------------------------------ #
    # 6. Phase gate
    # ------------------------------------------------------------------ #
    if elapsed_s < _PHASE_OBSERVE_END:
        print(f"  [agent] SKIP — observe phase ({elapsed_s:.1f}s < {_PHASE_OBSERVE_END}s)")
        return None

    if not analysis.guess:
        print("  [agent] SKIP — model returned no guess")
        return None

    threshold = _CONF_AGGRESSIVE if elapsed_s >= _PHASE_AGGRESSIVE_START else _CONF_NORMAL
    if analysis.confidence < threshold:
        print(
            f"  [agent] SKIP — confidence {analysis.confidence:.2f} < {threshold:.2f} "
            f"(t={elapsed_s:.1f}s)"
        )
        return None

    # ------------------------------------------------------------------ #
    # 7. Semantic dedup
    # ------------------------------------------------------------------ #
    normalized = analysis.guess.strip().lower()

    if normalized in _submitted_guesses:
        print(f"  [agent] SKIP — already submitted {analysis.guess!r}")
        return None

    if _is_semantic_dup(analysis.guess, _wrong_guesses):
        print(f"  [agent] SKIP — {analysis.guess!r} is semantically similar to a wrong guess")
        return None

    if _is_semantic_dup(analysis.guess, _submitted_guesses):
        print(f"  [agent] SKIP — {analysis.guess!r} is semantically similar to a submitted guess")
        return None

    # ------------------------------------------------------------------ #
    # 8. Commit the guess
    # ------------------------------------------------------------------ #
    _submitted_guesses.append(normalized)
    print(f"  [agent] GUESS → {analysis.guess!r} (conf={analysis.confidence:.2f}, t={elapsed_s:.1f}s)")
    return analysis.guess
