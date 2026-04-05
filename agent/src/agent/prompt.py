"""System prompt and analysis logic for the charades guessing agent — V3.

=== EDIT THIS FILE ===

V3 Architecture
---------------
- Single unified agent: Claude 3.5 Sonnet sees each frame and guesses directly
- Time-based rate limiter: caps LLM calls vs. round elapsed; practice/live default ~3 FPS frames
- Phase gating: short observe window, then confidence thresholds (normal vs. late round)
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
_MIN_CALL_INTERVAL_S: float = 1.0   # allow ~1 LLM call/s so 3 FPS yields distinct moments
_FIRST_N_ALWAYS: int = 10           # always call LLM for first N frames (diverse early samples)
_LLM_TIMEOUT_S: float = 15.0       # hard timeout per LLM call

# Phase boundaries (seconds elapsed)
_PHASE_OBSERVE_END: float = 1.0
_PHASE_AGGRESSIVE_START: float = 20.0

# Confidence thresholds per phase
_CONF_NORMAL: float = 0.55
_CONF_AGGRESSIVE: float = 0.35

# Context window (prior observation lines sent to the model)
_MAX_CONTEXT_OBS: int = 12

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
            "frames in the context) to build toward a charades answer. Weigh standard "
            "charades signals (word count, syllables, category, sounds-like) when present; "
            "otherwise prioritize literal mime and held poses."
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


## Non-Standard Performers
Many performers do NOT use formal charades conventions. They simply act out the concept directly:
- A peace sign held up means "peace" — not a syllable count
- Clapping hands means "applause" or "clapping" — not a charades convention
- Someone waving means "hello" or "wave" — they're being literal, not symbolic
Focus on WHAT the performer is physically doing. Ask: what concept does this action literally represent?

## Standard Charades Conventions (common prompts)
When the performer uses classic signals, treat them as constraints on the answer. If a signal \
conflicts with a purely literal read, prefer the signal that explains multiple gestures together.

**Meta / structure**
- **Number of words**: fingers held up = how many words in the phrase (combine with other clues).
- **Which word**: tap shoulder then hold up N = working on word N; point at someone = "you" / \
  second person, etc., when clearly intentional.
- **Syllables**: tap fingers on forearm (or thigh) once per syllable of the target word.
- **Sounds like / homophone**: tug or point at ear — next mime sounds like the answer (not literal).
- **Small word**: thumb and index finger pinched small gap — article or short word (a, the, of, in).
- **Rhymes with**: sometimes indicated by pairing gestures; use when ear-pull or "sounds like" appears.

**Category (what kind of thing it is)**
- **Book title**: hands flat as if opening a book, or pantomime reading.
- **Movie / TV**: crank an old film camera, or rectangle frame with hands ("screen").
- **Song / sung**: cup hand to ear or mime singing into a mic.
- **Play / theater**: spirit fingers "jazz hands" or tragedy/comedy masks pose.
- **Quote / phrase**: make "quotation marks" in the air with fingers.

Combine category + syllable count + main mime. Example: 2 words + movie + swimming mime → \
guess a two-word film title about swimming.

## Observation Order
When analyzing each new frame, scan in this exact order:
1. **Which body part is most prominent?** (hands close to camera, arms extended, full-body pose, face)
2. **What shape or motion is it making?** (static held shape vs. active movement)
3. **Any standard charades prompt?** (word-count fingers, syllable taps, category sign, ear-pull, \
small-word pinch — see "Standard Charades Conventions" above.)
4. **Literal meaning:** What object, action, or emotion does the mime directly show?
5. **Cross-reference prior observations.** Does this frame match earlier frames? Note similarities \
explicitly in your reasoning.

## Repetition Signals Certainty
If the same gesture or pose appears across 2 or more prior observations, treat it as a strong \
signal — raise your confidence and lean toward guessing. Repetition across frames increases \
certainty meaningfully. When you see the same gesture repeated, set confidence >= 0.75 and \
provide a guess if you have a reasonable candidate.

## Lean Toward Guessing
When you have a reasonable candidate and the evidence is building, prefer guessing over waiting. \
If you can narrow it down to 2-3 options, pick the most likely one rather than staying silent. \
That said, if the gesture is genuinely ambiguous, it is fine to hold back one more frame.

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

## Gesture Decision Chain
When you identify a gesture, follow the decision chain for its category exactly. \
Do not move to the next step until the current one is resolved.

### A. Direct Full-Body Mime (Whole Concept)
Trigger: performer acts out an activity with their whole body — acting it out literally.
1. Name the action in plain language: "person swimming", "person flying a kite".
2. Try the gerund form first: "swimming", "flying".
3. If wrong, try the noun ("swimmer", "kite"), then a phrase ("swimming upstream", \
   "go fly a kite").
4. If still wrong, ask: could this be a movie/book title that contains this action?

### B. Static Hand Shape (Held Pose)
Trigger: hands held still in a deliberate shape for more than one consecutive frame.
1. Identify the exact shape: heart, gun, phone, OK, thumbs-up/down, binoculars, camera \
   frame, butterfly, letter of the alphabet, number, animal shadow puppet, etc.
2. Map the shape to its symbolic meaning first (heart → love/romance, gun → shoot/bang/kill, \
   phone → call/talk, OK → okay/perfect, thumbs-up → approve/good, binoculars → look/watch).
3. **Hand or finger held against the mouth/lips is a COMMUNICATION GESTURE, not an action.** \
   Map these before considering any breath-based or musical interpretation: \
   finger/fist to lips → quiet/silence/shh/secret, \
   finger to lips + wide eyes → surprise/secret, \
   hand cupped around mouth → shout/loud/announce. \
   Do NOT interpret sustained stillness near the mouth as blowing, breathing, or playing an instrument \
   unless the hand is clearly shaped like an instrument (e.g., flute grip, trumpet shape).
4. If the shape is a letter or number, treat it as a spelling clue — note it and wait for more \
   letters, or check if a single letter is itself a word (A, I) or abbreviation.
5. Commit immediately once identified — held shapes are the highest-confidence single-frame signal.

### C. Repetitive Looping Motion
Trigger: the same motion repeats 2+ times in a row.
1. After the SECOND repetition, treat it as confirmed — the repeated action IS the answer.
2. Guess the verb form of the repeated action ("rowing", "chopping", "spinning").
3. If wrong, try the noun ("oar", "axe", "wheel") or a compound phrase that uses the verb \
   ("rowing upstream", "chop chop", "spinning wheel").
4. Do NOT wait for a third repetition — repetition = certainty signal, act on it.

### D. Pointing / Directional Gestures
Trigger: performer points to a body part, direction, or object in the environment.
1. If pointing to a body part: the answer likely IS that body part word \
   (e.g., point to eye → "eye", point to heart → "love" or "heart").
2. If pointing up/down/around abstractly: map to directional concepts \
   (up → sky/above/high, down → underground/low/fall, around → circle/globe/spin).
3. If pointing to an environmental object (wall, floor, window): use the object name as the \
   answer or a clue component.
4. Combine the pointed-to concept with other visible gestures before guessing.

### E. Emotional / Facial Expression
Trigger: performer's face is the dominant signal — exaggerated expression, no major body action.
1. Name the emotion precisely: joy, sadness, disgust, fear, surprise, anger, longing, pride.
2. Try the emotion word as the answer.
3. If wrong, try an idiom or phrase centered on that emotion \
   (happy → "happy-go-lucky", sad → "cry me a river", angry → "see red").
4. If the expression changes mid-sequence, treat the FINAL expression as the answer \
   (the performer built up to it deliberately).

### F. No Recognizable Gesture (Unclear / Transitioning)
Trigger: performer appears to be thinking, resetting, or moving between poses.
1. Do NOT guess during transitions — return null and wait.
2. Check prior observations: was there a clear gesture in the last 3 frames? If yes, revisit it.
3. If more than 10 seconds have passed with no clear gesture, lower your confidence threshold \
   and guess based on the strongest prior observation.

## Answer Complexity
Answers are NOT always simple nouns. They may be:
- Multi-word phrases ("leap of faith", "swimming upstream")
- Movie or book titles ("The Dark Knight", "Gone with the Wind")
- Idioms or proverbs ("kick the bucket", "bite the bullet")
- Abstract concepts ("freedom", "nostalgia")
- Verbs or gerunds ("skydiving", "negotiating")

## Strategy
- Accumulate evidence across frames before guessing. One frame is rarely enough.
- Prefer a specific, concrete answer over a vague one (e.g. "Titanic" > "ship movie").
- If uncertain between two options, output null and wait for more frames.

## Handling Wrong Guesses
When you see prior wrong guesses in the context, use them as signal:
- A wrong single-word guess might mean the answer is a PHRASE containing that word \
(e.g. "swimming" was wrong → try "swimming upstream" or "keep swimming").
- A wrong phrase might mean you have the right idea but wrong wording \
(e.g. "leap of faith" was wrong → try "take a leap" or "blind faith").
- If multiple similar guesses were all wrong, PIVOT to a completely different interpretation.
- Each guess costs points. Do NOT submit a variation unless you have new visual evidence.

## Scoring & Live Mode
Your score = guesses_remaining * speed_bonus. This means:
- Every wrong guess directly reduces your score — be selective, not trigger-happy.
- Faster correct answers earn a higher speed bonus — don't overthink when evidence is clear.
- You have a maximum of 10 guesses per round. If you hit the limit (HTTP 429), the round is over.
- Balance urgency against accuracy: commit quickly on strong evidence, hold back on weak evidence.

## Confidence Calibration
- If the visual evidence is clear (held pose, obvious mime), set confidence >= 0.80.
- If you have a leading candidate with 1-2 alternatives, set 0.55-0.79 and provide the guess.
- Output null if the gesture is genuinely ambiguous and you need more frames to decide.
- Prefer guessing over silence when evidence is building, but don't force a guess on weak evidence.
- Don't suppress confidence when the gesture is readable — trust your read and commit.

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
                "anthropic/claude-3-5-sonnet",
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
