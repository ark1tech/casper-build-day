"""System prompt and analysis logic for the charades guessing agent — V2.

=== EDIT THIS FILE ===

V2 Architecture
---------------
- Frame Screener  : PIL pixel-diff (<10 ms); gates LLM calls; maintains key-frame buffer
- Observer Agent  : Gemini 2.5 Flash, structured observation-only (no guessing)
- Synthesizer     : Gemini 2.5 Flash + CoT; background asyncio.Task, fires every 4 s
- Convention Det. : State machine (unclear → standard_charades / free_form / mixed)
- Guess Arbiter   : 4-phase time gating, rolling guess budget, semantic dedup, early-exit

Scoring model reminder
----------------------
  score = guesses_remaining * speed_bonus
  => fewer wasted guesses + faster correct answer = maximum points
"""

from __future__ import annotations

import asyncio
import io
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal

from PIL import ImageChops, ImageStat
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core import Frame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Frame screener
_MOTION_THRESHOLD: float = 0.015       # pixel-diff / max_possible; calibrate in practice
_KEY_FRAME_BUFFER_SIZE: int = 10        # max 3-s windows retained
_HEARTBEAT_INTERVAL_S: float = 3.0     # force-process even if no motion
_FIRST_N_FRAMES_ALWAYS: int = 10       # always process the very first second

# Synthesizer
_SYNTH_CADENCE_S: float = 4.0
_MAX_CONTEXT_OBS: int = 12

# Phase boundaries (seconds elapsed)
_PHASE_OBSERVE_END: float = 5.0
_PHASE_PATIENT_END: float = 40.0
_PHASE_ACTIVE_END: float = 80.0

# Confidence thresholds per phase
_CONF_PATIENT: float = 0.85
_CONF_ACTIVE: float = 0.65
_CONF_DESPERATION: float = 0.40

# Guess budgets (soft caps; unused patient rolls into active)
_BUDGET_PATIENT: int = 3
_BUDGET_ACTIVE: int = 4              # combined patient+active max = 7

# Early-exit: same top candidate at >= 0.90 for this many consecutive observer calls
_EARLY_EXIT_MIN_FRAMES: int = 3
_EARLY_EXIT_CONF: float = 0.90

# Semantic dedup Jaccard similarity threshold
_SEMANTIC_DUP_J: float = 0.70

# Convention detector
_STYLE_LOCK_S: float = 5.0           # lock style after this many seconds
_STYLE_DOWNGRADE_S: float = 20.0     # standard_charades with no signals → mixed

# ---------------------------------------------------------------------------
# Model factory (lazy — created on first call so the module imports cleanly)
# ---------------------------------------------------------------------------


def _make_openrouter_model(model_id: str = "google/gemini-2.5-flash") -> OpenAIChatModel:
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("LLM_API_KEY", ""),
        ),
    )


_observer_agent: Agent[Any, Observation] | None = None
_synthesizer_agent: Agent[Any, SynthesisResult] | None = None


def _get_observer() -> Agent[Any, Observation]:
    global _observer_agent
    if _observer_agent is None:
        _observer_agent = Agent(
            _make_openrouter_model(),
            system_prompt=_OBSERVER_SYSTEM_PROMPT,
            retries=3,
        )
    return _observer_agent


def _get_synthesizer() -> Agent[Any, SynthesisResult]:
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = Agent(
            _make_openrouter_model(),
            system_prompt=_SYNTHESIZER_SYSTEM_PROMPT,
            retries=3,
        )
    return _synthesizer_agent


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Structured output from the observer agent for a single frame."""

    gesture_phase: Literal["rest", "preparation", "stroke", "hold", "retraction"] = Field(
        description="Which phase of the gesture cycle the performer is in."
    )
    hand_position: str = Field(
        description=(
            'Where the hands/arms are. E.g. "near face", "extended forward", '
            '"at sides", "above head", "touching ear".'
        )
    )
    charades_signal: str | None = Field(
        default=None,
        description=(
            "Conventional charades signal detected, or null. "
            'Examples: "syllable_count:3", "sounds_like", "category:movie", '
            '"word_count:2", "which_word:1", "small_word".'
        ),
    )
    action_description: str = Field(
        description="The physical action being depicted (e.g. 'swinging a bat')."
    )
    scene_interpretation: str = Field(
        description="The scene or concept this could represent (e.g. 'baseball')."
    )
    negative_space: str = Field(
        description="What the performer is clearly NOT doing (narrows the guess space)."
    )
    performance_style: Literal["standard_charades", "free_form", "mixed", "unclear"] = Field(
        description=(
            "How the performer communicates: conventional charades signals, "
            "free-form acting, a mix, or unclear so far."
        )
    )


class Candidate(BaseModel):
    answer: str
    confidence: float = Field(description="Confidence 0.0–1.0.")
    reasoning: str

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class SynthesisResult(BaseModel):
    """Ranked guess candidates produced by the synthesizer agent."""

    candidates: list[Candidate] = Field(
        description="Ranked guess candidates, highest confidence first."
    )
    top_pick: str | None = Field(
        default=None,
        description=(
            "The single best guess to submit now, or null if not confident enough. "
            "Only set this when the best candidate has a clear confidence gap (>0.15) over the next."
        ),
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_OBSERVER_SYSTEM_PROMPT = """\
You are a precision charades observer. You watch a LIVE CAMERA FEED and produce \
structured observations. There is NO audio.

Your ONLY job: observe and classify. Do NOT guess the answer.

## Spatial Anchor Priority (check in this order)
1. Finger count near chin → syllable count ("syllable_count:N")
2. Finger count held up → word count or which word ("word_count:N", "which_word:N")
3. Ear touch → sounds-like modifier ("sounds_like")
4. Repeated cyclic motion → ongoing activity (swimming, stirring, running)
5. Single decisive gesture → category sign or specific mime
6. Facial expression → only relevant for emotions/characters; low priority otherwise

## Gesture Phase Classification
- rest: hands at sides, neutral posture. LOW SIGNAL.
- preparation: hands moving toward starting position. LOW SIGNAL.
- stroke: the meaningful gesture stroke. HIGHEST SIGNAL.
- hold: performer frozen at apex deliberately. HIGH SIGNAL.
- retraction: hands returning to rest. LOW SIGNAL.

## Performance Style
- standard_charades: uses finger counting, ear pulling, deliberate symbolic gestures with pauses
- free_form: continuous narrative action, no symbolic gestures
- mixed: some conventional signals with free-form acting
- unclear: not enough evidence yet

## Negative Space
Explicitly state what the performer is clearly NOT doing. Examples:
- "not holding any imaginary object" → eliminates tool/prop-based answers
- "not moving from their spot" → eliminates locomotion answers
- "no exaggerated facial expression" → eliminates emotion/character answers

Return valid JSON matching the Observation schema. Be precise about body parts and positions.
"""

_SYNTHESIZER_SYSTEM_PROMPT = """\
You are a charades synthesis expert. You receive structured observations of a performer \
and must reason carefully to produce ranked guess candidates.

## Your Process (think step by step)
1. Review all observations chronologically.
   Weight stroke/hold phase observations 3× higher than rest/transition.
2. Check for charades signals (syllable count, word count, sounds-like).
   If found, treat them as hard constraints on all candidates.
3. Note the performance style and apply channel weighting:
   - standard_charades → 70% charades signals + 30% scene interpretation
   - free_form → 10% charades signals + 90% scene interpretation
   - mixed / unclear → 50/50
4. Cross-reference negative_space fields — eliminate answers that contradict them.
5. Exclude confirmed-wrong guesses AND semantically similar answers.
6. Produce 2–4 ranked Candidates with specific, evidence-backed reasoning.

## Candidate Ranking Rules
- Rank by overall evidence strength across all frames, not just the latest.
- Set top_pick only if the best candidate has a clear gap (confidence > 0.15 above second).
- Be specific: "Titanic" beats "ship movie". "skydiving" beats "falling from height".
- Answers can be single words, phrases, movie/book titles, idioms, or abstract concepts.
- No filler ("I think", "maybe") in the answer field — just the answer text itself.
"""

# ---------------------------------------------------------------------------
# Module-level state  (resets automatically on process restart = new round)
# ---------------------------------------------------------------------------

# Round timing
_round_start: datetime | None = None

# Observer accumulator
_observations: list[tuple[float, Observation]] = []   # (elapsed_s, obs)

# Guess tracking
_submitted_guesses: list[str] = []    # normalized lowercase, in submission order
_wrong_guesses: list[str] = []        # confirmed 409 from judge
_round_solved: bool = False

# Frame screener
_prev_frame_small: Any = None          # PIL.Image grayscale 160×120
_frame_count: int = 0
_last_heartbeat_s: float = -999.0
_observer_running: bool = False

# Key-frame buffer: (elapsed_s, motion_magnitude, jpeg_bytes)
_key_frames: list[tuple[float, float, bytes]] = []

# Convention detector
_performance_style: str = "unclear"
_style_votes: list[str] = []
_last_standard_signal_s: float = -999.0

# Synthesizer
_synth_task: asyncio.Task[None] | None = None
_synth_candidates: list[Candidate] = []
_synth_last_run_s: float = -999.0

# Early-exit streak tracking
_early_exit_used: bool = False
_early_exit_streak_answer: str = ""
_early_exit_streak_count: int = 0

# Phase guess counts (for rolling budget)
_phase_guess_counts: dict[str, int] = {"patient": 0, "active": 0, "desperation": 0}


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
# Frame screener
# ---------------------------------------------------------------------------


def _screen_frame(img: Any, elapsed_s: float) -> tuple[bool, float, str]:
    """Compute motion between current and previous frame.

    Returns (is_significant, motion_magnitude, motion_region).
    Always updates the key-frame buffer as a side effect.
    """
    global _prev_frame_small, _frame_count, _last_heartbeat_s

    _frame_count += 1

    curr_small = img.resize((160, 120)).convert("L")
    motion_magnitude: float = 1.0
    motion_region: str = "full"

    if _prev_frame_small is not None:
        diff = ImageChops.difference(_prev_frame_small, curr_small)
        total = ImageStat.Stat(diff).sum[0]
        motion_magnitude = total / (160 * 120 * 255)

        w, h = diff.size  # 160 × 120
        region_sums = {
            "upper": ImageStat.Stat(diff.crop((0, 0, w, h // 2))).sum[0],
            "lower": ImageStat.Stat(diff.crop((0, h // 2, w, h))).sum[0],
            "left":  ImageStat.Stat(diff.crop((0, 0, w // 2, h))).sum[0],
            "right": ImageStat.Stat(diff.crop((w // 2, 0, w, h))).sum[0],
        }
        motion_region = max(region_sums, key=lambda k: region_sums[k])

    _prev_frame_small = curr_small

    # Always update the key-frame buffer regardless of significance
    _update_key_frame_buffer(img, elapsed_s, motion_magnitude)

    # Determine significance
    if _frame_count <= _FIRST_N_FRAMES_ALWAYS:
        return True, motion_magnitude, motion_region

    if elapsed_s - _last_heartbeat_s >= _HEARTBEAT_INTERVAL_S:
        _last_heartbeat_s = elapsed_s
        return True, motion_magnitude, "heartbeat"

    if motion_magnitude >= _MOTION_THRESHOLD:
        return True, motion_magnitude, motion_region

    return False, motion_magnitude, motion_region


def _update_key_frame_buffer(img: Any, elapsed_s: float, motion_magnitude: float) -> None:
    """Store highest-motion frame per 3-second window."""
    window = int(elapsed_s // 3.0)

    existing_idx = next(
        (i for i, entry in enumerate(_key_frames) if int(entry[0] // 3.0) == window),
        None,
    )

    buf = io.BytesIO()
    thumb = img.copy()
    thumb.thumbnail((640, 360))
    thumb.save(buf, format="JPEG", quality=70)
    jpeg_bytes = buf.getvalue()

    if existing_idx is not None:
        if motion_magnitude > _key_frames[existing_idx][1]:
            _key_frames[existing_idx] = (elapsed_s, motion_magnitude, jpeg_bytes)
    else:
        _key_frames.append((elapsed_s, motion_magnitude, jpeg_bytes))
        if len(_key_frames) > _KEY_FRAME_BUFFER_SIZE:
            _key_frames.pop(0)


def _select_key_frames(n: int = 3) -> list[tuple[float, bytes]]:
    """Pick up to n key frames with >=2 s temporal spread, preferring highest motion."""
    if not _key_frames:
        return []

    sorted_by_motion = sorted(_key_frames, key=lambda x: x[1], reverse=True)
    selected: list[tuple[float, bytes]] = []

    for t, _mag, jpeg in sorted_by_motion:
        if all(abs(t - sel_t) >= 2.0 for sel_t, _ in selected):
            selected.append((t, jpeg))
        if len(selected) >= n:
            break

    return sorted(selected, key=lambda x: x[0])  # chronological order


# ---------------------------------------------------------------------------
# Convention detector
# ---------------------------------------------------------------------------


def _update_style(obs: Observation, elapsed_s: float) -> None:
    """Update the performance style state machine from a new observation."""
    global _performance_style, _style_votes, _last_standard_signal_s

    _style_votes.append(obs.performance_style)

    if obs.charades_signal is not None:
        _last_standard_signal_s = elapsed_s

    # Lock style after STYLE_LOCK_S seconds (majority vote, excluding "unclear")
    if elapsed_s >= _STYLE_LOCK_S and _performance_style == "unclear":
        votes = [v for v in _style_votes if v != "unclear"]
        if votes:
            _performance_style = Counter(votes).most_common(1)[0][0]
            print(f"  [style] Locked → {_performance_style}")

    # Downgrade: standard_charades with no signals for STYLE_DOWNGRADE_S → mixed
    if (
        _performance_style == "standard_charades"
        and _last_standard_signal_s > 0
        and elapsed_s - _last_standard_signal_s > _STYLE_DOWNGRADE_S
    ):
        _performance_style = "mixed"
        print("  [style] Downgraded: standard_charades → mixed (no signals)")

    # Upgrade: free_form but a signal appears → mixed
    if _performance_style == "free_form" and obs.charades_signal is not None:
        _performance_style = "mixed"
        print("  [style] Upgraded: free_form → mixed (signal detected)")


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
    return any(_jaccard(candidate, g) >= _SEMANTIC_DUP_J for g in existing)


# ---------------------------------------------------------------------------
# Observer agent
# ---------------------------------------------------------------------------


async def _run_observer(jpeg_bytes: bytes, elapsed_s: float) -> Observation | None:
    """Call the observer LLM. Always resets _observer_running in finally."""
    global _observer_running
    try:
        recent_obs = _observations[-5:]
        obs_summary = "\n".join(
            f"  t={t:.1f}s [{o.gesture_phase}] {o.action_description} "
            f"| signal={o.charades_signal} | style={o.performance_style}"
            for t, o in recent_obs
        )

        context = (
            f"t={elapsed_s:.1f}s | style={_performance_style} | "
            f"guesses_remaining={10 - len(_submitted_guesses)}"
        )
        if obs_summary:
            context += f"\n\nRecent observations:\n{obs_summary}"
        if _submitted_guesses:
            context += "\n\nSubmitted: " + ", ".join(f'"{g}"' for g in _submitted_guesses)
        context += "\n\nAnalyze this frame → fill the Observation schema."

        result = await _get_observer().run(
            [context, BinaryContent(data=jpeg_bytes, media_type="image/jpeg")],
            output_type=Observation,
        )
        return result.output
    except Exception as exc:
        print(f"  [observer] Error: {exc}")
        return None
    finally:
        _observer_running = False


# ---------------------------------------------------------------------------
# Synthesizer agent (background asyncio.Task)
# ---------------------------------------------------------------------------


async def _synthesizer_loop() -> None:
    """Background loop: fire the synthesizer every _SYNTH_CADENCE_S seconds."""
    while not _round_solved:
        await asyncio.sleep(_SYNTH_CADENCE_S)
        if _round_solved or len(_observations) < 2:
            continue
        await _run_synthesizer()


async def _run_synthesizer() -> None:
    """One synthesizer pass — updates _synth_candidates with ranked guesses."""
    global _synth_candidates, _synth_last_run_s

    if _round_start is None:
        return

    elapsed_s = (datetime.now(timezone.utc) - _round_start).total_seconds()

    _style_weight = {
        "standard_charades": "Weight charades signals 70%, scene interpretation 30%.",
        "free_form":          "Weight scene interpretation 90%, charades signals 10%.",
        "mixed":              "Weight both channels equally (50/50).",
        "unclear":            "Style not yet determined — consider both channels equally.",
    }.get(_performance_style, "")

    obs_lines = []
    for t, obs in _observations[-_MAX_CONTEXT_OBS:]:
        weight = "⚡HIGH" if obs.gesture_phase in ("stroke", "hold") else "low"
        obs_lines.append(
            f"t={t:.1f}s [{obs.gesture_phase}/{weight}] "
            f"hands={obs.hand_position} | signal={obs.charades_signal} | "
            f"action={obs.action_description} | scene={obs.scene_interpretation} | "
            f"NOT: {obs.negative_space}"
        )

    msg = (
        f"t={elapsed_s:.1f}s | guesses_remaining={10 - len(_submitted_guesses)}\n"
        f"Style: {_performance_style}. {_style_weight}\n\n"
        "Observations:\n" + "\n".join(obs_lines)
    )
    if _wrong_guesses:
        msg += "\n\nCONFIRMED WRONG (exclude + semantically similar):\n" + ", ".join(
            f'"{g}"' for g in _wrong_guesses
        )
    if _submitted_guesses:
        msg += "\n\nAlready submitted (do not repeat):\n" + ", ".join(
            f'"{g}"' for g in _submitted_guesses
        )
    msg += "\n\nThink step by step, then output SynthesisResult JSON."

    parts: list[Any] = [msg]
    for t, jpeg in _select_key_frames(3):
        parts.append(f"\n[Key frame @ t={t:.1f}s — highest motion in its 3-second window]")
        parts.append(BinaryContent(data=jpeg, media_type="image/jpeg"))

    try:
        result = await _get_synthesizer().run(parts, output_type=SynthesisResult)
        synthesis: SynthesisResult = result.output
        _synth_candidates = synthesis.candidates
        _synth_last_run_s = elapsed_s

        top_str = (
            f"{synthesis.candidates[0].answer!r} conf={synthesis.candidates[0].confidence:.2f}"
            if synthesis.candidates
            else "none"
        )
        print(
            f"\n  [synth] t={elapsed_s:.1f}s | top={top_str} | top_pick={synthesis.top_pick!r}"
            f" | all={[(c.answer, f'{c.confidence:.2f}') for c in synthesis.candidates]}"
        )
    except Exception as exc:
        print(f"  [synth] Error: {exc}")


# ---------------------------------------------------------------------------
# Guess arbiter
# ---------------------------------------------------------------------------


def _get_phase(elapsed_s: float) -> str:
    if elapsed_s < _PHASE_OBSERVE_END:
        return "observe"
    if elapsed_s < _PHASE_PATIENT_END:
        return "patient"
    if elapsed_s < _PHASE_ACTIVE_END:
        return "active"
    return "desperation"


_PHASE_THRESHOLDS: dict[str, float] = {
    "observe":     9999.0,
    "patient":     _CONF_PATIENT,
    "active":      _CONF_ACTIVE,
    "desperation": _CONF_DESPERATION,
}


def _run_arbiter(elapsed_s: float) -> str | None:
    """Apply phase gates, rolling budget, and semantic dedup. Return a guess or None."""
    global _phase_guess_counts

    if not _synth_candidates:
        return None

    phase = _get_phase(elapsed_s)
    if phase == "observe":
        return None

    if 10 - len(_submitted_guesses) <= 0:
        return None

    # Rolling budget: patient + active combined ≤ 7 before entering desperation
    if phase != "desperation":
        non_desp_used = _phase_guess_counts["patient"] + _phase_guess_counts["active"]
        if non_desp_used >= _BUDGET_PATIENT + _BUDGET_ACTIVE:
            print("  [arbiter] SKIP — pre-desperation budget exhausted")
            return None

    threshold = _PHASE_THRESHOLDS[phase]

    for candidate in _synth_candidates:
        normalized = candidate.answer.strip().lower()

        if normalized in _submitted_guesses:
            continue

        if _is_semantic_dup(candidate.answer, _wrong_guesses):
            print(f"  [arbiter] SKIP {candidate.answer!r} — semantic dup of wrong guess")
            continue

        if _is_semantic_dup(candidate.answer, _submitted_guesses):
            print(f"  [arbiter] SKIP {candidate.answer!r} — semantic dup of submitted guess")
            continue

        if candidate.confidence < threshold:
            print(
                f"  [arbiter] SKIP {candidate.answer!r} — "
                f"conf={candidate.confidence:.2f} < {threshold:.2f} ({phase})"
            )
            return None  # Don't fall through to lower-confidence candidates

        # Commit
        _submitted_guesses.append(normalized)
        _phase_guess_counts[phase] = _phase_guess_counts.get(phase, 0) + 1
        print(
            f"  [arbiter] GUESS → {candidate.answer!r} "
            f"(conf={candidate.confidence:.2f}, {phase}, t={elapsed_s:.1f}s)"
        )
        return candidate.answer

    return None


# ---------------------------------------------------------------------------
# Early-exit check
# ---------------------------------------------------------------------------


def _check_early_exit(elapsed_s: float) -> str | None:
    """Submit immediately if top synthesizer candidate is >= 0.90 for 3 consecutive
    observer calls. One early-exit per round."""
    global _early_exit_used, _early_exit_streak_answer, _early_exit_streak_count

    if _early_exit_used or not _synth_candidates or elapsed_s < _PHASE_OBSERVE_END:
        return None

    top = _synth_candidates[0]
    if top.confidence < _EARLY_EXIT_CONF:
        # Reset streak on low confidence
        _early_exit_streak_answer = ""
        _early_exit_streak_count = 0
        return None

    normalized = top.answer.strip().lower()
    if normalized == _early_exit_streak_answer:
        _early_exit_streak_count += 1
    else:
        _early_exit_streak_answer = normalized
        _early_exit_streak_count = 1

    if _early_exit_streak_count >= _EARLY_EXIT_MIN_FRAMES:
        if normalized not in _submitted_guesses and not _is_semantic_dup(
            top.answer, _submitted_guesses
        ):
            _early_exit_used = True
            _submitted_guesses.append(normalized)
            _phase_guess_counts["patient"] = _phase_guess_counts.get("patient", 0) + 1
            print(
                f"  [arbiter] EARLY EXIT → {top.answer!r} "
                f"(conf={top.confidence:.2f}, "
                f"{_early_exit_streak_count} consecutive high-conf observer calls)"
            )
            return top.answer

    return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_observation(elapsed_s: float, obs: Observation, motion: float, region: str) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  [observer] t={elapsed_s:.1f}s | motion={motion:.4f} region={region}")
    print(f"  phase={obs.gesture_phase} | hands={obs.hand_position}")
    print(f"  signal={obs.charades_signal} | style_vote={obs.performance_style} → locked={_performance_style}")
    print(f"  action : {obs.action_description}")
    print(f"  scene  : {obs.scene_interpretation}")
    print(f"  NOT    : {obs.negative_space}")
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def analyze(frame: Frame) -> str | None:
    """Analyze a single charades frame and return a guess, or None to skip.

    Called by the main loop for every sampled frame. Coordinates the screener,
    observer, synthesizer, and arbiter.
    """
    global _round_start, _observer_running, _synth_task

    if _round_solved:
        return None

    now = frame.timestamp
    if _round_start is None:
        _round_start = now

    elapsed_s = (now - _round_start).total_seconds()

    # Ensure synthesizer background task is running
    if _synth_task is None or _synth_task.done():
        _synth_task = asyncio.create_task(_synthesizer_loop())

    # ------------------------------------------------------------------ #
    # 1. Frame screener (~5 ms)
    # ------------------------------------------------------------------ #
    is_significant, motion_magnitude, motion_region = _screen_frame(frame.image, elapsed_s)

    if not is_significant:
        # Even on skipped frames check for fresh synthesizer output
        return _run_arbiter(elapsed_s)

    # ------------------------------------------------------------------ #
    # 2. Observer gate — skip if another observer call is already in-flight
    # ------------------------------------------------------------------ #
    if _observer_running:
        return _run_arbiter(elapsed_s)

    # ------------------------------------------------------------------ #
    # 3. Run observer LLM (~1–2 s)
    # ------------------------------------------------------------------ #
    _observer_running = True  # reset in _run_observer's finally block

    buf = io.BytesIO()
    img = frame.image
    if img.width > 1280 or img.height > 720:
        img.thumbnail((1280, 720))
    img.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    observation = await _run_observer(jpeg_bytes, elapsed_s)

    if observation is None:
        return _run_arbiter(elapsed_s)

    # ------------------------------------------------------------------ #
    # 4. Record observation and update convention detector
    # ------------------------------------------------------------------ #
    _observations.append((elapsed_s, observation))
    if len(_observations) > 20:
        _observations.pop(0)

    _update_style(observation, elapsed_s)
    _log_observation(elapsed_s, observation, motion_magnitude, motion_region)

    # ------------------------------------------------------------------ #
    # 5. Early-exit check (before standard arbiter)
    # ------------------------------------------------------------------ #
    early_guess = _check_early_exit(elapsed_s)
    if early_guess:
        return early_guess

    # ------------------------------------------------------------------ #
    # 6. Standard arbiter
    # ------------------------------------------------------------------ #
    return _run_arbiter(elapsed_s)
