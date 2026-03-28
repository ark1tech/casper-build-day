# Baseline Agent — Version 2 Context

> **Purpose**: This document is the handoff brief for agents building Version 3 of the charades predictor. It describes what V2 does, why each design decision was made, what V2 improved over V1, and the full list of deferred optimizations to tackle next.

---

## Event Summary

**Agent Charades** — March 28, 2026.

- The performer acts silently; the agent watches a live camera feed and guesses what is being acted out.
- Guesses are submitted via `POST /api/guess` (plain-text body).
- **10 guesses per round, 120 seconds per round.**
- Scoring: `score = guesses_remaining × speed_bonus` — fewer wasted guesses + faster solve = max points.
- The answer is judged by the server: HTTP 201 = correct, HTTP 409 = wrong.

---

## Repository Layout

```
casper/
├── agent/src/agent/
│   ├── __main__.py      # CLI entry point — do not edit
│   └── prompt.py        # V2 implementation lives here (edit this for V3)
├── api/src/api/
│   ├── client.py        # CasperAPI — REST client, do not edit
│   └── models.py        # Pydantic models + exception types
├── core/src/core/
│   ├── frame.py         # Frame dataclass: .image (PIL RGB), .timestamp (UTC datetime)
│   ├── practice.py      # Local camera capture via ffmpeg
│   └── stream.py        # LiveKit stream subscription
├── .env                 # Secret credentials (git-ignored)
├── .env.example         # Template — copy to .env
└── pyproject.toml       # uv workspace root
```

**The only file V3 needs to change is `agent/src/agent/prompt.py`.** Everything else is infrastructure.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `API_URL` | Dashboard base URL (no trailing slash), e.g. `https://your-app.workers.dev` |
| `TEAM_TOKEN` | Team API key sent as `Authorization: Bearer <token>` |
| `LLM_API_KEY` | OpenRouter API key — get one at [openrouter.ai/keys](https://openrouter.ai/keys) |

`python-dotenv` loads `.env` automatically on startup. `LLM_API_KEY` is read explicitly via `os.environ.get("LLM_API_KEY")` and passed to `OpenAIProvider(api_key=...)` which points at `https://openrouter.ai/api/v1`.

---

## V2 Architecture

V2 replaces V1's single-model monolith with a **multi-agent pipeline** with five discrete subsystems:

```
Frame (PIL Image + UTC timestamp)
  │
  ▼
analyze(frame)          ← the single async function V3 must implement
  │
  ├─ 1. Frame Screener   (~5 ms, PIL pixel-diff)
  │     ├─ Always process: first 10 frames + heartbeat every 3 s
  │     ├─ Process if motion_magnitude ≥ 0.015
  │     └─ Maintains a key-frame buffer (best frame per 3-s window, max 10)
  │
  ├─ 2. Observer Agent   (Gemini 2.5 Flash, structured output)
  │     ├─ Observation-ONLY — explicitly told NOT to guess
  │     ├─ Fills: gesture_phase, hand_position, charades_signal,
  │     │         action_description, scene_interpretation,
  │     │         negative_space, performance_style
  │     └─ One call in-flight at a time (_observer_running gate)
  │
  ├─ 3. Convention Detector   (state machine, no LLM)
  │     ├─ Accumulates performance_style votes
  │     ├─ Locks style after 5 s (majority vote, "unclear" excluded)
  │     ├─ Downgrades standard_charades → mixed if no signal for 20 s
  │     └─ Upgrades free_form → mixed if a charades_signal appears
  │
  ├─ 4. Synthesizer Agent   (Gemini 2.5 Flash + CoT, background asyncio.Task)
  │     ├─ Fires every 4 s regardless of frame activity
  │     ├─ Receives last 12 observations + 3 key frames (highest motion, ≥2 s apart)
  │     ├─ Applies style-weighted channel reasoning (signals vs. scene)
  │     ├─ Excludes confirmed-wrong guesses + semantic near-duplicates
  │     └─ Outputs ranked Candidates with confidence + reasoning
  │
  └─ 5. Guess Arbiter   (pure Python, ~0 ms)
        ├─ Phase 1 — observe (0–5 s): always skip
        ├─ Phase 2 — patient (5–40 s): guess only if confidence ≥ 0.85
        ├─ Phase 3 — active (40–80 s): guess only if confidence ≥ 0.65
        ├─ Phase 4 — desperation (80–120 s): guess if confidence ≥ 0.40
        ├─ Rolling budget: patient + active combined ≤ 7 guesses before desperation
        ├─ Semantic deduplication (Jaccard ≥ 0.70 → skip)
        └─ Early-exit: same top candidate ≥ 0.90 for 3 consecutive observer calls → fire immediately
```

---

## What V2 Improved Over V1

| Area | V1 | V2 |
|---|---|---|
| **Model calls per frame** | 1 (observe + guess in one call) | 1 observer + async synthesizer (decoupled) |
| **Temporal context** | Text summaries of prior frames | Real JPEG key-frames sent to synthesizer |
| **Guess confidence** | Self-reported by the model that also guesses | Produced by a separate synthesizer that never sees gate logic |
| **Charades signals** | Detected ad-hoc in unstructured text | Structured `charades_signal` field; treated as hard constraints |
| **Performance style** | Implicit in prompt | Explicit state machine; weights LLM interpretation channels accordingly |
| **Phase strategy** | 2 phases (observe / guess) | 4 phases with a rolling pre-desperation budget |
| **Deduplication** | Exact string match (lowercase) | Jaccard semantic similarity on stemmed word sets |
| **Feedback loop** | None | `set_last_result()` — confirmed-wrong guesses excluded from all future candidates |
| **Negative space** | Not captured | `negative_space` field in Observation; synthesizer filters contradicting candidates |
| **Motion detection** | None | PIL pixel-diff frame screener; heartbeat fallback |
| **Frame selection** | Latest frame only | Key-frame buffer: best frame per 3-s window, highest motion |
| **Early exit** | None | Fires immediately if top candidate ≥ 0.90 for 3 consecutive frames |

---

## Structured Schemas

### `Observation` — Observer output per frame

```python
class Observation(BaseModel):
    gesture_phase: Literal["rest", "preparation", "stroke", "hold", "retraction"]
    hand_position: str          # "near face", "above head", etc.
    charades_signal: str | None # "syllable_count:3", "sounds_like", "category:movie", etc.
    action_description: str     # physical action being depicted
    scene_interpretation: str   # scene or concept this could represent
    negative_space: str         # what the performer is clearly NOT doing
    performance_style: Literal["standard_charades", "free_form", "mixed", "unclear"]
```

### `Candidate` — Single ranked guess from the synthesizer

```python
class Candidate(BaseModel):
    answer: str
    confidence: float           # 0.0–1.0, clamped
    reasoning: str
```

### `SynthesisResult` — Synthesizer output

```python
class SynthesisResult(BaseModel):
    candidates: list[Candidate] # ranked highest confidence first
    top_pick: str | None        # best answer if confidence gap > 0.15 over second; else null
```

---

## Key Constants

### Frame Screener

| Constant | Value | Meaning |
|---|---|---|
| `_MOTION_THRESHOLD` | `0.015` | Pixel-diff / max_possible; below this → skip frame |
| `_KEY_FRAME_BUFFER_SIZE` | `10` | Max 3-s windows retained in key-frame buffer |
| `_HEARTBEAT_INTERVAL_S` | `3.0` | Force-process a frame even with no motion |
| `_FIRST_N_FRAMES_ALWAYS` | `10` | Always pass the first 10 frames to the observer |

### Synthesizer

| Constant | Value | Meaning |
|---|---|---|
| `_SYNTH_CADENCE_S` | `4.0` | Synthesizer fires every N seconds |
| `_MAX_CONTEXT_OBS` | `12` | Max observations sent to synthesizer per call |

### Phase Boundaries

| Constant | Value | Phase |
|---|---|---|
| `_PHASE_OBSERVE_END` | `5.0 s` | End of observe phase |
| `_PHASE_PATIENT_END` | `40.0 s` | End of patient phase |
| `_PHASE_ACTIVE_END` | `80.0 s` | End of active phase |

### Confidence Thresholds

| Constant | Value | Phase |
|---|---|---|
| `_CONF_PATIENT` | `0.85` | Required confidence in patient phase |
| `_CONF_ACTIVE` | `0.65` | Required confidence in active phase |
| `_CONF_DESPERATION` | `0.40` | Required confidence in desperation phase |

### Guess Budget

| Constant | Value | Meaning |
|---|---|---|
| `_BUDGET_PATIENT` | `3` | Max guesses during patient phase |
| `_BUDGET_ACTIVE` | `4` | Patient + active combined ≤ 7 before desperation |

### Early Exit

| Constant | Value | Meaning |
|---|---|---|
| `_EARLY_EXIT_MIN_FRAMES` | `3` | Consecutive high-conf observer calls needed |
| `_EARLY_EXIT_CONF` | `0.90` | Confidence required for each frame in the streak |

### Convention Detector

| Constant | Value | Meaning |
|---|---|---|
| `_STYLE_LOCK_S` | `5.0 s` | Lock performance style after this many seconds |
| `_STYLE_DOWNGRADE_S` | `20.0 s` | Downgrade standard_charades → mixed if no signals for this long |

### Semantic Deduplication

| Constant | Value | Meaning |
|---|---|---|
| `_SEMANTIC_DUP_J` | `0.70` | Jaccard similarity threshold; above this → treat as duplicate |

---

## Module-Level State (resets on process restart = new round)

```python
_round_start: datetime | None          # set on first frame; drives all elapsed_s calculations

# Observer
_observations: list[tuple[float, Observation]]   # (elapsed_s, obs), capped at 20
_observer_running: bool                          # one-in-flight gate

# Key-frame buffer
_prev_frame_small: PIL.Image | None              # 160×120 grayscale for motion diff
_frame_count: int
_last_heartbeat_s: float
_key_frames: list[tuple[float, float, bytes]]    # (elapsed_s, motion_magnitude, jpeg_bytes)

# Convention detector
_performance_style: str                          # "unclear" → locked style
_style_votes: list[str]
_last_standard_signal_s: float

# Synthesizer
_synth_task: asyncio.Task | None
_synth_candidates: list[Candidate]
_synth_last_run_s: float

# Guess tracking
_submitted_guesses: list[str]                    # lowercase-normalized, in order
_wrong_guesses: list[str]                        # confirmed 409 responses
_round_solved: bool

# Phase budgets
_phase_guess_counts: dict[str, int]              # {"patient": 0, "active": 0, "desperation": 0}

# Early-exit streak
_early_exit_used: bool
_early_exit_streak_answer: str
_early_exit_streak_count: int
```

---

## `set_last_result(guess, correct)` — Feedback Hook

Called by `__main__.py` immediately after `client.guess()` returns:

```python
set_last_result(guess, result.correct)
```

- **Correct** → sets `_round_solved = True`; `analyze()` returns `None` for all subsequent frames.
- **Wrong** → appends normalized guess to `_wrong_guesses`; synthesizer context includes this list as a hard exclusion constraint.

---

## Observer System Prompt Design

The observer is instructed to **observe only, never guess**. Key elements:

1. **Spatial anchor priority** — ordered checklist: finger count near chin → syllable count → word count → ear touch → cyclic motion → decisive gesture → facial expression.
2. **Gesture phase classification** — `stroke` and `hold` are the highest-signal phases; `rest`, `preparation`, `retraction` are low-signal.
3. **Performance style detection** — observer classifies style per frame; convention detector aggregates across frames.
4. **Negative space** — observer explicitly records what the performer is NOT doing; synthesizer uses this to eliminate candidate families.

---

## Synthesizer System Prompt Design

The synthesizer receives structured observations (text) + 3 key JPEG frames and applies chain-of-thought reasoning:

1. **Chronological review** — stroke/hold phases weighted 3× higher than rest/transition.
2. **Charades signal constraints** — if `syllable_count:3` or `which_word:2` is detected, all candidates must satisfy these as hard constraints.
3. **Style-weighted channel mixing**:
   - `standard_charades` → 70% signals + 30% scene
   - `free_form` → 10% signals + 90% scene
   - `mixed` / `unclear` → 50/50
4. **Cross-reference negative space** — eliminates answer families contradicted by `negative_space` fields.
5. **Specificity requirement** — "Titanic" beats "ship movie"; "skydiving" beats "falling from height".
6. **`top_pick` gating** — only set when best candidate has a >0.15 confidence gap over second place.

---

## Running V2

```bash
# Practice (local camera, no network)
uv run -m agent --practice --fps 1

# Live (event day)
uv run -m agent --live

# Higher frame rate
uv run -m agent --practice --fps 2
```

`LLM_API_KEY` must be set in `.env` before running.

---

## API Reference (for V3)

### `GET /api/feed` → `Feed`
Returns LiveKit credentials when a round is active.
```python
class Feed(BaseModel):
    livekit_url: str  # wss://...
    token: str        # subscribe-only JWT
    round_id: str
```

### `POST /api/guess` (plain-text body) → `GuessResult`
| Status | Meaning |
|---|---|
| 201 | Correct — body is the guess row ID |
| 409 | Wrong |
| 429 | Max guesses reached |
| 404 | No active round |
| 401 | Bad token |
| 503 | Judge unavailable — safe to retry (main loop already does exponential backoff) |

```python
class GuessResult(BaseModel):
    correct: bool
    guess_id: int | None
```

### Exceptions (importable from `api`)
`NoActiveRound`, `Unauthorized`, `MaxGuessesReached`, `JudgeUnavailable`

---

## Dependency Stack

| Package | Role |
|---|---|
| `pydantic-ai` | LLM orchestration, structured output, `BinaryContent` image attachment |
| `openrouter.ai` | Model routing layer — provides Gemini 2.5 Flash via OpenAI-compatible API |
| `livekit` | LiveKit room subscription (bundled with `core`) |
| `Pillow` | PIL image manipulation + `ImageChops`/`ImageStat` for motion detection |
| `httpx` | HTTP client for REST API |
| `python-dotenv` | `.env` loading |

Install: `uv sync` from the workspace root.

---

## Known Limitations / What V3 Should Improve

These were identified during V2 design as deferred optimizations.

### 1. Module-level state is never explicitly reset between rounds

All state variables are module-level globals. If the process is reused across rounds (e.g. via a restart wrapper), stale `_observations`, `_wrong_guesses`, `_submitted_guesses`, and `_round_start` carry over. V3 should add an explicit `reset_round()` function and call it when a new `round_id` is detected.

### 2. Observer is still called per-frame with no cross-frame image comparison

The observer sees only ONE frame per call plus recent observation text. It cannot detect the *direction* of a gesture (arm moving up vs. down) or identify transition between gesture phases from visual motion alone. V3 could:
- Send 2–3 consecutive frames in a single observer call to expose motion direction
- Add a dedicated optical-flow or motion-vector step before the observer to generate `motion_vector: (dx, dy)` that is injected into the context string

### 3. Synthesizer fires on a fixed 4-second cadence regardless of signal quality

The synthesizer runs every 4 seconds even when there are no new high-signal observations. V3 could:
- Trigger the synthesizer immediately after a `stroke` or `hold` phase observation (high-signal events)
- Suppress the synthesizer if fewer than 2 new observations have arrived since its last run
- Reduce cadence to 6–8 s early in the round when there is little evidence, increasing to 2 s near desperation

### 4. Confidence is not calibrated — model self-reports its own certainty

The synthesizer's `confidence` field is the model's own estimate. Miscalibration leads to either conservative under-guessing or aggressive over-guessing. V3 could:
- Add a **consistency calibration layer**: only increase confidence if the same top answer appears across N consecutive synthesizer runs (not observer frames)
- Track historical accuracy per confidence band across rounds (requires persistent logging)

### 5. Semantic dedup uses a shallow Jaccard on stemmed words

`_jaccard()` misses synonyms ("baseball" vs. "bat sport") and morphological variants the stemmer doesn't catch. V3 could:
- Use a small embedding model (e.g. `text-embedding-3-small` via OpenRouter) to compute cosine similarity for dedup instead of Jaccard
- Or cache a small synonym dictionary for common charades answer families

### 6. No round-over detection — agent continues running after `_round_solved`

After `_round_solved = True`, `analyze()` returns `None` every frame but the observer and synthesizer tasks keep running until the process exits. V3 should:
- Cancel `_synth_task` immediately on correct solve
- Stop accepting frames (or exit the `async for` loop in `__main__.py`) once solved

### 7. Key-frame selection uses motion magnitude as the only quality signal

High-motion frames may be blurry (fast movement = camera blur) or transitional (between gestures). V3 could:
- Add a blur detector (Laplacian variance on grayscale thumbnail) and prefer sharp frames over simply highest-motion frames
- Score key frames as `sharpness_weight * motion_magnitude` with a tunable `alpha`

### 8. Single model (Gemini 2.5 Flash) used for both observer and synthesizer

Both agents use the same model via two lazy-initialized `Agent` instances. V3 could:
- Use a **fast, cheap model** (e.g. `gemini-2.0-flash-lite` or `gpt-4o-mini`) for the observer — observation is a lower-reasoning task
- Reserve **Gemini 2.5 Pro** for synthesizer calls near desperation phase when accuracy matters most
- OpenRouter makes this a one-line change per agent factory

### 9. The `negative_space` field is not used to proactively narrow the key-frame selection

Currently, `negative_space` is sent to the synthesizer as text but doesn't affect which frames are selected or which parts of the frame the observer focuses on. V3 could crop or annotate frames to direct model attention away from confirmed-negative regions.

### 10. No prompt versioning or A/B testing harness

V2's observer and synthesizer prompts were hand-written without systematic evaluation. V3 should instrument a simple eval loop:
- Record `(frame_jpeg, observation_output)` pairs in practice mode
- Allow offline replay of observation runs against prompt variants
- Track synthesizer accuracy by comparing `top_pick` against the true answer after a round
