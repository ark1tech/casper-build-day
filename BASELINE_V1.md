# Baseline Agent — Version 1 Context

> **Purpose**: This document is the handoff brief for agents building Version 2 of the charades predictor. It describes what V1 does, why each design decision was made, what was confirmed working, and the full list of deferred optimizations to tackle next.

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
│   └── prompt.py        # V1 implementation lives here (edit this for V2)
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

**The only file V2 needs to change is `agent/src/agent/prompt.py`.** Everything else is infrastructure.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `API_URL` | Dashboard base URL (no trailing slash), e.g. `https://your-app.workers.dev` |
| `TEAM_TOKEN` | Team API key sent as `Authorization: Bearer <token>` |
| `LLM_API_KEY` | OpenRouter API key — get one at [openrouter.ai/keys](https://openrouter.ai/keys) |

`python-dotenv` loads `.env` automatically on startup. `LLM_API_KEY` is read explicitly via `os.environ.get("LLM_API_KEY")` and passed to `OpenAIProvider(api_key=...)` which points at `https://openrouter.ai/api/v1`.

---

## V1 Architecture

```
Frame (PIL Image + UTC timestamp)
  │
  ▼
analyze(frame)          ← the single function V2 must implement
  │
  ├─ 1. Convert PIL Image → JPEG bytes (resized to ≤ 1280×720)
  ├─ 2. Build user message: elapsed time + prior observations + new image
  ├─ 3. await _get_agent().run([context_str, BinaryContent(jpeg)], output_type=FrameAnalysis)
  ├─ 4. Append observation to rolling history (cap 15)
  ├─ 5. Gate A — hold period: skip first 5 s
  ├─ 6. Gate B — model must have returned a non-null guess
  ├─ 7. Gate C — confidence ≥ 0.75 (drops to 0.50 after 60 s)
  ├─ 8. Gate D — deduplicate (lowercase normalized)
  └─ 9. Return guess string → __main__.py submits to API
```

### Model

- **Google Gemini 2.5 Flash via OpenRouter** (`google/gemini-2.5-flash`)
- Accessed through `OpenAIChatModel` + `OpenAIProvider(base_url="https://openrouter.ai/api/v1")` — OpenRouter speaks the OpenAI wire format, so pydantic-ai's structured output and `BinaryContent` image attachment work without any additional provider packages.
- API key read from `LLM_API_KEY` env var and passed explicitly to the provider.
- Temperature `0.2` for consistency.
- Agent is **lazy-initialized** (created on first `analyze()` call, not at import time).

### Structured Output: `FrameAnalysis`

```python
class FrameAnalysis(BaseModel):
    observation: str         # what is literally happening in THIS frame
    accumulated_reasoning: str  # synthesis across all frames toward an answer
    guess: str | None        # committed answer, or null
    confidence: float        # 0.0–1.0; gates in analyze() use this
```

Using structured output (not raw string) eliminates parsing fragility and gives explicit confidence routing.

### Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `_HOLD_PERIOD_S` | `5.0` | Always skip the first 5 s to accumulate observations |
| `_CONFIDENCE_HIGH` | `0.75` | Required confidence before guessing (0–60 s) |
| `_ESCALATION_AFTER_S` | `60.0` | After this many seconds, lower the confidence gate |
| `_CONFIDENCE_LOW` | `0.50` | Relaxed confidence threshold after 60 s |
| `_MAX_CONTEXT_FRAMES` | `10` | How many prior observations to include in each prompt |

### Per-Round Module-Level State

```python
_frame_observations: list[str]   # rolling window, capped at 15
_submitted_guesses: set[str]     # lowercase-normalized; prevents duplicates
_round_start: datetime | None    # set on first frame; drives elapsed time
```

State persists across frames within a single process run. A new round = restart the process.

---

## SYSTEM_PROMPT Design Rationale

The prompt is **charades-specific**, not generic object identification. Key elements:

1. **Charades conventions dictionary** — teaches the model standard hand signals (ear pull = "sounds like", finger count for syllables/words, pinch = small word, etc.) that Gemini won't know without being told.
2. **Complexity warning** — explicitly says answers may be multi-word phrases, movie titles, idioms, abstract concepts, gerunds. Prevents the model from always returning simple nouns.
3. **Null discipline** — instructs the model to output `null` guess when uncertain, which is the right default given the 10-guess limit.
4. **Temporal accumulation hint** — tells the model to synthesize prior frame observations in `accumulated_reasoning`, not just react to the current frame in isolation.

---

## Guessing Strategy (V1)

| Phase | Elapsed | Behavior |
|---|---|---|
| Observe | 0–5 s | Always SKIP. Every LLM call records an observation but no guess fires. |
| High-confidence only | 5–60 s | Guess only if `confidence ≥ 0.75`. Model must commit a non-null answer. |
| Escalation | 60–120 s | Confidence gate drops to `0.50`. Prevents ending the round with unused guesses. |

The scoring formula makes each wasted guess permanently reduce the ceiling score by one multiplier step. V1's strategy is: **be patient, be right, be fast** — in that priority order.

---

## Known Limitations / What V2 Should Improve

These were explicitly deferred from V1 as out-of-scope until the baseline was confirmed working.

### 1. Single-frame temporal context is shallow
V1 feeds back prior `observation` strings as text. This loses spatial and motion information. V2 could:
- Maintain a sequence of frames and run a diff (detect motion, gesture direction)
- Feed the last 2–3 raw images to the model in one call instead of text summaries

### 2. No multi-model pipeline
V1 uses one model (Gemini 2.5 Flash) for all decisions. A fast screener → strong guesser pipeline could:
- Use `gemini-2.0-flash-lite` or similar to quickly classify gesture type
- Invoke `gemini-2.5-pro` only when the screener flags high-signal frames

### 3. Fixed FPS — no adaptive sampling
The frame rate is set at CLI invocation (`--fps 1` or `--fps 2`) and never changes. V2 could:
- Start at 1 FPS to conserve tokens
- Temporarily boost to 2 FPS when motion/activity is detected in a frame
- Drop back to 0.5 FPS during pauses (no movement detected)

### 4. Module-level state doesn't reset between rounds
If the process is reused across rounds, `_frame_observations`, `_submitted_guesses`, and `_round_start` will carry over. V2 should add an explicit `reset()` function or detect round boundaries via the API.

### 5. Confidence is self-reported by the model
The `confidence` field is the model's own estimate — not a calibrated probability. V2 could add an independent calibration step: ask the model to list its top 3 candidate answers and only submit the one it picks across consecutive frames.

### 6. No prompt iteration
V1's `SYSTEM_PROMPT` was written once and untested against real charades footage. After running in practice mode, it should be tuned based on observed output quality.

---

## Running V1

```bash
# Practice (local camera, no network)
uv run -m agent --practice --fps 1

# Live (event day)
uv run -m agent --live

# Higher frame rate
uv run -m agent --practice --fps 2
```

`GOOGLE_API_KEY` must be set in `.env` before running live mode.

---

## API Reference (for V2 reference)

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
| `Pillow` | PIL image manipulation |
| `httpx` | HTTP client for REST API |
| `python-dotenv` | `.env` loading |

Install: `uv sync` from the workspace root.
