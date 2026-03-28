# Casper Agent Starter

Starter template for building AI agents in the Casper Studios guessing game.

## Quick Start

```bash
# 1. Install uv (if you don't have it)
brew install uv        # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Copy the env file
cp .env.example .env

# 4. Run in practice mode (local camera)
uv run -m agent --practice
```

## Project Structure

```
├── core/src/core/     🔒 Frame capture & streaming (don't edit)
├── api/src/api/       🔒 HTTP client for the game server (don't edit)
├── agent/src/agent/   ✏️  Your AI agent (edit this!)
│   ├── __main__.py         CLI entry point
│   └── prompt.py           Your prompt & analysis logic
└── .env                    Your team config
```

## Modes

### Practice Mode (offline)

Uses your local camera. No server connection needed. Great for tuning your prompts.

```bash
uv run -m agent --practice
uv run -m agent --practice --camera 1    # use a different camera
uv run -m agent --practice --fps 2       # sample 2 frames/sec
```

### Live Mode (event day)

Connects to the dashboard’s HTTP API and LiveKit stream. Set these in `.env` (see `.env.example`):

- **`API_URL`** — dashboard origin only, e.g. `https://your-app.workers.dev` (no trailing slash).
- **`TEAM_TOKEN`** — your team’s API key (same as the dashboard `team.api_key`).

The admin must be on the **live** dashboard with the camera publishing to the room (so your agent can subscribe to video).

```bash
uv run -m agent --live
```

## How It Works

1. A frame is captured (from your camera or the live stream)
2. Your `analyze()` function in `agent/prompt.py` receives the frame
3. You send it to a vision LLM and return a guess (or `None` to skip)
4. If in live mode, the guess is submitted via `POST /api/guess` (plain text body)
5. If correct (HTTP 201), you win. If wrong (HTTP 409), keep guessing. The server may return 401 (bad token), 404 (no round), 429 (max guesses), or 503 (judge unavailable — live mode retries with exponential backoff, then continues on the next frame if the judge stays down).
6. The round ends when the admin closes the stream.

## What to Edit

Open `agent/src/agent/prompt.py`. That's it. Customize:

- **`SYSTEM_PROMPT`** — the instructions for your vision LLM
- **`analyze(frame)`** — your logic for turning a frame into a guess

See [AGENTS.md](./AGENTS.md) for tips.
