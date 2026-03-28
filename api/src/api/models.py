"""Pydantic models and exceptions for the Casper API."""

from __future__ import annotations

from pydantic import BaseModel


class Feed(BaseModel):
    """Response from GET /api/feed when a round is active."""

    livekit_url: str
    """LiveKit server URL (e.g. wss://project.livekit.cloud)."""

    token: str
    """Subscribe-only LiveKit JWT for the current room."""

    round_id: str
    """Identifier for the current round."""


class GuessResult(BaseModel):
    """Result of a POST /api/guess submission."""

    correct: bool
    """Whether the guess was correct (201 = True, 409 = False)."""

    guess_id: int | None = None
    """Database row id of the guess (returned as plain text on 201)."""


class NoActiveRound(Exception):
    """Raised when GET /api/feed returns 404 (no round in progress)."""

    def __str__(self) -> str:
        return "No active round. Wait for the admin to start a new round."


class Unauthorized(Exception):
    """Raised when the server returns 401 (invalid or missing team token)."""

    def __str__(self) -> str:
        return "Unauthorized. Check TEAM_TOKEN matches your team's API key."


class MaxGuessesReached(Exception):
    """Raised when POST /api/guess returns 429 (max guesses per round)."""

    def __str__(self) -> str:
        return "Maximum guesses reached for this round."


class JudgeUnavailable(Exception):
    """Raised when POST /api/guess returns 503.

    The server could not judge the guess (e.g. judge LLM failure); no guess row
    was persisted. Safe to retry the same guess.
    """

    def __str__(self) -> str:
        return "Judge unavailable (503). The server could not score this guess; try again."
