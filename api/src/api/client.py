"""Typed HTTP client for the Casper guessing game API."""

from __future__ import annotations

import os

import httpx

from api.models import (
    Feed,
    GuessResult,
    JudgeUnavailable,
    MaxGuessesReached,
    NoActiveRound,
    Unauthorized,
)


class CasperAPI:
    """Client for interacting with the Casper game server.

    Usage::

        client = CasperAPI.from_env()
        feed = await client.get_feed()
        result = await client.guess("golden retriever")
    """

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10.0,
        )

    @classmethod
    def from_env(cls) -> CasperAPI:
        """Create a client from API_URL and TEAM_TOKEN environment variables."""
        base_url = os.environ.get("API_URL")
        token = os.environ.get("TEAM_TOKEN")

        if not base_url:
            raise EnvironmentError("API_URL is not set. Check your .env file.")
        if not token:
            raise EnvironmentError("TEAM_TOKEN is not set. Check your .env file.")

        return cls(base_url=base_url, token=token)

    async def get_feed(self) -> Feed:
        """Get LiveKit credentials for the current round.

        Returns:
            Feed with livekit_url, token, and round_id.

        Raises:
            NoActiveRound: If no round is currently active (404).
            Unauthorized: If the team token is invalid (401).
        """
        resp = await self._client.get("/api/feed")

        if resp.status_code == 401:
            raise Unauthorized()

        if resp.status_code == 404:
            raise NoActiveRound()

        resp.raise_for_status()
        return Feed.model_validate(resp.json())

    async def guess(self, answer: str) -> GuessResult:
        """Submit a guess for the current round.

        Args:
            answer: The text guess to submit.

        Returns:
            GuessResult indicating whether the guess was correct.
            - 201 Created → correct = True (body is guess id as plain text)
            - 409 Conflict → correct = False

        Raises:
            NoActiveRound: If no round is active (404).
            Unauthorized: If the team token is invalid (401).
            MaxGuessesReached: If the team hit the per-round guess limit (429).
            JudgeUnavailable: If the server could not judge the guess (503).
        """
        resp = await self._client.post(
            "/api/guess",
            content=answer,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        if resp.status_code == 401:
            raise Unauthorized()

        if resp.status_code == 404:
            raise NoActiveRound()

        if resp.status_code == 429:
            raise MaxGuessesReached()

        if resp.status_code == 201:
            guess_id: int | None = None
            text = resp.text.strip()
            if text:
                try:
                    guess_id = int(text)
                except ValueError:
                    guess_id = None
            return GuessResult(correct=True, guess_id=guess_id)

        if resp.status_code == 409:
            return GuessResult(correct=False, guess_id=None)

        if resp.status_code == 503:
            raise JudgeUnavailable()

        resp.raise_for_status()
        # Unreachable but keeps type checker happy
        return GuessResult(correct=False)  # pragma: no cover

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
