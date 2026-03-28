"""Casper Agent CLI entry point.

Usage:
    uv run -m agent --practice     # Local camera, no network
    uv run -m agent --live         # Connect to live game
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

_JUDGE_UNAVAILABLE_BACKOFF_CAP_S = 30.0
_MAX_JUDGE_UNAVAILABLE_RETRIES = 5
_JUDGE_UNAVAILABLE_BACKOFF_S = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Casper guessing game AI agent",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--practice",
        action="store_true",
        help="Use local camera for offline development",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Connect to a live game round",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for practice mode (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to sample (default: 1)",
    )
    return parser.parse_args()


async def run_practice(camera: int, fps: int) -> None:
    """Run the agent in practice mode with a local camera."""
    from core import start_practice

    from agent.prompt import analyze

    print("=" * 50)
    print("  PRACTICE MODE")
    print("  Local camera — no network required")
    print("=" * 50)
    print()

    async for frame in start_practice(camera_index=camera, fps=fps):
        guess = await analyze(frame)
        if guess:
            print(f"  [guess] {guess}")
        else:
            print("  [skip]  No guess this frame")


async def run_live() -> None:
    """Run the agent in live mode against the game server."""
    from api import (
        CasperAPI,
        JudgeUnavailable,
        MaxGuessesReached,
        NoActiveRound,
        Unauthorized,
    )
    from core import start_stream

    from agent.prompt import analyze

    print("=" * 50)
    print("  LIVE MODE")
    print("  Connecting to game server...")
    print("=" * 50)
    print()

    client = CasperAPI.from_env()

    try:
        feed = await client.get_feed()
    except Unauthorized:
        print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
        sys.exit(1)
    except NoActiveRound:
        print("[!] No active round. Wait for the admin to start one.")
        sys.exit(1)
    except Exception as exc:
        print(f"[!] Could not connect to game server: {exc}")
        sys.exit(1)

    print(f"[+] Joined round: {feed.round_id}")
    print(f"[+] LiveKit URL:  {feed.livekit_url}")
    print()

    guess_count = 0

    try:
        async for frame in start_stream(feed.livekit_url, feed.token):
            guess = await analyze(frame)

            if guess:
                result = None
                n_503 = 0
                try:
                    while True:
                        try:
                            result = await client.guess(guess)
                            break
                        except JudgeUnavailable:
                            if n_503 >= _MAX_JUDGE_UNAVAILABLE_RETRIES:
                                break
                            delay = min(
                                _JUDGE_UNAVAILABLE_BACKOFF_S * (2**n_503),
                                _JUDGE_UNAVAILABLE_BACKOFF_CAP_S,
                            )
                            await asyncio.sleep(delay)
                            n_503 += 1
                except Unauthorized:
                    print("[!] Unauthorized. Check TEAM_TOKEN matches your team's API key.")
                    break
                except NoActiveRound:
                    print("[!] No active round (round may have ended).")
                    break
                except MaxGuessesReached:
                    print("[!] Maximum guesses reached for this round.")
                    break

                if result is None:
                    attempts = 1 + _MAX_JUDGE_UNAVAILABLE_RETRIES
                    print(
                        f"[!] Judge unavailable (503) after {attempts} attempt(s). "
                        "Skipping this guess; will try again on the next frame."
                    )
                    continue

                guess_count += 1
                id_suffix = f" id={result.guess_id}" if result.guess_id is not None else ""
                print(f"  [guess #{guess_count}{id_suffix}] {guess}")

                if result.correct:
                    print()
                    print("=" * 50)
                    print(f"  CORRECT! Solved in {guess_count} guesses.")
                    print("=" * 50)
                    break
            else:
                print("  [skip] No guess this frame")

    except (KeyboardInterrupt, ConnectionError):
        print("\n[!] Disconnected from stream.")
    finally:
        await client.close()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.practice:
        await run_practice(camera=args.camera, fps=args.fps)
    else:
        await run_live()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
        import os
        os._exit(0)
