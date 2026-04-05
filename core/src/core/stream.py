"""Live mode: subscribe to a LiveKit room and sample remote video frames."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from PIL import Image

from core.frame import Frame

# Wait for admin (or any remote) to publish camera video.
_FIRST_FRAME_TIMEOUT_S = 120.0


def _min_frame_interval_s(fps: float) -> float:
    """Seconds between yielded frames (higher fps → more temporal variety for the agent)."""
    f = max(0.5, min(30.0, float(fps)))
    return 1.0 / f


async def start_stream(
    url: str,
    token: str,
    *,
    fps: float = 3.0,
) -> AsyncIterator[Frame]:
    """Connect to a LiveKit room and yield video frames from remote participants.

    Args:
        url: The LiveKit server URL (e.g. wss://project.livekit.cloud).
        token: A subscribe-only JWT token from GET /api/feed.
        fps: Target sampling rate for yielded frames (default 3). Higher values give the agent
            more distinct moments; very high values increase CPU load.

    Yields:
        Frame objects with a PIL Image (RGB) and timestamp.

    Raises:
        ConnectionError: If no remote video track appears within the timeout.
    """
    from livekit import rtc
    from livekit.rtc import TrackKind

    min_interval_s = _min_frame_interval_s(fps)

    room = rtc.Room()
    video_streams: list[rtc.VideoStream] = []
    pump_tasks: list[asyncio.Task[None]] = []
    started_track_sids: set[str] = set()
    frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)

    def start_pump_for_track(track: rtc.Track) -> None:
        if track.kind != TrackKind.KIND_VIDEO:
            return
        if track.sid in started_track_sids:
            return
        started_track_sids.add(track.sid)
        vs = rtc.VideoStream(track)
        video_streams.append(vs)
        pump_tasks.append(
            asyncio.create_task(_pump_video_to_queue(vs, frame_queue, min_interval_s))
        )

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        _participant: rtc.RemoteParticipant,
    ) -> None:
        start_pump_for_track(track)

    try:
        await room.connect(url, token)

        for _identity, participant in room.remote_participants.items():
            for _tid, publication in participant.track_publications.items():
                t = publication.track
                if (
                    publication.kind == TrackKind.KIND_VIDEO
                    and t is not None
                ):
                    start_pump_for_track(t)

        try:
            first = await asyncio.wait_for(
                frame_queue.get(),
                timeout=_FIRST_FRAME_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise ConnectionError(
                "No remote video track received within "
                f"{_FIRST_FRAME_TIMEOUT_S:.0f}s. Is the admin publishing camera "
                "from the live dashboard?"
            ) from exc

        yield first

        while True:
            yield await frame_queue.get()

    finally:
        for task in pump_tasks:
            task.cancel()
        if pump_tasks:
            await asyncio.gather(*pump_tasks, return_exceptions=True)
        for vs in video_streams:
            await vs.aclose()
        await room.disconnect()


async def _pump_video_to_queue(
    stream,
    queue: asyncio.Queue[Frame],
    min_interval_s: float,
) -> None:
    from livekit.rtc import VideoBufferType

    last_emit_monotonic: float | None = None
    try:
        async for event in stream:
            now_m = time.monotonic()
            if (
                last_emit_monotonic is not None
                and now_m - last_emit_monotonic < min_interval_s
            ):
                continue
            vf = event.frame
            rgb = vf.convert(VideoBufferType.RGB24)
            raw = rgb.data.tobytes()
            image = Image.frombytes("RGB", (rgb.width, rgb.height), raw)
            ts = datetime.fromtimestamp(
                event.timestamp_us / 1_000_000.0,
                tz=timezone.utc,
            )
            last_emit_monotonic = now_m
            frame = Frame(image=image, timestamp=ts)
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(frame)
    except asyncio.CancelledError:
        raise
