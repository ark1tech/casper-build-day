"""Practice mode: capture frames from a local camera via ffmpeg subprocess."""

from __future__ import annotations

import asyncio
import platform
import shutil
from datetime import datetime, timezone
from typing import AsyncIterator

from PIL import Image

from core.frame import Frame


def _detect_ffmpeg() -> str:
    """Find usable ffmpeg binary, preferring system install over imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FileNotFoundError(
        "ffmpeg not found. Install it:\n"
        "  Linux:  sudo apt install ffmpeg\n"
        "  macOS:  brew install ffmpeg\n"
        "  Windows: winget install ffmpeg"
    )


def _camera_input_args(camera_index: int) -> tuple[list[str], str]:
    """Return (ffmpeg input option list, device string) for the local camera."""
    system = platform.system()

    if system == "Linux":
        input_fmt = ["-f", "v4l2"]
        device = f"/dev/video{camera_index}"
    elif system == "Darwin":
        # avfoundation defaults to ~29.97 fps; many Mac cameras only allow 30.0.
        input_fmt = ["-f", "avfoundation", "-framerate", "30"]
        device = str(camera_index)
    elif system == "Windows":
        input_fmt = ["-f", "dshow"]
        device = f"video={camera_index}"
    else:
        input_fmt = ["-f", "v4l2"]
        device = f"/dev/video{camera_index}"

    return input_fmt, device


def _build_probe_cmd(ffmpeg: str, camera_index: int) -> list[str]:
    """Single-frame probe to validate the device and learn frame dimensions."""
    input_fmt, device = _camera_input_args(camera_index)
    return [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        *input_fmt,
        "-i", device,
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]


def _build_stream_cmd(ffmpeg: str, camera_index: int, output_fps: int) -> list[str]:
    """Continuous raw RGB stream at ``output_fps`` — keeps the camera open."""
    input_fmt, device = _camera_input_args(camera_index)
    fps = max(1, output_fps)
    return [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        *input_fmt,
        "-i", device,
        "-vf", f"fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]


async def _read_exact(stream: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly ``n`` bytes from a stream (async)."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = await stream.read(remaining)
        if not chunk:
            raise RuntimeError(f"EOF after {n - remaining} of {n} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def _drain_stderr(stream: asyncio.StreamReader) -> None:
    """Consume stderr so a long-running ffmpeg process does not block."""
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
    except Exception:
        pass


async def _capture_one_frame(cmd: list[str]) -> Image.Image:
    """Run ffmpeg once to grab a single frame, return as PIL Image."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg capture failed (exit {proc.returncode}): {err}")

    if not stdout:
        raise RuntimeError("ffmpeg returned no data")

    raw = stdout
    num_bytes = len(raw)
    for w, h in [(640, 480), (1280, 720), (1920, 1080), (320, 240), (800, 600)]:
        if w * h * 3 == num_bytes:
            return Image.frombytes("RGB", (w, h), raw)

    raise RuntimeError(
        f"Could not determine frame dimensions from {num_bytes} bytes of raw data. "
        "Try specifying resolution with -video_size in the ffmpeg command."
    )


async def start_practice(
    camera_index: int = 0,
    fps: int = 1,
) -> AsyncIterator[Frame]:
    """Yield frames from the local camera at the given FPS.

    Args:
        camera_index: Which camera device to use (default 0).
        fps: Frames per second to sample (default 1).

    Yields:
        Frame objects with a PIL Image and timestamp.
    """
    print(f"[practice] Opening camera {camera_index}...")
    print(f"[practice] Sampling at {fps} FPS. Press Ctrl+C to stop.\n")

    try:
        ffmpeg = _detect_ffmpeg()
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return

    probe_cmd = _build_probe_cmd(ffmpeg, camera_index)

    try:
        test_frame = await _capture_one_frame(probe_cmd)
    except Exception as exc:
        print(f"[!] Could not capture from camera {camera_index}: {exc}")
        return

    w, h = test_frame.size
    frame_bytes = w * h * 3

    stream_cmd = _build_stream_cmd(ffmpeg, camera_index, fps)
    proc = await asyncio.create_subprocess_exec(
        *stream_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))

    print(
        f"[practice] Camera {camera_index} streaming "
        f"({w}x{h}) — one ffmpeg process, camera stays on.\n"
    )

    try:
        while True:
            try:
                raw = await _read_exact(proc.stdout, frame_bytes)
            except KeyboardInterrupt:
                print("\n[practice] Stopped.")
                break
            except Exception as exc:
                print(f"[practice] Error capturing frame: {exc}")
                break

            image = Image.frombytes("RGB", (w, h), raw)
            yield Frame(
                image=image,
                timestamp=datetime.now(timezone.utc),
            )
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass
