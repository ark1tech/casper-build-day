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


def _build_capture_cmd(ffmpeg: str, camera_index: int) -> list[str]:
    """Build a platform-appropriate ffmpeg command for single-frame capture."""
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
    interval = 1.0 / fps

    print(f"[practice] Opening camera {camera_index}...")
    print(f"[practice] Sampling at {fps} FPS. Press Ctrl+C to stop.\n")

    try:
        ffmpeg = _detect_ffmpeg()
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return

    cmd = _build_capture_cmd(ffmpeg, camera_index)

    try:
        test_frame = await _capture_one_frame(cmd)
        print(f"[practice] Camera {camera_index} ready "
              f"({test_frame.size[0]}x{test_frame.size[1]}).\n")
    except Exception as exc:
        print(f"[!] Could not capture from camera {camera_index}: {exc}")
        return

    while True:
        try:
            image = await _capture_one_frame(cmd)

            yield Frame(
                image=image,
                timestamp=datetime.now(timezone.utc),
            )

            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n[practice] Stopped.")
            break
        except Exception as exc:
            print(f"[practice] Error capturing frame: {exc}")
            break
