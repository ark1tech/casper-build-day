"""Frame dataclass shared across practice and live modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PIL import Image


@dataclass(frozen=True)
class Frame:
    """A single captured video frame."""

    image: Image.Image
    """The frame as a PIL Image (RGB)."""

    timestamp: datetime
    """When the frame was captured."""
