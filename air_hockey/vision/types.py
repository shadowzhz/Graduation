"""Data contracts for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from camera.types import Frame


@dataclass(frozen=True)
class ROI:
    """图像坐标中的矩形感兴趣区域，原点位于左上角。"""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    def clamp(self, image_width: int, image_height: int) -> Optional["ROI"]:
        """Return the part inside an image, or None if it is outside."""
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(image_width, self.x + self.width)
        bottom = min(image_height, self.y + self.height)

        if right <= left or bottom <= top:
            return None

        return ROI(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class Detection:
    """从单帧中选出的最佳目标候选及其几何评分。"""

    center_x: float
    center_y: float
    radius: float
    area: float
    timestamp: float
    circularity: float = 0.0
    score: float = 0.0


class TrackState(Enum):
    """目标轨迹的生命周期状态。"""

    ACTIVE = "active"
    LOST = "lost"


@dataclass
class Track:
    """Tracker 对一个连续目标轨迹的状态记录。"""

    track_id: int

    center_x: float
    center_y: float
    radius: float

    vx: float
    vy: float

    last_timestamp: float

    age: int = 1
    missed_frames: int = 0
    state: TrackState = TrackState.ACTIVE


__all__ = ["Detection", "Frame", "ROI", "Track", "TrackState"]
