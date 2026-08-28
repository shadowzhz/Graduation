"""视觉层数据结构。"""

from dataclasses import dataclass
from enum import Enum

from camera.types import Frame


@dataclass
class ROI:
    """图像坐标里的矩形感兴趣区域。"""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    def clamp(self, image_width: int, image_height: int):
        """裁到图像范围内，整体在图像外时返回 None。"""
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(image_width, self.x + self.width)
        bottom = min(image_height, self.y + self.height)
        if right <= left or bottom <= top:
            return None
        return ROI(left, top, right - left, bottom - top)


@dataclass
class Detection:
    """一帧里选出的最佳目标。"""

    center_x: float
    center_y: float
    radius: float
    area: float
    timestamp: float
    circularity: float = 0.0
    score: float = 0.0


class TrackState(Enum):
    ACTIVE = "active"
    LOST = "lost"


@dataclass
class Track:
    """一条连续目标的轨迹。"""

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
