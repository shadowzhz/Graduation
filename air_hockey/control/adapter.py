"""将固定场地的 AI 目标限幅到实机球槌可达半场。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import core_config as core


@dataclass(frozen=True)
class PlcTarget:
    """交给 PLCLink 的游戏场地目标；timestamp 是决策所用帧的 perf_counter 时刻，用于计算有效期。"""

    target_x: float
    target_y: float
    timestamp: float

    def to_dict(self) -> dict:
        return {"target_x": self.target_x, "target_y": self.target_y, "timestamp": self.timestamp}


class PlcControlAdapter:
    """只允许 AI 半场内、球槌不贴墙的有限目标。"""

    X_MIN = core.RINK_LEFT + core.MALLET_RADIUS
    X_MAX = core.RINK_RIGHT - core.MALLET_RADIUS
    Y_MIN = core.RINK_TOP + core.MALLET_RADIUS
    Y_MAX = core.RINK_CENTER_Y - core.MALLET_RADIUS

    def __init__(self, bounds: tuple[float, float, float, float] | None = None) -> None:
        if bounds is None:
            self.x_min, self.x_max = self.X_MIN, self.X_MAX
            self.y_min, self.y_max = self.Y_MIN, self.Y_MAX
        else:
            left, right, top, bottom = map(float, bounds)
            self.x_min = max(self.X_MIN, left)
            self.x_max = min(self.X_MAX, right)
            self.y_min = max(self.Y_MIN, top)
            self.y_max = min(self.Y_MAX, bottom)

    def build_target(self, target_x: float, target_y: float, timestamp: float) -> PlcTarget:
        target_x, target_y = float(target_x), float(target_y)
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            raise ValueError("target must be finite")
        return PlcTarget(min(max(target_x, self.x_min), self.x_max),
                         min(max(target_y, self.y_min), self.y_max), float(timestamp))
