"""将固定场地的 AI 目标限幅到实机球槌可达半场。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .. import core_config as core


@dataclass(frozen=True)
class PlcTarget:
    """交给 PLCLink 的游戏场地目标；timestamp 仅用于本地记录。"""

    target_x: float
    target_y: float
    timestamp: float

    def to_dict(self) -> dict:
        return {"target_x": self.target_x, "target_y": self.target_y, "timestamp": self.timestamp}


class PlcControlAdapter:
    """只允许 AI 半场的有限目标，选配单次目标跳变限制。"""

    def __init__(
        self,
        bounds: Optional[Sequence[float]] = None,
        margin: Optional[float] = None,
        max_target_jump: Optional[float] = None,
    ) -> None:
        self.bounds = tuple(float(value) for value in bounds) if bounds is not None else (
            core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_CENTER_Y
        )
        self.margin = core.MALLET_RADIUS if margin is None else float(margin)
        if len(self.bounds) != 4 or not all(math.isfinite(value) for value in self.bounds):
            raise ValueError("bounds must contain four finite coordinates")
        left, right, top, bottom = self.bounds
        if left >= right or top >= bottom:
            raise ValueError("bounds must enclose a positive area")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        x_min = max(left + self.margin, core.RINK_LEFT + core.MALLET_RADIUS)
        x_max = min(right - self.margin, core.RINK_RIGHT - core.MALLET_RADIUS)
        y_min = max(top + self.margin, core.RINK_TOP + core.MALLET_RADIUS)
        y_max = min(bottom - self.margin, core.RINK_CENTER_Y - core.MALLET_RADIUS)
        if x_min > x_max or y_min > y_max:
            raise ValueError("no reachable target within AI half")
        self._limits = (x_min, x_max, y_min, y_max)
        if max_target_jump is not None and (not math.isfinite(max_target_jump) or max_target_jump <= 0.0):
            raise ValueError("max_target_jump must be positive when provided")
        self.max_target_jump = max_target_jump
        self._last_target: Optional[tuple[float, float]] = None

    def clamp_target(self, target_x: float, target_y: float) -> tuple[float, float]:
        target_x, target_y = float(target_x), float(target_y)
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            raise ValueError("target must be finite")
        x_min, x_max, y_min, y_max = self._limits
        return min(max(target_x, x_min), x_max), min(max(target_y, y_min), y_max)

    def build_target(self, decision, *, timestamp: float) -> PlcTarget:
        target = self.clamp_target(decision.target_x, decision.target_y)
        x, y = self.clamp_target(*self._limit_jump(target))
        return PlcTarget(x, y, float(timestamp))

    def _limit_jump(self, target: tuple[float, float]) -> tuple[float, float]:
        if self.max_target_jump is None:
            return target
        if self._last_target is not None:
            delta_x = target[0] - self._last_target[0]
            delta_y = target[1] - self._last_target[1]
            distance = math.hypot(delta_x, delta_y)
            if distance > self.max_target_jump:
                scale = self.max_target_jump / distance
                target = (
                    self._last_target[0] + delta_x * scale,
                    self._last_target[1] + delta_y * scale,
                )
        self._last_target = target
        return target

    def reset(self) -> None:
        self._last_target = None
