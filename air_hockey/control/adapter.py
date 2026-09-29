"""AI 目标到 PLC 载荷的控制适配层：坐标限幅与字段映射。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .. import core_config as core
from game_state import CurlingState


@dataclass
class PlcWriteRequest:
    """一次 PLC 写入的完整载荷。

    timestamp 用于本地上报/记录（PLC DB 布局固定为 36 字节，不含时间戳字段），
    因此 as_kwargs() 不会把它传给 write_game_state。
    """

    ai_target_x: float
    ai_target_y: float
    ai_x: float
    ai_y: float
    stone_x: float
    stone_y: float
    stone_vx: float
    stone_vy: float
    player_score: int
    ai_score: int
    timestamp: float

    def as_kwargs(self) -> dict:
        """展开为 PLCInterface.write_game_state 的关键字参数。"""
        return {
            "ai_target_x": self.ai_target_x,
            "ai_target_y": self.ai_target_y,
            "ai_x": self.ai_x,
            "ai_y": self.ai_y,
            "stone_x": self.stone_x,
            "stone_y": self.stone_y,
            "stone_vx": self.stone_vx,
            "stone_vy": self.stone_vy,
            "player_score": int(self.player_score),
            "ai_score": int(self.ai_score),
        }

    def to_dict(self) -> dict:
        payload = self.as_kwargs()
        payload["timestamp"] = float(self.timestamp)
        return payload


class PlcControlAdapter:
    """把 AI 决策目标限幅后打包成 PLC 写入请求。"""

    def __init__(
        self,
        bounds: Optional[Sequence[float]] = None,
        margin: Optional[float] = None,
        max_target_jump: Optional[float] = None,
    ) -> None:
        self.bounds = tuple(float(value) for value in bounds) if bounds is not None else (
            core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM
        )
        self.margin = core.MALLET_RADIUS if margin is None else float(margin)
        if len(self.bounds) != 4 or not all(math.isfinite(value) for value in self.bounds):
            raise ValueError("bounds must contain four finite coordinates")
        left, right, top, bottom = self.bounds
        if left >= right or top >= bottom:
            raise ValueError("bounds must enclose a positive area")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        # 自定义范围不能放宽 PLC 可达的物理场地边界。
        x_min = max(left + self.margin, core.RINK_LEFT + core.MALLET_RADIUS)
        x_max = min(right - self.margin, core.RINK_RIGHT - core.MALLET_RADIUS)
        y_min = max(top + self.margin, core.RINK_TOP + core.MALLET_RADIUS)
        y_max = min(bottom - self.margin, core.RINK_BOTTOM - core.MALLET_RADIUS)
        if x_min > x_max or y_min > y_max:
            raise ValueError("no reachable target within rink bounds")
        self._limits = (x_min, x_max, y_min, y_max)
        if max_target_jump is not None and (not math.isfinite(max_target_jump) or max_target_jump <= 0.0):
            raise ValueError("max_target_jump must be positive when provided")
        self.max_target_jump = max_target_jump
        self._last_target: Optional[tuple[float, float]] = None

    def clamp_target(self, target_x: float, target_y: float) -> tuple[float, float]:
        """检查非有限坐标，将目标限制在自定义范围和实际场地内。"""
        target_x, target_y = float(target_x), float(target_y)
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            raise ValueError("target must be finite")
        x_min, x_max, y_min, y_max = self._limits
        return min(max(target_x, x_min), x_max), min(max(target_y, y_min), y_max)

    def build_request(
        self,
        decision,
        *,
        ai_position: Sequence[float],
        curling_state: CurlingState,
        timestamp: float,
        player_score: int = 0,
        ai_score: int = 0,
    ) -> PlcWriteRequest:
        """AI 决策 -> PLC 写入请求；下发前再次收进实际场地边界。"""
        target = self.clamp_target(decision.target_x, decision.target_y)
        target_x, target_y = self.clamp_target(*self._limit_jump(target))
        return PlcWriteRequest(
            ai_target_x=float(target_x),
            ai_target_y=float(target_y),
            ai_x=float(ai_position[0]),
            ai_y=float(ai_position[1]),
            stone_x=float(curling_state.x),
            stone_y=float(curling_state.y),
            stone_vx=float(curling_state.vx),
            stone_vy=float(curling_state.vy),
            player_score=int(player_score),
            ai_score=int(ai_score),
            timestamp=float(timestamp),
        )

    def _limit_jump(self, target: tuple[float, float]) -> tuple[float, float]:
        """限制单次下发的目标跳变距离，避免实机球槌目标突变。"""
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
