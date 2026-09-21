"""AI 决策到 PLC 的最小控制适配层。

    AIDecision(target_x, target_y)
        -> ControlCommand(planning.TrajectoryPlanner)
        -> PlcWriteRequest
        -> PLCInterface.write_game_state(**request.as_kwargs())

只做坐标范围检查、输出限幅与字段映射，不修改 AI、Predictor 或 PLCInterface。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from .. import core_config as core
from ..planning import ControlCommand, TrajectoryPlanner
from game_state import CurlingState

_EPSILON = 1e-9


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
    """把 AI 决策转成控制指令，再打包成 PLC 写入请求。"""

    def __init__(
        self,
        max_speed: float = 600.0,
        speed_gain: float = 1.0,
        min_travel_time: float = 0.1,
        bounds: Optional[Sequence[float]] = None,
        margin: Optional[float] = None,
        max_target_jump: Optional[float] = None,
    ) -> None:
        self.planner = TrajectoryPlanner(
            max_speed=max_speed,
            speed_gain=speed_gain,
            min_travel_time=min_travel_time,
            bounds=bounds,
            margin=margin,
        )
        self.bounds = self.planner.bounds
        self.margin = self.planner.margin
        if max_target_jump is not None and max_target_jump <= 0.0:
            raise ValueError("max_target_jump must be positive when provided")
        self.max_target_jump = max_target_jump
        self._last_target: Optional[tuple[float, float]] = None

    def clamp_target(self, target_x: float, target_y: float) -> tuple[float, float]:
        """坐标范围检查 + 输出限幅：把目标收进场内可达范围。"""
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            raise ValueError("target must be finite")
        return self.planner.clamp_target((target_x, target_y))

    def build_command(
        self,
        decision,
        curling_state: CurlingState,
        prediction,
        *,
        timestamp: float = 0.0,
    ) -> ControlCommand:
        """AIDecision -> ControlCommand。

        decision 只需提供 target_x / target_y；预测时长用于换算所需速度。
        """
        target = self.clamp_target(float(decision.target_x), float(decision.target_y))
        command = self.planner.plan(curling_state, prediction, target)
        limited = self._limit_jump(command.target_position)
        if limited != command.target_position:
            command = replace(command, target_position=limited)
        return command

    def to_write_request(
        self,
        command: ControlCommand,
        *,
        ai_position: Sequence[float],
        curling_state: CurlingState,
        timestamp: float,
        player_score: int = 0,
        ai_score: int = 0,
    ) -> PlcWriteRequest:
        """ControlCommand -> PLC 写入请求（再次限幅，保证下发坐标合法）。"""
        target_x, target_y = self.clamp_target(*command.target_position)
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
