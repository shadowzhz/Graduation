"""轨迹规划层。

把状态估计与预测结果连接到控制接口：

    CurlingState + PredictionState + target_position  ->  ControlCommand

ControlCommand 给出控制侧需要的目标点、方向与速度，供上层控制/PLC 写值使用。
本模块不依赖也不修改 Detector / Tracker / KalmanFilter / Predictor / PLC。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .. import core_config as core
from ..prediction import PredictionState
from game_state import CurlingState

_EPSILON = 1e-9


@dataclass
class ControlCommand:
    """控制指令：目标位置、方向（单位向量）与速度。"""

    target_position: tuple[float, float]
    direction: tuple[float, float]
    speed: float

    @property
    def angle_degrees(self) -> float:
        """方向角（度，x 轴正向为 0，顺时针为正）。"""
        return math.degrees(math.atan2(self.direction[1], self.direction[0]))

    def to_dict(self) -> dict:
        return {
            "target_position": [float(self.target_position[0]), float(self.target_position[1])],
            "direction": [float(self.direction[0]), float(self.direction[1])],
            "speed": float(self.speed),
        }


class TrajectoryPlanner:
    """把预测结果转成控制指令的规划器。"""

    def __init__(
        self,
        max_speed: float = 600.0,
        speed_gain: float = 1.0,
        min_travel_time: float = 0.1,
        bounds: Optional[Sequence[float]] = None,
        margin: Optional[float] = None,
    ) -> None:
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if speed_gain <= 0.0:
            raise ValueError("speed_gain must be positive")
        if min_travel_time <= 0.0:
            raise ValueError("min_travel_time must be positive")
        self.max_speed = float(max_speed)
        self.speed_gain = float(speed_gain)
        self.min_travel_time = float(min_travel_time)
        self.bounds = tuple(bounds) if bounds is not None else (core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM)
        self.margin = core.MALLET_RADIUS if margin is None else float(margin)

    def clamp_target(self, target_position: Sequence[float]) -> tuple[float, float]:
        """把目标点收进场内可达范围（考虑球槌半径）。"""
        left, right, top, bottom = self.bounds
        target_x = min(max(float(target_position[0]), left + self.margin), right - self.margin)
        target_y = min(max(float(target_position[1]), top + self.margin), bottom - self.margin)
        return (target_x, target_y)

    def plan(
        self,
        curling_state: CurlingState,
        prediction: PredictionState,
        target_position: Sequence[float],
    ) -> ControlCommand:
        """规划一条控制指令。

        - target_position: 收进场内的可达目标点
        - direction:       当前冰壶状态指向目标点的单位向量（重合时为 (0, 0)）
        - speed:           在预测时长内抵达目标所需速度，按 speed_gain 缩放并限幅
        """
        target = self.clamp_target(target_position)
        delta_x = target[0] - curling_state.x
        delta_y = target[1] - curling_state.y
        distance = math.hypot(delta_x, delta_y)

        if distance <= _EPSILON:
            direction = (0.0, 0.0)
        else:
            direction = (delta_x / distance, delta_y / distance)

        # 用预测时长作为可用时间预算：冰壶越早停下，留给控制的时间越短，需求速度越高。
        travel_time = max(float(prediction.duration), self.min_travel_time)
        speed = min(self.max_speed, self.speed_gain * distance / travel_time)

        return ControlCommand(target_position=target, direction=direction, speed=speed)
