"""统一轨迹预测物理核心。

视觉与仿真共用此预测核心，遵循同一套物理规则（摩擦阻尼、四周边界反弹、球门开口穿透、球门柱碰撞与停止速度）。
物理常量统一在运行时从 core_config 获取，无任何硬编码 fallback。

状态统一使用 CurlingState 描述，状态转换集中在 CurlingState.from_any，预测器本身不再重复解析坐标。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

from .. import core_config as core
from ..physics import StoneMotion, goal_scorer
from game_state import CurlingState


class TrajectoryPredictor:
    """基于物理微步模拟的冰壶轨迹预测器。"""

    def __init__(
        self,
        substep: float | None = None,
        point_interval: float | None = None,
        max_points: int | None = None,
        max_bends: int | None = None,
        max_simulation_steps: int | None = None,
        stop_speed: float | None = None,
    ):
        self._substep = substep
        self._point_interval = point_interval
        self._max_points = max_points
        self._max_bends = max_bends
        self._max_simulation_steps = max_simulation_steps
        self._stop_speed = stop_speed

    def predict(
        self,
        stone_or_x: Any,
        y: float | None = None,
        vx: float | None = None,
        vy: float | None = None,
        *,
        obstacles: Sequence[tuple[float, float]] = (),
        obstacle_radius: float | None = None,
    ) -> list[tuple[float, float]]:
        """输入统一冰壶状态，返回按 PREDICTION_POINT_INTERVAL 采样的预测轨迹点。

        stone_or_x 可以是 CurlingState（视觉/控制侧）、StoneMotion（仿真侧）或裸坐标。
        """
        substep = self._substep if self._substep is not None else core.PREDICTION_SUBSTEP
        point_interval = self._point_interval if self._point_interval is not None else core.PREDICTION_POINT_INTERVAL
        max_points = self._max_points if self._max_points is not None else core.PREDICTION_POINT_COUNT
        max_bends = self._max_bends if self._max_bends is not None else core.PREDICTION_MAX_BENDS
        max_simulation_steps = self._max_simulation_steps if self._max_simulation_steps is not None else core.PREDICTION_MAX_SIMULATION_STEPS
        stop_speed = self._stop_speed if self._stop_speed is not None else core.STONE_STOP_SPEED
        actual_obstacle_radius = obstacle_radius if obstacle_radius is not None else core.MALLET_RADIUS

        motion = self._normalize_stone(stone_or_x, y, vx, vy)
        origin = (motion.x, motion.y)

        current_speed = math.hypot(motion.vx, motion.vy)
        target_speed = math.hypot(motion.target_vx, motion.target_vy)
        if current_speed <= stop_speed and (not motion.response_active or target_speed <= stop_speed):
            return [origin]

        trajectory: list[tuple[float, float]] = [origin]
        sample_elapsed = 0.0
        bend_count = 0
        simulation_steps = 0
        collision_distance_sq = (core.STONE_RADIUS + actual_obstacle_radius) ** 2

        while len(trajectory) < max_points and simulation_steps < max_simulation_steps:
            simulation_steps += 1
            motion.x += motion.vx * substep
            motion.y += motion.vy * substep

            if goal_scorer(motion):
                trajectory.append((motion.x, motion.y))
                break

            bounced_this_step = motion.resolve_walls()
            bounced_this_step = motion.resolve_goal_posts() or bounced_this_step
            if bounced_this_step:
                bend_count += 1
                if math.hypot(motion.x - trajectory[-1][0], motion.y - trajectory[-1][1]) > 1.0:
                    trajectory.append((motion.x, motion.y))
                sample_elapsed = 0.0
                if bend_count > max_bends or len(trajectory) >= max_points:
                    break

            if obstacles:
                if any((motion.x - ox) ** 2 + (motion.y - oy) ** 2 <= collision_distance_sq for ox, oy in obstacles):
                    trajectory.append((motion.x, motion.y))
                    break

            motion.advance_velocity(substep)

            speed = math.hypot(motion.vx, motion.vy)
            if speed <= stop_speed and not motion.response_active:
                if math.hypot(motion.x - trajectory[-1][0], motion.y - trajectory[-1][1]) > 1.0:
                    trajectory.append((motion.x, motion.y))
                break

            sample_elapsed += substep
            if sample_elapsed + 1e-9 >= point_interval:
                sample_elapsed -= point_interval
                if math.hypot(motion.x - trajectory[-1][0], motion.y - trajectory[-1][1]) > 1.0:
                    trajectory.append((motion.x, motion.y))

        return trajectory

    def predict_endpoint(
        self,
        stone_or_x: Any,
        y: float | None = None,
        vx: float | None = None,
        vy: float | None = None,
        *,
        obstacles: Sequence[tuple[float, float]] = (),
        obstacle_radius: float | None = None,
    ) -> tuple[float, float]:
        """输入统一冰壶状态，返回预测轨迹的终点（冰壶最终停在/离场的位置）。"""
        trajectory = self.predict(
            stone_or_x,
            y,
            vx,
            vy,
            obstacles=obstacles,
            obstacle_radius=obstacle_radius,
        )
        return trajectory[-1]

    @staticmethod
    def _normalize_stone(stone_or_x: Any, y: float | None = None, vx: float | None = None, vy: float | None = None) -> StoneMotion:
        # 仿真侧的 StoneMotion 自带目标速度/响应状态，直接复制保留。
        if isinstance(stone_or_x, StoneMotion):
            return replace(stone_or_x)
        # 其余（CurlingState / 裸坐标）统一走 CurlingState 归一化，避免各调用点重复转换。
        state = CurlingState.from_any(stone_or_x, y, vx, vy)
        return StoneMotion(
            x=state.x,
            y=state.y,
            vx=state.vx,
            vy=state.vy,
            target_vx=float(getattr(stone_or_x, "target_vx", state.vx)),
            target_vy=float(getattr(stone_or_x, "target_vy", state.vy)),
            response_active=bool(getattr(stone_or_x, "response_active", False)),
        )


_default_predictor = TrajectoryPredictor()


def predict_trajectory(
    stone_or_x: Any,
    y: float | None = None,
    vx: float | None = None,
    vy: float | None = None,
    *,
    obstacles: Sequence[tuple[float, float]] = (),
    obstacle_radius: float | None = None,
) -> list[tuple[float, float]]:
    """生成包含摩擦减速与碰撞反弹的预测轨迹。"""
    return _default_predictor.predict(
        stone_or_x,
        y,
        vx,
        vy,
        obstacles=obstacles,
        obstacle_radius=obstacle_radius,
    )
