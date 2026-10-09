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
from .state import PredictionState


class TrajectoryPredictor:
    """基于物理微步模拟的冰壶轨迹预测器。"""

    def __init__(
        self,
        substep: float | None = None,
        point_interval: float | None = None,
        max_simulation_steps: int | None = None,
        stop_speed: float | None = None,
    ):
        for name, value in (("substep", substep), ("point_interval", point_interval)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if stop_speed is not None and (not math.isfinite(stop_speed) or stop_speed < 0.0):
            raise ValueError("stop_speed must be finite and non-negative")
        if max_simulation_steps is not None and (
            isinstance(max_simulation_steps, bool)
            or not isinstance(max_simulation_steps, int)
            or max_simulation_steps <= 0
        ):
            raise ValueError("max_simulation_steps must be a positive integer")
        self._substep = substep
        self._point_interval = point_interval
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
    ) -> PredictionState:
        """预测到停止、进球或障碍接触，返回轨迹、准确终点、完整时长和初始状态。

        stone_or_x 可以是 CurlingState（视觉/控制侧）、StoneMotion（仿真侧）或裸坐标。
        超过计算步数预算时抛出 RuntimeError，不返回非终态的伪终点。
        """
        substep = self._substep if self._substep is not None else core.PREDICTION_SUBSTEP
        point_interval = self._point_interval if self._point_interval is not None else core.PREDICTION_POINT_INTERVAL
        max_simulation_steps = self._max_simulation_steps if self._max_simulation_steps is not None else core.PREDICTION_MAX_SIMULATION_STEPS
        stop_speed = self._stop_speed if self._stop_speed is not None else core.STONE_STOP_SPEED
        actual_obstacle_radius = obstacle_radius if obstacle_radius is not None else core.MALLET_RADIUS
        if not math.isfinite(actual_obstacle_radius) or actual_obstacle_radius < 0.0:
            raise ValueError("obstacle_radius must be finite and non-negative")
        if any(not math.isfinite(value) for obstacle in obstacles for value in obstacle):
            raise ValueError("obstacle positions must be finite")

        motion = self._normalize_stone(stone_or_x, y, vx, vy)
        source_state = stone_or_x if isinstance(stone_or_x, CurlingState) else CurlingState(x=motion.x, y=motion.y, vx=motion.vx, vy=motion.vy)
        origin = (motion.x, motion.y)
        collision_distance_sq = (core.STONE_RADIUS + actual_obstacle_radius) ** 2

        current_speed = math.hypot(motion.vx, motion.vy)
        target_speed = math.hypot(motion.target_vx, motion.target_vy)
        if (
            goal_scorer(motion)
            or any((motion.x - ox) ** 2 + (motion.y - oy) ** 2 <= collision_distance_sq for ox, oy in obstacles)
            or (current_speed <= stop_speed and (not motion.response_active or target_speed <= stop_speed))
        ):
            return PredictionState([origin], origin, 0.0, source_state)

        trajectory: list[tuple[float, float]] = [origin]
        sample_elapsed = 0.0
        duration = 0.0

        for _ in range(max_simulation_steps):
            duration += substep
            motion.x += motion.vx * substep
            motion.y += motion.vy * substep

            if goal_scorer(motion):
                trajectory.append((motion.x, motion.y))
                break

            bounced_this_step = motion.resolve_walls()
            bounced_this_step = motion.resolve_goal_posts() or bounced_this_step
            if bounced_this_step:
                if (motion.x, motion.y) != trajectory[-1]:
                    trajectory.append((motion.x, motion.y))
                sample_elapsed = 0.0

            if obstacles:
                if any((motion.x - ox) ** 2 + (motion.y - oy) ** 2 <= collision_distance_sq for ox, oy in obstacles):
                    trajectory.append((motion.x, motion.y))
                    break

            motion.advance_velocity(substep)

            speed = math.hypot(motion.vx, motion.vy)
            if speed <= stop_speed and not motion.response_active:
                if (motion.x, motion.y) != trajectory[-1]:
                    trajectory.append((motion.x, motion.y))
                break

            sample_elapsed += substep
            if sample_elapsed + 1e-9 >= point_interval:
                sample_elapsed -= point_interval
                if (motion.x, motion.y) != trajectory[-1]:
                    trajectory.append((motion.x, motion.y))
        else:
            raise RuntimeError(f"prediction exceeded max_simulation_steps={max_simulation_steps} before reaching a terminal state")

        return PredictionState(trajectory, (motion.x, motion.y), duration, source_state)

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
        """输入统一冰壶状态，返回预测终点（冰壶最终停在/离场的位置）。"""
        return self.predict(
            stone_or_x,
            y,
            vx,
            vy,
            obstacles=obstacles,
            obstacle_radius=obstacle_radius,
        ).endpoint

    @staticmethod
    def _normalize_stone(stone_or_x: Any, y: float | None = None, vx: float | None = None, vy: float | None = None) -> StoneMotion:
        # 仿真侧的 StoneMotion 自带目标速度/响应状态，直接复制保留。
        if isinstance(stone_or_x, StoneMotion):
            motion = replace(stone_or_x)
        else:
            # 其余输入复用 CurlingState 归一化。
            state = CurlingState.from_any(stone_or_x, y, vx, vy)
            motion = StoneMotion(
                x=state.x,
                y=state.y,
                vx=state.vx,
                vy=state.vy,
                target_vx=float(getattr(stone_or_x, "target_vx", state.vx)),
                target_vy=float(getattr(stone_or_x, "target_vy", state.vy)),
                response_active=bool(getattr(stone_or_x, "response_active", False)),
            )
        if not all(math.isfinite(value) for value in (motion.x, motion.y, motion.vx, motion.vy, motion.target_vx, motion.target_vy)):
            raise ValueError("stone position and velocities must be finite")
        motion.vx, motion.vy = motion._limited_velocity(motion.vx, motion.vy)
        motion.target_vx, motion.target_vy = motion._limited_velocity(motion.target_vx, motion.target_vy)
        return motion


_default_predictor = TrajectoryPredictor()


def predict_trajectory(
    stone_or_x: Any,
    y: float | None = None,
    vx: float | None = None,
    vy: float | None = None,
    *,
    obstacles: Sequence[tuple[float, float]] = (),
    obstacle_radius: float | None = None,
) -> PredictionState:
    """生成包含摩擦减速与碰撞反弹的预测结果。"""
    return _default_predictor.predict(
        stone_or_x,
        y,
        vx,
        vy,
        obstacles=obstacles,
        obstacle_radius=obstacle_radius,
    )
