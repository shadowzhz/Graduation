"""统一轨迹预测物理核心。

视觉与仿真共用此预测核心，遵循同一套物理规则（摩擦阻尼、四周边界反弹、球门开口穿透、球门柱碰撞与停止速度）。
物理常量统一从 air_hockey_config 获取，无任何硬编码 fallback。
"""

import math
from dataclasses import replace
from typing import Any, Sequence

import air_hockey_config as layout
from air_hockey_physics import StoneMotion, goal_scorer


class TrajectoryPredictor:
    """基于物理微步模拟的冰壶轨迹预测器。"""

    def __init__(
        self,
        substep: float = layout.PREDICTION_SUBSTEP,
        point_interval: float = layout.PREDICTION_POINT_INTERVAL,
        max_points: int = layout.PREDICTION_POINT_COUNT,
        max_bends: int = layout.PREDICTION_MAX_BENDS,
        max_simulation_steps: int = layout.PREDICTION_MAX_SIMULATION_STEPS,
        stop_speed: float = layout.STONE_STOP_SPEED,
    ):
        self.substep = substep
        self.point_interval = point_interval
        self.max_points = max_points
        self.max_bends = max_bends
        self.max_simulation_steps = max_simulation_steps
        self.stop_speed = stop_speed

    def predict(
        self,
        stone_or_x: Any,
        y: float | None = None,
        vx: float | None = None,
        vy: float | None = None,
        *,
        obstacles: Sequence[tuple[float, float]] = (),
        obstacle_radius: float = layout.MALLET_RADIUS,
    ) -> list[tuple[float, float]]:
        motion = self._normalize_stone(stone_or_x, y, vx, vy)
        origin = (motion.x, motion.y)

        current_speed = math.hypot(motion.vx, motion.vy)
        target_speed = math.hypot(motion.target_vx, motion.target_vy)
        if current_speed <= self.stop_speed and (not motion.response_active or target_speed <= self.stop_speed):
            return [origin]

        trajectory: list[tuple[float, float]] = [origin]
        sample_elapsed = 0.0
        bend_count = 0
        simulation_steps = 0
        collision_distance_sq = (layout.STONE_RADIUS + obstacle_radius) ** 2

        while len(trajectory) < self.max_points and simulation_steps < self.max_simulation_steps:
            simulation_steps += 1
            motion.x += motion.vx * self.substep
            motion.y += motion.vy * self.substep

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
                if bend_count > self.max_bends or len(trajectory) >= self.max_points:
                    break

            if obstacles:
                if any((motion.x - ox) ** 2 + (motion.y - oy) ** 2 <= collision_distance_sq for ox, oy in obstacles):
                    trajectory.append((motion.x, motion.y))
                    break

            motion.advance_velocity(self.substep)

            speed = math.hypot(motion.vx, motion.vy)
            if speed <= self.stop_speed and not motion.response_active:
                if math.hypot(motion.x - trajectory[-1][0], motion.y - trajectory[-1][1]) > 1.0:
                    trajectory.append((motion.x, motion.y))
                break

            sample_elapsed += self.substep
            if sample_elapsed + 1e-9 >= self.point_interval:
                sample_elapsed -= self.point_interval
                if math.hypot(motion.x - trajectory[-1][0], motion.y - trajectory[-1][1]) > 1.0:
                    trajectory.append((motion.x, motion.y))

        return trajectory

    @staticmethod
    def _normalize_stone(stone_or_x: Any, y: float | None = None, vx: float | None = None, vy: float | None = None) -> StoneMotion:
        if y is not None:
            return StoneMotion(
                x=float(stone_or_x),
                y=float(y),
                vx=float(vx or 0.0),
                vy=float(vy or 0.0),
            )
        if isinstance(stone_or_x, StoneMotion):
            return replace(stone_or_x)
        return StoneMotion(
            x=float(stone_or_x.x),
            y=float(stone_or_x.y),
            vx=float(stone_or_x.vx),
            vy=float(stone_or_x.vy),
            target_vx=float(getattr(stone_or_x, "target_vx", stone_or_x.vx)),
            target_vy=float(getattr(stone_or_x, "target_vy", stone_or_x.vy)),
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
    obstacle_radius: float = layout.MALLET_RADIUS,
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
