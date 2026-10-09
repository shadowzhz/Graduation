"""卡尔曼滤波状态估计层。

用常量速度（CV）模型平滑 Detector/Tracker 给出的位置观测，估计更稳定的位置与速度，
默认模型输出统一的 CurlingState，降低视觉抖动；friction 用于自由滑动 EKF，collision_aware 增加已知球台坐标下的墙法向反弹。三种模型都不处理门柱碰撞，实时默认仍为 CV。

状态向量 x = [px, py, vx, vy]^T，观测 z = [px, py]^T。
"""

from __future__ import annotations

import math

import numpy as np

from .. import core_config as core
from ..physics import apply_friction_velocity, resolve_wall_axis, stone_inside_goal_mouth
from game_state import CurlingState

_STATE_DIM = 4
_MEASUREMENT_DIM = 2
# 单步时间上限，避免长时间丢帧后一次性推进过大
_MAX_DT = 0.5


class KalmanFilter:
    """默认常速度 Kalman；可选 friction / collision_aware EKF。"""

    def __init__(
        self,
        measurement_noise: float = 4.0,
        acceleration_noise: float = 200.0,
        initial_velocity_variance: float = 1_000_000.0,
        motion_model: str = "constant_velocity",
    ) -> None:
        if measurement_noise <= 0.0:
            raise ValueError("measurement_noise must be positive")
        if acceleration_noise < 0.0:
            raise ValueError("acceleration_noise must be non-negative")
        if initial_velocity_variance <= 0.0:
            raise ValueError("initial_velocity_variance must be positive")
        if motion_model not in ("constant_velocity", "friction", "collision_aware"):
            raise ValueError("motion_model must be 'constant_velocity', 'friction' or 'collision_aware'")
        self.measurement_noise = float(measurement_noise)
        self.acceleration_noise = float(acceleration_noise)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.motion_model = motion_model
        self.reset()

    def reset(self) -> None:
        """清空滤波器，等待下一次观测重新初始化。"""
        self._x = np.zeros(_STATE_DIM)
        self._P = np.eye(_STATE_DIM)
        self._last_timestamp = None
        self._initialized = False
        self._friction_started = False
        self._confidence = 0.0
        self._radius = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def state_vector(self) -> np.ndarray:
        """当前状态估计 [px, py, vx, vy] 的副本。"""
        return self._x.copy()

    def predict(self, timestamp: float, *, confidence: float | None = None, radius: float | None = None) -> CurlingState:
        """没有新观测时按模型外推到 timestamp，返回平滑后的 CurlingState。"""
        timestamp = float(timestamp)
        if not self._initialized:
            return CurlingState(x=0.0, y=0.0, timestamp=timestamp)
        self._advance(self._elapsed(timestamp))
        self._last_timestamp = timestamp
        self._remember(confidence, radius)
        return self._to_curling_state()

    def update(
        self,
        x: float,
        y: float,
        timestamp: float,
        *,
        confidence: float | None = None,
        radius: float | None = None,
    ) -> CurlingState:
        """用新的位置观测 (x, y, timestamp) 校正状态，返回平滑后的 CurlingState。"""
        measurement = np.array([float(x), float(y)])
        timestamp = float(timestamp)

        if not self._initialized:
            self._initialize(measurement, timestamp)
            self._remember(confidence, radius)
            return self._to_curling_state()

        self._advance(self._elapsed(timestamp))

        H = self._measurement_matrix()
        Ht = H.T
        R = np.eye(_MEASUREMENT_DIM) * self.measurement_noise**2
        innovation = measurement - H @ self._x
        S = H @ self._P @ Ht + R
        gain = self._P @ Ht @ np.linalg.inv(S)
        self._x = self._x + gain @ innovation
        self._P = (np.eye(_STATE_DIM) - gain @ H) @ self._P

        self._last_timestamp = timestamp
        self._remember(confidence, radius)
        return self._to_curling_state()

    # --- 内部实现 ---

    def _initialize(self, measurement: np.ndarray, timestamp: float) -> None:
        self._x = np.array([measurement[0], measurement[1], 0.0, 0.0])
        self._P = np.diag(
            [
                self.measurement_noise**2,
                self.measurement_noise**2,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ]
        )
        self._last_timestamp = timestamp
        self._initialized = True

    def _elapsed(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            return 0.0
        return min(max(timestamp - self._last_timestamp, 0.0), _MAX_DT)

    def _advance(self, dt: float) -> None:
        if self.motion_model == "collision_aware" and self._friction_started and dt > 0.0:
            self._advance_walls(dt)
            return
        if self.motion_model in ("friction", "collision_aware") and self._friction_started:
            vx, vy = float(self._x[2]), float(self._x[3])
            jacobian = self._friction_velocity_jacobian(vx, vy, dt)
            next_vx, next_vy = apply_friction_velocity(vx, vy, dt)
            transition = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, jacobian[0, 0], jacobian[0, 1]],
                    [0.0, 0.0, jacobian[1, 0], jacobian[1, 1]],
                ]
            )
            self._x[0] += vx * dt
            self._x[1] += vy * dt
            self._x[2] = next_vx
            self._x[3] = next_vy
            self._P = transition @ self._P @ transition.T + self._process_noise(dt)
            return

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._x = transition @ self._x
        self._P = transition @ self._P @ transition.T + self._process_noise(dt)
        if self.motion_model in ("friction", "collision_aware") and dt > 0.0:
            # 首个位置差用来估算未知初速度，再启用摩擦模型。
            self._friction_started = True

    def _advance_walls(self, dt: float) -> None:
        """Position, wall-normal reflection, friction; never use measurement/truth collision flags."""
        # Only normal wall contacts are modeled, not corner-ejection impulses or goal posts.
        vx, vy = float(self._x[2]), float(self._x[3])
        x, y = self._x[0] + vx * dt, self._x[1] + vy * dt
        next_x, vx, hit_x = resolve_wall_axis(
            x, vx, core.RINK_LEFT + core.STONE_RADIUS, core.RINK_RIGHT - core.STONE_RADIUS
        )
        next_y, hit_y = y, False
        if not stone_inside_goal_mouth(next_x):
            next_y, vy, hit_y = resolve_wall_axis(
                y, vy, core.RINK_TOP + core.STONE_RADIUS, core.RINK_BOTTOM - core.STONE_RADIUS
            )
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        if hit_x or next_x != x:
            transition[0, :] = 0.0
        if hit_y or next_y != y:
            transition[1, :] = 0.0
        if hit_x:
            transition[2, 2] = -core.WALL_RESTITUTION
        if hit_y:
            transition[3, 3] = -core.WALL_RESTITUTION
        transition[2:, :] = self._friction_velocity_jacobian(vx, vy, dt) @ transition[2:, :]
        next_vx, next_vy = apply_friction_velocity(vx, vy, dt)
        self._x[:] = next_x, next_y, next_vx, next_vy
        self._P = transition @ self._P @ transition.T + self._process_noise(dt)

    @staticmethod
    def _friction_velocity_jacobian(vx: float, vy: float, dt: float) -> np.ndarray:
        speed = math.hypot(vx, vy)
        deceleration = core.STONE_FRICTION_DECELERATION * dt
        if speed <= 1e-9 or speed - deceleration <= 1e-9:
            return np.zeros((2, 2))
        scale = 1.0 - deceleration / speed
        radial = deceleration / (speed * speed * speed)
        return np.array(
            [
                [scale + radial * vx * vx, radial * vx * vy],
                [radial * vx * vy, scale + radial * vy * vy],
            ]
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        variance = self.acceleration_noise**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return variance * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ]
        )

    @staticmethod
    def _measurement_matrix() -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

    def _remember(self, confidence: float | None, radius: float | None) -> None:
        if confidence is not None:
            self._confidence = float(confidence)
        if radius is not None:
            self._radius = float(radius)

    def _to_curling_state(self) -> CurlingState:
        return CurlingState(
            x=float(self._x[0]),
            y=float(self._x[1]),
            vx=float(self._x[2]),
            vy=float(self._x[3]),
            timestamp=float(self._last_timestamp),
            confidence=self._confidence,
            radius=self._radius,
        )
