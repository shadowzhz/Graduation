"""无摄像头运动轨迹仿真。

生成真实冰壶轨迹（共享物理 StoneMotion），叠加视觉位置噪声得到模拟观测，
再跑完整数据链路：

    真实轨迹 -> 模拟观测 -> KalmanFilter -> CurlingState -> Predictor -> PredictionState

输出结构化结果：真实轨迹 / 观测轨迹 / 滤波轨迹 / 预测终点。
可直接用于在无相机条件下验证 KalmanFilter 与 Predictor。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .. import core_config as core
from ..estimation import KalmanFilter
from ..physics import StoneMotion
from ..prediction import PredictionState, TrajectoryPredictor
from game_state import CurlingState


@dataclass
class SimulationConfig:
    """仿真输入：初始位置、初始速度、时间步长与观测噪声。"""

    initial_x: float
    initial_y: float
    initial_vx: float
    initial_vy: float
    dt: float = 1.0 / 60.0
    steps: int = 220
    position_noise: float = 3.0
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.position_noise < 0.0:
            raise ValueError("position_noise must be non-negative")


@dataclass
class SimulationResult:
    """仿真结果：真实/观测/滤波轨迹与预测终点。"""

    true_trajectory: list[tuple[float, float]]
    observed_trajectory: list[tuple[float, float]]
    filtered_trajectory: list[tuple[float, float]]
    predicted_endpoint: tuple[float, float]
    prediction: PredictionState
    filtered_state: CurlingState
    timestamps: list[float]

    @property
    def true_endpoint(self) -> tuple[float, float]:
        return self.true_trajectory[-1]

    def _mean_error(self, sequence: list[tuple[float, float]]) -> float:
        if not sequence:
            return 0.0
        return sum(
            math.hypot(point[0] - truth[0], point[1] - truth[1])
            for point, truth in zip(sequence, self.true_trajectory)
        ) / len(sequence)

    def mean_observation_error(self) -> float:
        """观测轨迹相对真实轨迹的平均位置误差。"""
        return self._mean_error(self.observed_trajectory)

    def mean_filtered_error(self) -> float:
        """滤波轨迹相对真实轨迹的平均位置误差。"""
        return self._mean_error(self.filtered_trajectory)

    def endpoint_error(self) -> float:
        """预测终点与真实终点的距离。"""
        return math.hypot(
            self.predicted_endpoint[0] - self.true_endpoint[0],
            self.predicted_endpoint[1] - self.true_endpoint[1],
        )

    def summary(self) -> str:
        truth_x, truth_y = self.true_endpoint
        predict_x, predict_y = self.predicted_endpoint
        return (
            f"samples={len(self.true_trajectory)} | "
            f"true_endpoint=({truth_x:.1f}, {truth_y:.1f}) | "
            f"predicted_endpoint=({predict_x:.1f}, {predict_y:.1f}) | "
            f"endpoint_error={self.endpoint_error():.2f} | "
            f"obs_error={self.mean_observation_error():.2f} | "
            f"filtered_error={self.mean_filtered_error():.2f}"
        )


class MotionSimulator:
    """基于共享物理的运动轨迹仿真器。"""

    def __init__(self, config: SimulationConfig, predictor: Optional[TrajectoryPredictor] = None) -> None:
        self.config = config
        self.predictor = predictor or TrajectoryPredictor()

    @classmethod
    def default(cls, **overrides) -> "MotionSimulator":
        """默认直线上滑、无碰撞、会在界内停下的场景。"""
        config = SimulationConfig(
            initial_x=core.RINK_CENTER_X,
            initial_y=core.RINK_BOTTOM - 60.0,
            initial_vx=0.0,
            initial_vy=-220.0,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        config.__post_init__()
        return cls(config)

    @classmethod
    def random(cls, seed: Optional[int] = 0, **overrides) -> "MotionSimulator":
        """在场地内随机生成初始位置与初始速度（朝上方球门）。"""
        generator = random.Random(seed)
        radius = core.STONE_RADIUS
        initial_x = generator.uniform(core.RINK_LEFT + radius, core.RINK_RIGHT - radius)
        initial_y = generator.uniform(core.RINK_CENTER_Y, core.RINK_BOTTOM - radius)
        angle = generator.uniform(math.radians(210.0), math.radians(330.0))
        speed = generator.uniform(150.0, 300.0)
        config = SimulationConfig(
            initial_x=initial_x,
            initial_y=initial_y,
            initial_vx=speed * math.cos(angle),
            initial_vy=speed * math.sin(angle),
            seed=seed,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        config.__post_init__()
        return cls(config)

    def true_trajectory(self) -> list[tuple[float, float]]:
        """用共享物理（StoneMotion）推进得到真实轨迹。"""
        stone = StoneMotion(
            x=self.config.initial_x,
            y=self.config.initial_y,
            vx=self.config.initial_vx,
            vy=self.config.initial_vy,
        )
        trajectory = [(stone.x, stone.y)]
        for _ in range(self.config.steps):
            stone.advance_velocity(self.config.dt)
            stone.x += stone.vx * self.config.dt
            stone.y += stone.vy * self.config.dt
            stone.resolve_walls()
            trajectory.append((stone.x, stone.y))
        return trajectory

    def observe(
        self,
        true_positions: list[tuple[float, float]],
        rng: Optional[random.Random] = None,
    ) -> list[tuple[float, float]]:
        """叠加高斯位置扰动，模拟视觉检测噪声。"""
        generator = rng if rng is not None else random.Random(self.config.seed)
        noise = self.config.position_noise
        return [
            (x + generator.gauss(0.0, noise), y + generator.gauss(0.0, noise))
            for x, y in true_positions
        ]

    def simulate(self, predict_step: Optional[int] = None) -> SimulationResult:
        """跑完整链路并返回结构化结果。

        predict_step 指定从第几个滤波状态发起预测；默认用最后一个状态。
        """
        true_trajectory = self.true_trajectory()
        rng = random.Random(self.config.seed)
        observed_trajectory = self.observe(true_trajectory, rng)

        kalman = KalmanFilter()
        states: list[CurlingState] = []
        timestamps: list[float] = []
        for index, (observed_x, observed_y) in enumerate(observed_trajectory):
            timestamp = index * self.config.dt
            state = kalman.update(observed_x, observed_y, timestamp)
            states.append(state)
            timestamps.append(timestamp)

        filtered_trajectory = [(state.x, state.y) for state in states]
        source_index = len(states) - 1 if predict_step is None else min(max(predict_step, 0), len(states) - 1)
        filtered_state = states[source_index]
        prediction = self.predictor.predict(filtered_state)

        return SimulationResult(
            true_trajectory=true_trajectory,
            observed_trajectory=observed_trajectory,
            filtered_trajectory=filtered_trajectory,
            predicted_endpoint=prediction.endpoint,
            prediction=prediction,
            filtered_state=filtered_state,
            timestamps=timestamps,
        )


def run_simulation(config: Optional[SimulationConfig] = None, predict_step: Optional[int] = None) -> SimulationResult:
    """便捷入口：用给定配置（默认场景）运行一次仿真。"""
    simulator = MotionSimulator(config) if config is not None else MotionSimulator.default()
    return simulator.simulate(predict_step=predict_step)
