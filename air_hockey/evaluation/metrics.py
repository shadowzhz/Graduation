"""算法评估指标。

输入一次仿真的 SimulationResult（内含 PredictionState），输出：
观测与滤波位置误差、滤波速度误差、oracle 终点误差及估计状态端到端终点误差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..prediction import PredictionState
from ..simulation import SimulationResult
from game_state import CurlingState


@dataclass
class ErrorStats:
    """一组误差的统计量。"""

    mean: float
    maximum: float
    minimum: float
    count: int
    rmse: float = 0.0

    @classmethod
    def from_errors(cls, errors: Iterable[float]) -> "ErrorStats":
        values = [float(error) for error in errors]
        if not values:
            return cls(0.0, 0.0, 0.0, 0, 0.0)
        count = len(values)
        mean = sum(values) / count
        rmse = math.sqrt(sum(value * value for value in values) / count)
        return cls(mean=mean, maximum=max(values), minimum=min(values), count=count, rmse=rmse)

    @classmethod
    def combine(cls, stats: Sequence["ErrorStats"]) -> "ErrorStats":
        """按样本数加权合并多组统计量（用于多组实验聚合）。"""
        present = [item for item in stats if item.count > 0]
        if not present:
            return cls(0.0, 0.0, 0.0, 0, 0.0)
        total = sum(item.count for item in present)
        mean = sum(item.mean * item.count for item in present) / total
        rmse = math.sqrt(sum(item.rmse**2 * item.count for item in present) / total)
        return cls(
            mean=mean,
            maximum=max(item.maximum for item in present),
            minimum=min(item.minimum for item in present),
            count=total,
            rmse=rmse,
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "max": self.maximum,
            "min": self.minimum,
            "rmse": self.rmse,
            "count": self.count,
        }


@dataclass
class EvaluationResult:
    """单次仿真的状态误差与预测终点误差。"""

    observation_error: ErrorStats
    filtered_error: ErrorStats
    estimated_endpoint_error: float
    source: EvaluationSource | None = None
    velocity_error: ErrorStats | None = None
    steady_state_filtered_error: ErrorStats | None = None
    oracle_endpoint_error: float | None = None
    prediction_frame: int | None = None
    true_prediction_state: CurlingState | None = None
    estimated_prediction_state: CurlingState | None = None
    motion_events: tuple[str, ...] = ()

    @property
    def endpoint_error(self) -> float:
        """Compatibility alias for end-to-end error from the estimated state."""
        return self.estimated_endpoint_error

    def to_dict(self) -> dict:
        return {
            "observation_position_error": self.observation_error.to_dict(),
            "filtered_position_error": self.filtered_error.to_dict(),
            "steady_state_filtered_position_error": (
                None if self.steady_state_filtered_error is None else self.steady_state_filtered_error.to_dict()
            ),
            "velocity_error": None if self.velocity_error is None else self.velocity_error.to_dict(),
            "oracle_endpoint_error": self.oracle_endpoint_error,
            "estimated_endpoint_error": self.estimated_endpoint_error,
            "prediction_frame": self.prediction_frame,
            "true_prediction_state": _state_to_dict(self.true_prediction_state),
            "estimated_prediction_state": _state_to_dict(self.estimated_prediction_state),
            "motion_events": list(self.motion_events),
            "source": None if self.source is None else self.source.to_dict(),
        }


def _state_to_dict(state: CurlingState | None) -> dict | None:
    if state is None:
        return None
    return {
        "x": state.x,
        "y": state.y,
        "vx": state.vx,
        "vy": state.vy,
        "timestamp": state.timestamp,
    }


@dataclass
class EvaluationSource:
    """评估来源的可追溯信息。"""

    initial_x: float
    initial_y: float
    initial_vx: float
    initial_vy: float
    dt: float
    steps: int
    position_noise: float
    seed: int | None

    def to_dict(self) -> dict:
        return {
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "initial_vx": self.initial_vx,
            "initial_vy": self.initial_vy,
            "dt": self.dt,
            "steps": self.steps,
            "position_noise": self.position_noise,
            "seed": self.seed,
        }


def position_errors(
    estimated: Sequence[tuple[float, float]],
    truth: Sequence[tuple[float, float]],
) -> list[float]:
    """逐点位置误差（欧氏距离）。"""
    return [
        math.hypot(est_x - true_x, est_y - true_y)
        for (est_x, est_y), (true_x, true_y) in zip(estimated, truth)
    ]


def velocity_errors(
    estimated: Sequence[tuple[float, float]],
    truth: Sequence[tuple[float, float]],
) -> list[float]:
    """逐帧二维速度向量误差（欧氏距离）。"""
    return [
        math.hypot(est_vx - true_vx, est_vy - true_vy)
        for (est_vx, est_vy), (true_vx, true_vy) in zip(estimated, truth)
    ]


def endpoint_error(prediction: PredictionState, true_endpoint: tuple[float, float]) -> float:
    """预测终点误差：PredictionState.endpoint 与真实终点的距离。"""
    predicted_x, predicted_y = prediction.endpoint
    return math.hypot(predicted_x - true_endpoint[0], predicted_y - true_endpoint[1])


def evaluate_result(
    result: SimulationResult,
    *,
    source: "EvaluationSource | None" = None,
    oracle_prediction: PredictionState | None = None,
    warmup_frames: int = 5,
) -> EvaluationResult:
    """评估逐帧状态误差，以及 oracle 和估计状态发起的终点误差。"""
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    observation_errors = position_errors(result.observed_trajectory, result.true_trajectory)
    filtered_errors = position_errors(result.filtered_trajectory, result.true_trajectory)
    true_velocities = result.true_velocity_trajectory or []
    filtered_velocities = result.filtered_velocity_trajectory or []
    frame = result.prediction_frame
    true_prediction_state = None
    estimated_prediction_state = result.filtered_state
    if frame is not None and frame < len(true_velocities):
        true_x, true_y = result.true_trajectory[frame]
        true_vx, true_vy = true_velocities[frame]
        true_prediction_state = CurlingState(
            x=true_x,
            y=true_y,
            vx=true_vx,
            vy=true_vy,
            timestamp=result.timestamps[frame],
        )
    estimate = endpoint_error(result.prediction, result.true_endpoint)
    oracle = None if oracle_prediction is None else endpoint_error(oracle_prediction, result.true_endpoint)

    return EvaluationResult(
        observation_error=ErrorStats.from_errors(observation_errors),
        filtered_error=ErrorStats.from_errors(filtered_errors),
        estimated_endpoint_error=estimate,
        source=source,
        velocity_error=ErrorStats.from_errors(velocity_errors(filtered_velocities, true_velocities)),
        steady_state_filtered_error=ErrorStats.from_errors(filtered_errors[warmup_frames:]),
        oracle_endpoint_error=oracle,
        prediction_frame=frame,
        true_prediction_state=true_prediction_state,
        estimated_prediction_state=estimated_prediction_state,
        motion_events=result.motion_events,
    )
