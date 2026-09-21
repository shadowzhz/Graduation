"""算法评估指标。

输入一次仿真的 SimulationResult（内含 PredictionState），输出：
观测位置误差、Kalman 滤波误差、预测终点误差，以及各自的平均/最大误差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..prediction import PredictionState
from ..simulation import SimulationResult


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
    """单次仿真的评估结果。"""

    observation_error: ErrorStats
    filtered_error: ErrorStats
    endpoint_error: float
    source: EvaluationSource | None = None

    def to_dict(self) -> dict:
        return {
            "observation_error": self.observation_error.to_dict(),
            "filtered_error": self.filtered_error.to_dict(),
            "endpoint_error": self.endpoint_error,
            "source": None if self.source is None else self.source.to_dict(),
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


def endpoint_error(prediction: PredictionState, true_endpoint: tuple[float, float]) -> float:
    """预测终点误差：PredictionState.endpoint 与真实终点的距离。"""
    predicted_x, predicted_y = prediction.endpoint
    return math.hypot(predicted_x - true_endpoint[0], predicted_y - true_endpoint[1])


def evaluate_result(result: SimulationResult, *, source: "EvaluationSource | None" = None) -> EvaluationResult:
    """评估一次仿真：观测误差、滤波误差、预测终点误差。"""
    observation_errors = position_errors(result.observed_trajectory, result.true_trajectory)
    filtered_errors = position_errors(result.filtered_trajectory, result.true_trajectory)
    conclusion = endpoint_error(result.prediction, result.true_endpoint)

    return EvaluationResult(
        observation_error=ErrorStats.from_errors(observation_errors),
        filtered_error=ErrorStats.from_errors(filtered_errors),
        endpoint_error=conclusion,
        source=source,
    )
