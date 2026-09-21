"""多组实验与实验报告。

批量运行无摄像头仿真评估，聚合统计并输出 JSON 与控制台摘要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..simulation import MotionSimulator, SimulationResult
from .metrics import ErrorStats, EvaluationResult, EvaluationSource, evaluate_result


@dataclass
class ExperimentConfig:
    """多组实验配置。"""

    runs: int = 5
    seed: int = 0
    predict_step: Optional[int] = 20
    steps: int = 220
    dt: float = 1.0 / 60.0
    position_noise: float = 3.0
    random_initial: bool = True

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.position_noise < 0.0:
            raise ValueError("position_noise must be non-negative")

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "seed": self.seed,
            "predict_step": self.predict_step,
            "steps": self.steps,
            "dt": self.dt,
            "position_noise": self.position_noise,
            "random_initial": self.random_initial,
        }


@dataclass
class ExperimentReport:
    """一次多组实验的完整报告。"""

    config: ExperimentConfig
    runs: list[EvaluationResult] = field(default_factory=list)
    observation_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    filtered_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    endpoint_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    mean_error: float = 0.0
    max_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "summary": {
                "observation_error": self.observation_error.to_dict(),
                "filtered_error": self.filtered_error.to_dict(),
                "endpoint_error": self.endpoint_error.to_dict(),
                "mean_error": self.mean_error,
                "max_error": self.max_error,
            },
            "runs": [result.to_dict() for result in self.runs],
        }

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        """序列化为 JSON；给定 path 时同时写入文件。"""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def console_summary(self) -> str:
        """控制台摘要。"""
        config = self.config
        lines = [
            "=== 算法评估报告 ===",
            f"实验组数 {len(self.runs)} | seed {config.seed} | predict_step {config.predict_step} "
            f"| noise {config.position_noise:.1f} | dt {config.dt:.4f}s",
            f"观测位置误差: 平均 {self.observation_error.mean:.3f} | 最大 {self.observation_error.maximum:.3f} "
            f"| RMSE {self.observation_error.rmse:.3f}",
            f"Kalman滤波误差: 平均 {self.filtered_error.mean:.3f} | 最大 {self.filtered_error.maximum:.3f} "
            f"| RMSE {self.filtered_error.rmse:.3f}",
            f"预测终点误差: 平均 {self.endpoint_error.mean:.3f} | 最大 {self.endpoint_error.maximum:.3f}",
            f"平均误差 {self.mean_error:.3f} | 最大误差 {self.max_error:.3f}",
        ]
        return "\n".join(lines)


class Experiment:
    """多组实验运行器。"""

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        simulator_factory: Optional[Callable[[int], MotionSimulator]] = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self._factory = simulator_factory or self._default_factory

    def _default_factory(self, seed: int) -> MotionSimulator:
        overrides = {
            "steps": self.config.steps,
            "dt": self.config.dt,
            "position_noise": self.config.position_noise,
        }
        if self.config.random_initial:
            return MotionSimulator.random(seed=seed, **overrides)
        return MotionSimulator.default(seed=seed, **overrides)

    def _source_of(self, simulator: MotionSimulator) -> EvaluationSource:
        config = simulator.config
        return EvaluationSource(
            initial_x=config.initial_x,
            initial_y=config.initial_y,
            initial_vx=config.initial_vx,
            initial_vy=config.initial_vy,
            dt=config.dt,
            steps=config.steps,
            position_noise=config.position_noise,
            seed=config.seed,
        )

    def _run_once(self, seed: int) -> EvaluationResult:
        simulator = self._factory(seed)
        result: SimulationResult = simulator.simulate(predict_step=self.config.predict_step)
        return evaluate_result(result, source=self._source_of(simulator))

    def run(self) -> ExperimentReport:
        """运行全部实验组并聚合统计。"""
        runs = [self._run_once(self.config.seed + index) for index in range(self.config.runs)]

        observation = ErrorStats.combine([result.observation_error for result in runs])
        filtered = ErrorStats.combine([result.filtered_error for result in runs])
        endpoint = ErrorStats.from_errors([result.endpoint_error for result in runs])
        overall = ErrorStats.combine([observation, filtered, endpoint])

        return ExperimentReport(
            config=self.config,
            runs=runs,
            observation_error=observation,
            filtered_error=filtered,
            endpoint_error=endpoint,
            mean_error=overall.mean,
            max_error=overall.maximum,
        )


def run_experiment(
    config: Optional[ExperimentConfig] = None,
    simulator_factory: Optional[Callable[[int], MotionSimulator]] = None,
) -> ExperimentReport:
    """便捷入口：运行一次多组实验。"""
    return Experiment(config=config, simulator_factory=simulator_factory).run()
