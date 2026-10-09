"""多组实验与实验报告。

批量运行无摄像头仿真评估，聚合统计并输出 JSON 与控制台摘要。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from ..simulation import MotionSimulator, SimulationResult
from game_state import CurlingState
from .metrics import ErrorStats, EvaluationResult, EvaluationSource, evaluate_result


@dataclass
class ExperimentConfig:
    """多组实验配置。"""

    runs: int = 5
    seed: int = 42
    prediction_fraction: float = 0.25
    predict_step: Optional[int] = None
    steps: int = 220
    dt: float = 1.0 / 60.0
    position_noise: float = 3.0
    random_initial: bool = True
    estimator_model: str = "constant_velocity"
    compare_estimators: bool = False
    warmup_frames: int = 5

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(self.prediction_fraction) or not 0.0 <= self.prediction_fraction <= 1.0:
            raise ValueError("prediction_fraction must be finite and between 0 and 1")
        if self.predict_step is not None and (isinstance(self.predict_step, bool) or not isinstance(self.predict_step, int)):
            raise ValueError("predict_step must be an integer or None")
        if isinstance(self.warmup_frames, bool) or not isinstance(self.warmup_frames, int) or self.warmup_frames < 0:
            raise ValueError("warmup_frames must be a non-negative integer")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.position_noise) or self.position_noise < 0.0:
            raise ValueError("position_noise must be finite and non-negative")
        if self.estimator_model not in ("constant_velocity", "friction", "collision_aware"):
            raise ValueError("estimator_model must be 'constant_velocity', 'friction' or 'collision_aware'")

    def frame_for(self, steps: int) -> int:
        if self.predict_step is not None:
            return steps if self.predict_step < 0 else min(self.predict_step, steps)
        return int(steps * self.prediction_fraction)

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "seed": self.seed,
            "prediction_fraction": self.prediction_fraction,
            "predict_step": self.predict_step,
            "steps": self.steps,
            "dt": self.dt,
            "position_noise": self.position_noise,
            "random_initial": self.random_initial,
            "warmup_frames": self.warmup_frames,
            "estimator_model": self.estimator_model,
            "compare_estimators": self.compare_estimators,
        }


@dataclass
class ExperimentReport:
    """一次多组实验的完整报告。"""

    config: ExperimentConfig
    runs: list[EvaluationResult] = field(default_factory=list)
    observation_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    filtered_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    steady_state_filtered_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    velocity_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    oracle_endpoint_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    estimated_endpoint_error: ErrorStats = field(default_factory=lambda: ErrorStats(0.0, 0.0, 0.0, 0, 0.0))
    prediction_frame: int = 0
    motion_event_errors: dict[str, ErrorStats] = field(default_factory=dict)
    estimator_comparison: dict[str, "ExperimentReport"] = field(default_factory=dict)

    @property
    def endpoint_error(self) -> ErrorStats:
        """Compatibility alias for end-to-end error from the estimated state."""
        return self.estimated_endpoint_error

    @property
    def mean_error(self) -> float:
        """Compatibility alias for estimated endpoint mean, not an overall metric."""
        return self.estimated_endpoint_error.mean

    @property
    def max_error(self) -> float:
        """Compatibility alias for estimated endpoint maximum, not an overall metric."""
        return self.estimated_endpoint_error.maximum

    def to_dict(self) -> dict:
        payload = {
            "config": self.config.to_dict(),
            "summary": {
                "observation_position_error": self.observation_error.to_dict(),
                "filtered_position_error": self.filtered_error.to_dict(),
                "steady_state_filtered_position_error": self.steady_state_filtered_error.to_dict(),
                "velocity_error": self.velocity_error.to_dict(),
                "oracle_endpoint_error": self.oracle_endpoint_error.to_dict(),
                "estimated_endpoint_error": self.estimated_endpoint_error.to_dict(),
                "prediction_frame": self.prediction_frame,
                "motion_event_estimated_endpoint_error": {
                    event: stats.to_dict() for event, stats in self.motion_event_errors.items()
                },
            },
            "runs": [result.to_dict() for result in self.runs],
        }
        if self.estimator_comparison:
            payload["estimator_comparison"] = {
                model: report.to_dict() for model, report in self.estimator_comparison.items()
            }
        return payload

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        """序列化为 JSON；给定 path 时同时写入文件。"""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def console_summary(self) -> str:
        """控制台摘要。"""
        config = self.config
        steps = self.runs[0].source.steps if self.runs and self.runs[0].source is not None else config.steps
        lines = [
            "=== 算法评估报告 ===",
            f"runs {len(self.runs)} | seed {config.seed} | prediction frame {self.prediction_frame}/{steps} "
            f"| fraction {config.prediction_fraction:.2f} | warmup {config.warmup_frames} frames "
            f"| noise {config.position_noise:.1f} | dt {config.dt:.4f}s",
        ]
        reports = [self, *self.estimator_comparison.values()]
        for report in reports:
            lines.append(f"=== Estimator: {report.config.estimator_model} ===")
            lines.extend(report._metric_lines())
        return "\n".join(lines)

    def _metric_lines(self) -> list[str]:
        config = self.config
        lines = [
            "=== Position observation (px) ===",
            f"Mean {self.observation_error.mean:.3f} | RMSE {self.observation_error.rmse:.3f} | Max {self.observation_error.maximum:.3f}",
            "=== Position after Kalman (px) ===",
            f"Mean {self.filtered_error.mean:.3f} | RMSE {self.filtered_error.rmse:.3f} | Max {self.filtered_error.maximum:.3f}",
            f"Post-warmup RMSE (frames >= {config.warmup_frames}): {self.steady_state_filtered_error.rmse:.3f} px",
            "=== Velocity after Kalman (px/s) ===",
            f"Mean {self.velocity_error.mean:.3f} | RMSE {self.velocity_error.rmse:.3f} | Max {self.velocity_error.maximum:.3f}",
            "=== Oracle endpoint prediction (px) ===",
            f"Mean {self.oracle_endpoint_error.mean:.3f} | RMSE {self.oracle_endpoint_error.rmse:.3f} | Max {self.oracle_endpoint_error.maximum:.3f}",
            "=== Estimated-state endpoint prediction (end-to-end, px) ===",
            f"Mean {self.estimated_endpoint_error.mean:.3f} | RMSE {self.estimated_endpoint_error.rmse:.3f} | Max {self.estimated_endpoint_error.maximum:.3f}",
            "=== Estimated endpoint error by truth event (px; categories may overlap) ===",
        ]
        lines.extend(
            f"{event}: n={stats.count} | mean={stats.mean:.3f} | RMSE={stats.rmse:.3f} | max={stats.maximum:.3f}"
            for event, stats in self.motion_event_errors.items()
        )
        return lines


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

    def _run_once(self, seed: int, models: tuple[str, ...]) -> dict[str, EvaluationResult]:
        simulator = self._factory(seed)
        prediction_frame = self.config.frame_for(simulator.config.steps)
        simulations: dict[str, SimulationResult] = simulator.simulate_estimators(
            models,
            predict_step=prediction_frame,
        )
        reference = simulations[models[0]]
        true_velocities = reference.true_velocity_trajectory
        if true_velocities is None or reference.prediction_frame is None:
            raise ValueError("simulation result must include true velocities and its prediction frame")
        index = reference.prediction_frame
        true_x, true_y = reference.true_trajectory[index]
        true_vx, true_vy = true_velocities[index]
        true_state = CurlingState(
            x=true_x,
            y=true_y,
            vx=true_vx,
            vy=true_vy,
            timestamp=reference.timestamps[index],
        )
        oracle_prediction = simulator.predictor.predict(true_state)
        source = self._source_of(simulator)
        return {
            model: evaluate_result(
                result,
                source=source,
                oracle_prediction=oracle_prediction,
                warmup_frames=self.config.warmup_frames,
            )
            for model, result in simulations.items()
        }

    def _report(self, config: ExperimentConfig, runs: list[EvaluationResult]) -> ExperimentReport:
        observation = ErrorStats.combine([result.observation_error for result in runs])
        filtered = ErrorStats.combine([result.filtered_error for result in runs])
        steady_state = ErrorStats.combine(
            [result.steady_state_filtered_error for result in runs if result.steady_state_filtered_error is not None]
        )
        velocity = ErrorStats.combine([result.velocity_error for result in runs if result.velocity_error is not None])
        oracle = ErrorStats.from_errors(
            [result.oracle_endpoint_error for result in runs if result.oracle_endpoint_error is not None]
        )
        estimated = ErrorStats.from_errors([result.estimated_endpoint_error for result in runs])
        event_names = ("no_bounce", "wall_bounce", "goal_post_bounce", "goal")
        event_errors = {
            event: ErrorStats.from_errors(
                [result.estimated_endpoint_error for result in runs if event in result.motion_events]
            )
            for event in event_names
        }
        return ExperimentReport(
            config=config,
            runs=runs,
            observation_error=observation,
            filtered_error=filtered,
            steady_state_filtered_error=steady_state,
            velocity_error=velocity,
            oracle_endpoint_error=oracle,
            estimated_endpoint_error=estimated,
            prediction_frame=runs[0].prediction_frame or 0,
            motion_event_errors=event_errors,
        )

    def run(self) -> ExperimentReport:
        """运行全部实验组并聚合统计。"""
        models = (
            ("constant_velocity", "friction", "collision_aware")
            if self.config.compare_estimators
            else (self.config.estimator_model,)
        )
        runs_by_model = {model: [] for model in models}
        for index in range(self.config.runs):
            for model, result in self._run_once(self.config.seed + index, models).items():
                runs_by_model[model].append(result)

        if self.config.compare_estimators:
            cv_config = replace(self.config, estimator_model="constant_velocity")
            cv_report = self._report(cv_config, runs_by_model["constant_velocity"])
            for model in models[1:]:
                model_config = replace(self.config, estimator_model=model, compare_estimators=False)
                cv_report.estimator_comparison[model] = self._report(model_config, runs_by_model[model])
            return cv_report
        return self._report(self.config, runs_by_model[self.config.estimator_model])


def run_experiment(
    config: Optional[ExperimentConfig] = None,
    simulator_factory: Optional[Callable[[int], MotionSimulator]] = None,
) -> ExperimentReport:
    """便捷入口：运行一次多组实验。"""
    return Experiment(config=config, simulator_factory=simulator_factory).run()
