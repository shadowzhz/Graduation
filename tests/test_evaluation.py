"""算法评估模块测试。

覆盖：
- ErrorStats 统计与多组聚合
- 观测/滤波/终点误差计算
- 单次仿真评估（手工构造 SimulationResult 校验数值）
- 多组实验、JSON 报告与控制台摘要
"""

import json
import math
import tempfile
from pathlib import Path

from air_hockey.evaluation import (
    ErrorStats,
    EvaluationResult,
    Experiment,
    ExperimentConfig,
    endpoint_error,
    evaluate_result,
    position_errors,
    run_experiment,
    velocity_errors,
)
from air_hockey.prediction import PredictionState
from air_hockey.simulation import MotionSimulator, SimulationConfig, SimulationResult
from game_state import CurlingState


def _make_result():
    true_trajectory = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
    observed_trajectory = [(1.0, 0.0), (3.0, 5.0), (6.0, 8.0)]
    filtered_trajectory = [(0.5, 0.0), (3.0, 4.5), (6.0, 8.0)]
    prediction = PredictionState(
        trajectory=[(6.0, 8.0), (5.0, 7.0)],
        endpoint=(5.0, 7.0),
        duration=0.1,
        source_state=CurlingState(x=6.0, y=8.0),
    )

    return SimulationResult(
        true_trajectory=true_trajectory,
        observed_trajectory=observed_trajectory,
        filtered_trajectory=filtered_trajectory,
        predicted_endpoint=(5.0, 7.0),
        prediction=prediction,
        filtered_state=CurlingState(x=6.0, y=8.0),
        timestamps=[0.0, 1.0, 2.0],
    )


def test_error_stats_from_errors():
    stats = ErrorStats.from_errors([1.0, 2.0, 3.0, 4.0])
    assert stats.count == 4
    assert stats.mean == 2.5
    assert stats.maximum == 4.0
    assert stats.minimum == 1.0
    assert abs(stats.rmse - math.sqrt((1 + 4 + 9 + 16) / 4)) < 1e-9


def test_error_stats_from_empty_errors():
    stats = ErrorStats.from_errors([])
    assert stats.count == 0
    assert stats.mean == 0.0 and stats.maximum == 0.0


def test_error_stats_combine_is_weighted():
    first = ErrorStats.from_errors([1.0, 1.0])
    second = ErrorStats.from_errors([3.0, 3.0, 3.0, 3.0])
    combined = ErrorStats.combine([first, second])
    assert combined.count == 6
    assert abs(combined.mean - (1.0 * 2 + 3.0 * 4) / 6) < 1e-9
    assert combined.maximum == 3.0
    assert combined.minimum == 1.0
    assert abs(combined.rmse - math.sqrt((2.0 * 1.0 + 4.0 * 9.0) / 6.0)) < 1e-9


def test_position_errors_and_endpoint_error_helpers():
    errors = position_errors([(0.0, 0.0), (3.0, 4.0)], [(0.0, 0.0), (0.0, 0.0)])
    assert errors == [0.0, 5.0]
    assert velocity_errors([(3.0, 4.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]) == [5.0, 0.0]

    prediction = PredictionState(
        trajectory=[(3.0, 4.0)],
        endpoint=(3.0, 4.0),
        duration=0.0,
        source_state=CurlingState(x=0.0, y=0.0),
    )
    assert endpoint_error(prediction, (0.0, 0.0)) == 5.0


def test_evaluate_result_metrics_match_manual_calculation():
    result = evaluate_result(_make_result())
    assert isinstance(result, EvaluationResult)

    assert abs(result.observation_error.mean - (1.0 + 1.0 + 0.0) / 3) < 1e-9
    assert result.observation_error.maximum == 1.0

    assert abs(result.filtered_error.mean - (0.5 + 0.5 + 0.0) / 3) < 1e-9
    assert result.filtered_error.maximum == 0.5

    assert abs(result.endpoint_error - math.hypot(1.0, 1.0)) < 1e-9
    assert result.estimated_endpoint_error == result.endpoint_error
    assert result.oracle_endpoint_error is None


def test_evaluate_result_does_not_mutate_input():
    result = _make_result()
    snapshot = list(result.true_trajectory)
    evaluate_result(result)
    assert result.true_trajectory == snapshot


def test_experiment_config_validation():
    invalid = [
        {"runs": 0}, {"runs": -1}, {"steps": 0}, {"dt": 0.0},
        {"position_noise": -1.0}, {"prediction_fraction": -0.1},
        {"prediction_fraction": 1.1}, {"warmup_frames": -1}, {"estimator_model": "unknown"},
    ]
    invalid.extend({name: value} for name in ("dt", "position_noise") for value in (math.nan, math.inf, -math.inf))
    invalid.extend({"prediction_fraction": value} for value in (math.nan, math.inf, -math.inf))
    for kwargs in invalid:
        try:
            ExperimentConfig(**kwargs)
            assert False, f"{kwargs} 应该抛 ValueError"
        except ValueError:
            pass


def test_experiment_supports_multiple_runs_and_aggregates():
    config = ExperimentConfig(runs=3, seed=0, predict_step=20, steps=120)
    report = Experiment(config).run()

    assert len(report.runs) == 3
    assert report.observation_error.count == 3 * (config.steps + 1)
    assert report.filtered_error.count == 3 * (config.steps + 1)
    assert report.endpoint_error.count == 3
    assert report.estimated_endpoint_error.count == 3
    assert report.oracle_endpoint_error.count == 3
    assert report.velocity_error.count == 3 * (config.steps + 1)
    assert report.steady_state_filtered_error.count == 3 * (config.steps + 1 - config.warmup_frames)
    # 滤波相对观测显著降噪
    assert report.filtered_error.mean < report.observation_error.mean
    # 每组都带可追溯来源
    assert all(result.source is not None for result in report.runs)


def test_experiment_report_to_json_and_console_summary():
    report = run_experiment(ExperimentConfig(runs=2, seed=1, predict_step=10, steps=120))

    payload = json.loads(report.to_json())
    assert set(payload.keys()) == {"config", "summary", "runs"}
    summary = payload["summary"]
    for key in (
        "observation_position_error",
        "filtered_position_error",
        "steady_state_filtered_position_error",
        "velocity_error",
        "oracle_endpoint_error",
        "estimated_endpoint_error",
        "prediction_frame",
        "motion_event_estimated_endpoint_error",
    ):
        assert key in summary
    assert "mean_error" not in summary and "max_error" not in summary
    assert len(payload["runs"]) == 2
    assert payload["config"]["runs"] == 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        report.to_json(path)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["summary"]["estimated_endpoint_error"] == payload["summary"]["estimated_endpoint_error"]

    text = report.console_summary()
    assert "Position observation" in text
    assert "Position after Kalman" in text
    assert "Velocity after Kalman" in text
    assert "Oracle endpoint prediction" in text
    assert "Estimated-state endpoint prediction" in text
    assert "Overall error" not in text


def test_prediction_fraction_sets_an_explicit_frame_and_source_state():
    report = Experiment(ExperimentConfig(runs=1, steps=40, prediction_fraction=0.25, position_noise=0.0)).run()
    result = report.runs[0]

    assert report.prediction_frame == 10
    assert result.prediction_frame == 10
    assert result.true_prediction_state.timestamp == result.estimated_prediction_state.timestamp == 10 / 60
    assert result.to_dict()["true_prediction_state"]["timestamp"] == 10 / 60
    assert result.to_dict()["estimated_prediction_state"]["timestamp"] == 10 / 60


def test_evaluation_is_deterministic_for_same_seed():
    config = ExperimentConfig(runs=4, seed=123, position_noise=3.0)
    first = Experiment(config).run()
    second = Experiment(config).run()
    different = Experiment(ExperimentConfig(runs=4, seed=124, position_noise=3.0)).run()

    assert first.to_json() == second.to_json()
    assert first.runs[0].source.initial_x != different.runs[0].source.initial_x


def test_oracle_metrics_are_identical_between_estimators():
    report = Experiment(
        ExperimentConfig(runs=2, seed=42, steps=80, position_noise=3.0, compare_estimators=True)
    ).run()
    friction = report.estimator_comparison["friction"]

    assert report.config.estimator_model == "constant_velocity"
    assert friction.config.estimator_model == "friction"
    assert report.observation_error == friction.observation_error
    assert report.oracle_endpoint_error == friction.oracle_endpoint_error
    for cv_run, friction_run in zip(report.runs, friction.runs):
        assert cv_run.source == friction_run.source
        assert cv_run.oracle_endpoint_error == friction_run.oracle_endpoint_error
        assert cv_run.true_prediction_state == friction_run.true_prediction_state
        assert cv_run.prediction_frame == friction_run.prediction_frame

    payload = json.loads(report.to_json())
    assert payload["estimator_comparison"]["friction"]["summary"]["oracle_endpoint_error"] == payload["summary"]["oracle_endpoint_error"]
    assert "Estimator: constant_velocity" in report.console_summary()
    assert "Estimator: friction" in report.console_summary()
    collision = report.estimator_comparison["collision_aware"]
    assert report.oracle_endpoint_error == collision.oracle_endpoint_error
    assert report.observation_error == collision.observation_error
    for cv_run, collision_run in zip(report.runs, collision.runs):
        assert cv_run.source == collision_run.source
        assert cv_run.true_prediction_state == collision_run.true_prediction_state
        assert cv_run.prediction_frame == collision_run.prediction_frame
        assert cv_run.oracle_endpoint_error == collision_run.oracle_endpoint_error
    assert "Estimator: collision_aware" in report.console_summary()


def test_experiment_uses_injected_simulator_factory():
    seen_seeds = []

    def factory(seed):
        seen_seeds.append(seed)
        return MotionSimulator.default(seed=seed, steps=80)

    report = Experiment(ExperimentConfig(runs=3, seed=5, predict_step=5), simulator_factory=factory).run()
    assert seen_seeds == [5, 6, 7]
    assert len(report.runs) == 3


def test_collision_model_improves_wall_errors_without_changing_no_bounce():
    for noise in (0.0, 3.0):
        report = Experiment(
            ExperimentConfig(runs=20, seed=42, position_noise=noise, compare_estimators=True)
        ).run()
        friction = report.estimator_comparison["friction"]
        collision = report.estimator_comparison["collision_aware"]
        assert collision.motion_event_errors["no_bounce"] == friction.motion_event_errors["no_bounce"]
        assert collision.motion_event_errors["wall_bounce"].mean < friction.motion_event_errors["wall_bounce"].mean
        assert collision.motion_event_errors["wall_bounce"].maximum < friction.motion_event_errors["wall_bounce"].maximum
        assert collision.oracle_endpoint_error == friction.oracle_endpoint_error == report.oracle_endpoint_error


def _noise_free_model_comparison():
    initial = (369.7134, 388.0034, -83.2889, -163.4884)

    def factory(seed):
        return MotionSimulator(
            SimulationConfig(*initial, dt=1.0 / 60.0, steps=220, position_noise=0.0, seed=seed)
        )

    return Experiment(
        ExperimentConfig(
            runs=1,
            seed=42,
            prediction_fraction=0.25,
            steps=220,
            dt=1.0 / 60.0,
            position_noise=0.0,
            random_initial=False,
            compare_estimators=True,
        ),
        simulator_factory=factory,
    ).run()


def test_friction_model_reduces_velocity_error_on_noise_free_decelerating_motion():
    report = _noise_free_model_comparison()
    friction = report.estimator_comparison["friction"]

    assert report.runs[0].motion_events == ("no_bounce",)
    assert friction.velocity_error.rmse < report.velocity_error.rmse


def test_friction_model_reduces_endpoint_error_on_noise_free_decelerating_motion():
    report = _noise_free_model_comparison()
    friction = report.estimator_comparison["friction"]

    assert friction.estimated_endpoint_error.mean < report.estimated_endpoint_error.mean
