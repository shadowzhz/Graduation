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
)
from air_hockey.prediction import PredictionState
from air_hockey.simulation import MotionSimulator, SimulationResult
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


def test_position_errors_and_endpoint_error_helpers():
    errors = position_errors([(0.0, 0.0), (3.0, 4.0)], [(0.0, 0.0), (0.0, 0.0)])
    assert errors == [0.0, 5.0]

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


def test_evaluate_result_does_not_mutate_input():
    result = _make_result()
    snapshot = list(result.true_trajectory)
    evaluate_result(result)
    assert result.true_trajectory == snapshot


def test_experiment_config_validation():
    for kwargs in ({"runs": 0}, {"runs": -1}, {"steps": 0}, {"dt": 0.0}, {"position_noise": -1.0}):
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
    # 滤波相对观测显著降噪
    assert report.filtered_error.mean < report.observation_error.mean
    assert report.mean_error > 0.0
    assert report.max_error >= report.endpoint_error.maximum - 1e-9
    # 每组都带可追溯来源
    assert all(result.source is not None for result in report.runs)


def test_experiment_report_to_json_and_console_summary():
    report = run_experiment(ExperimentConfig(runs=2, seed=1, predict_step=10, steps=120))

    payload = json.loads(report.to_json())
    assert set(payload.keys()) == {"config", "summary", "runs"}
    summary = payload["summary"]
    for key in ("observation_error", "filtered_error", "endpoint_error", "mean_error", "max_error"):
        assert key in summary
    assert len(payload["runs"]) == 2
    assert payload["config"]["runs"] == 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        report.to_json(path)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["summary"]["mean_error"] == payload["summary"]["mean_error"]

    text = report.console_summary()
    assert "观测位置误差" in text
    assert "Kalman滤波误差" in text
    assert "预测终点误差" in text
    assert "平均误差" in text and "最大误差" in text


def test_experiment_uses_injected_simulator_factory():
    seen_seeds = []

    def factory(seed):
        seen_seeds.append(seed)
        return MotionSimulator.default(seed=seed, steps=80)

    report = Experiment(ExperimentConfig(runs=3, seed=5, predict_step=5), simulator_factory=factory).run()
    assert seen_seeds == [5, 6, 7]
    assert len(report.runs) == 3
