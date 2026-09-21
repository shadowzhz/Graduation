"""运动轨迹仿真测试环境测试。

覆盖：
- 仿真配置生成（初始位置/速度/步长）与参数校验
- 视觉位置噪声模拟
- 完整链路：真实轨迹 -> 观测 -> KalmanFilter -> CurlingState -> Predictor -> PredictionState
- 输出结果：真实/观测/滤波轨迹与预测终点，滤波相对观测降低误差
"""

import math

from air_hockey import core_config as core
from air_hockey.estimation import KalmanFilter
from air_hockey.prediction import PredictionState
from air_hockey.simulation import (
    MotionSimulator,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from game_state import CurlingState


def test_config_generates_initial_conditions_and_validates():
    config = SimulationConfig(initial_x=10.0, initial_y=20.0, initial_vx=30.0, initial_vy=-40.0)
    assert (config.initial_x, config.initial_y) == (10.0, 20.0)
    assert (config.initial_vx, config.initial_vy) == (30.0, -40.0)
    assert config.dt > 0.0

    for kwargs in ({"dt": 0.0}, {"dt": -1.0}, {"steps": 0}, {"position_noise": -1.0}):
        base = dict(initial_x=0.0, initial_y=0.0, initial_vx=1.0, initial_vy=1.0)
        base.update(kwargs)
        try:
            SimulationConfig(**base)
            assert False, f"{kwargs} 应该抛 ValueError"
        except ValueError:
            pass


def test_true_trajectory_shape_and_start():
    simulator = MotionSimulator.default()
    trajectory = simulator.true_trajectory()
    assert len(trajectory) == simulator.config.steps + 1
    assert trajectory[0] == (simulator.config.initial_x, simulator.config.initial_y)
    # 直线上滑：x 不变，y 单调减小
    assert all(abs(point[0] - simulator.config.initial_x) < 1e-9 for point in trajectory)
    assert trajectory[-1][1] < trajectory[0][1]


def test_zero_noise_observation_matches_true():
    simulator = MotionSimulator.default(position_noise=0.0)
    result = simulator.simulate()
    assert result.observed_trajectory == result.true_trajectory


def test_observation_adds_position_noise():
    simulator = MotionSimulator.default(position_noise=5.0)
    truth = simulator.true_trajectory()
    observed = simulator.observe(truth)
    assert len(observed) == len(truth)
    assert observed != truth
    # 扰动幅度与噪声量级一致
    max_offset = max(math.hypot(ox - tx, oy - ty) for (ox, oy), (tx, ty) in zip(observed, truth))
    assert 0.0 < max_offset < 5.0 * 5.0


def test_chain_lengths_and_timestamps_aligned():
    result = MotionSimulator.default().simulate()
    size = len(result.true_trajectory)
    assert len(result.observed_trajectory) == size
    assert len(result.filtered_trajectory) == size
    assert len(result.timestamps) == size
    assert result.timestamps[0] == 0.0
    assert all(b > a for a, b in zip(result.timestamps, result.timestamps[1:]))


def test_filtering_reduces_position_error():
    result = MotionSimulator.default().simulate()
    assert result.mean_filtered_error() < result.mean_observation_error()


def test_prediction_recovers_true_endpoint_from_final_state():
    result = MotionSimulator.default().simulate()
    # 冰壶已停稳，从最终滤波状态预测应回到真实终点附近
    assert result.endpoint_error() < 5.0
    assert result.predicted_endpoint[1] < result.true_trajectory[0][1]


def test_prediction_recovers_true_endpoint_without_noise():
    result = MotionSimulator.default(position_noise=0.0).simulate()
    assert result.endpoint_error() < 1.0


def test_prediction_state_and_source_state():
    result = MotionSimulator.default().simulate(predict_step=10)
    pred = result.prediction
    assert isinstance(pred, PredictionState)
    assert pred.endpoint == result.predicted_endpoint
    assert pred.source_state is result.filtered_state
    assert isinstance(result.filtered_state, CurlingState)
    # 早期预测仍应是有限的轨迹
    assert len(pred.trajectory) >= 1
    assert all(math.isfinite(value) for value in result.predicted_endpoint)


def test_simulation_is_deterministic_with_seed():
    first = MotionSimulator.default(seed=0).simulate()
    second = MotionSimulator.default(seed=0).simulate()
    assert first.filtered_trajectory == second.filtered_trajectory
    assert first.observed_trajectory == second.observed_trajectory
    assert first.predicted_endpoint == second.predicted_endpoint

    other = MotionSimulator.default(seed=1).simulate()
    assert other.observed_trajectory != first.observed_trajectory


def test_random_generator_respects_floor_bounds():
    simulator = MotionSimulator.random(seed=3)
    config = simulator.config
    assert core.RINK_LEFT + core.STONE_RADIUS <= config.initial_x <= core.RINK_RIGHT - core.STONE_RADIUS
    assert core.RINK_CENTER_Y <= config.initial_y <= core.RINK_BOTTOM - core.STONE_RADIUS
    speed = math.hypot(config.initial_vx, config.initial_vy)
    assert 0.0 < speed <= core.MAX_STONE_SPEED
    assert config.initial_vy < 0.0


def test_run_simulation_helper_and_summary():
    result = run_simulation()
    assert isinstance(result, SimulationResult)
    text = result.summary()
    assert "true_endpoint" in text
    assert "predicted_endpoint" in text
    assert "endpoint_error" in text


def test_simulator_uses_injected_kalman_compatible_predictor():
    # 仿真链路复用共享的 KalmanFilter 与 TrajectoryPredictor，接口无需改动
    simulator = MotionSimulator.default()
    result = simulator.simulate()
    # 滤波轨迹条数应与 KalmanFilter 逐帧输出一致
    kalman = KalmanFilter()
    expected = []
    for timestamp, (ox, oy) in zip(result.timestamps, result.observed_trajectory):
        expected.append(kalman.update(ox, oy, timestamp))
    assert [(state.x, state.y) for state in expected] == result.filtered_trajectory
