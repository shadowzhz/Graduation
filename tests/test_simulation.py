"""运动轨迹仿真测试环境测试。

覆盖：
- 仿真配置生成（初始位置/速度/步长）与参数校验
- 视觉位置噪声模拟
- 完整链路：真实轨迹 -> 观测 -> KalmanFilter -> CurlingState -> Predictor -> PredictionState
- 输出结果：真实/观测/滤波轨迹与预测终点，滤波相对观测降低误差
"""

import math

from air_hockey import core_config as core
from air_hockey.evaluation import Experiment, ExperimentConfig
from air_hockey.estimation import KalmanFilter
from air_hockey.physics import StoneMotion, goal_scorer
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

    invalid = [{"dt": 0.0}, {"dt": -1.0}, {"steps": 0}, {"position_noise": -1.0}]
    invalid.extend({name: value} for name in ("dt", "position_noise") for value in (math.nan, math.inf, -math.inf))
    for kwargs in invalid:
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


def test_motion_simulator_true_trajectory_matches_repeated_stone_steps():
    config = SimulationConfig(
        initial_x=300.0, initial_y=400.0, initial_vx=180.0, initial_vy=-120.0,
        dt=1 / 60, steps=50, position_noise=0.0,
    )
    trajectory = MotionSimulator(config).true_trajectory()
    stone = StoneMotion(x=config.initial_x, y=config.initial_y,
                        vx=config.initial_vx, vy=config.initial_vy)
    stone.set_immediate_velocity(stone.vx, stone.vy)
    expected = [(stone.x, stone.y)]
    scored = goal_scorer(stone) is not None
    for _ in range(config.steps):
        if not scored:
            stone.step(config.dt)
            scored = goal_scorer(stone) is not None
        expected.append((stone.x, stone.y))

    assert trajectory == expected


def test_zero_noise_observation_matches_true():
    simulator = MotionSimulator.default(position_noise=0.0)
    result = simulator.simulate()
    assert result.observed_trajectory == result.true_trajectory


def test_physical_endpoint_continues_beyond_recorded_horizon():
    simulator = MotionSimulator(
        SimulationConfig(100.0, 400.0, 0.0, -150.0, steps=20, position_noise=0.0)
    )
    result = simulator.simulate(predict_step=0)

    assert result.true_trajectory[-1][1] > result.true_endpoint[1]
    assert result.true_velocity_trajectory[-1][1] < -core.STONE_STOP_SPEED


def test_scored_true_trajectory_holds_position_and_timestamps():
    config = SimulationConfig(300.0, 80.0, 0.0, -150.0, dt=1 / 60, steps=120, position_noise=0.0)
    result = MotionSimulator(config).simulate()
    trajectory = result.true_trajectory
    scored_index = next(index for index, (_, y) in enumerate(trajectory) if y + core.STONE_RADIUS < core.RINK_TOP)
    assert scored_index < config.steps
    assert all(point == trajectory[scored_index] for point in trajectory[scored_index:])
    assert len(trajectory) == config.steps + 1
    assert result.timestamps == [index * config.dt for index in range(config.steps + 1)]


def test_true_trajectory_deflects_grazing_hits_and_rebounds_head_on():
    trajectory = MotionSimulator(SimulationConfig(223.0, 80.0, 0.0, -150.0, steps=120)).true_trajectory()
    assert len(trajectory) == 121
    assert max(x for x, _ in trajectory) > 240.0
    head_on = MotionSimulator(SimulationConfig(core.GOAL_LEFT, 80.0, 0.0, -150.0, steps=120)).true_trajectory()
    contact_y = core.RINK_TOP + core.STONE_RADIUS + core.GOAL_POST_RADIUS
    assert abs(min(y for _, y in head_on) - contact_y) < 1e-9
    assert head_on[-1][1] > 80.0
    assert all(y + core.STONE_RADIUS >= core.RINK_TOP for _, y in head_on)


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
    assert len(result.true_velocity_trajectory) == size
    assert len(result.filtered_velocity_trajectory) == size


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
    assert result.prediction_frame == 10
    assert (pred.source_state.x, pred.source_state.y) == result.filtered_trajectory[10]
    assert (pred.source_state.vx, pred.source_state.vy) == result.filtered_velocity_trajectory[10]
    assert pred.source_state.timestamp == result.timestamps[10]
    assert isinstance(result.filtered_state, CurlingState)
    # 早期预测仍应是有限的轨迹
    assert len(pred.trajectory) >= 1
    assert all(math.isfinite(value) for value in result.predicted_endpoint)


def test_zero_noise_diagnostic_frames_keep_position_and_velocity_states_aligned():
    result = MotionSimulator.random(seed=42, steps=220, position_noise=0.0).simulate(predict_step=20)

    assert result.observed_trajectory == result.true_trajectory
    for index in (0, 1, 2, 5, 10, 20, 220):
        assert math.isfinite(result.true_velocity_trajectory[index][0])
        assert math.isfinite(result.filtered_velocity_trajectory[index][1])
        assert result.timestamps[index] == index * result.timestamps[1]


def test_simulation_is_deterministic_with_seed():
    first = MotionSimulator.default(seed=0).simulate()
    second = MotionSimulator.default(seed=0).simulate()
    assert first.filtered_trajectory == second.filtered_trajectory
    assert first.observed_trajectory == second.observed_trajectory
    assert first.predicted_endpoint == second.predicted_endpoint

    other = MotionSimulator.default(seed=1).simulate()
    assert other.observed_trajectory != first.observed_trajectory


def test_friction_model_uses_same_observations_as_cv():
    simulator = MotionSimulator.random(seed=42, steps=80, position_noise=3.0)
    results = simulator.simulate_estimators(("constant_velocity", "friction", "collision_aware"), predict_step=20)
    cv = results["constant_velocity"]
    friction = results["friction"]


    assert cv.observed_trajectory == friction.observed_trajectory
    assert cv.true_trajectory == friction.true_trajectory
    assert cv.true_velocity_trajectory == friction.true_velocity_trajectory
    assert cv.timestamps == friction.timestamps
    assert cv.physical_endpoint == friction.physical_endpoint
    collision = results["collision_aware"]
    assert cv.observed_trajectory == collision.observed_trajectory
    assert cv.true_trajectory == collision.true_trajectory
    assert cv.true_velocity_trajectory == collision.true_velocity_trajectory
    assert cv.timestamps == collision.timestamps
    assert cv.physical_endpoint == collision.physical_endpoint


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


def _oracle_report(initial_state, *, steps=700):
    dt = core.PREDICTION_SUBSTEP
    simulation_config = SimulationConfig(
        *initial_state,
        dt=dt,
        steps=steps,
        position_noise=0.0,
        seed=19,
    )
    experiment_config = ExperimentConfig(
        runs=1,
        seed=19,
        prediction_fraction=0.0,
        steps=steps,
        dt=dt,
        position_noise=0.0,
        random_initial=False,
    )
    return Experiment(
        experiment_config,
        simulator_factory=lambda _seed: MotionSimulator(simulation_config),
    ).run()


def test_oracle_endpoint_error_is_near_zero_without_noise():
    report = _oracle_report((300.0, 400.0, 40.0, -100.0))
    result = report.runs[0]

    assert result.motion_events == ("no_bounce",)
    assert report.motion_event_errors["no_bounce"].count == 1
    assert result.oracle_endpoint_error < 0.1
    assert result.estimated_endpoint_error > result.oracle_endpoint_error


def test_oracle_prediction_matches_wall_bounce_truth():
    report = _oracle_report((core.RINK_RIGHT - core.STONE_RADIUS - 1.0, core.RINK_CENTER_Y, 120.0, 0.0))
    result = report.runs[0]

    assert "wall_bounce" in result.motion_events
    assert report.motion_event_errors["wall_bounce"].count == 1
    assert result.oracle_endpoint_error < 0.1


def test_oracle_prediction_matches_goal_post_truth():
    report = _oracle_report((core.GOAL_LEFT + core.STONE_RADIUS + 1.0, 80.0, 0.0, -150.0))
    result = report.runs[0]

    assert "goal_post_bounce" in result.motion_events
    assert report.motion_event_errors["goal_post_bounce"].count == 1
    assert result.oracle_endpoint_error < 0.1


def test_scored_endpoint_uses_first_held_goal_position():
    report = _oracle_report((core.RINK_CENTER_X, 80.0, 0.0, -150.0))
    result = report.runs[0]
    simulation = MotionSimulator(
        SimulationConfig(
            core.RINK_CENTER_X,
            80.0,
            0.0,
            -150.0,
            dt=core.PREDICTION_SUBSTEP,
            steps=700,
            position_noise=0.0,
        )
    ).simulate(predict_step=0)
    score_frame = next(
        index
        for index, (_, y) in enumerate(simulation.true_trajectory)
        if y + core.STONE_RADIUS < core.RINK_TOP
    )

    assert "goal" in result.motion_events
    assert report.motion_event_errors["goal"].count == 1
    assert result.oracle_endpoint_error < 0.1
    assert simulation.true_endpoint == simulation.true_trajectory[score_frame]
    assert all(point == simulation.true_endpoint for point in simulation.true_trajectory[score_frame:])
