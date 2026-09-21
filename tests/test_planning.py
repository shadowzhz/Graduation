"""轨迹规划层测试。

覆盖：
- ControlCommand 字段与序列化
- plan() 的方向、限幅、速度与预测时长的关系
- 目标点收敛到场内
- 参数校验与输入不可变性
- 与真实 Predictor 输出的联通
"""

import math

from air_hockey import core_config as core
from air_hockey.planning import ControlCommand, TrajectoryPlanner
from air_hockey.prediction import PredictionState, TrajectoryPredictor
from game_state import CurlingState


def _prediction(duration, endpoint=(0.0, 0.0)):
    return PredictionState(
        trajectory=[(300.0, 300.0), tuple(endpoint)],
        endpoint=tuple(endpoint),
        duration=duration,
        source_state=CurlingState(x=300.0, y=300.0),
    )


def test_control_command_fields_and_serialization():
    command = ControlCommand(target_position=(1.0, 2.0), direction=(0.6, 0.8), speed=10.0)
    assert command.target_position == (1.0, 2.0)
    assert command.direction == (0.6, 0.8)
    assert command.speed == 10.0
    assert abs(command.angle_degrees - math.degrees(math.atan2(0.8, 0.6))) < 1e-9

    payload = command.to_dict()
    assert set(payload.keys()) == {"target_position", "direction", "speed"}
    assert payload["target_position"] == [1.0, 2.0]
    assert payload["direction"] == [0.6, 0.8]
    assert payload["speed"] == 10.0


def test_plan_direction_points_from_state_to_target():
    curling = CurlingState(x=300.0, y=300.0, vx=10.0, vy=0.0)
    planner = TrajectoryPlanner(max_speed=1000.0)

    command = planner.plan(curling, _prediction(1.0), (400.0, 300.0))
    assert abs(command.direction[0] - 1.0) < 1e-9
    assert abs(command.direction[1]) < 1e-9
    assert abs(math.hypot(*command.direction) - 1.0) < 1e-9

    down = planner.plan(curling, _prediction(1.0), (300.0, 400.0))
    assert abs(down.direction[0]) < 1e-9
    assert abs(down.direction[1] - 1.0) < 1e-9


def test_plan_clamps_target_into_rink():
    planner = TrajectoryPlanner()
    low = planner.plan(CurlingState(x=300.0, y=300.0), _prediction(1.0), (-100.0, -100.0))
    assert low.target_position == (core.RINK_LEFT + core.MALLET_RADIUS, core.RINK_TOP + core.MALLET_RADIUS)

    high = planner.plan(CurlingState(x=300.0, y=300.0), _prediction(1.0), (10_000.0, 10_000.0))
    assert high.target_position == (core.RINK_RIGHT - core.MALLET_RADIUS, core.RINK_BOTTOM - core.MALLET_RADIUS)

    inside = planner.plan(CurlingState(x=300.0, y=300.0), _prediction(1.0), (320.0, 340.0))
    assert inside.target_position == (320.0, 340.0)


def test_plan_speed_scales_with_distance():
    curling = CurlingState(x=300.0, y=300.0)
    planner = TrajectoryPlanner(max_speed=10_000.0, speed_gain=1.0)
    near = planner.plan(curling, _prediction(1.0), (350.0, 300.0))
    far = planner.plan(curling, _prediction(1.0), (400.0, 300.0))
    assert near.speed < far.speed
    assert abs(near.speed - 50.0) < 1e-9
    assert abs(far.speed - 100.0) < 1e-9


def test_plan_speed_uses_prediction_duration_as_time_budget():
    curling = CurlingState(x=300.0, y=300.0)
    planner = TrajectoryPlanner(max_speed=10_000.0)
    slow = planner.plan(curling, _prediction(1.0), (400.0, 300.0))
    medium = planner.plan(curling, _prediction(0.5), (400.0, 300.0))
    fast = planner.plan(curling, _prediction(0.25), (400.0, 300.0))
    assert slow.speed < medium.speed < fast.speed
    assert abs(slow.speed - 100.0) < 1e-9
    assert abs(medium.speed - 200.0) < 1e-9
    assert abs(fast.speed - 400.0) < 1e-9


def test_plan_speed_is_clamped_and_handles_stopped_stone():
    curling = CurlingState(x=300.0, y=300.0)
    planner = TrajectoryPlanner(max_speed=300.0, min_travel_time=0.1)
    # 停下的冰壶 duration=0 -> 使用最小时间预算，速度被 max_speed 限幅
    stopped = planner.plan(curling, _prediction(0.0), (400.0, 300.0))
    assert stopped.speed == 300.0

    tiny = planner.plan(curling, _prediction(0.001), (400.0, 300.0))
    assert tiny.speed == 300.0


def test_plan_coincident_position_is_stationary():
    curling = CurlingState(x=400.0, y=300.0)
    planner = TrajectoryPlanner()
    command = planner.plan(curling, _prediction(1.0), (400.0, 300.0))
    assert command.direction == (0.0, 0.0)
    assert command.speed == 0.0


def test_planner_parameter_validation():
    for kwargs in (
        {"max_speed": 0.0},
        {"max_speed": -1.0},
        {"speed_gain": 0.0},
        {"min_travel_time": 0.0},
    ):
        try:
            TrajectoryPlanner(**kwargs)
            assert False, f"{kwargs} 应该抛 ValueError"
        except ValueError:
            pass


def test_plan_does_not_modify_inputs():
    curling = CurlingState(x=300.0, y=300.0, vx=10.0, vy=-20.0, confidence=0.5)
    pred = _prediction(1.0)
    before_curling = (curling.x, curling.y, curling.vx, curling.vy, curling.confidence)
    before_prediction = (pred.endpoint, pred.duration, len(pred.trajectory))

    TrajectoryPlanner().plan(curling, pred, (400.0, 300.0))

    assert (curling.x, curling.y, curling.vx, curling.vy, curling.confidence) == before_curling
    assert (pred.endpoint, pred.duration, len(pred.trajectory)) == before_prediction


def test_plan_connects_real_prediction_output():
    curling = CurlingState(x=300.0, y=500.0, vx=120.0, vy=-80.0)
    prediction = TrajectoryPredictor().predict(curling)
    assert prediction.duration > 0.0

    planner = TrajectoryPlanner()
    command = planner.plan(curling, prediction, (350.0, 200.0))

    assert isinstance(command, ControlCommand)
    assert abs(math.hypot(*command.direction) - 1.0) < 1e-9
    assert 0.0 < command.speed <= planner.max_speed
    left, right, top, bottom = planner.bounds
    assert left + planner.margin <= command.target_position[0] <= right - planner.margin
    assert top + planner.margin <= command.target_position[1] <= bottom - planner.margin
