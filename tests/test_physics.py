import math
from pathlib import Path

import air_hockey_config as layout
from air_hockey_physics import (
    StoneMotion,
    clamp,
    goal_scorer,
    stone_inside_goal_mouth,
)
from game_state import StoneState
from prediction import TrajectoryPredictor, predict_trajectory


def test_clamp():
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0


def test_wall_bounce_reflects_velocity():
    stone = StoneMotion(x=layout.RINK_LEFT + layout.STONE_RADIUS - 5.0, y=layout.RINK_CENTER_Y)
    stone.set_immediate_velocity(-300.0, 0.0)
    stone.x += stone.vx * 0.01
    assert stone.resolve_walls()
    assert stone.x == layout.RINK_LEFT + layout.STONE_RADIUS
    assert abs(stone.vx - 300.0 * layout.WALL_RESTITUTION) < 1e-9


def test_friction_eventually_stops_stone():
    stone = StoneMotion()
    stone.set_immediate_velocity(100.0, 0.0)
    for _ in range(600):
        stone.advance_velocity(1 / 60)
    assert stone.vx == 0.0 and stone.vy == 0.0


def test_goal_detection():
    # 球门口内整体越线才算得分
    top_stone = StoneMotion(x=layout.RINK_CENTER_X, y=layout.RINK_TOP - layout.STONE_RADIUS - 1.0)
    assert goal_scorer(top_stone) == "player"
    bottom_stone = StoneMotion(x=layout.RINK_CENTER_X, y=layout.RINK_BOTTOM + layout.STONE_RADIUS + 1.0)
    assert goal_scorer(bottom_stone) == "ai"
    side_stone = StoneMotion(x=layout.RINK_LEFT + 10.0, y=layout.RINK_TOP - layout.STONE_RADIUS - 1.0)
    assert goal_scorer(side_stone) is None
    assert stone_inside_goal_mouth(layout.RINK_CENTER_X)
    assert not stone_inside_goal_mouth(layout.RINK_LEFT + 10.0)


def test_straight_line_friction_deceleration():
    """无碰撞直线减速：步间距随阻尼减小，最终速度低于停止速度。"""
    predictor = TrajectoryPredictor()
    traj = predictor.predict(
        layout.RINK_CENTER_X,
        layout.RINK_CENTER_Y,
        150.0,
        0.0,
    )
    assert len(traj) > 3
    # y 保持不变
    for _, py in traj:
        assert abs(py - layout.RINK_CENTER_Y) < 1e-5

    # 前半段平均步长显著大于后半段平均步长（摩擦减速）
    step_lengths = [
        math.hypot(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1])
        for i in range(len(traj) - 1)
    ]
    half = len(step_lengths) // 2
    first_half_avg = sum(step_lengths[:half]) / half
    second_half_avg = sum(step_lengths[half:]) / (len(step_lengths) - half)
    assert first_half_avg > second_half_avg
    assert step_lengths[0] > step_lengths[5] > step_lengths[-1]


def test_left_right_wall_bounce():
    """左右边墙反弹：碰到边墙产生折返，且反弹后速度按恢复系数折减。"""
    predictor = TrajectoryPredictor()
    start_x = layout.RINK_LEFT + layout.STONE_RADIUS + 25.0
    traj = predictor.predict(
        start_x,
        layout.RINK_CENTER_Y,
        -200.0,
        0.0,
    )
    assert len(traj) > 2
    # 必须反弹，即 x 必须先减小到边界，随后增大
    min_x = min(pt[0] for pt in traj)
    assert abs(min_x - (layout.RINK_LEFT + layout.STONE_RADIUS)) < 1e-4
    last_x = traj[-1][0]
    assert last_x > min_x


def test_goal_mouth_no_wall_bounce():
    """球门开口处不发生普通墙壁反弹，而是穿过门线进球。"""
    predictor = TrajectoryPredictor()
    # 位于球门横坐标范围内部，向上冲向球门
    start_x = layout.RINK_CENTER_X
    start_y = layout.RINK_TOP + 20.0
    traj = predictor.predict(
        start_x,
        start_y,
        0.0,
        -250.0,
    )
    assert len(traj) > 1
    # y 应该持续减小并穿过 RINK_TOP + STONE_RADIUS，最终越过门线进球，而不被当成墙反弹
    for pt in traj:
        assert abs(pt[0] - start_x) < 1e-5
    min_y = min(pt[1] for pt in traj)
    assert min_y < layout.RINK_TOP + layout.STONE_RADIUS
    # 最终进入球门被判定进球并终止
    assert traj[-1][1] <= layout.RINK_TOP - layout.STONE_RADIUS or min_y < layout.RINK_TOP


def test_goal_post_collision():
    """球门柱碰撞：球撞击门柱发生圆形接触反弹。"""
    predictor = TrajectoryPredictor()
    # 目标门柱：左上门柱 (layout.GOAL_LEFT, layout.RINK_TOP)
    post_x, post_y = layout.GOAL_LEFT, layout.RINK_TOP
    # 放置在门柱稍左下侧，朝门柱运动
    start_x = post_x - 15.0
    start_y = post_y + 15.0
    traj = predictor.predict(
        start_x,
        start_y,
        150.0,
        -150.0,
    )
    assert len(traj) > 2
    # 轨迹应偏折，反弹后速度方向发生显著变化
    initial_dx = traj[1][0] - traj[0][0]
    initial_dy = traj[1][1] - traj[0][1]
    last_dx = traj[-1][0] - traj[-2][0]
    last_dy = traj[-1][1] - traj[-2][1]
    dot = initial_dx * last_dx + initial_dy * last_dy
    initial_len = math.hypot(initial_dx, initial_dy)
    last_len = math.hypot(last_dx, last_dy)
    # 发生碰撞偏折后方向与初始方向夹角应明显改变
    cos_angle = dot / (initial_len * last_len)
    assert cos_angle < 0.95


def test_stone_eventually_stops():
    """冰壶速度低于停止速度时停止，初始速度为 0 时直接返回起始点。"""
    predictor = TrajectoryPredictor()
    # 零速
    zero_traj = predictor.predict(100.0, 100.0, 0.0, 0.0)
    assert len(zero_traj) == 1
    assert zero_traj[0] == (100.0, 100.0)

    # 极低速 (低于 STOP_SPEED)
    low_traj = predictor.predict(100.0, 100.0, 2.0, 0.0)
    assert len(low_traj) == 1
    assert low_traj[0] == (100.0, 100.0)

    # 低速在冰面滑动很短距离后停止
    short_traj = predictor.predict(layout.RINK_CENTER_X, layout.RINK_CENTER_Y, 20.0, 0.0)
    assert 1 < len(short_traj) < layout.PREDICTION_POINT_COUNT


def test_vision_and_sim_produce_identical_trajectory():
    """相同初始状态下，视觉入口 (StoneState) 和仿真入口 (StoneMotion) 输出完全相同的轨迹。"""
    predictor = TrajectoryPredictor()
    x, y, vx, vy = 300.0, 400.0, 260.0, -180.0

    # 视觉侧传入 StoneState
    vision_stone = StoneState(x=x, y=y, vx=vx, vy=vy)
    vision_traj = predictor.predict(vision_stone)

    # 仿真侧传入 StoneMotion
    sim_stone = StoneMotion(x=x, y=y, vx=vx, vy=vy)
    sim_traj = predictor.predict(sim_stone)

    # 函数入口 predict_trajectory
    func_traj = predict_trajectory(x, y, vx, vy)

    assert vision_traj == sim_traj
    assert vision_traj == func_traj


def test_no_old_predictor_references():
    """删除旧 Predictor 后，确保不存在旧引用与旧模块。"""
    import importlib

    # 确保 vision.predictor 不存在
    try:
        importlib.import_module("vision.predictor")
        assert False, "vision.predictor 应已被删除"
    except ModuleNotFoundError:
        pass

    try:
        importlib.import_module("prediction.trajectory")
        assert False, "prediction.trajectory 应已被删除"
    except ModuleNotFoundError:
        pass

    # 扫描工程下所有 py 文件，确保没有引用 vision.predictor 或旧 predict_position
    root = Path(__file__).resolve().parents[1]
    for py_file in root.rglob("*.py"):
        if ".git" in py_file.parts or "__pycache__" in py_file.parts or py_file == Path(__file__).resolve():
            continue
        content = py_file.read_text(encoding="utf-8")
        assert "vision.predictor" not in content, f"{py_file} 仍包含 vision.predictor 引用"
        assert "prediction.trajectory" not in content, f"{py_file} 仍包含 prediction.trajectory 引用"
