import math
from pathlib import Path

import core_config as layout
from air_hockey_physics import (
    StoneMotion,
    clamp,
    goal_scorer,
    stone_inside_goal_mouth,
)
from game_state import GameState, StoneState
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


def test_responsive_scaling_updates_predictor_parameters():
    """验证响应式缩放后，Predictor 在运行时读取当前配置，无需重新初始化。"""
    predictor = TrajectoryPredictor()
    orig_stop = layout.STONE_STOP_SPEED
    orig_mallet = layout.MALLET_RADIUS
    try:
        # 1. 验证 stop_speed 动态生效
        stone = StoneState(x=layout.RINK_CENTER_X, y=layout.RINK_CENTER_Y, vx=50.0, vy=0.0)
        layout.STONE_STOP_SPEED = 8.0
        traj1 = predictor.predict(stone)
        assert len(traj1) > 1, "速度 50 > 8 时应产生移动轨迹"

        layout.STONE_STOP_SPEED = 60.0
        traj2 = predictor.predict(stone)
        assert len(traj2) == 1, "速度 50 <= 60 时应立即判定停止并返回单点"
        layout.STONE_STOP_SPEED = orig_stop

        # 2. 验证 obstacle_radius 动态生效
        moving_stone = StoneState(x=layout.RINK_CENTER_X, y=layout.RINK_CENTER_Y, vx=100.0, vy=0.0)
        # 放置障碍物在侧前方 (纵向偏移 28px，横向偏移 30px)
        obs_x = moving_stone.x + 30.0
        obs_y = moving_stone.y + 28.0
        # 冰壶半径 14，若球槌半径为 10，总半径 24 < 28，不碰
        layout.MALLET_RADIUS = 10.0
        traj_no_hit = predictor.predict(moving_stone, obstacles=((obs_x, obs_y),))
        # 若球槌半径增至 20，总半径 34 >= 28，必撞
        layout.MALLET_RADIUS = 20.0
        traj_hit = predictor.predict(moving_stone, obstacles=((obs_x, obs_y),))
        # 撞击会截断轨迹
        assert len(traj_hit) < len(traj_no_hit)
    finally:
        layout.STONE_STOP_SPEED = orig_stop
        layout.MALLET_RADIUS = orig_mallet


def test_ai_uses_shared_predictor_without_independent_physics():
    """验证 AI 不再保留任何独立物理公式（无 _reflect_coordinate），统一使用 TrajectoryPredictor。"""
    import air_hockey_ai

    # 模块和类中绝无 _reflect_coordinate
    assert not hasattr(air_hockey_ai, "_reflect_coordinate")
    assert not hasattr(air_hockey_ai.AirHockeyAI, "_reflect_coordinate")

    ai = air_hockey_ai.AirHockeyAI()
    assert hasattr(ai, "predictor")
    assert isinstance(ai.predictor, TrajectoryPredictor)

    # 验证 AI 的预测直接由其共享的 predictor 提供
    state = StoneState(x=300.0, y=500.0, vx=80.0, vy=-120.0)
    game_state = GameState.from_vision(state)
    target_y = 420.0
    predicted_x = ai._predict_stone_x(game_state, target_y=target_y)

    # 手动用同个 predictor 算一遍截距，应完全一致
    traj = ai.predictor.predict(state)
    expected_x = None
    for (x0, y0), (x1, y1) in zip(traj[:-1], traj[1:]):
        if (y0 <= target_y <= y1) or (y1 <= target_y <= y0):
            t = (target_y - y0) / (y1 - y0)
            expected_x = x0 + t * (x1 - x0)
            break
    assert expected_x is not None
    assert abs(predicted_x - expected_x) < 1e-9


def test_vision_sim_ai_share_same_prediction_core():
    """验证视觉、仿真、AI 对相同状态使用同一预测核心，轨迹 100% 一致。"""
    import air_hockey_ai

    vision_predictor = TrajectoryPredictor()
    sim_predictor = TrajectoryPredictor()
    ai = air_hockey_ai.AirHockeyAI(predictor=sim_predictor)

    x, y, vx, vy = 280.0, 480.0, 150.0, -220.0

    # 1. 视觉输入 (StoneState)
    vision_stone = StoneState(x=x, y=y, vx=vx, vy=vy)
    vision_traj = vision_predictor.predict(vision_stone)

    # 2. 仿真输入 (StoneMotion)
    sim_stone = StoneMotion(x=x, y=y, vx=vx, vy=vy)
    sim_traj = sim_predictor.predict(sim_stone)

    # 3. AI 输入 (GameState.stone)
    game_state = GameState.from_vision(vision_stone)
    ai_traj = ai.predictor.predict(game_state.stone)

    assert vision_traj == sim_traj
    assert vision_traj == ai_traj


def test_gui_scale_change_does_not_affect_physics_and_predictor():
    """验证调整 GUI UI_SCALE 后，共享物理参数和 Predictor 结果不发生变化。"""
    import air_hockey_config as gui_config
    import core_config as core

    # 1. 记录共享物理核心配置快照
    initial_params = {
        "RINK_LEFT": core.RINK_LEFT,
        "RINK_RIGHT": core.RINK_RIGHT,
        "RINK_TOP": core.RINK_TOP,
        "RINK_BOTTOM": core.RINK_BOTTOM,
        "RINK_CENTER_X": core.RINK_CENTER_X,
        "RINK_CENTER_Y": core.RINK_CENTER_Y,
        "STONE_RADIUS": core.STONE_RADIUS,
        "MALLET_RADIUS": core.MALLET_RADIUS,
        "GOAL_LEFT": core.GOAL_LEFT,
        "GOAL_RIGHT": core.GOAL_RIGHT,
        "MAX_STONE_SPEED": core.MAX_STONE_SPEED,
        "STONE_STOP_SPEED": core.STONE_STOP_SPEED,
        "STONE_FRICTION_DECELERATION": core.STONE_FRICTION_DECELERATION,
        "MIN_WALL_BOUNCE_SPEED": core.MIN_WALL_BOUNCE_SPEED,
        "WALL_RESTITUTION": core.WALL_RESTITUTION,
    }

    # 2. 预测器在原始状态下的预测结果
    predictor = TrajectoryPredictor()
    stone = StoneState(x=300.0, y=400.0, vx=200.0, vy=-150.0)
    base_traj = predictor.predict(stone)

    orig_ui_scale = gui_config.UI_SCALE
    orig_canvas_w = gui_config.CANVAS_WIDTH
    orig_canvas_h = gui_config.CANVAS_HEIGHT

    try:
        # 3. 模拟 GUI 屏幕缩放（例如缩小到 0.6 或调用响应式布局计算）
        class MockScreenRoot:
            @staticmethod
            def winfo_screenwidth():
                return 400

            @staticmethod
            def winfo_screenheight():
                return 400

        gui_config._apply_responsive_layout(MockScreenRoot())
        assert gui_config.UI_SCALE < 1.0, "GUI 缩放应小于 1.0"
        assert gui_config.CANVAS_WIDTH < core.BASE_CANVAS_WIDTH

        # 4. 验证共享物理参数完全未发生变化
        for param, expected_val in initial_params.items():
            actual_val = getattr(core, param)
            assert actual_val == expected_val, f"物理参数 {param} 受到 GUI 缩放影响: {actual_val} != {expected_val}"

        # 5. 验证 Predictor 预测结果 100% 保持一致
        scaled_gui_traj = predictor.predict(stone)
        assert scaled_gui_traj == base_traj, "GUI 缩放导致 Predictor 输出结果发生变化"

    finally:
        gui_config.UI_SCALE = orig_ui_scale
        gui_config.CANVAS_WIDTH = orig_canvas_w
        gui_config.CANVAS_HEIGHT = orig_canvas_h

