import math

from air_hockey import core_config as layout
from air_hockey.physics import (
    StoneMotion,
    clamp,
    goal_scorer,
    stone_inside_goal_mouth,
)
from game_state import GameState, StoneState
from air_hockey.prediction import TrajectoryPredictor, predict_trajectory


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


def test_step_matches_position_then_velocity_order_without_collisions():
    dt = 0.02
    stone = StoneMotion(x=300.0, y=400.0, vx=120.0, vy=-80.0)
    expected_x = stone.x + stone.vx * dt
    expected_y = stone.y + stone.vy * dt
    expected_velocity = StoneMotion(vx=stone.vx, vy=stone.vy)
    expected_velocity.advance_velocity(dt)

    assert not stone.step(dt)
    assert (stone.x, stone.y) == (expected_x, expected_y)
    assert (stone.vx, stone.vy) == (expected_velocity.vx, expected_velocity.vy)


def test_step_resolves_wall_before_friction():
    left_limit = layout.RINK_LEFT + layout.STONE_RADIUS
    stone = StoneMotion(x=left_limit + 1.0, y=layout.RINK_CENTER_Y)
    stone.set_immediate_velocity(-200.0, 0.0)

    assert stone.step(0.01)
    assert stone.x == left_limit
    expected_speed = 200.0 * layout.WALL_RESTITUTION - layout.STONE_FRICTION_DECELERATION * 0.01
    assert abs(stone.vx - expected_speed) < 1e-9


def test_step_resolves_goal_post_before_velocity_update():
    post_x, post_y = layout.GOAL_POSTS[0]
    dt = 0.01
    stone = StoneMotion(x=post_x + 14.1, y=post_y + 15.0, vx=0.0, vy=-200.0)
    expected = StoneMotion(x=stone.x, y=stone.y, vx=stone.vx, vy=stone.vy)
    expected.x += expected.vx * dt
    expected.y += expected.vy * dt
    expected.resolve_walls()
    assert expected.resolve_goal_posts()
    expected.advance_velocity(dt)

    assert stone.step(dt)
    assert (stone.x, stone.y, stone.vx, stone.vy) == (
        expected.x, expected.y, expected.vx, expected.vy,
    )


def test_step_zero_dt_preserves_all_motion_state():
    stone = StoneMotion(x=300.0, y=400.0, vx=12.0, vy=-8.0,
                        target_vx=40.0, target_vy=-20.0, response_active=True)
    before = (stone.x, stone.y, stone.vx, stone.vy, stone.target_vx, stone.target_vy,
              stone.response_active)
    assert not stone.step(0.0)
    assert (stone.x, stone.y, stone.vx, stone.vy, stone.target_vx, stone.target_vy,
            stone.response_active) == before


def test_step_rejects_negative_and_nonfinite_dt_without_mutation():
    for dt in (-1.0, math.nan, math.inf):
        stone = StoneMotion(x=300.0, y=400.0, vx=12.0, vy=-8.0)
        before = (stone.x, stone.y, stone.vx, stone.vy)
        try:
            stone.step(dt)
            assert False, f"dt={dt} 应该抛 ValueError"
        except ValueError:
            assert (stone.x, stone.y, stone.vx, stone.vy) == before


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
    assert len(short_traj) > 1
    assert 0.0 < short_traj.endpoint[0] - layout.RINK_CENTER_X < 3.0


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


def test_predictor_reads_core_config_at_runtime():
    """验证 Predictor 在运行时读取 core_config 的当前值，无需重新初始化。"""
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


def test_ai_crossing_matches_shared_prediction():
    """AI 横线截距与共享预测的轨迹插值一致。"""
    from air_hockey import ai

    ai = ai.AirHockeyAI()

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
    from air_hockey import ai

    vision_predictor = TrajectoryPredictor()
    sim_predictor = TrajectoryPredictor()
    ai = ai.AirHockeyAI(predictor=sim_predictor)

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


def test_default_forecast_reaches_stop_after_multiple_bounces():
    source = StoneMotion(x=300.0, y=380.0, vx=layout.MAX_STONE_SPEED)
    prediction = TrajectoryPredictor().predict(source)
    expected = StoneMotion(x=source.x, y=source.y, vx=source.vx)
    step = layout.PREDICTION_SUBSTEP
    steps = 0
    while math.hypot(expected.vx, expected.vy) > layout.STONE_STOP_SPEED:
        expected.x += expected.vx * step
        expected.y += expected.vy * step
        expected.resolve_walls()
        expected.resolve_goal_posts()
        expected.advance_velocity(step)
        steps += 1
    assert prediction.endpoint == (expected.x, expected.y)
    assert prediction[-1] == prediction.endpoint
    assert abs(prediction.duration - steps * step) < 1e-9
    assert prediction.duration > 1024 * step
    assert sum(x in (50.0, 550.0) for x, _ in prediction) > 2
    assert source.vx == layout.MAX_STONE_SPEED


def test_predictor_matches_repeated_shared_motion_steps():
    dt = layout.PREDICTION_SUBSTEP
    source = StoneMotion(x=300.0, y=400.0, vx=180.0, vy=-120.0)
    prediction = TrajectoryPredictor().predict(source)
    truth = StoneMotion(x=source.x, y=source.y, vx=source.vx, vy=source.vy)
    truth.set_immediate_velocity(truth.vx, truth.vy)
    duration = 0.0

    for _ in range(layout.PREDICTION_MAX_SIMULATION_STEPS):
        truth.step(dt)
        duration += dt
        if goal_scorer(truth) or math.hypot(truth.vx, truth.vy) <= layout.STONE_STOP_SPEED:
            break
    else:
        assert False, "Expected motion to reach a terminal state"

    assert prediction.endpoint == (truth.x, truth.y)
    assert prediction.duration == duration


def test_overspeed_is_capped_before_first_position_step():
    predictor = TrajectoryPredictor()
    source = StoneState(x=300.0, y=50.0, vx=0.0, vy=-5000.0)
    prediction = predictor.predict(source)
    expected = predictor.predict(300.0, 50.0, 0.0, -layout.MAX_STONE_SPEED)
    assert prediction.trajectory == expected.trajectory
    assert prediction.duration == expected.duration
    assert prediction.source_state is source
    assert source.vy == -5000.0


def test_current_and_target_velocity_are_capped_without_losing_response():
    source = StoneMotion(x=300.0, y=380.0, vx=5000.0, target_vx=-5000.0, response_active=True)
    prediction = TrajectoryPredictor().predict(source)
    expected = TrajectoryPredictor().predict(
        StoneMotion(x=300.0, y=380.0, vx=1100.0, target_vx=-1100.0, response_active=True)
    )
    assert prediction.trajectory == expected.trajectory
    assert prediction.duration == expected.duration
    assert (source.vx, source.target_vx, source.response_active) == (5000.0, -5000.0, True)


def test_stationary_current_velocity_with_active_response_still_moves():
    source = StoneMotion(x=300.0, y=380.0, target_vx=150.0, response_active=True)
    prediction = TrajectoryPredictor().predict(source)
    assert prediction.endpoint[0] > source.x + 100.0
    assert prediction.duration > 1.0
    assert (source.vx, source.target_vx, source.response_active) == (0.0, 150.0, True)

