from dataclasses import replace
import math
from unittest.mock import patch

from air_hockey import ai as ai_module
from air_hockey import core_config as layout
from air_hockey.ai import AirHockeyAI
from air_hockey.control import PLCLink
from air_hockey.prediction import TrajectoryPredictor
from air_hockey.game_state import GameState, StoneState

# 关掉瞄准误差，决策才可复现
NO_ERROR = replace(layout.DIFFICULTIES["普通"], aim_error=0.0)


class CountingPredictor(TrajectoryPredictor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, stone, *args, **kwargs):
        self.calls += 1
        return super().predict(stone, *args, **kwargs)


def ai_home_y():
    return layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28


def make_state(stone_x, stone_y, vx=0.0, vy=0.0, ai_y=None):
    home = ai_home_y()
    return GameState(
        ai_x=layout.RINK_CENTER_X,
        ai_y=home if ai_y is None else ai_y,
        ai_home_y=home,
        target_x=layout.RINK_CENTER_X,
        target_y=home,
        stone=StoneState(x=stone_x, y=stone_y, vx=vx, vy=vy),
        awaiting_serve=False,
        current_server="player",
        serve_phase="idle",
        stalled_stone_phase="idle",
        reaction_timer=0.0,
        difficulty=NO_ERROR,
    )


def test_behind_puck_recovery_avoids_crossing_and_contacts_from_goal_side():
    minimum_distance = layout.STONE_RADIUS + layout.MALLET_RADIUS
    for x, y in ((300, 180), (100, 180), (500, 180), (300, 100)):
        ai = AirHockeyAI()
        state = make_state(x, y, ai_y=250)
        for _ in range(1000):
            decision = ai.choose_target(state)
            dx, dy = decision.target_x - state.ai_x, decision.target_y - state.ai_y
            distance = math.hypot(dx, dy)
            travel = min(distance, state.difficulty.ai_speed / 240)
            next_x = state.ai_x + dx * travel / distance if distance else state.ai_x
            next_y = state.ai_y + dy * travel / distance if distance else state.ai_y
            if decision.stalled_stone_phase in ("sidestepping", "positioning"):
                assert math.hypot(x - next_x, y - next_y) >= minimum_distance
            state.ai_x, state.ai_y = next_x, next_y
            state.stalled_stone_phase = decision.stalled_stone_phase
            if math.hypot(x - next_x, y - next_y) < minimum_distance:
                assert next_y < y  # 接触法线朝玩家端，不能从前方推向自家球门。
                break
        else:
            raise AssertionError("AI did not reach the behind puck")


def test_rear_recovery_contacts_from_reachable_goal_side():
    link = PLCLink("offline")
    bounds = link.reachable_bounds
    minimum_distance = layout.STONE_RADIUS + layout.MALLET_RADIUS
    for x, y, mallet_x in ((300, 140, 300), (480, 180, 480), (70, 180, 150)):
        ai = AirHockeyAI(recovery_bounds=bounds)
        state = make_state(x, y, ai_y=250)
        state.ai_x = mallet_x
        for _ in range(1000):
            decision = ai.choose_target(state)
            actual_target = link.physical_to_game(*link.game_to_physical(decision.target_x, decision.target_y))
            assert math.dist(actual_target, (decision.target_x, decision.target_y)) < 1e-8
            dx, dy = actual_target[0] - state.ai_x, actual_target[1] - state.ai_y
            distance = math.hypot(dx, dy)
            travel = min(distance, state.difficulty.ai_speed / 240)
            state.ai_x += dx * travel / distance if distance else 0.0
            state.ai_y += dy * travel / distance if distance else 0.0
            state.target_x, state.target_y = decision.target_x, decision.target_y
            state.stalled_stone_phase = decision.stalled_stone_phase
            if math.hypot(x - state.ai_x, y - state.ai_y) < minimum_distance:
                assert state.ai_y < y
                break
        else:
            raise AssertionError(("Rear recovery never contacted puck", (x, y), state))


def test_recovery_keeps_chosen_side_while_observation_crosses_mallet_center():
    ai = AirHockeyAI(recovery_bounds=PLCLink("offline").reachable_bounds)
    state = make_state(300, 180, ai_y=250)
    first = ai.choose_target(state)
    assert first.target_x > state.stone.x
    state.target_x, state.target_y = first.target_x, first.target_y
    state.stalled_stone_phase = first.stalled_stone_phase
    state.stone.x = 301
    following = ai.choose_target(state)
    assert following.target_x > state.stone.x
    assert following.stalled_stone_phase == "sidestepping"


def test_front_interception_is_unchanged_by_recovery_travel_limits():
    state = make_state(420, 250, vx=100, vy=-80)
    free = AirHockeyAI(mallet_speed=500)
    bounded = AirHockeyAI(mallet_speed=500, recovery_bounds=PLCLink("offline").reachable_bounds)
    assert bounded.choose_target(state) == free.choose_target(state)


def test_moving_behind_puck_is_bypassed_and_outgoing_puck_ends_recovery():
    ai = AirHockeyAI()
    state = make_state(300, 180, vy=-20, ai_y=250)
    decision = ai.choose_target(state)
    assert abs(decision.target_x - state.stone.x) > layout.STONE_RADIUS + layout.MALLET_RADIUS
    assert decision.target_y == state.ai_y
    state.stalled_stone_phase = decision.stalled_stone_phase
    state.stone.y, state.stone.vy = 450, 100
    assert ai.choose_target(state).stalled_stone_phase == "idle"


def test_recovery_aligns_to_puck_near_own_goal():
    """现场要求"全部追球"：冰壶离自家球门很近时也对齐 X 去回收。"""
    state = make_state(300, 70, ai_y=91)
    state.ai_x = 353
    state.stalled_stone_phase = "positioning"
    decision = AirHockeyAI().choose_target(state)
    assert abs(decision.target_x - state.stone.x) <= layout.STONE_RADIUS + layout.MALLET_RADIUS


def test_mallet_speed_adds_lead_in_front_of_moving_puck():
    """提前量：目标要落在冰壶**行进方向的前方**，而不是它现在的位置。

    回归现场反馈："球离开这个位置，轴才追上去"。
    """
    state = make_state(420.0, 250.0, vx=400.0, vy=-200.0)
    plain = AirHockeyAI().choose_target(state)
    lead = AirHockeyAI(mallet_speed=500.0).choose_target(state)
    assert (plain.target_x, plain.target_y) == (420.0, 250.0)           # 无参数 = 原行为
    assert lead.target_x > plain.target_x + 20.0                        # 球在往右跑
    assert lead.target_y < plain.target_y - 10.0                        # 球在往上跑


def test_mallet_speed_does_not_lead_a_still_puck():
    state = make_state(420.0, 250.0, vx=0.0, vy=0.0)
    lead = AirHockeyAI(mallet_speed=500.0).choose_target(state)
    assert (lead.target_x, lead.target_y) == (420.0, 250.0)


def test_lead_aims_ahead_on_path_when_puck_outruns_mallet():
    """球比球槌快时要瞄它**将要经过**的位置，不能退回瞄它现在的位置（那样永远追在后面），
    也不能发散到场地角落。

    回归：实时链路里冰壶进入机器半场、又不是正对球门时，旧写法无解就瞄冰壶当前位置，
    球槌离开防守线上本来正确的拦截点，往前冲 200 多单位后一路追在球后面。
    """
    state = make_state(420.0, 250.0, vx=400.0, vy=-200.0)
    decision = AirHockeyAI(mallet_speed=300.0).choose_target(state)     # 比球慢
    ahead = (decision.target_x - 420.0) * 400.0 + (decision.target_y - 250.0) * -200.0
    assert ahead > 0.0, decision
    assert layout.RINK_LEFT + layout.MALLET_RADIUS <= decision.target_x <= layout.RINK_RIGHT - layout.MALLET_RADIUS
    assert layout.RINK_TOP + layout.MALLET_RADIUS <= decision.target_y


def test_lead_target_accounts_for_wall_bounce():
    """提前量遇到侧墙反弹时：必须按物理微步计算反弹后的位置，而不是向墙外线性外推，
    并且取球槌**来得及赶到**的最早那个点。"""
    state = make_state(530.0, 250.0, vx=400.0, vy=0.0, ai_y=139.52)
    state = replace(state, ai_x=300.0)

    plain = AirHockeyAI().choose_target(state)
    assert (plain.target_x, plain.target_y) == (530.0, 250.0)

    decision = AirHockeyAI(mallet_speed=500.0).choose_target(state)
    # 冰壶撞右墙后往左弹回：目标在反弹后的路径上（x < 530），没有被钉在右极限
    assert decision.target_x < 530.0 and abs(decision.target_y - 250.0) < 0.1
    # 是"最早来得及"的点：比旧写法的 0.6s 位置（约 350.9）更靠前
    assert decision.target_x > 360.0, decision

def test_defense_uses_injected_predictor():
    # 注入不同参数的预测器，防守预测的目标横坐标必须不同
    from air_hockey.prediction import TrajectoryPredictor

    state = make_state(300.0, 500.0, vx=60.0, vy=-80.0)
    normal = AirHockeyAI().choose_target(state)
    short_stop = AirHockeyAI(predictor=TrajectoryPredictor(stop_speed=90.0)).choose_target(state)
    assert normal.target_x != short_stop.target_x


def test_default_forecast_reaches_home_line_and_guides_defense():
    state = make_state(260.0, 500.0, vx=180.0, vy=-250.0)
    ai = AirHockeyAI()
    prediction = ai.predictor.predict(state.stone)
    home = state.ai_home_y
    crossings = [
        x0 + (home - y0) / (y1 - y0) * (x1 - x0)
        for (x0, y0), (x1, y1) in zip(prediction[:-1], prediction[1:])
        if y1 <= home < y0
    ]
    assert crossings
    assert abs(crossings[0] - 519.55) < 0.1
    assert abs(ai._predict_stone_x(state, home, prediction) - crossings[0]) < 1e-9
    decision = ai.choose_target(state, prediction=prediction)
    expected_x = layout.RINK_CENTER_X * (1.0 - NO_ERROR.prediction) + crossings[0] * NO_ERROR.prediction
    assert abs(decision.target_x - expected_x) < 1e-9
    assert decision.target_y == home


def _assert_realtime_threat_reuses_prediction(stone_y):
    state = make_state(260.0, stone_y, vx=180.0, vy=-250.0)
    predictor = CountingPredictor()
    ai = AirHockeyAI(predictor=predictor)

    # Standalone callers still compute their own prediction on every reaction tick.
    baseline = ai.update(state, 1 / 60)
    assert predictor.calls == 1

    prediction = predictor.predict(state.stone)
    assert predictor.calls == 2
    reused = ai.update(state, 1 / 60, prediction=prediction)
    assert (reused.target_x, reused.target_y) == (baseline.target_x, baseline.target_y)
    assert reused.stalled_stone_phase == baseline.stalled_stone_phase
    assert predictor.calls == 2

    direct = ai.choose_target(state, prediction=prediction)
    assert (direct.target_x, direct.target_y) == (baseline.target_x, baseline.target_y)
    assert predictor.calls == 2
    standalone = ai.choose_target(state)
    assert (standalone.target_x, standalone.target_y) == (baseline.target_x, baseline.target_y)
    assert predictor.calls == 3


def test_realtime_threat_outside_attack_zone_reuses_prediction():
    _assert_realtime_threat_reuses_prediction(500.0)


def test_reaction_timer_holds_previous_target():
    ai = AirHockeyAI()
    state = make_state(300.0, 200.0)
    state.reaction_timer = 0.05
    decision = ai.update(state, 0.01)
    assert decision.target_x == state.target_x
    assert abs(decision.reaction_timer - 0.04) < 1e-9


def test_serve_phase_switches_to_striking_after_arrival():
    ai = AirHockeyAI()
    state = make_state(0.0, 0.0)
    state.awaiting_serve = True
    state.current_server = "ai"
    state.serve_phase = "positioning"
    state.ai_x = layout.RINK_CENTER_X
    state.ai_y = layout.RINK_CENTER_Y - layout.MALLET_RADIUS - layout.STONE_RADIUS - 6.0
    assert ai.advance_serve_phase(state, 1 / 60) == "striking"


LIVE = replace(layout.DIFFICULTIES["普通"], aim_error=0.0, prediction=1.0,
               reaction_delay=0.0, attack_line=layout.RINK_CENTER_Y)


def _live_state(stone_x, stone_y, vx, vy):
    return replace(make_state(stone_x, stone_y, vx, vy), difficulty=LIVE)


def test_fast_shot_guard_has_hysteresis_against_velocity_noise():
    """冲门球的预测穿线点在门柱附近抖动时，Y 目标不能在防守线和冰壶之间来回跳。"""
    home = ai_home_y()
    inside = layout.GOAL_RIGHT + layout.STONE_RADIUS * 0.5    # 进入门限内
    wobble = layout.GOAL_RIGHT + layout.STONE_RADIUS * 2.0    # 进入门限外、退出门限内
    outside = layout.GOAL_RIGHT + layout.STONE_RADIUS * 4.0   # 明确偏离球门
    ai = AirHockeyAI()
    # 没进入守门模式时，门柱外 2R 的球不算冲门（上前击打）
    assert ai.choose_target(_live_state(wobble, 330.0, 0.0, -300.0)).target_y > home + 100.0
    # 明确冲门：进入守门线模式；之后穿线点在门柱附近抖动也保持
    for x in (inside, wobble, inside, wobble, wobble):
        assert ai.choose_target(_live_state(x, 330.0, 0.0, -300.0)).target_y == home
    # 明确偏离球门才退出
    assert ai.choose_target(_live_state(outside, 330.0, 0.0, -300.0)).target_y > home + 100.0
    # 速度迟滞：守门模式下 vy 落在进入阈值以上、退出阈值以下仍保持（这里只看守门判断，关掉迎击）
    ai = AirHockeyAI()
    with patch.object(ai_module, "STRIKE_THROUGH", 0.0):
        assert ai.choose_target(_live_state(300.0, 200.0, 0.0, -300.0)).target_y == home
        assert ai.choose_target(_live_state(300.0, 180.0, 0.0, -100.0)).target_y == home
        assert ai.choose_target(_live_state(300.0, 150.0, 0.0, -45.0)).target_y == home


def test_goal_guard_strikes_through_only_when_puck_is_about_to_arrive():
    """守门线等球时，冰壶快到了才迎着来球方向冲出去打，远的时候老老实实等在防守线。"""
    home = ai_home_y()
    contact = layout.STONE_RADIUS + layout.MALLET_RADIUS
    far = AirHockeyAI().choose_target(_live_state(300.0, 330.0, 0.0, -600.0))
    assert far.target_y == home
    # 再过约 0.05s 就碰到球槌：目标沿来球方向（+y）往前推
    near = AirHockeyAI().choose_target(_live_state(300.0, home + contact + 30.0, 0.0, -600.0))
    assert near.target_x == 300.0
    assert near.target_y == home + ai_module.STRIKE_THROUGH


def test_slow_ball_that_stops_before_defense_line_is_attacked_not_guarded():
    """vy 足够负但滑不到防守线就会停下的慢球，应上前击打而不是退守。"""
    ai = AirHockeyAI()
    decision = ai.choose_target(_live_state(300.0, 330.0, 0.0, -70.0))
    assert decision.target_y > ai_home_y() + 100.0
