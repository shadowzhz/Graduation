from dataclasses import replace
import math

from air_hockey import core_config as layout
from air_hockey.ai import AirHockeyAI
from air_hockey.prediction import TrajectoryPredictor
from game_state import GameState, StoneState

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


def test_moving_behind_puck_is_bypassed_and_outgoing_puck_ends_recovery():
    ai = AirHockeyAI()
    state = make_state(300, 180, vy=-20, ai_y=250)
    decision = ai.choose_target(state)
    assert abs(decision.target_x - state.stone.x) > layout.STONE_RADIUS + layout.MALLET_RADIUS
    assert decision.target_y == state.ai_y
    state.stalled_stone_phase = decision.stalled_stone_phase
    state.stone.y, state.stone.vy = 450, 100
    assert ai.choose_target(state).stalled_stone_phase == "idle"


def test_recovery_does_not_strike_toward_goal_below_safe_back_line():
    state = make_state(300, 70, ai_y=91)
    state.ai_x = 353
    state.stalled_stone_phase = "positioning"
    decision = AirHockeyAI().choose_target(state)
    assert decision.target_y == state.ai_y
    assert abs(decision.target_x - state.stone.x) > layout.STONE_RADIUS + layout.MALLET_RADIUS


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
