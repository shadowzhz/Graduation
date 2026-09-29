from dataclasses import replace

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


def test_stone_behind_ai_does_not_chase():
    ai = AirHockeyAI()
    home = ai_home_y()
    # 冰壶在 AI 身后直奔球门：只守家门横线，不追球
    state = make_state(layout.RINK_CENTER_X, home - 50.0, vx=0.0, vy=-300.0, ai_y=home + 10.0)
    decision = ai.choose_target(state)
    assert decision.target_y == home


def test_defense_uses_injected_predictor():
    # 注入不同参数的预测器，防守预测的目标横坐标必须不同
    from air_hockey.prediction import TrajectoryPredictor

    state = make_state(300.0, 500.0, vx=60.0, vy=-80.0)
    normal = AirHockeyAI().choose_target(state)
    short_stop = AirHockeyAI(predictor=TrajectoryPredictor(stop_speed=90.0)).choose_target(state)
    assert normal.target_x != short_stop.target_x


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


def test_realtime_threat_behind_ai_reuses_prediction():
    _assert_realtime_threat_reuses_prediction(ai_home_y() - 50.0)


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
