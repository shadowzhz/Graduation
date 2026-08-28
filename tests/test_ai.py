from dataclasses import replace

import air_hockey_config as layout
from air_hockey_ai import AirHockeyAI
from game_state import GameState, PuckState

# 关掉瞄准误差，决策才可复现
NO_ERROR = replace(layout.DIFFICULTIES["普通"], aim_error=0.0)


def ai_home_y():
    return layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28


def make_state(puck_x, puck_y, vx=0.0, vy=0.0, ai_y=None):
    home = ai_home_y()
    return GameState(
        ai_x=layout.RINK_CENTER_X,
        ai_y=home if ai_y is None else ai_y,
        ai_home_y=home,
        target_x=layout.RINK_CENTER_X,
        target_y=home,
        puck=PuckState(x=puck_x, y=puck_y, vx=vx, vy=vy),
        awaiting_serve=False,
        current_server="player",
        serve_phase="idle",
        stalled_puck_phase="idle",
        reaction_timer=0.0,
        difficulty=NO_ERROR,
    )


def test_puck_behind_ai_does_not_chase():
    ai = AirHockeyAI()
    home = ai_home_y()
    # 冰球在 AI 身后直奔球门：只守家门横线，不追球
    state = make_state(layout.RINK_CENTER_X, home - 50.0, vx=0.0, vy=-300.0, ai_y=home + 10.0)
    decision = ai.choose_target(state)
    assert decision.target_y == home


def test_defense_uses_injected_friction():
    # 同一状态，冰面摩擦不同，防守预测的横坐标必须不同
    state = make_state(300.0, 500.0, vx=60.0, vy=-80.0)
    loose = AirHockeyAI().choose_target(state)
    stiff = AirHockeyAI(friction_deceleration=10000.0).choose_target(state)
    assert loose.target_x != stiff.target_x


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
    state.ai_y = layout.RINK_CENTER_Y - layout.MALLET_RADIUS - layout.PUCK_RADIUS - 6.0
    assert ai.advance_serve_phase(state, 1 / 60) == "striking"
