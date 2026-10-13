"""直接防守/主动击球、回位及 PLC 目标边界。"""

from dataclasses import replace
import math

import numpy as np

from air_hockey import core_config as core
from air_hockey.ai import AirHockeyAI
from air_hockey.app.rally import READY_POSITION, RallyController, RallyState
from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
from air_hockey.control import PLCLink, PlcControlAdapter
from air_hockey.control.plc import PLCFeedback
from air_hockey.prediction import TrajectoryPredictor
from air_hockey.vision.types import Detection, Track, TrackState
from game_state import CurlingState, GameState
from main import submit_rally_target


def observation(t, *, x=420.0, y=500.0, vx=0.0, vy=-300.0, hard=True,
                confirmed=True, identity=1, mallet=READY_POSITION):
    stone = CurlingState(x, y, vx, vy, timestamp=t)
    prediction = TrajectoryPredictor().predict(stone)
    state = GameState(
        ai_x=mallet[0], ai_y=mallet[1], ai_home_y=READY_POSITION[1],
        target_x=mallet[0], target_y=mallet[1], stone=stone,
        awaiting_serve=False, current_server="player", serve_phase="idle",
        stalled_stone_phase="idle", reaction_timer=0,
        difficulty=replace(core.DIFFICULTIES["普通"], aim_error=0, prediction=1,
                           reaction_delay=0, attack_line=core.RINK_CENTER_Y),
    )
    decision = AirHockeyAI().choose_target(state, prediction=prediction)
    return VisionResult(
        frame=Frame(np.zeros((20, 20, 3), np.uint8), timestamp=t),
        detection=Detection(x, y, 14, 700, t) if hard else None,
        track=Track(identity, x, y, 14, vx=vx, vy=vy, last_timestamp=t,
                    state=TrackState.ACTIVE if confirmed else TrackState.LOST),
        track_confirmed=confirmed, curling_state=stone, prediction=prediction,
        ai_decision=decision,
    )


def tick(controller, result, feedback=None, now=None, mallet=READY_POSITION):
    result.rally = controller.update(result, now=result.frame.timestamp if now is None else now,
                                     feedback=feedback, mallet_position=mallet)
    return result.rally


def test_first_machine_half_detection_is_immediately_attacked():
    controller = RallyController()
    result = observation(1, x=420, y=250, vy=0)
    decision = tick(controller, result)
    assert result.track.hits == 1
    assert decision.state == RallyState.ATTACKING
    assert decision.target == (420, 250)
    request = submit_rally_target(result, PlcControlAdapter(), None)
    assert (request.target_x, request.target_y) == (420, 250)


def test_stopped_machine_half_puck_is_retrieved():
    """自己半场里**已经停住**的冰壶必须上去回收，否则回合会停住。"""
    for y, vy in ((340, 0), (core.RINK_CENTER_Y, 0)):
        result = observation(1, x=420, y=y, vy=vy)
        decision = tick(RallyController(), result)
        assert decision.state == RallyState.ATTACKING
        assert decision.target == (420, min(y, core.RINK_CENTER_Y - core.MALLET_RADIUS))


def test_moving_machine_half_puck_pursuit_is_capped_by_depth_limit():
    """主防御：冰壶在自己半场还在运动时，追击深度被 defense_forward_limit 截住。

    注意 Y 仍然会动 —— 这是"把 Y 钉死在防守线上"那一版的回归点，
    完全固定 Y 会让轴看起来像不动。
    """
    controller = RallyController(defense_forward_limit=90.0)
    decision = tick(controller, observation(1, x=420, y=250, vy=200))
    assert decision.state == RallyState.ATTACKING
    assert decision.target == (420.0, READY_POSITION[1] + 90.0)


def test_zero_forward_limit_pins_to_defense_line():
    """上限为 0 时等价于钉死在防守线上（Y 完全不动）。"""
    controller = RallyController(defense_forward_limit=0.0)
    decision = tick(controller, observation(1, x=420, y=250, vy=200))
    assert decision.target == (420.0, READY_POSITION[1])


def test_unlimited_pursuit_matches_ai_target():
    """不设上限时等于原来的主动追球行为。"""
    decision = tick(RallyController(defense_forward_limit=None),
                    observation(1, x=420, y=250, vy=200))
    assert decision.target == (420.0, 250.0)


def test_depth_limit_does_not_block_dead_ball_retrieve():
    """冰壶停在自己半场（死球）必须能回收 —— 深度上限不能让它够不着。"""
    for limit in (0.0, 90.0, None):
        controller = RallyController(defense_forward_limit=limit)
        decision = tick(controller, observation(1, x=420, y=320, vy=0))
        assert decision.state == RallyState.ATTACKING
        assert decision.target == (420.0, 320.0), f"limit={limit} 挡住了回收"


def test_human_half_unknown_velocity_prepares_defense_immediately():
    result = observation(1, x=420, y=500, vy=0)
    decision = tick(RallyController(), result)
    assert decision.state == RallyState.DEFENDING
    assert decision.target == (420, READY_POSITION[1])


def test_stationary_human_half_puck_does_not_flip_state_every_frame():
    """静止冰壶在人类半场时，状态与目标不能每帧翻转。

    回归：这一支原来判的是 `stone.vy <= 0`，而 stone.vy 是卡尔曼滤波的估计值 ——
    静止冰壶的 vy 会在 0 附近抖正负。vy 略正时走 else（状态留在 RECOVERING、
    target 保持 READY），vy 略负时走 if（状态 DEFENDING、target 换成 AI 目标），
    于是每帧翻转。

    实机 165 FPS 下实测：DEFENDING<->RECOVERING 各 119/118 次，
    目标 (300,140)<->(338,140) 交替 207/102 次；机械臂被反复中止而几乎不动。
    """
    controller = RallyController()
    current = (364, 344)          # 不在 READY 位，所以每帧会先被置为 RECOVERING
    states, targets = set(), set()
    for i in range(80):
        vy = 0.5 if i % 2 == 0 else -0.5      # 模拟卡尔曼估计在 0 附近抖动
        result = observation(1 + i * 0.01, x=338, y=536, vx=0.0, vy=vy, mallet=current)
        decision = tick(controller, result, mallet=current)
        states.add(decision.state)
        targets.add(decision.target)

    assert states == {RallyState.DEFENDING}, f"状态在每帧抖动：{states}"
    assert len(targets) == 1, f"目标在每帧抖动：{targets}"


def test_incoming_can_interrupt_recovery_without_ready_busy_or_rearm_wait():
    controller = RallyController()
    current = (480, 300)
    outgoing = observation(1, x=400, y=500, vy=200, mallet=current)
    assert tick(controller, outgoing, mallet=current).state == RallyState.RECOVERING
    incoming = observation(1.1, x=100, y=500, vy=-300, mallet=current)
    feedback = PLCFeedback(stamp=1.1, valid=True, x=current[0], y=current[1], busy=True)
    decision = tick(controller, incoming, feedback)
    assert decision.state == RallyState.DEFENDING
    assert decision.target == (100, READY_POSITION[1])
    assert decision.mallet_position == current and decision.mallet_source == "PLC"


def test_new_identity_and_predict_frame_do_not_delay_active_target():
    controller = RallyController()
    tick(controller, observation(1))
    result = observation(1.1, x=200, y=300, vy=0, identity=2, hard=False)
    decision = tick(controller, result)
    assert decision.state == RallyState.ATTACKING and decision.target == (200, 300)


def test_behind_mallet_puck_uses_existing_goal_side_retrieval():
    current = (300, 250)
    result = observation(1, x=300, y=180, vy=0, mallet=current)
    decision = tick(RallyController(), result, mallet=current)
    assert decision.state == RallyState.ATTACKING
    assert result.ai_decision.stalled_stone_phase == "sidestepping"
    assert decision.target == (353, 250)


def test_returned_human_half_puck_requests_ready_on_same_observation():
    controller = RallyController()
    tick(controller, observation(1, y=250, vy=0))
    current = (420, 250)
    result = observation(1.1, x=420, y=430, vy=200, mallet=current)
    decision = tick(controller, result, mallet=current)
    assert decision.state == RallyState.RECOVERING and decision.target == READY_POSITION
    again = observation(1.2, x=420, y=460, vy=200)
    assert tick(controller, again).state == RallyState.WAITING
    assert tick(controller, observation(1.3, x=100, y=500, vy=-300)).state == RallyState.DEFENDING


def test_missing_puck_returns_ready_without_timeout():
    controller = RallyController()
    result = VisionResult(Frame(np.zeros((20, 20, 3), np.uint8), timestamp=1))
    decision = tick(controller, result, mallet=(420, 250))
    assert decision.state == RallyState.RECOVERING and decision.target == READY_POSITION
    assert tick(controller, result).state == RallyState.WAITING


def test_real_detection_loss_stops_chasing_estimated_puck():
    result = observation(1, x=420, y=250, vy=0, confirmed=False)
    decision = tick(RallyController(), result, mallet=(420, 250))
    assert decision.state == RallyState.RECOVERING and decision.target == READY_POSITION


def test_stale_and_nonfinite_input_clears_motion_target():
    for field in ("x", "y", "vx", "vy", "timestamp"):
        for value in (math.nan, math.inf):
            result = observation(1)
            result.curling_state = replace(result.curling_state, **{field: value})
            assert tick(RallyController(), result).target is None
    for now in (0.5, 1.5):
        assert tick(RallyController(), observation(1), now=now).target is None
    for x, y in ((core.RINK_LEFT - 1, 300), (300, core.RINK_BOTTOM + 1)):
        result = observation(1)
        result.curling_state = replace(result.curling_state, x=x, y=y)
        assert tick(RallyController(), result).target is None


def test_missing_or_nonfinite_ai_target_cannot_be_sent():
    for decision in (None, replace(observation(1).ai_decision, target_x=math.nan)):
        result = observation(1)
        result.ai_decision = decision
        assert tick(RallyController(), result).target is None
        assert submit_rally_target(result, PlcControlAdapter(), None) is None


def test_active_target_needs_no_prediction_timing_or_reachability_gate():
    result = observation(1, x=480, y=500, vy=-1000)
    result.prediction = None
    decision = tick(RallyController(), result, mallet=(100, 300))
    assert decision.state == RallyState.DEFENDING
    assert decision.target == (480, READY_POSITION[1])


def test_dry_run_and_manual_plc_arm_remain_independent_of_play():
    class Transport:
        connected = True
        last_send_state = "idle"
        def __init__(self, *args):
            pass
    plc = PLCLink("offline", client_factory=Transport)
    adapter = PlcControlAdapter()
    result = observation(1, x=420, y=250, vy=0)
    tick(RallyController(), result)
    request = submit_rally_target(result, adapter, plc, dry_run=True)
    assert (request.target_x, request.target_y) == (420, 250)
    assert plc._target is None and not plc.armed
    submit_rally_target(result, adapter, plc)
    assert plc._target == (420, 250) and not plc.armed
    result.track_confirmed = False
    assert submit_rally_target(result, adapter, plc) is None
    assert plc._target is None
    result.rally = replace(result.rally, state=RallyState.RECOVERING, target=READY_POSITION)
    assert submit_rally_target(result, adapter, plc) is not None
    assert plc._target == READY_POSITION and not plc.armed
