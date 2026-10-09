"""回合阶段、真实检测确认、可达性和 PLC 路由安全。"""

from dataclasses import replace
import math

import numpy as np

from air_hockey import core_config as core
from air_hockey.ai import AIDecision
from air_hockey.app.rally import READY_POSITION, RallyController, RallyState
from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
from air_hockey.control import PLCLink, PlcControlAdapter
from air_hockey.control.plc import PLCFeedback
from air_hockey.prediction import PredictionState, TrajectoryPredictor
from air_hockey.vision.types import Detection, Track, TrackState
from game_state import CurlingState
from main import submit_rally_target


def observation(t, *, x=300.0, y=500.0, vx=0.0, vy=-300.0, hard=True,
                confirmed=True, identity=1):
    stone = CurlingState(x, y, vx, vy, timestamp=t)
    return VisionResult(
        frame=Frame(np.zeros((20, 20, 3), np.uint8), timestamp=t),
        detection=Detection(x, y, 14, 700, t) if hard else None,
        track=Track(identity, x, y, 14, vx=vx, vy=vy, last_timestamp=t,
                    state=TrackState.ACTIVE if confirmed else TrackState.TENTATIVE),
        track_confirmed=confirmed, curling_state=stone,
        prediction=TrajectoryPredictor().predict(stone),
        ai_decision=AIDecision(300, READY_POSITION[1], "idle"),
    )


def tick(controller, result, feedback=None, now=None):
    result.rally = controller.update(result, now=result.frame.timestamp if now is None else now,
                                     feedback=feedback)
    return result.rally


def returned(controller):
    def observed(t, **kwargs):
        feedback = PLCFeedback(stamp=t, valid=True, x=300, y=READY_POSITION[1]) if controller.hardware else None
        return tick(controller, observation(t, **kwargs), feedback)
    observed(1.0)
    observed(1.1, y=READY_POSITION[1] + 45)
    first = observed(1.12, y=READY_POSITION[1] + 45, vy=250)
    assert first.state == RallyState.DEFENDING
    assert first.target == READY_POSITION  # 不再追 outgoing。
    second = observed(1.18, y=READY_POSITION[1] + 60, vy=250)
    assert second.state == RallyState.RETURNED
    return second


def test_invalid_mallet_speed_is_rejected():
    for speed in (0, -1, math.nan, math.inf):
        try:
            RallyController(mallet_speed=speed)
        except ValueError:
            continue
        raise AssertionError(f"invalid mallet speed accepted: {speed}")


def test_nonfinite_puck_state_revokes_dynamic_intercept():
    for field in ("x", "y", "vx", "vy", "timestamp"):
        for value in (math.nan, math.inf):
            controller = RallyController()
            tick(controller, observation(1))
            result = observation(1.1)
            result.curling_state = replace(result.curling_state, **{field: value})
            decision = tick(controller, result)
            assert decision.state == RallyState.RECOVERING and decision.target is None


def test_normal_rally_recovers_and_accepts_next_incoming():
    controller = RallyController()
    returned(controller)
    assert tick(controller, observation(1.2, y=230, vy=250)).state == RallyState.RECOVERING
    assert tick(controller, observation(1.25, y=245, vy=250)).state == RallyState.WAITING
    assert tick(controller, observation(1.3, y=250, vy=-300)).state == RallyState.WAITING
    assert tick(controller, observation(1.35, y=500, vy=-300)).state == RallyState.DEFENDING
    assert controller.counters == dict(incoming_detected=2, defense_entered=2, intercept_attempted=2,
                                       successful_return=1, failed_intercept=0, recovered=1)


def test_tentative_outgoing_slow_and_stale_cannot_start_defense():
    for result in (observation(1, confirmed=False), observation(1, vy=300),
                   observation(1, vx=500, vy=-5), observation(1, vy=-core.STONE_STOP_SPEED)):
        controller = RallyController()
        decision = tick(controller, result)
        assert decision.state == RallyState.WAITING
        assert decision.target == READY_POSITION
        assert controller.counters["defense_entered"] == 0
    controller = RallyController()
    stale = tick(controller, observation(1), now=1.5)
    assert stale.state == RallyState.WAITING and stale.target is None
    future = tick(RallyController(), observation(2), now=1)
    assert future.target is None and future.state == RallyState.WAITING


def test_return_requires_real_observations_displacement_and_time():
    controller = RallyController()
    tick(controller, observation(1))
    tick(controller, observation(1.1, y=READY_POSITION[1] + 45))
    tick(controller, observation(1.12, y=READY_POSITION[1] + 45, vy=300))
    for t in (1.14, 1.16, 1.18):
        assert tick(controller, observation(t, y=205, vy=300, hard=False)).state == RallyState.DEFENDING
    assert tick(controller, observation(1.2, y=208, vy=300)).state == RallyState.RETURNED
    controller = RallyController()
    tick(controller, observation(2))
    tick(controller, observation(2.1, y=184, vy=300))
    assert tick(controller, observation(2.15, y=186, vy=300)).state == RallyState.DEFENDING
    # 速度符号扰动后要重新累积连续硬检测。
    tick(controller, observation(2.17, y=184, vy=-50))
    assert tick(controller, observation(2.2, y=190, vy=300)).state == RallyState.DEFENDING
    assert tick(controller, observation(2.26, y=210, vy=300)).state == RallyState.RETURNED


def test_wall_return_far_from_mallet_cannot_confirm_contact():
    controller = RallyController(hardware=True)
    feedback = PLCFeedback(stamp=1, valid=True, x=300, y=READY_POSITION[1])
    tick(controller, observation(1), feedback)
    for t, y in ((1.1, 185), (1.17, 205), (1.24, 225)):
        decision = tick(controller, observation(t, x=core.RINK_LEFT + core.STONE_RADIUS,
                                               y=y, vy=300), replace(feedback, stamp=t))
        assert decision.state == RallyState.DEFENDING
    assert controller.counters["successful_return"] == 0


def test_identity_change_and_loss_abort_old_interception():
    for next_result in (observation(1.1, identity=2), observation(1.1, confirmed=False)):
        controller = RallyController()
        tick(controller, observation(1))
        decision = tick(controller, next_result)
        assert decision.state == RallyState.RECOVERING
        assert decision.target is None
        assert controller.counters["failed_intercept"] == 1


def test_intercept_is_front_contact_not_endpoint_and_updates():
    controller = RallyController()
    result = observation(1, x=400, vx=-60, vy=-400)
    decision = tick(controller, result)
    assert decision.state == RallyState.DEFENDING and decision.reachable
    assert decision.target == decision.intercept
    assert decision.target[1] == READY_POSITION[1]
    assert decision.target != result.prediction.endpoint
    assert decision.time_to_mallet + core.DIFFICULTIES["普通"].reaction_delay <= decision.time_to_intercept
    updated = tick(controller, observation(1.1, x=420, vx=-20, vy=-400, hard=False))
    assert updated.state == RallyState.DEFENDING
    assert updated.intercept != decision.intercept
    assert core.RINK_LEFT + core.MALLET_RADIUS <= updated.target[0] <= core.RINK_RIGHT - core.MALLET_RADIUS


def test_unreachable_intercept_holds_ready_using_actual_feedback():
    controller = RallyController(mallet_speed=10, hardware=True)
    feedback = PLCFeedback(stamp=1, valid=True, x=300, y=READY_POSITION[1])
    result = observation(1, x=480, y=250, vy=-1000)
    decision = tick(controller, result, feedback)
    assert decision.state == RallyState.DEFENDING
    assert decision.intercept is not None and not decision.reachable
    assert decision.target == READY_POSITION
    assert decision.mallet_source == "PLC"
    assert decision.time_to_mallet > decision.time_to_intercept
    assert controller.counters["intercept_attempted"] == 0


def test_arrived_mallet_does_not_withdraw_at_contact_deadline():
    controller = RallyController(hardware=True)
    feedback = PLCFeedback(stamp=1, valid=True, x=300, y=READY_POSITION[1])
    early = tick(controller, observation(1, x=420, vy=-400), feedback)
    assert early.target == (420, READY_POSITION[1])
    for t, y in ((1.1, READY_POSITION[1] + 49), (1.12, READY_POSITION[1] + 40)):
        actual = replace(feedback, stamp=t, x=420)
        late = tick(controller, observation(t, x=420, y=y, vy=-400), actual)
        assert late.target == early.target and late.reachable
        assert late.time_to_intercept < core.DIFFICULTIES["普通"].reaction_delay


def test_single_outgoing_sign_holds_intercept_until_confirmed_return():
    controller = RallyController(hardware=True)
    feedback = PLCFeedback(stamp=1, valid=True, x=300, y=READY_POSITION[1])
    early = tick(controller, observation(1, x=420, vy=-400), feedback)
    tick(controller, observation(1.1, x=420, y=184.52, vy=-400), replace(feedback, stamp=1.1, x=420))
    pending = tick(controller, observation(1.12, x=420, y=185, vy=300), replace(feedback, stamp=1.12, x=420))
    assert pending.state == RallyState.DEFENDING and pending.target == early.target
    predicted = tick(controller, observation(1.15, x=450, y=215, vy=300, hard=False),
                     replace(feedback, stamp=1.15, x=420))
    assert predicted.state == RallyState.DEFENDING and predicted.target == early.target
    confirmed = tick(controller, observation(1.18, x=420, y=203, vy=300), replace(feedback, stamp=1.18, x=420))
    assert confirmed.state == RallyState.RETURNED and confirmed.target == READY_POSITION


def test_missing_or_mismatched_prediction_cannot_produce_intercept():
    for missing in ("prediction", "ai_decision"):
        result = observation(1)
        setattr(result, missing, None)
        decision = tick(RallyController(), result)
        assert decision.target == READY_POSITION and decision.intercept is None
    result = observation(1)
    result.prediction.source_state = replace(result.curling_state, timestamp=0.9)
    assert tick(RallyController(), result).intercept is None


def test_bounce_samples_do_not_create_fictitious_timestamp_spacing():
    result = observation(1, vy=-400)
    points = [(300, 500), (400, 450), (core.RINK_RIGHT - core.STONE_RADIUS, 300), (400, 100)]
    result.prediction = PredictionState(points, points[-1], 4, result.curling_state)
    decision = tick(RallyController(), result)
    line = READY_POSITION[1] + core.STONE_RADIUS + core.MALLET_RADIUS
    f = (line - 300) / (100 - 300)
    path = math.dist(points[0], points[1]) + math.dist(points[1], points[2]) + f * math.dist(points[2], points[3])
    assert math.isclose(decision.time_to_intercept, path / 400, abs_tol=1e-12)


def test_hardware_recovery_requires_actual_fresh_nonbusy_arrival():
    controller = RallyController(hardware=True)
    returned(controller)
    tick(controller, observation(1.2, vy=300))
    for t, feedback in ((1.3, None),
                        (1.4, PLCFeedback(stamp=1, valid=True, x=300, y=READY_POSITION[1])),
                        (1.5, PLCFeedback(stamp=1.5, valid=True, x=300, y=READY_POSITION[1], busy=True)),
                        (1.6, PLCFeedback(stamp=1.6, valid=True, x=450, y=READY_POSITION[1]))):
        assert tick(controller, observation(t, vy=300), feedback).state == RallyState.RECOVERING
    feedback = PLCFeedback(stamp=1.7, valid=True, x=300, y=READY_POSITION[1])
    assert tick(controller, observation(1.7, vy=300), feedback).state == RallyState.WAITING


def test_new_confirmed_identity_can_rearm_after_recovery():
    controller = RallyController()
    returned(controller)
    tick(controller, observation(1.2, y=230, vy=300))
    tick(controller, observation(1.25, y=250, vy=300))
    assert tick(controller, observation(1.3, y=300, identity=2)).state == RallyState.DEFENDING


def test_rally_timeout_returns_ready_and_counts_failed_intercept_once():
    controller = RallyController()
    tick(controller, observation(1))
    result = observation(6.1, x=450, y=250, vy=300)
    assert tick(controller, result).state == RallyState.RECOVERING
    assert result.rally.target == READY_POSITION
    tick(controller, observation(6.2, y=400, vy=300))
    assert controller.counters["failed_intercept"] == 1


def test_dry_run_and_unconfirmed_targets_cannot_reach_plc_motion():
    class Transport:
        connected = True
        last_send_state = "idle"
        def __init__(self, *args):
            pass
    plc = PLCLink("offline", client_factory=Transport)
    adapter = PlcControlAdapter()
    result = observation(1, x=410)
    tick(RallyController(), result)
    assert result.rally.state == RallyState.DEFENDING
    request = submit_rally_target(result, adapter, plc, dry_run=True)
    assert request is not None and plc._target is None and not plc.armed
    submit_rally_target(result, adapter, plc)
    assert plc._target == result.rally.target
    assert not plc.armed  # 发布意图不授予轴运动资格。
    result.track_confirmed = False
    assert submit_rally_target(result, adapter, plc) is None
    assert plc._target is None
    # 回位目标独立于 puck；仍只能通过未被绕过的 PLCLink 安全层。
    result.rally = replace(result.rally, state=RallyState.RECOVERING, target=READY_POSITION)
    assert submit_rally_target(result, adapter, plc) is not None
    assert plc._target == READY_POSITION and not plc.armed
