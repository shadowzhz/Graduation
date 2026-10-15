"""针对 VisionRuntime 及其数据流的单元测试。

验证链路：
frame -> detection -> track -> table StoneState -> predictor -> AI
"""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import math
import sys
import tempfile
import time
import cv2
import numpy as np

from air_hockey import core_config as core
from air_hockey.ai import AIDecision, AirHockeyAI
from air_hockey.app.rally import (
    DEFAULT_DEFENSE_FORWARD_LIMIT,
    READY_POSITION,
    RallyController,
    RallyState,
)
from air_hockey.app.vision_runtime import AI_HOME_Y, VisionResult, VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.control import PlcControlAdapter
from air_hockey.control.plc import PLCFeedback
from air_hockey.prediction import TrajectoryPredictor
from game_state import CurlingState, GameState, StoneState
from air_hockey.vision.tracker import StoneTracker
from air_hockey.vision.types import Detection, TrackState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


class MockDetector:
    def __init__(self, detections):
        self._detections = list(detections)
        self._index = 0
        self._frame = None
        self._cached = None

    def detect(self, frame, dynamic_roi=None):
        if self._frame is frame:
            return self._cached
        self._frame = frame
        self._cached = None
        if self._index < len(self._detections):
            self._cached = self._detections[self._index]
            self._index += 1
        return self._cached


class CountingPredictor(TrajectoryPredictor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, stone, *args, **kwargs):
        self.calls += 1
        return super().predict(stone, *args, **kwargs)


class FixedKalman:
    def __init__(self, x, y):
        self.state = CurlingState(x, y, vx=180.0, vy=-250.0)

    def update(self, *args, **kwargs):
        return self.state


def test_stone_tracker_predict_position_public():
    """测试 StoneTracker 的公开接口 predict_position(timestamp)。"""
    tracker = StoneTracker()
    tracker.update(Detection(center_x=100.0, center_y=200.0, radius=25.0, area=1960.0, timestamp=1.0))
    tracker.update(Detection(center_x=120.0, center_y=210.0, radius=25.0, area=1960.0, timestamp=1.1))

    # vx = (120 - 100) / 0.1 * 0.2 = 40.0
    # vy = (210 - 200) / 0.1 * 0.2 = 20.0
    pred_x, pred_y = tracker.predict_position(1.2)
    assert pred_x > 120.0
    assert pred_y > 210.0


def test_default_detector_finds_red_puck_on_both_hue_ends():
    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True)

    for hue, saturation, value, expected in (
        (5, 157, 186, True), (178, 180, 135, True), (90, 180, 135, False),
    ):
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        red = cv2.cvtColor(
            np.array([[[hue, saturation, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.circle(image, (715, 200), 35, tuple(int(channel) for channel in red), -1)
        detection = runtime.detector.detect(Frame(image))

        assert (detection is not None) is expected
        if expected:
            assert abs(detection.center_x - 715) < 1.0
            assert abs(detection.center_y - 200) < 1.0
            assert detection.radius >= 25

    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    red = cv2.cvtColor(np.array([[[1, 167, 166]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    color = tuple(int(channel) for channel in red)
    line_only = np.full((720, 1280, 3), 255, dtype=np.uint8)
    cv2.line(line_only, (540, 0), (540, 650), color, 7)
    assert runtime.detector.detect(Frame(line_only)) is None

    cv2.line(image, (540, 140), (540, 190), color, 7)
    cv2.circle(image, (580, 190), 39, color, -1)
    detection = runtime.detector.detect(Frame(image))

    assert detection is not None
    assert abs(detection.center_x - 580) < 15
    assert abs(detection.center_y - 190) < 15


def test_vision_runtime_dataflow_track_to_ai():
    """验证 track -> table StoneState -> predictor -> AI 的完整调用链。"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    det1 = Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)
    det2 = Detection(center_x=520.0, center_y=310.0, radius=25.0, area=1960.0, timestamp=1.05)

    detector = MockDetector([det1, det2])
    runtime = VisionRuntime(
        detection_interval=3,
        tracker=StoneTracker(min_confirmed_hits=2),
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=detector,
    )

    frame1 = Frame(image=img, timestamp=1.0, sequence=1)
    res1 = runtime.process_frame(frame1)

    assert isinstance(res1, VisionResult)
    assert res1.detection is not None
    assert res1.detection.center_x == 500.0
    assert res1.track is not None
    assert res1.track.center_x == 500.0
    assert res1.track.state == TrackState.TENTATIVE
    assert not res1.track_confirmed

    # 验证经过坐标变换生成的 StoneState
    assert isinstance(res1.stone_state, StoneState)
    assert res1.stone_state.radius == 25.0

    # 验证 TrajectoryPredictor 输出的桌面轨迹
    assert res1.trajectory is not None
    assert len(res1.trajectory) >= 1

    # 验证 AI 决策输出的目标坐标
    assert res1.ai_target is not None
    assert len(res1.ai_target) == 2
    assert isinstance(res1.ai_target[0], float)
    assert isinstance(res1.ai_target[1], float)

    # 处理第二帧：检测间隔帧，Tracker 进行常速预测
    frame2 = Frame(image=img, timestamp=1.033, sequence=2)
    res2 = runtime.process_frame(frame2)
    assert res2.detection is None
    assert res2.track is not None
    assert res2.stone_state is not None
    assert res2.trajectory is not None
    assert res2.ai_target is not None

    # 处理第三帧：第 3 帧到达检测周期 (frame_index % 3 == 0)，执行硬检测并更新速度
    frame3 = Frame(image=img, timestamp=1.066, sequence=3)
    res3 = runtime.process_frame(frame3)

    assert res3.detection is not None
    assert res3.track is not None
    assert res3.track.state == TrackState.ACTIVE
    assert res3.track_confirmed
    assert res3.track.vx > 0
    assert res3.stone_state is not None
    assert res3.stone_state.vx > 0
    assert res3.trajectory is not None
    assert len(res3.trajectory) >= 1
    assert res3.ai_target is not None
    assert res3.fps > 0.0


def _assert_vision_runtime_threat_predicts_once(stone_y):
    predictor = CountingPredictor()
    runtime = VisionRuntime(
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=MockDetector([Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)]),
        kalman=FixedKalman(260.0, stone_y),
        predictor=predictor,
    )
    runtime.ai._random.seed(120)
    result = runtime.process_frame(Frame(image=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=1.0, sequence=1))

    assert predictor.calls == 1
    assert result.prediction is not None
    assert result.prediction.source_state is result.curling_state
    assert isinstance(result.ai_decision, AIDecision)
    assert result.ai_target == (result.ai_decision.target_x, result.ai_decision.target_y)
    assert abs(result.ai_decision.reaction_timer + 0.001) < 1e-9

    baseline_ai = AirHockeyAI()
    baseline_ai._random.seed(120)
    baseline_state = GameState(
        ai_x=core.RINK_CENTER_X,
        ai_y=AI_HOME_Y,
        ai_home_y=AI_HOME_Y,
        target_x=core.RINK_CENTER_X,
        target_y=AI_HOME_Y,
        stone=result.curling_state,
        awaiting_serve=False,
        current_server="player",
        serve_phase="idle",
        stalled_stone_phase="idle",
        reaction_timer=0.0,
        difficulty=runtime._ai_difficulty,
    )
    baseline = baseline_ai.update(baseline_state, 0.001)
    assert result.ai_target == (baseline.target_x, baseline.target_y)
    assert result.ai_decision.stalled_stone_phase == baseline.stalled_stone_phase


def test_realtime_ai_target_stays_fixed_for_a_stationary_puck():
    timestamps = [1.0 + 0.1 * index for index in range(5)]
    detection = lambda timestamp: Detection(
        center_x=640.0, center_y=360.0, radius=30.0, area=2800.0, timestamp=timestamp
    )
    runtime = VisionRuntime(
        table_calibration_file=None,
        disable_undistort=True,
        detection_interval=1,
        detector=MockDetector([detection(timestamp) for timestamp in timestamps]),
    )
    runtime.ai._random.seed(120)
    results = [
        runtime.process_frame(
            Frame(np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=timestamp, sequence=index + 1)
        )
        for index, timestamp in enumerate(timestamps)
    ]

    targets = [result.ai_target for result in results]
    assert all(target is not None for target in targets)
    assert all(target == targets[0] for target in targets)


def test_vision_runtime_threat_behind_ai_predicts_once():
    _assert_vision_runtime_threat_predicts_once(AI_HOME_Y - 50.0)


def test_default_runtime_accepts_and_moves_on_first_machine_half_detection():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for x, y, vy in (
        (260.0, 250.0, 0.0),
        (260.0, core.RINK_CENTER_Y - 50.0, 100.0),
        (core.RINK_CENTER_X, AI_HOME_Y - 30.0, 0.0),
    ):
        estimator = FixedKalman(x, y)
        estimator.state = CurlingState(x, y, vx=0.0, vy=vy)
        runtime = VisionRuntime(
            table_calibration_file=None, disable_undistort=True, kalman=estimator,
            detector=MockDetector([
                Detection(center_x=500.0, center_y=300.0, radius=25.0,
                          area=1960.0, timestamp=timestamp)
                for timestamp in (1.0, 1.01)
            ]),
        )
        result = runtime.process_frame(Frame(image, timestamp=1.0))
        assert result.track.state == TrackState.ACTIVE and result.track_confirmed
        assert result.ai_target != READY_POSITION
        if y < AI_HOME_Y:
            assert result.ai_decision.stalled_stone_phase == "sidestepping"
            assert result.ai_target[1] == AI_HOME_Y
        else:
            assert result.ai_target == (x, y)
        second = runtime.process_frame(Frame(image, timestamp=1.01))
        assert second.detection is not None and second.track_confirmed


def test_explicit_two_hit_tracker_keeps_first_detection_tentative():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True,
        tracker=StoneTracker(min_confirmed_hits=2),
        detector=MockDetector([
            Detection(center_x=500.0, center_y=300.0, radius=25.0,
                      area=1960.0, timestamp=1.0),
        ]),
    )
    result = runtime.process_frame(Frame(np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=1.0))

    assert result.track is not None
    assert result.track.state == TrackState.TENTATIVE
    assert not result.track_confirmed


def test_explicit_two_hit_tracker_confirms_on_detection_not_prediction():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True, detection_interval=3,
        tracker=StoneTracker(min_confirmed_hits=2),
        detector=MockDetector([
            Detection(center_x=500.0, center_y=300.0, radius=25.0,
                      area=1960.0, timestamp=1.0),
            Detection(center_x=501.0, center_y=300.0, radius=25.0,
                      area=1960.0, timestamp=1.066),
        ]),
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = []
    states = []
    for index in range(3):
        result = runtime.process_frame(Frame(image, timestamp=1.0 + index * 0.033))
        results.append(result)
        states.append(result.track.state)

    assert states == [TrackState.TENTATIVE, TrackState.TENTATIVE, TrackState.ACTIVE]
    assert results[1].detection is None
    assert results[2].detection is not None
    assert [result.track_confirmed for result in results] == [False, False, True]


def test_new_far_identity_is_immediately_confirmed():
    detections = [
        Detection(center_x=x, center_y=300.0, radius=25.0, area=1960.0,
                  timestamp=1.0 + index * 0.033)
        for index, x in enumerate((500.0, 501.0, 900.0, 901.0))
    ]
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True, detection_interval=1,
        detector=MockDetector(detections),
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = []
    identities = []
    states = []
    for index in range(len(detections)):
        result = runtime.process_frame(Frame(image, timestamp=1.0 + index * 0.033))
        results.append(result)
        identities.append(result.track.track_id)
        states.append(result.track.state)

    assert identities == [1, 1, 2, 2]
    assert states == [TrackState.ACTIVE] * 4
    assert [result.track_confirmed for result in results] == [True] * 4


def test_vision_runtime_threat_outside_attack_zone_predicts_once():
    _assert_vision_runtime_threat_predicts_once(500.0)


def test_vision_runtime_supports_custom_ai_without_prediction_parameter():
    class IndependentAI:
        def update(self, state, dt):
            return AIDecision(225.0, 155.0, "striking", 0.35)

    runtime = VisionRuntime(
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=MockDetector([Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)]),
        ai=IndependentAI(),
    )
    result = runtime.process_frame(Frame(image=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=1.0, sequence=1))
    assert result.ai_decision == AIDecision(225.0, 155.0, "striking", 0.35)
    assert result.ai_target == (225.0, 155.0)
    assert runtime.stalled_phase == "striking"
    assert runtime.reaction_timer == 0.35


def test_vision_runtime_ai_uses_fresh_axis_feedback_not_software_position():
    class CapturingAI:
        def __init__(self):
            self.positions = []

        def update(self, state, dt):
            self.positions.append((state.ai_x, state.ai_y))
            return AIDecision(400.0, 200.0, "idle", 0.0)

    ai = CapturingAI()
    runtime = VisionRuntime(
        table_calibration_file=None,
        disable_undistort=True,
        detector=MockDetector([Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)]),
        ai=ai,
    )
    frame = lambda sequence, timestamp: Frame(
        image=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=timestamp, sequence=sequence
    )
    runtime.set_ai_feedback(SimpleNamespace(valid=True, stamp=time.monotonic(), x=210.0, y=160.0))
    runtime.process_frame(frame(1, 1.0))
    assert ai.positions[0] == (210.0, 160.0)
    assert runtime.ai_current_pos == [210.0, 160.0]

    runtime.set_ai_feedback(SimpleNamespace(valid=True, stamp=time.monotonic() - 1, x=999.0, y=999.0))
    runtime.process_frame(frame(2, 1.05))
    assert ai.positions[1] == (210.0, 160.0)
    assert runtime.ai_current_pos != [999.0, 999.0]


def test_plc_track_confirmation_survives_one_missed_detection():
    """漏检一帧时 PLC 侧确认**保持**（宽限期内），状态估计继续可用。

    ★ 这是"接不到球"的直接回归点：原来这里 results[2].track_confirmed 是 False，
    rally 会立刻切 RECOVERING、把目标设回 READY，机械臂被叫回去。
    """
    detector = MockDetector([
        Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0),
        Detection(center_x=501.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.03),
        None,
        Detection(center_x=501.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.15),
    ])
    runtime = VisionRuntime(
        calibration_file=str(CALIB_FILE), table_calibration_file=None,
        disable_undistort=True, detection_interval=1, detector=detector,
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = [runtime.process_frame(Frame(image=image, timestamp=1 + (i - 1) * 0.03, sequence=i))
               for i in range(1, 5)]
    assert results[0].track_confirmed
    assert results[1].track_confirmed
    # ★ 漏检那一帧：AI 仍然给出决策，而且确认保持
    assert results[2].ai_decision is not None and results[2].track_confirmed
    assert results[3].track_confirmed
    # tracking_state 仍如实报告"这一帧没有新观测"
    assert [result.curling_state.tracking_state.value for result in results] == [
        "active", "active", "lost", "active",
    ]


def test_vision_runtime_without_detection():
    """验证未检测到目标时，调用链输出空状态并保持稳健。"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = MockDetector([])
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=detector,
    )

    frame = Frame(image=img, timestamp=1.0, sequence=1)
    result = runtime.process_frame(frame)

    assert isinstance(result, VisionResult)
    assert result.detection is None
    assert result.track is None
    assert result.stone_state is None
    assert result.trajectory is None
    assert result.ai_target == READY_POSITION
    assert result.ai_decision is None


def test_low_fps_moving_puck_keeps_identity_and_uses_2d_target_within_depth_limit():
    from main import submit_rally_target

    runtime = VisionRuntime(
        table_roi=(0, 0, 1280, 720), table_calibration_file=None, disable_undistort=True,
        detection_interval=3,
    )
    controller = RallyController()
    adapter = PlcControlAdapter()
    identities = []
    active_targets = []
    states = []
    for sequence, y in enumerate((600, 600, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150), 1):
        timestamp = 10.0 + (sequence - 1) * 0.1
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (900, y), 30, (0, 0, 255), -1)
        result = runtime.process_frame(Frame(image, timestamp=timestamp, sequence=sequence))
        identities.append(result.track.track_id)
        assert result.detection is not None
        assert result.track.state == TrackState.ACTIVE and result.track_confirmed
        feedback = PLCFeedback(stamp=timestamp, valid=True, x=READY_POSITION[0], y=READY_POSITION[1])
        result.rally = controller.update(result, now=timestamp + 0.001, feedback=feedback)
        request = submit_rally_target(result, adapter, None)
        states.append(result.rally.state)
        if result.rally.state in (RallyState.DEFENDING, RallyState.ATTACKING):
            if sequence <= 3:
                assert result.curling_state.vy == 0.0
                assert result.rally.state == RallyState.DEFENDING
                assert result.ai_target == (result.curling_state.x, READY_POSITION[1])
            else:
                assert result.curling_state.vy < -25
            assert request is not None
            # 默认不限制深入（现场选的是"全部追"），x/y 都跟 AI 目标走。
            # ⚠️ 回归点：默认曾经是 90，把 Y 卡在 AI_HOME_Y+90 = 229.52，
            # 而 ai.py 允许到 349 —— 少了 111mm 前向行程，症状是"左右能到极限、Y 到不了"。
            assert DEFAULT_DEFENSE_FORWARD_LIMIT is None
            assert request.target_x == result.ai_target[0]
            assert request.target_y == result.ai_target[1]
            assert request.target_x > READY_POSITION[0] + core.MALLET_RADIUS
            active_targets.append((request.target_x, request.target_y))
    assert set(identities) == {1}
    assert RallyState.ATTACKING in states and RallyState.DEFENDING in states
    # 关键回归点：Y 必须真的在动，而且要用满 ai.py 允许的行程。
    ys = {round(y, 2) for _x, y in active_targets}
    assert len(ys) > 1, f"Y 没有动，只有 {ys}"
    assert max(ys) > AI_HOME_Y + 100.0, f"Y 的前向行程被压掉了：最大只有 {max(ys)}"
    targets = {(round(x, 2), round(y, 2)) for x, y in active_targets}
    assert (407.25, READY_POSITION[1]) in targets


def test_low_fps_real_miss_keeps_confirmation_within_grace_period():
    """漏检一帧**不再撤销**确认 —— 只要还在 max_missed_frames 宽限期内。

    ★ 回归现场问题：追踪器在漏检第一帧就把 state 置 LOST，而 track_confirmed
    原来绑在 `state == ACTIVE` 上 → 漏检一帧就 False → rally 每两帧切回 READY
    → 机械臂刚起步就被叫回 (300,140) → **接不到球**。
    宽限期内位置由 Kalman 外推，对"球往哪走"的判断依然有效。
    """
    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True)
    states = []
    confirmed = []
    for sequence, visible in enumerate((True, False, True, True), 1):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        if visible:
            cv2.circle(image, (640, 360), 30, (0, 0, 255), -1)
        result = runtime.process_frame(Frame(image, timestamp=1.0 + (sequence - 1) * 0.1))
        states.append(result.track.state)
        confirmed.append(result.track_confirmed)
    # state 仍然是 LOST（追踪器语义没动），但确认必须保持
    assert states == [TrackState.ACTIVE, TrackState.LOST, TrackState.ACTIVE, TrackState.ACTIVE]
    assert confirmed == [True, True, True, True]


def test_confirmation_is_revoked_after_the_grace_period_expires():
    """宽限期用完、轨道被真正丢弃之后必须撤销确认 —— 不能永远认为有球。"""
    detector = MockDetector([
        Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0),
    ] + [None] * 20)
    runtime = VisionRuntime(
        calibration_file=str(CALIB_FILE), table_calibration_file=None,
        disable_undistort=True, detection_interval=1, detector=detector,
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = [runtime.process_frame(Frame(image=image, timestamp=1 + i * 0.03, sequence=i))
               for i in range(1, 22)]
    assert results[0].track_confirmed
    # 宽限期内（detection_interval * 8 = 8 帧）仍然确认
    assert results[5].track_confirmed
    assert results[7].track_confirmed
    # 超过宽限期之后必须撤销，且轨道被丢弃
    assert not results[20].track_confirmed
    assert results[20].track is None


def test_fast_reversal_reacquires_outside_local_roi_without_confirmation_wait():
    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True)
    old_identity = None
    for sequence, y in enumerate((500, 450, 570), 1):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (640, y), 30, (0, 0, 255), -1)
        result = runtime.process_frame(Frame(image, timestamp=1.0 + (sequence - 1) * 0.1))
        if sequence == 2:
            old_identity = result.track.track_id
    assert abs(result.detection.center_y - 570) < 1
    assert result.track.track_id != old_identity
    assert result.track.state == TrackState.ACTIVE and result.track.hits == 1
    assert result.track_confirmed
    expected = runtime.camera_geometry.undistorted_to_table(640, 570)
    assert math.dist(result.curling_state.position, expected) < 1e-6


def test_real_detector_misses_advance_clock_and_new_identity_resets_estimator():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True, detection_interval=1,
    )
    results = []
    for sequence, x in enumerate([500, 503, 506, 509, 512, None, None, None, 800], 1):
        hsv = np.zeros((720, 1280, 3), dtype=np.uint8)
        if x is not None:
            cv2.circle(hsv, (x, 300), 30, (175, 255, 255), -1)
        frame = Frame(
            image=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
            timestamp=1.0 + (sequence - 1) * 0.005, sequence=sequence,
        )
        results.append(runtime.process_frame(frame))

    assert results[4].curling_state.vx > 0.0
    for previous, missed in zip(results[4:7], results[5:8]):
        assert missed.detection is None
        assert missed.track.track_id == 1
        assert missed.curling_state.timestamp == missed.frame.timestamp
        assert missed.curling_state.x > previous.curling_state.x
        assert math.isclose(missed.curling_state.x,
                            previous.curling_state.x + previous.curling_state.vx * 0.005)
        assert math.isclose(missed.curling_state.vx, previous.curling_state.vx)

    new = results[-1]
    assert new.track.track_id == 2
    undistorted = runtime.camera_geometry.raw_to_undistorted(800.0, 300.0)
    expected = runtime.camera_geometry.undistorted_to_table(*undistorted)
    assert math.dist(new.curling_state.position, expected) < 1e-6
    assert new.curling_state.timestamp == new.frame.timestamp
    assert new.curling_state.vx == 0.0
    assert new.curling_state.vy == 0.0


def test_estimator_reset_on_no_track_preserves_new_observation_initialization():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True, detection_interval=1,
        tracker=StoneTracker(max_missed_frames=0),
        detector=MockDetector([
            Detection(center_x=500.0, center_y=300.0, radius=30.0, area=2800.0, timestamp=1.0),
            Detection(center_x=503.0, center_y=300.0, radius=30.0, area=2800.0, timestamp=1.005),
            None,
            Detection(center_x=800.0, center_y=300.0, radius=30.0, area=2800.0, timestamp=1.015),
        ]),
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    runtime.process_frame(Frame(image=image, timestamp=1.0))
    moving = runtime.process_frame(Frame(image=image, timestamp=1.005))
    assert moving.curling_state.vx > 0.0
    runtime.stalled_phase = "striking"
    runtime.reaction_timer = 0.35
    missing = runtime.process_frame(Frame(image=image, timestamp=1.01))
    assert missing.track is None and missing.curling_state is None
    assert missing.ai_target == READY_POSITION
    assert runtime.stalled_phase == "idle" and runtime.reaction_timer == 0.0
    new = runtime.process_frame(Frame(image=image, timestamp=1.015))
    expected = runtime.camera_geometry.undistorted_to_table(800.0, 300.0)
    assert math.dist(new.curling_state.position, expected) < 1e-6
    assert new.curling_state.vx == new.curling_state.vy == 0.0


def test_new_identity_does_not_inherit_recovery_phase_or_reaction_timer():
    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(image, (500, 300), 30, (0, 0, 255), -1)
    first = runtime.process_frame(Frame(image, timestamp=1.0))
    old_identity = first.track.track_id
    runtime.stalled_phase = "striking"
    runtime.reaction_timer = 10.0
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(image, (900, 220), 30, (0, 0, 255), -1)
    new = runtime.process_frame(Frame(image, timestamp=1.1))
    assert new.track.track_id != old_identity and new.track_confirmed
    assert new.ai_target == new.curling_state.position
    assert new.ai_decision.stalled_stone_phase == "idle"


def test_no_puck_updates_feedback_pose_and_smoothly_returns_fallback_to_ready():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True,
    )
    runtime.stalled_phase = "striking"
    runtime.reaction_timer = 0.35
    runtime.target[:] = (400.0, 200.0)
    runtime.set_ai_feedback(SimpleNamespace(valid=True, stamp=time.monotonic(), x=210.0, y=160.0))
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = runtime.process_frame(Frame(image, timestamp=1.0))
    assert result.track is None and result.ai_decision is None
    assert result.ai_target == READY_POSITION
    assert runtime.ai_current_pos == [210.0, 160.0]

    runtime.set_ai_feedback(None)
    for index in range(1, 31):
        result = runtime.process_frame(Frame(image, timestamp=1.0 + index * 0.02))
    assert result.ai_target == READY_POSITION
    assert math.dist(runtime.ai_current_pos, READY_POSITION) < core.STONE_RADIUS
    cv2.circle(image, (900, 220), 30, (0, 0, 255), -1)
    new = runtime.process_frame(Frame(image, timestamp=1.7))
    assert new.ai_target == new.curling_state.position


def test_camera_calibration_capture_failure_exits_and_cleans_up():
    from air_hockey.tools import calibrate_camera as tool

    failure = RuntimeError('GStreamer capture failed')

    class Camera:
        error = None
        stopped = False

        def start(self):
            if failure_stage == 'start':
                raise failure

        def get_latest_frame(self):
            self.error = failure
            return Frame(np.zeros((240, 320, 3), dtype=np.uint8), sequence=1)

        def stop(self):
            self.stopped = True

    for failure_stage in ('start', 'capture'):
        camera = Camera()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as patches:
            output = Path(tmp) / 'calibration.npz'
            patches.enter_context(patch.object(sys, 'argv', ['calibrate_camera.py', '--output', str(output)]))
            patches.enter_context(patch.object(tool, 'CameraManager', return_value=camera))
            for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
                patches.enter_context(patch.object(tool.cv2, name, return_value=None))
            patches.enter_context(patch.object(tool.cv2, 'waitKey', return_value=-1))
            with TestCase().assertRaises(RuntimeError) as caught:
                tool.main()
            assert camera.stopped and not output.exists()
            if failure_stage == 'capture':
                assert caught.exception.__cause__ is failure
            else:
                assert caught.exception is failure


def test_camera_calibration_duplicate_sequence_is_not_resampled_and_quit_responds():
    from air_hockey.tools import calibrate_camera as tool

    image = np.full((300, 420, 3), 255, dtype=np.uint8)
    for row in range(8):
        for col in range(11):
            if (row + col) % 2 == 0:
                image[30 + row * 24:30 + (row + 1) * 24,
                      30 + col * 24:30 + (col + 1) * 24] = 0
    shifted = np.roll(image, 60, axis=1)
    frames = iter([Frame(image, sequence=7), Frame(shifted, sequence=7)])
    keys = iter([-1, -1, ord('q')])
    clock = iter([1.0, 2.0])
    calibrator = tool.CameraCalibrator((10, 7), 25)

    class Camera:
        error = None
        stopped = False

        def start(self):
            pass

        def get_latest_frame(self):
            return next(frames)

        def stop(self):
            self.stopped = True

    camera = Camera()
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as patches:
        output = Path(tmp) / 'calibration.npz'
        patches.enter_context(patch.object(sys, 'argv', ['calibrate_camera.py', '--output', str(output)]))
        patches.enter_context(patch.object(tool, 'CameraManager', return_value=camera))
        patches.enter_context(patch.object(tool, 'CameraCalibrator', return_value=calibrator))
        patches.enter_context(patch.object(tool.time, 'monotonic', side_effect=lambda: next(clock)))
        for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
            patches.enter_context(patch.object(tool.cv2, name, return_value=None))
        patches.enter_context(patch.object(tool.cv2, 'waitKey', side_effect=lambda delay: next(keys)))
        tool.main()
        assert camera.stopped and not output.exists()
    assert calibrator.sample_count == 1


def test_adaptive_roi_stationary_puck_local_detection_maintains_track():
    """静止冰壶动态 ROI 包含目标并由局部 detect_candidates 找回，维持同一轨迹。"""
    from air_hockey.app.vision_runtime import _compute_dynamic_roi

    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True, detection_interval=1)
    track_id = None

    for i in range(3):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (600, 350), 30, (0, 0, 255), -1)
        t = 1.0 + i * 0.05
        frame = Frame(image, timestamp=t)

        if runtime.tracker.track is not None:
            pred_pos = runtime.tracker.predict_position(t)
            roi = _compute_dynamic_roi(runtime.tracker.track, pred_pos, 1280, 720)
            assert roi is not None
            assert roi.x <= 600 - 30 and roi.x + roi.width >= 600 + 30
            assert roi.y <= 350 - 30 and roi.y + roi.height >= 350 + 30
            candidates = runtime.detector.detect_candidates(frame, dynamic_roi=roi)
            assert len(candidates) > 0
            assert abs(candidates[0].x - 600) < 5

        result = runtime.process_frame(frame)
        assert result.detection is not None
        assert abs(result.detection.center_x - 600) < 5
        if track_id is None:
            track_id = result.track.track_id
        else:
            assert result.track is not None
            assert result.track.track_id == track_id
            assert result.track_confirmed


def test_adaptive_roi_fast_puck_and_dropped_frames_reacquired_locally():
    """高速球与掉帧时根据速度和时间差扩展搜索框，局部候选找回并维持轨迹。"""
    from air_hockey.app.vision_runtime import _compute_dynamic_roi

    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True, detection_interval=1)

    track_id = None
    x = 200
    t = 1.00
    for i in range(8):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (x, 300), 30, (0, 0, 255), -1)
        res = runtime.process_frame(Frame(image, timestamp=t))
        if track_id is None:
            track_id = res.track.track_id
        else:
            assert res.track.track_id == track_id
        last_x = x
        x += 35
        t += 0.035

    last_t = t - 0.035
    t_drop = last_t + 0.08
    x_drop = last_x + 60
    image_drop = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(image_drop, (x_drop, 300), 30, (0, 0, 255), -1)
    frame_drop = Frame(image_drop, timestamp=t_drop)

    pred_pos = runtime.tracker.predict_position(t_drop)
    roi = _compute_dynamic_roi(runtime.tracker.track, pred_pos, 1280, 720)
    assert roi is not None
    assert roi.width > 140
    assert roi.x <= x_drop - 30 and roi.x + roi.width >= x_drop + 30
    assert roi.x <= last_x - 30

    candidates = runtime.detector.detect_candidates(frame_drop, dynamic_roi=roi)
    assert len(candidates) > 0
    assert abs(candidates[0].x - x_drop) < 5

    res_drop = runtime.process_frame(frame_drop)
    assert res_drop.detection is not None
    assert abs(res_drop.detection.center_x - x_drop) < 5
    assert res_drop.track is not None and res_drop.track.track_id == track_id
    assert res_drop.track_confirmed


def test_adaptive_roi_fast_reversal_contains_previous_observation():
    """快速反向时 ROI 同时包含上一观测点与预测位置，局部找回并维持轨迹。"""
    from air_hockey.app.vision_runtime import _compute_dynamic_roi

    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True, detection_interval=1)

    track_id = None
    x = 200
    t = 1.00
    for i in range(10):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (x, 300), 30, (0, 0, 255), -1)
        res = runtime.process_frame(Frame(image, timestamp=t))
        if track_id is None:
            track_id = res.track.track_id
        else:
            assert res.track.track_id == track_id
        last_x = x
        x += 35
        t += 0.035

    last_t = t - 0.035
    rev_t = last_t + 0.08
    rev_x = 520
    image_rev = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(image_rev, (rev_x, 300), 30, (0, 0, 255), -1)
    frame_rev = Frame(image_rev, timestamp=rev_t)

    pred_pos = runtime.tracker.predict_position(rev_t)
    roi = _compute_dynamic_roi(runtime.tracker.track, pred_pos, 1280, 720)
    old_rx = int(pred_pos[0] - 70)

    assert old_rx > rev_x - 30
    assert roi is not None
    assert roi.x <= rev_x - 30 and roi.x + roi.width >= rev_x + 30
    assert roi.x <= last_x - 30

    candidates = runtime.detector.detect_candidates(frame_rev, dynamic_roi=roi)
    assert len(candidates) > 0
    assert abs(candidates[0].x - rev_x) < 5

    res_rev = runtime.process_frame(frame_rev)
    assert res_rev.detection is not None
    assert abs(res_rev.detection.center_x - rev_x) < 5
    assert res_rev.track is not None and res_rev.track.track_id == track_id
    assert res_rev.track_confirmed

def test_adaptive_roi_boundary_and_edge_cases():
    """测试图像边缘裁剪、非有限值保护和原框下限保护。"""
    from air_hockey.app.vision_runtime import _compute_dynamic_roi
    from air_hockey.vision.types import Track

    track_edge = Track(1, center_x=30.0, center_y=40.0, radius=25.0, vx=-200.0, vy=-200.0, last_timestamp=1.0)
    roi_edge = _compute_dynamic_roi(track_edge, pred_pos=(10.0, 20.0), img_w=1280, img_h=720)
    assert roi_edge is not None
    assert roi_edge.x == 0
    assert roi_edge.y == 0

    track_rb = Track(2, center_x=1260.0, center_y=700.0, radius=25.0, vx=100.0, vy=100.0, last_timestamp=1.0)
    roi_rb = _compute_dynamic_roi(track_rb, pred_pos=(1270.0, 710.0), img_w=1280, img_h=720)
    assert roi_rb is not None
    assert roi_rb.x + roi_rb.width <= 1280
    assert roi_rb.y + roi_rb.height <= 720

    assert _compute_dynamic_roi(track_edge, pred_pos=(float('nan'), 20.0), img_w=1280, img_h=720) is None
    assert _compute_dynamic_roi(track_edge, pred_pos=(10.0, float('inf')), img_w=1280, img_h=720) is None
    assert _compute_dynamic_roi(None, pred_pos=(10.0, 20.0), img_w=1280, img_h=720) is None

    track_nan_obs = Track(3, center_x=float('nan'), center_y=float('nan'), radius=25.0, vx=0.0, vy=0.0, last_timestamp=1.0)
    roi_safe = _compute_dynamic_roi(track_nan_obs, pred_pos=(500.0, 300.0), img_w=1280, img_h=720)
    assert roi_safe is not None

    orig_rx = max(0, min(int(500.0 - 70), 1279))
    track_fast = Track(4, center_x=300.0, center_y=300.0, radius=30.0, vx=2000.0, vy=0.0, last_timestamp=1.0)
    roi_fast = _compute_dynamic_roi(track_fast, pred_pos=(500.0, 300.0), img_w=1280, img_h=720)
    assert roi_fast.x <= orig_rx
    assert roi_fast.x + roi_fast.width >= orig_rx + 140
