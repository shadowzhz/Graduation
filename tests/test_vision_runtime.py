"""针对 VisionRuntime 及其数据流的单元测试。

验证链路：
frame -> detection -> track -> table StoneState -> predictor -> AI
"""

from contextlib import ExitStack
from dataclasses import replace
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
from air_hockey.app.vision_runtime import AI_HOME_Y, VisionResult, VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.prediction import TrajectoryPredictor
from game_state import CurlingState, GameState, StoneState
from air_hockey.vision.tracker import StoneTracker
from air_hockey.vision.types import Detection, TrackState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


class MockDetector:
    def __init__(self, detections):
        self._detections = list(detections)
        self._index = 0

    def detect(self, frame, dynamic_roi=None):
        if self._index < len(self._detections):
            det = self._detections[self._index]
            self._index += 1
            return det
        return None


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
    assert abs(result.ai_decision.reaction_timer - (core.DIFFICULTIES["普通"].reaction_delay - 0.001)) < 1e-9

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
        difficulty=replace(core.DIFFICULTIES["普通"], aim_error=0.0),
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


def test_vision_runtime_does_not_confirm_single_detection():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True,
        detector=MockDetector([
            Detection(center_x=500.0, center_y=300.0, radius=25.0,
                      area=1960.0, timestamp=1.0),
        ]),
    )
    result = runtime.process_frame(Frame(np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=1.0))

    assert result.track is not None
    assert result.track.state == TrackState.TENTATIVE
    assert not result.track_confirmed


def test_vision_runtime_confirms_after_second_hard_detection():
    runtime = VisionRuntime(
        table_calibration_file=None, disable_undistort=True, detection_interval=3,
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


def test_new_far_identity_revokes_confirmation():
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
    assert states == [
        TrackState.TENTATIVE, TrackState.ACTIVE, TrackState.TENTATIVE, TrackState.ACTIVE,
    ]
    assert [result.track_confirmed for result in results] == [False, True, False, True]


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


def test_plc_track_confirmation_stops_on_missed_detection_and_recovers():
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
    assert not results[0].track_confirmed
    assert results[1].track_confirmed
    assert results[2].ai_decision is not None and not results[2].track_confirmed
    assert results[3].track_confirmed
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
    assert result.ai_target is None
    assert result.ai_decision is None


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
    missing = runtime.process_frame(Frame(image=image, timestamp=1.01))
    assert missing.track is None and missing.curling_state is None
    new = runtime.process_frame(Frame(image=image, timestamp=1.015))
    expected = runtime.camera_geometry.undistorted_to_table(800.0, 300.0)
    assert math.dist(new.curling_state.position, expected) < 1e-6
    assert new.curling_state.vx == new.curling_state.vy == 0.0


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
