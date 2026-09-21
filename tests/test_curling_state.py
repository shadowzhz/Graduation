"""统一 CurlingState 数据链路测试。

覆盖：
- CurlingState 的 position/velocity/timestamp/confidence 语义与方向属性
- Detector 输出统一为 CurlingState
- Predictor 输入 CurlingState、输出预测终点
- VisionRuntime 暴露 CurlingState 与轨迹调试信息（当前坐标/速度方向/预测终点）
"""

from pathlib import Path

import cv2
import numpy as np

from air_hockey.app.vision_runtime import TrajectoryDebug, VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.physics import StoneMotion
from air_hockey.prediction import TrajectoryPredictor
from air_hockey.vision.detector import StoneDetector
from air_hockey.vision.types import Detection
from game_state import CurlingState, StoneState, TrackingState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


def test_curling_state_geometry_properties():
    state = CurlingState(x=3.0, y=4.0, vx=3.0, vy=4.0, timestamp=1.5, confidence=0.9)
    assert state.position == (3.0, 4.0)
    assert state.velocity == (3.0, 4.0)
    assert state.speed == 5.0
    assert state.direction == (0.6, 0.8)


def test_curling_state_still_has_zero_direction():
    state = CurlingState(x=0.0, y=0.0)
    assert state.speed == 0.0
    assert state.direction == (0.0, 0.0)


def test_curling_state_from_detection_maps_score_to_confidence():
    detection = Detection(
        center_x=10.0, center_y=20.0, radius=5.0, area=78.0, timestamp=0.5, score=0.8
    )
    state = CurlingState.from_detection(detection)
    assert state.position == (10.0, 20.0)
    assert state.radius == 5.0
    assert state.timestamp == 0.5
    assert state.confidence == 0.8
    # 视觉像素命名别名保持可用
    assert state.center_x == 10.0 and state.center_y == 20.0


def test_curling_state_from_any_normalizes_all_entries():
    scalars = CurlingState.from_any(1.0, 2.0, 3.0, 4.0)
    assert (scalars.x, scalars.y, scalars.vx, scalars.vy) == (1.0, 2.0, 3.0, 4.0)

    from_stone = CurlingState.from_any(StoneState(x=5.0, y=6.0, vx=7.0, vy=8.0))
    assert isinstance(from_stone, CurlingState)
    assert from_stone.position == (5.0, 6.0)
    assert from_stone.velocity == (7.0, 8.0)

    from_motion = CurlingState.from_any(StoneMotion(x=9.0, y=10.0, vx=11.0, vy=12.0))
    assert (from_motion.x, from_motion.y, from_motion.vx, from_motion.vy) == (9.0, 10.0, 11.0, 12.0)


def test_stone_state_is_a_curling_state():
    stone = StoneState(x=1.0, y=2.0, vx=3.0, vy=4.0, tracking_state=TrackingState.ACTIVE)
    assert isinstance(stone, CurlingState)
    assert stone.position == (1.0, 2.0)
    assert stone.tracking_state == TrackingState.ACTIVE


def test_detector_returns_curling_state():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 20, (0, 0, 255), -1)
    detector = StoneDetector(
        color_space="hsv",
        lower=(0, 100, 100),
        upper=(10, 255, 255),
        min_area=100.0,
        min_radius=5.0,
        min_circularity=0.3,
    )
    result = detector.detect(Frame(image=image, timestamp=2.0))
    assert isinstance(result, CurlingState)
    assert abs(result.x - 100.0) < 3.0
    assert abs(result.y - 100.0) < 3.0
    assert result.radius > 5.0
    assert result.timestamp == 2.0
    assert result.confidence > 0.0


def test_predict_endpoint_matches_trajectory_tail():
    predictor = TrajectoryPredictor()
    state = CurlingState(x=300.0, y=400.0, vx=200.0, vy=-100.0)
    endpoint = predictor.predict_endpoint(state)
    assert endpoint == predictor.predict(state)[-1]


def test_predict_accepts_curling_state_like_motion():
    predictor = TrajectoryPredictor()
    x, y, vx, vy = 280.0, 480.0, 150.0, -220.0
    curling_traj = predictor.predict(CurlingState(x=x, y=y, vx=vx, vy=vy))
    motion_traj = predictor.predict(StoneMotion(x=x, y=y, vx=vx, vy=vy))
    assert curling_traj == motion_traj


class _MockDetector:
    """按顺序返回预设 Detection，模拟真实检测器的 CurlingState 输出边界。"""

    def __init__(self, detections):
        self._detections = list(detections)
        self._index = 0

    def detect(self, frame, dynamic_roi=None):
        if self._index < len(self._detections):
            detection = self._detections[self._index]
            self._index += 1
            return detection
        return None


def test_vision_runtime_exposes_curling_state_and_debug():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = _MockDetector(
        [
            Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0, score=0.7),
            Detection(center_x=520.0, center_y=310.0, radius=25.0, area=1960.0, timestamp=1.05, score=0.8),
        ]
    )
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=detector,
    )
    result = runtime.process_frame(Frame(image=image, timestamp=1.0, sequence=1))

    assert isinstance(result.curling_state, CurlingState)
    assert result.curling_state.confidence == 0.7
    assert isinstance(result.trajectory_debug, TrajectoryDebug)
    assert result.trajectory_debug.position == result.curling_state.position
    assert result.trajectory_debug.direction == result.curling_state.direction
    assert result.trajectory_debug.predicted_endpoint == result.trajectory[-1]
