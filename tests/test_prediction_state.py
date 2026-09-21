"""PredictionState 数据层测试。

覆盖：
- Predictor 输出 PredictionState（trajectory / endpoint / duration / source_state）
- 序列协议（len / 遍历 / 下标 / 切片）与裸列表比较
- predict_endpoint 与 PredictionState.endpoint 一致
- VisionRuntime 透传 PredictionState
"""

from pathlib import Path

import numpy as np

from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.prediction import PredictionState, TrajectoryPredictor, predict_trajectory
from air_hockey.vision.types import Detection
from game_state import CurlingState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


def test_predict_returns_prediction_state():
    predictor = TrajectoryPredictor()
    source = CurlingState(x=300.0, y=400.0, vx=200.0, vy=-100.0, confidence=0.6)
    pred = predictor.predict(source)

    assert isinstance(pred, PredictionState)
    assert pred.source_state is source
    assert pred.trajectory[0] == (300.0, 400.0)
    assert pred.endpoint == pred.trajectory[-1]
    assert pred.duration > 0.0
    assert len(pred.trajectory) > 1


def test_stationary_prediction_has_zero_duration_singleton():
    predictor = TrajectoryPredictor()
    pred = predictor.predict(100.0, 100.0, 0.0, 0.0)
    assert isinstance(pred, PredictionState)
    assert len(pred) == 1
    assert pred.endpoint == (100.0, 100.0)
    assert pred.duration == 0.0


def test_prediction_state_sequence_protocol_matches_trajectory():
    predictor = TrajectoryPredictor()
    pred = predictor.predict(CurlingState(x=300.0, y=400.0, vx=180.0, vy=-120.0))

    assert list(pred) == pred.trajectory
    assert pred[0] == pred.trajectory[0]
    assert pred[-1] == pred.trajectory[-1]
    assert pred[:-1] == pred.trajectory[:-1]
    assert pred == pred.trajectory
    assert len(pred) == len(pred.trajectory)


def test_predict_endpoint_matches_prediction_state_endpoint():
    predictor = TrajectoryPredictor()
    source = CurlingState(x=280.0, y=480.0, vx=150.0, vy=-220.0)
    assert predictor.predict_endpoint(source) == predictor.predict(source).endpoint


def test_predict_trajectory_function_returns_prediction_state():
    pred = predict_trajectory(300.0, 400.0, 120.0, -60.0)
    assert isinstance(pred, PredictionState)
    assert pred.endpoint == pred.trajectory[-1]


class _MockDetector:
    def __init__(self, detections):
        self._detections = list(detections)
        self._index = 0

    def detect(self, frame, dynamic_roi=None):
        if self._index < len(self._detections):
            detection = self._detections[self._index]
            self._index += 1
            return detection
        return None


def test_vision_runtime_forwards_prediction_state():
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

    assert isinstance(result.prediction, PredictionState)
    assert result.prediction.endpoint == result.trajectory[-1]
    assert result.trajectory_debug.predicted_endpoint == result.prediction.endpoint
