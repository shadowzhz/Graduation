"""PredictionState 数据层测试。

覆盖：
- Predictor 输出 PredictionState（trajectory / endpoint / duration / source_state）
- 序列协议（len / 遍历 / 下标 / 切片）与裸列表比较
- predict_endpoint 与 PredictionState.endpoint 一致
- VisionRuntime 透传 PredictionState
"""

import math
from pathlib import Path
from unittest import TestCase

import numpy as np

from air_hockey import core_config as core
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


def test_stop_endpoint_includes_last_subpixel_motion_at_budget_boundary():
    predictor = TrajectoryPredictor(substep=0.01, point_interval=1.0, max_simulation_steps=2)
    prediction = predictor.predict(300.0, 380.0, 9.0, 0.0)
    assert len(prediction.trajectory) == 2
    assert prediction[0] == (300.0, 380.0)
    assert math.dist(prediction.endpoint, (300.1728, 380.0)) < 1e-10
    assert prediction.endpoint == prediction[-1]
    assert prediction.duration == 0.02


def test_budget_exhaustion_never_returns_false_terminal_state():
    predictor = TrajectoryPredictor(max_simulation_steps=1)
    with TestCase().assertRaisesRegex(RuntimeError, "max_simulation_steps=1.*terminal"):
        predictor.predict(300.0, 380.0, 150.0, 0.0)
    with TestCase().assertRaisesRegex(RuntimeError, "max_simulation_steps=1.*terminal"):
        predictor.predict_endpoint(300.0, 380.0, 150.0, 0.0)


def test_goal_and_obstacle_terminal_positions_are_exact():
    predictor = TrajectoryPredictor()
    goal = predictor.predict(300.0, 80.0, 0.0, -150.0)
    assert goal.endpoint == goal[-1]
    assert goal.endpoint[1] + core.STONE_RADIUS < core.RINK_TOP
    assert goal.duration > 0.0
    already_scored = predictor.predict(300.0, 20.0, 0.0, -150.0)
    assert already_scored.trajectory == [(300.0, 20.0)]
    assert already_scored.duration == 0.0

    obstacle = (350.0, 380.0)
    contact = predictor.predict(300.0, 380.0, 150.0, 0.0, obstacles=(obstacle,))
    assert contact.endpoint == contact[-1]
    assert math.dist(contact.endpoint, obstacle) <= core.STONE_RADIUS + core.MALLET_RADIUS
    assert contact.duration > 0.0
    already_touching = predictor.predict(350.0, 380.0, 150.0, 0.0, obstacles=(obstacle,))
    assert already_touching.trajectory == [(350.0, 380.0)]
    assert already_touching.duration == 0.0


def test_predictor_rejects_invalid_numerical_and_budget_parameters():
    invalid = [
        {"substep": 0.0}, {"substep": -1.0}, {"point_interval": 0.0},
        {"stop_speed": -1.0}, {"max_simulation_steps": 0}, {"max_simulation_steps": -1},
        {"max_simulation_steps": 1.5}, {"max_simulation_steps": True},
    ]
    invalid.extend(
        {name: value}
        for name in ("substep", "point_interval", "stop_speed", "max_simulation_steps")
        for value in (math.nan, math.inf, -math.inf)
    )
    for kwargs in invalid:
        with TestCase().assertRaises(ValueError):
            TrajectoryPredictor(**kwargs)

    predictor = TrajectoryPredictor()
    for value in (math.nan, math.inf, -math.inf):
        for source in (CurlingState(x=value, y=380.0), CurlingState(x=300.0, y=380.0, vx=value)):
            with TestCase().assertRaisesRegex(ValueError, "finite"):
                predictor.predict(source)
        with TestCase().assertRaisesRegex(ValueError, "finite"):
            predictor.predict(300.0, 380.0, 150.0, 0.0, obstacle_radius=value)
        with TestCase().assertRaisesRegex(ValueError, "finite"):
            predictor.predict(300.0, 380.0, 150.0, 0.0, obstacles=((value, 380.0),))


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
