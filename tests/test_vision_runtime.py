"""针对 VisionRuntime 及其数据流的单元测试。

验证链路：
frame -> detection -> track -> table StoneState -> predictor -> AI
"""

from pathlib import Path
import numpy as np

from air_hockey.app.vision_runtime import VisionResult, VisionRuntime, track_to_rink_state
from air_hockey.camera.types import Frame
from game_state import StoneState
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


def test_vision_runtime_dataflow_track_to_ai():
    """验证 track -> table StoneState -> predictor -> AI 的完整调用链。"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    det1 = Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)
    det2 = Detection(center_x=520.0, center_y=310.0, radius=25.0, area=1960.0, timestamp=1.05)

    detector = MockDetector([det1, det2])
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
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
    assert res1.track.state == TrackState.ACTIVE

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
    assert res3.track.vx > 0
    assert res3.stone_state is not None
    assert res3.stone_state.vx > 0
    assert res3.trajectory is not None
    assert len(res3.trajectory) >= 1
    assert res3.ai_target is not None
    assert res3.fps > 0.0


def test_vision_runtime_without_detection():
    """验证未检测到目标时，调用链输出空状态并保持稳健。"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = MockDetector([])
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
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
