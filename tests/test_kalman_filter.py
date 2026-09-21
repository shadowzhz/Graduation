"""KalmanFilter 状态估计层测试。

覆盖：
- 单元行为：初始化、预测、校正、reset、参数校验、字段透传
- 估计质量：速度收敛、抖动抑制
- 状态连续性：逐帧估计连续、时间戳单调
- 与 VisionRuntime 的链路连续性
"""

import math
import random
from pathlib import Path

import numpy as np

from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.estimation import KalmanFilter
from air_hockey.vision.types import Detection
from game_state import CurlingState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


def test_first_update_initializes_at_measurement():
    kalman = KalmanFilter()
    assert not kalman.initialized
    state = kalman.update(10.0, 20.0, 1.0)
    assert kalman.initialized
    assert isinstance(state, CurlingState)
    assert state.x == 10.0 and state.y == 20.0
    assert state.vx == 0.0 and state.vy == 0.0
    assert state.timestamp == 1.0


def test_predict_before_initialization_returns_zero_state():
    kalman = KalmanFilter()
    state = kalman.predict(2.0)
    assert not kalman.initialized
    assert (state.x, state.y) == (0.0, 0.0)
    assert state.timestamp == 2.0


def test_predict_advances_by_estimated_velocity():
    kalman = KalmanFilter()
    kalman.update(0.0, 0.0, 0.0)
    previous = kalman.update(20.0, 0.0, 0.1)
    assert previous.vx > 0.0

    predicted = kalman.predict(0.2)
    assert predicted.x > previous.x
    assert predicted.timestamp == 0.2


def test_velocity_converges_to_constant_velocity():
    kalman = KalmanFilter()
    rng = random.Random(1)
    true_vx, true_vy = 200.0, -80.0
    x, y = 300.0, 500.0
    dt = 1.0 / 60.0
    for i in range(120):
        x += true_vx * dt
        y += true_vy * dt
        state = kalman.update(x + rng.gauss(0.0, 4.0), y + rng.gauss(0.0, 4.0), (i + 1) * dt)
    assert abs(state.vx - true_vx) < 40.0
    assert abs(state.vy - true_vy) < 40.0


def test_smoothing_reduces_jitter():
    kalman = KalmanFilter()
    rng = random.Random(7)
    true_vx, true_vy = 180.0, -40.0
    x, y = 250.0, 450.0
    dt = 1.0 / 60.0
    raw = []
    smoothed = []
    for i in range(100):
        x += true_vx * dt
        y += true_vy * dt
        noisy_x = x + rng.gauss(0.0, 4.0)
        noisy_y = y + rng.gauss(0.0, 4.0)
        state = kalman.update(noisy_x, noisy_y, (i + 1) * dt)
        raw.append((noisy_x, noisy_y))
        smoothed.append((state.x, state.y))

    def jitter(sequence):
        return sum(
            abs(sequence[i + 1][0] - 2.0 * sequence[i][0] + sequence[i - 1][0])
            for i in range(1, len(sequence) - 1)
        ) / (len(sequence) - 2)

    assert jitter(smoothed) < jitter(raw) * 0.5


def test_reset_reinitializes_on_next_update():
    kalman = KalmanFilter()
    kalman.update(5.0, 5.0, 0.0)
    assert kalman.initialized
    kalman.reset()
    assert not kalman.initialized
    state = kalman.update(9.0, 9.0, 1.0)
    assert state.x == 9.0 and state.y == 9.0
    assert state.vx == 0.0 and state.vy == 0.0


def test_confidence_and_radius_pass_through():
    kalman = KalmanFilter()
    state = kalman.update(1.0, 2.0, 0.0, confidence=0.7, radius=12.0)
    assert state.confidence == 0.7
    assert state.radius == 12.0

    carried = kalman.predict(0.1)
    assert carried.confidence == 0.7
    assert carried.radius == 12.0


def test_invalid_parameters_rejected():
    for kwargs in (
        {"measurement_noise": 0.0},
        {"measurement_noise": -1.0},
        {"acceleration_noise": -1.0},
        {"initial_velocity_variance": 0.0},
    ):
        try:
            KalmanFilter(**kwargs)
            assert False, f"{kwargs} 应该抛 ValueError"
        except ValueError:
            pass


def test_state_estimate_is_continuous_and_monotonic():
    kalman = KalmanFilter()
    dt = 1.0 / 60.0
    vx, vy = 150.0, -60.0
    speed = math.hypot(vx, vy)
    x, y = 200.0, 400.0
    previous = None
    max_step = 0.0
    for i in range(90):
        timestamp = (i + 1) * dt
        x += vx * dt
        y += vy * dt
        state = kalman.update(x, y, timestamp)
        assert math.isfinite(state.x) and math.isfinite(state.y)
        assert math.isfinite(state.vx) and math.isfinite(state.vy)
        if previous is not None:
            assert state.timestamp > previous.timestamp
            step = math.hypot(state.x - previous.x, state.y - previous.y)
            max_step = max(max_step, step)
        previous = state
    # 估计器无跳跃：单帧位移不显著超过真实位移
    assert max_step < 5.0 * speed * dt


def test_state_continues_through_detection_gaps():
    kalman = KalmanFilter()
    dt = 1.0 / 60.0
    vx = 160.0
    x = 100.0
    kalman.update(x, 300.0, 0.0)
    x += vx * dt
    kalman.update(x, 300.0, dt)

    # 连续数帧没有观测，仅用模型外推
    previous = kalman.update(x, 300.0, dt)
    for i in range(2, 8):
        state = kalman.predict(i * dt)
        assert state.x > previous.x
        assert math.isfinite(state.x)
        previous = state


class _MovingDetector:
    """按帧时间戳生成匀速移动的检测结果。"""

    def __init__(self, start_x=500.0, start_y=300.0, step_x=6.0, step_y=3.0):
        self._count = 0
        self._start_x = start_x
        self._start_y = start_y
        self._step_x = step_x
        self._step_y = step_y

    def detect(self, frame, dynamic_roi=None):
        self._count += 1
        return Detection(
            center_x=self._start_x + self._step_x * self._count,
            center_y=self._start_y + self._step_y * self._count,
            radius=25.0,
            area=1960.0,
            timestamp=frame.timestamp,
            score=0.8,
        )


def test_vision_runtime_curling_state_continuous_across_frames():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=_MovingDetector(),
    )
    previous = None
    for index in range(12):
        result = runtime.process_frame(Frame(image=image, timestamp=1.0 + 0.033 * index, sequence=index + 1))
        curling = result.curling_state
        assert isinstance(curling, CurlingState)
        assert math.isfinite(curling.x) and math.isfinite(curling.y)
        if previous is not None:
            assert curling.timestamp >= previous.timestamp
            # 状态连续：单帧位移有界
            assert math.hypot(curling.x - previous.x, curling.y - previous.y) < 100.0
        previous = curling
