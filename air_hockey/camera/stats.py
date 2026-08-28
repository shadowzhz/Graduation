"""采集帧率统计。"""

from dataclasses import dataclass
import time
from threading import Lock


@dataclass
class CameraStats:
    """某次快照时刻的采集速度。

    capture_fps 是采集线程实测值，别和 CameraInfo.negotiated_fps 混淆。
    """

    current_fps: float = 0.0
    average_fps: float = 0.0
    max_fps: float = 0.0
    frame_count: int = 0
    elapsed: float = 0.0
    requested_fps: float = 0.0

    @property
    def capture_fps(self) -> float:
        return self.current_fps


class FPSStats:
    """采集线程调 record_frame，GUI 线程调 snapshot 看结果。"""

    def __init__(self, requested_fps: float = 0.0) -> None:
        self.requested_fps = float(requested_fps)
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._frame_count = 0
            self._start_time = None
            self._last_sample_time = None
            self._last_sample_count = 0
            self._current_fps = 0.0
            self._average_fps = 0.0
            self._max_fps = 0.0

    def record_frame(self, timestamp=None) -> None:
        captured_at = time.perf_counter() if timestamp is None else float(timestamp)
        with self._lock:
            if self._start_time is None:
                # 第一帧才算计时起点，启动等待时间不计入平均帧率
                self._start_time = captured_at
                self._last_sample_time = captured_at
            self._frame_count += 1

    def snapshot(self, now=None) -> CameraStats:
        observed_at = time.perf_counter() if now is None else float(now)
        with self._lock:
            if self._start_time is None:
                return CameraStats(requested_fps=self.requested_fps)

            elapsed = max(0.0, observed_at - self._start_time)
            last_sample_time = (
                observed_at if self._last_sample_time is None else self._last_sample_time
            )
            interval = max(0.0, observed_at - last_sample_time)
            if interval > 0.0:
                self._current_fps = (self._frame_count - self._last_sample_count) / interval
                self._last_sample_count = self._frame_count
                self._last_sample_time = observed_at
                self._max_fps = max(self._max_fps, self._current_fps)
            if elapsed > 0.0:
                self._average_fps = self._frame_count / elapsed

            return CameraStats(
                current_fps=self._current_fps,
                average_fps=self._average_fps,
                max_fps=self._max_fps,
                frame_count=self._frame_count,
                elapsed=elapsed,
                requested_fps=self.requested_fps,
            )
