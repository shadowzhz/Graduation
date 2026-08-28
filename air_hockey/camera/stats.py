"""FPS measurement independent from camera I/O and GUI code."""

from __future__ import annotations

from dataclasses import dataclass
import time
from threading import Lock
from typing import Callable, Optional


@dataclass(frozen=True)
class CameraStats:
    """Measured capture performance at one point in time.

    ``capture_fps`` is measured from frames received by the capture loop and
    must not be confused with ``CameraInfo.negotiated_fps``.
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
    """Thread-safe FPS accumulator.

    ``record_frame`` is called by the capture thread.  ``snapshot`` can be
    called by a GUI timer and computes interval, average, and peak rates
    without sharing mutable counters with that GUI thread.
    """

    def __init__(
        self,
        requested_fps: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.requested_fps = float(requested_fps)
        self._clock = clock
        self._lock = Lock()
        self.reset()

    def reset(self, start_time: Optional[float] = None) -> None:
        with self._lock:
            # 采集线程写入计数，GUI 线程读取快照；所有共享状态都受同一把锁保护。
            self._frame_count = 0
            self._start_time: Optional[float] = start_time
            self._last_sample_time: Optional[float] = start_time
            self._last_sample_count = 0
            self._current_fps = 0.0
            self._average_fps = 0.0
            self._max_fps = 0.0

    def record_frame(self, timestamp: Optional[float] = None) -> None:
        captured_at = self._clock() if timestamp is None else float(timestamp)
        with self._lock:
            if self._start_time is None:
                # 第一帧作为统计起点，避免把启动等待时间算入平均帧率。
                self._start_time = captured_at
                self._last_sample_time = captured_at
            self._frame_count += 1

    def snapshot(self, now: Optional[float] = None) -> CameraStats:
        observed_at = self._clock() if now is None else float(now)
        with self._lock:
            if self._start_time is None:
                return CameraStats(requested_fps=self.requested_fps)

            elapsed = max(0.0, observed_at - self._start_time)
            last_sample_time = (
                observed_at
                if self._last_sample_time is None
                else self._last_sample_time
            )
            interval = max(0.0, observed_at - last_sample_time)
            if interval > 0.0:
                # current_fps 统计上一次快照以来收到的帧，average_fps 统计整个运行期。
                self._current_fps = (
                    self._frame_count - self._last_sample_count
                ) / interval
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
