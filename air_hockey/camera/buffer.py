"""Thread-safe latest-frame buffer."""

from __future__ import annotations

from threading import Condition, Lock
from typing import Optional, Union
import numpy as np
from .types import Frame


class FrameBuffer:
    """线程安全的最新帧缓存，只保留当前最新图像。"""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._latest: Optional[Frame] = None
        self._sequence = 0

    def put(self, frame: Union[Frame, np.ndarray], timestamp: Optional[float] = None) -> Frame:
        """Publish a Frame or an ndarray directly as the latest frame."""
        # 无论输入是 ndarray 还是 Frame，都在这里统一分配递增序号。
        if not isinstance(frame, Frame):
            frame = Frame(frame, timestamp=timestamp)
        with self._condition:
            self._sequence += 1
            stored = Frame(frame.image, frame.timestamp, self._sequence)
            self._latest = stored
            self._condition.notify_all()
            return stored

    def get_latest_frame(self) -> Optional[Frame]:
        with self._condition:
            return self._latest

    def wait_for_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        with self._condition:
            if self._latest is None:
                # 首帧到达时由 put() 唤醒等待线程，避免轮询消耗 CPU。
                self._condition.wait(timeout)
            return self._latest

    def clear(self) -> None:
        with self._condition:
            self._latest = None
