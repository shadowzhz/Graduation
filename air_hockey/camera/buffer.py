"""只保留最新帧的线程安全缓存，处理摄像头采集速度和视觉处理速度不一致的问题"""

from dataclasses import replace
from threading import Lock

from .types import Frame


class FrameBuffer:
    def __init__(self) -> None:
        self._lock = Lock()
        self._latest = None
        self._sequence = 0

    def put(self, frame: Frame) -> Frame:
        """存为最新帧（替换旧帧），序号在这里统一分配。"""
        with self._lock:
            self._sequence += 1
            self._latest = replace(frame, sequence=self._sequence)
            return self._latest

    def get_latest_frame(self):
        with self._lock:
            return self._latest

    def clear(self) -> None:
        with self._lock:
            self._latest = None
