"""只保留最新帧的线程安全缓存，处理摄像头采集速度和视觉处理速度不一致的问题"""

from threading import Condition, Lock

from .types import Frame

# 定义一个缓存对象
class FrameBuffer:
    def __init__(self) -> None:
        self._condition = Condition(Lock())     # 保证线程安全
        self._latest = None
        self._sequence = 0

    def put(self, frame, timestamp=None) -> Frame:
        # 传 Frame 或裸 ndarray 都行，序号在这里统一分配
        if not isinstance(frame, Frame):
            frame = Frame(frame, timestamp=timestamp)
        with self._condition:
            self._sequence += 1
            stored = Frame(frame.image, frame.timestamp, self._sequence)
            self._latest = stored       # 替换最新帧
            self._condition.notify_all()
            return stored

    def get_latest_frame(self):
        with self._condition:
            return self._latest

    def wait_for_frame(self, timeout=None):
        with self._condition:
            if self._latest is None:
                self._condition.wait(timeout)
            return self._latest

    def clear(self) -> None:
        with self._condition:
            self._latest = None
