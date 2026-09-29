"""非阻塞 PLC 输出：新请求唤醒线程，写入限频且只保留最新请求。"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional

from .adapter import PlcWriteRequest

DEFAULT_INTERVAL = 1.0 / 60.0


class PlcOutputWorker:
    """覆盖式最新请求 + 后台事件唤醒与最小写入间隔。"""

    def __init__(self, plc, interval: float = DEFAULT_INTERVAL) -> None:
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("interval must be positive")
        self.plc = plc
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest: Optional[PlcWriteRequest] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.write_count = 0
        self.error_count = 0
        self.last_request: Optional[PlcWriteRequest] = None
        self._write_ms = deque(maxlen=4096)
        self._frame_to_write_ms = deque(maxlen=4096)

    def submit(self, request: PlcWriteRequest) -> None:
        """投递最新请求（非阻塞，只保留最后一条）。"""
        with self._condition:
            self._latest = request
            self._condition.notify()

    def write_pending(self) -> bool:
        """取出最新请求写一次；无请求或未连接返回 False。"""
        with self._lock:
            request = self._latest
            self._latest = None
        if request is None:
            return False
        if not getattr(self.plc, "connected", False):
            return False
        write_start = time.perf_counter()
        try:
            ok = bool(self.plc.write_game_state(**request.as_kwargs()))
        except Exception:
            ok = False
        write_end = time.perf_counter()
        with self._lock:
            self._write_ms.append((write_end - write_start) * 1000.0)
            if ok:
                # request.timestamp 是读出/转换完成时的主机单调时钟，不是曝光时间。
                self._frame_to_write_ms.append(max(0.0, (write_end - request.timestamp) * 1000.0))
        if ok:
            self.write_count += 1
            self.last_request = request
        else:
            self.error_count += 1
        return ok

    def timing_samples(self) -> tuple[list[float], list[float]]:
        """返回最近写入耗时、从应用读出帧到成功写入的延迟样本。"""
        with self._lock:
            return list(self._write_ms), list(self._frame_to_write_ms)

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="plc-output", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        with self._condition:
            self._stop.set()
            self._latest = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        with self._condition:
            if thread is self._thread and (thread is None or not thread.is_alive()):
                self._thread = None

    @property
    def running(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        next_write = 0.0
        while True:
            with self._condition:
                while not self._stop.is_set():
                    if self._latest is None:
                        self._condition.wait()
                        continue
                    delay = next_write - time.perf_counter()
                    if delay <= 0.0:
                        break
                    self._condition.wait(delay)
                if self._stop.is_set():
                    return
            if self._stop.is_set():
                return
            self.write_pending()
            next_write = time.perf_counter() + self.interval
