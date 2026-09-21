"""非阻塞 PLC 输出。

处理线程只调用 submit() 覆盖最新请求（不排队、不阻塞）；后台线程按固定间隔
取最新请求写入 PLC。PLC 未连接或写入异常时只计数，不向上抛出。
"""

from __future__ import annotations

import threading
from typing import Optional

from .adapter import PlcWriteRequest

DEFAULT_INTERVAL = 1.0 / 60.0


class PlcOutputWorker:
    """覆盖式最新请求 + 后台定时写入。"""

    def __init__(self, plc, interval: float = DEFAULT_INTERVAL) -> None:
        if interval <= 0.0:
            raise ValueError("interval must be positive")
        self.plc = plc
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._latest: Optional[PlcWriteRequest] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.write_count = 0
        self.error_count = 0
        self.last_request: Optional[PlcWriteRequest] = None

    def submit(self, request: PlcWriteRequest) -> None:
        """投递最新请求（非阻塞，只保留最后一条）。"""
        with self._lock:
            self._latest = request

    def write_pending(self) -> bool:
        """取出最新请求写一次；无请求或未连接返回 False。"""
        with self._lock:
            request = self._latest
            self._latest = None
        if request is None:
            return False
        if not getattr(self.plc, "connected", False):
            return False
        try:
            ok = bool(self.plc.write_game_state(**request.as_kwargs()))
        except Exception:
            self.error_count += 1
            return False
        if ok:
            self.write_count += 1
            self.last_request = request
        else:
            self.error_count += 1
        return ok

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="plc-output", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write_pending()
