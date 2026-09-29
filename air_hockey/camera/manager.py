"""GStreamer-only latest-frame capture with observable startup and runtime failures."""

import glob
import threading
import time

from .buffer import FrameBuffer
from .gst_backend import GStreamerBackend
from .stats import FPSStats
from .types import CameraConfig, Frame


class CameraManager:
    def __init__(self, config=None) -> None:
        self.config = config if config is not None else CameraConfig()
        self.frame_buffer = FrameBuffer()
        self._stats = FPSStats(self.config.requested_fps)
        self._backend = None
        self._info = None
        self._error = None
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        # Serializes start/stop across the entire join and device-selection phase.
        self._lifecycle_lock = threading.RLock()
        self._running = False

    @property
    def info(self):
        with self._lock:
            return self._info

    @property
    def error(self):
        with self._lock:
            return self._error

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _devices(self):
        if self.config.device:
            return [self.config.device]
        devices = glob.glob("/dev/video*")
        return sorted(
            (device for device in devices if device.rsplit("video", 1)[-1].isdigit()),
            key=lambda device: int(device.rsplit("video", 1)[-1]),
        )

    def _stop_locked(self):
        with self._lock:
            self._stop.set()
            thread, backend = self._thread, self._backend
            self._running = False
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)
            if thread.is_alive():
                raise RuntimeError("摄像头采集线程未停止，不能重新打开设备")
        if thread is None and backend is not None:
            backend.release()
        with self._lock:
            self._thread = None
            self._backend = None
        self.frame_buffer.clear()

    def start(self, timeout: float = 2.0) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._running:
                    return
            if self._thread is not None or self._backend is not None:
                self._stop_locked()
            self.frame_buffer.clear()
            self._stats.reset()
            with self._lock:
                self._info = None
                self._error = None

            errors = []
            devices = self._devices()
            for device in devices:
                backend = GStreamerBackend(self.config)
                try:
                    backend.open(device)
                except Exception as exc:
                    errors.append(f"{device}: {exc}")
                    try:
                        backend.release()
                    except Exception as release_exc:
                        errors.append(f"{device} 清理失败: {release_exc}")
                    continue

                with self._lock:
                    self._stop.clear()
                    self._ready.clear()
                    self._error = None
                    self._backend = backend
                    self._running = True
                    self._thread = threading.Thread(
                        target=self.capture_loop, name="camera-capture", daemon=True
                    )
                    self._thread.start()
                ready = self._ready.wait(max(0.0, timeout))
                with self._lock:
                    failure = self._error
                    first_frame = self.frame_buffer.get_latest_frame() is not None
                if ready and first_frame and failure is None:
                    return
                errors.append(
                    f"{device}: {failure}" if failure is not None
                    else f"{device}: {timeout:g}s 内未返回首帧"
                )
                self._stop_locked()

            failure = RuntimeError(
                "GStreamer 摄像头启动失败；已尝试设备："
                + ("; ".join(errors) if errors else "找不到 /dev/video* 设备")
            )
            with self._lock:
                self._error = failure
            raise failure

    def capture_loop(self) -> None:
        backend = self._backend
        last_sample = time.perf_counter()
        first_frame = False
        try:
            while not self._stop.is_set():
                ok, image = backend.read()
                if self._stop.is_set():
                    break
                if not ok or image is None:
                    if time.perf_counter() - last_sample >= max(
                        5.0, 10.0 / max(1.0, self.config.requested_fps)
                    ):
                        raise RuntimeError("GStreamer 长时间没有返回样本")
                    continue
                now = time.perf_counter()
                last_sample = now
                if not first_frame and backend.info is None:
                    raise RuntimeError("GStreamer 首帧缺少实际协商 caps")
                frame = self.frame_buffer.put(
                    Frame(
                        image, timestamp=now,
                        capture_read_ms=backend.last_read_ms,
                        color_convert_ms=backend.last_convert_ms,
                        gst_pts_ns=backend.last_pts_ns,
                    )
                )
                self._stats.record_frame(frame.timestamp)
                if not first_frame:
                    with self._lock:
                        self._info = backend.info
                    first_frame = True
                    self._ready.set()
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._error = exc
        finally:
            try:
                backend.release()
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        if self._error is None:
                            self._error = exc
            with self._lock:
                self._running = False
            self._ready.set()

    def get_latest_frame(self):
        return self.frame_buffer.get_latest_frame()

    def get_stats(self):
        return self._stats.snapshot()

    def wait_for_frame(self, timeout=None):
        return self.frame_buffer.wait_for_frame(timeout)

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_locked()
