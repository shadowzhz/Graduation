"""摄像头采集管理：自动选后端，后台线程只保留最新一帧。"""

import glob
import threading
import time

import cv2

from .buffer import FrameBuffer
from .gst_backend import GStreamerBackend
from .stats import FPSStats
from .types import CameraConfig, CameraInfo, Frame
from .v4l2_backend import V4L2Backend


class CameraManager:
    def __init__(self, config=None, preview_window_handle=None) -> None:
        self.config = config if config is not None else CameraConfig()
        self.frame_buffer = FrameBuffer()
        self._stats = FPSStats(self.config.requested_fps)
        self._backend = None
        self._info = None
        self._preview_window_handle = preview_window_handle
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._running = False

    @property
    def info(self):
        return self._info

    @property
    def hardware_preview(self) -> bool:
        """GStreamer 后端当前是否在往嵌入窗口做硬件预览。"""
        return bool(getattr(self._backend, "hardware_preview", False))

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _devices(self):
        if self.config.device:
            return [self.config.device]
        return sorted(glob.glob("/dev/video*"), key=lambda p: int(p.rsplit("video", 1)[-1]))

    def _open(self) -> None:
        errors = []
        backend_types = {
            "gstreamer": (GStreamerBackend,),
            "v4l2": (V4L2Backend,),
            "auto": (GStreamerBackend, V4L2Backend),
        }.get(self.config.backend.strip().lower())
        if backend_types is None:
            raise ValueError("backend must be 'auto', 'gstreamer', or 'v4l2'")
        for device in self._devices():
            for backend_type in backend_types:
                if backend_type is GStreamerBackend:
                    backend = backend_type(self.config, self._preview_window_handle)
                else:
                    backend = backend_type(self.config)
                try:
                    self._info = backend.open(device)
                    self._backend = backend
                    print(
                        f"摄像头：{device}，后端：{self._info.backend}，"
                        f"实际模式：{self._info.width}x{self._info.height} @ "
                        f"{self._info.negotiated_fps:.2f} FPS，"
                        f"格式：{self._info.source_format} -> {self._info.output_format}"
                    )
                    return
                except Exception as exc:
                    errors.append(f"{backend_type.__name__}: {exc}")
                    backend.release()
        raise RuntimeError(
            "找不到可用摄像头，请检查 /dev/video* 和 video 组权限。"
            + "; ".join(errors[-4:])
        )

    def start(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._running:
                return
            # 重启时清掉上一轮的帧和统计
            self.frame_buffer.clear()
            self._stats.reset()
            self._stop.clear()
            self._ready.clear()
            self._open()
            self._running = True
            self._thread = threading.Thread(
                target=self.capture_loop, name="camera-capture", daemon=True
            )
            self._thread.start()
        if not self._ready.wait(max(0.0, timeout)):
            self.stop()
            raise RuntimeError("摄像头未返回首帧")

    def capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                backend = self._backend
                if backend is None:
                    break
                try:
                    ok, image = backend.read()
                except cv2.error:
                    if not self._stop.is_set():
                        print("读取摄像头画面失败")
                    break
                if not ok:
                    if self._stop.is_set():
                        break
                    time.sleep(0.001)
                    continue
                # Frame 时间戳在采集线程里生成，统计和检测共用这个时间
                frame = self.frame_buffer.put(Frame(image))
                self._stats.record_frame(frame.timestamp)
                self._ready.set()
        finally:
            with self._lock:
                if not self._stop.is_set():
                    self._running = False

    def get_latest_frame(self):
        return self.frame_buffer.get_latest_frame()

    def get_stats(self):
        return self._stats.snapshot()

    def wait_for_frame(self, timeout=None):
        return self.frame_buffer.wait_for_frame(timeout)

    def stop(self) -> None:
        with self._lock:
            backend, thread = self._backend, self._thread
            self._running = False
            self._stop.set()
            self._backend = None
            self._thread = None
        # 先发停止信号并摘掉后端，再等线程退出
        if backend is not None:
            backend.release()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(1.0)
        self.frame_buffer.clear()
