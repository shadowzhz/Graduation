"""统一管理摄像头生命周期、后端选择和后台采集线程。"""
from __future__ import annotations
import glob, threading, time
from typing import Optional, Any
import cv2
from .backend import CameraBackend
from .buffer import FrameBuffer
from .gst_backend import GStreamerBackend
from .stats import FPSStats, CameraStats
from .types import CameraConfig, CameraInfo, Frame
from .v4l2_backend import V4L2Backend

class CameraManager:
    """在后台采集摄像头，并向上层提供最新帧和统计信息。"""

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        preview_window_handle: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None: config = CameraConfig(**kwargs)
        elif kwargs: raise TypeError("pass either CameraConfig or keyword configuration, not both")
        self.config = config; self.frame_buffer = FrameBuffer(); self._stats = FPSStats(config.requested_fps)
        self._backend: Optional[CameraBackend] = None; self._info: Optional[CameraInfo] = None
        self._preview_window_handle = preview_window_handle
        self._thread: Optional[threading.Thread] = None; self._stop = threading.Event(); self._ready = threading.Event()
        self._lock = threading.RLock(); self._running = False
    @property
    def running(self) -> bool: return self.is_running()
    @property
    def info(self) -> Optional[CameraInfo]: return self._info
    @property
    def capture_device(self) -> Optional[str]: return self._info.device if self._info else None
    @property
    def backend(self) -> Optional[CameraBackend]: return self._backend
    @property
    def hardware_preview(self) -> bool:
        """当前 GStreamer 后端是否正在向嵌入窗口执行硬件预览。"""
        return bool(getattr(self._backend, "hardware_preview", False))
    def is_running(self) -> bool:
        with self._lock: return self._running
    def _devices(self):
        # 未指定设备时按 /dev/video 编号排序，保证尝试顺序稳定。
        if self.config.device: return [self.config.device]
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
                # 每个设备依次尝试两种后端，失败原因保留到最终异常中。
                if backend_type is GStreamerBackend:
                    backend = backend_type(self.config, self._preview_window_handle)
                else:
                    backend = backend_type(self.config)
                try:
                    self._info = backend.open(device); self._backend = backend
                    print(f"摄像头：{device}，后端：{self._info.backend}，实际模式：{self._info.width}x{self._info.height} @ {self._info.negotiated_fps:.2f} FPS，格式：{self._info.source_format} -> {self._info.output_format}")
                    return
                except Exception as exc:
                    errors.append(f"{backend_type.__name__}: {exc}"); backend.release()
        raise RuntimeError("找不到可用摄像头，请检查 /dev/video* 和 video 组权限。" + "; ".join(errors[-4:]))
    def start(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._running: return
            # 重启时清空旧帧和旧统计，避免调用方读到上一次运行的数据。
            self.frame_buffer.clear(); self._stats.reset(); self._stop.clear(); self._ready.clear(); self._open()
            self._running = True; self._thread = threading.Thread(target=self.capture_loop, name="camera-capture", daemon=True); self._thread.start()
        if not self._ready.wait(max(0.0, timeout)):
            self.stop(); raise RuntimeError("摄像头未返回首帧")
    def capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                backend = self._backend
                if backend is None: break
                try: ok, image = backend.read()
                except cv2.error:
                    if not self._stop.is_set(): print("读取摄像头画面失败")
                    break
                if not ok:
                    if self._stop.is_set(): break
                    time.sleep(0.001); continue
                # Frame 的时间戳在采集线程中生成，统计和检测共用这份采集时间。
                frame = self.frame_buffer.put(Frame(image)); self._stats.record_frame(frame.timestamp); self._ready.set()
        finally:
            with self._lock:
                if not self._stop.is_set(): self._running = False
    def get_latest_frame(self) -> Optional[Frame]: return self.frame_buffer.get_latest_frame()
    def get_stats(self) -> CameraStats: return self._stats.snapshot()
    def wait_for_frame(self, timeout: Optional[float] = None) -> Optional[Frame]: return self.frame_buffer.wait_for_frame(timeout)
    def stop(self) -> None:
        with self._lock:
            backend, thread = self._backend, self._thread; self._running = False; self._stop.set(); self._backend = None; self._thread = None
        # 先发停止信号并摘除后端，再等待线程退出，避免并发读取已释放的对象。
        if backend is not None: backend.release()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread(): thread.join(1.0)
        self.frame_buffer.clear()
    release = stop
