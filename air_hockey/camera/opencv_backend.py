"""Windows camera capture through OpenCV's DirectShow backend."""

import math
import time

import cv2
import numpy as np

from .types import CameraInfo


class OpenCVBackend:
    backend_name = "OpenCV/DirectShow"

    def __init__(self, config):
        self.config = config
        self._capture = None
        self._device = None
        self.info = None
        self.last_read_ms = 0.0
        self.last_convert_ms = 0.0
        self.last_pts_ns = None

    def open(self, device):
        try:
            index = int(device)
            if str(device).strip() != str(index) or index < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("Windows 相机设备必须是非负摄像头编号，例如 0")

        self.release()
        self._device = str(index)
        self.info = None
        self.last_read_ms = self.last_convert_ms = 0.0
        self.last_pts_ns = None
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"DirectShow 无法打开摄像头 {index}；检查设备编号和占用情况")
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.requested_fps)
        except Exception:
            capture.release()
            raise
        self._capture = capture

    def read(self):
        if self._capture is None:
            raise RuntimeError("DirectShow 摄像头尚未启动")
        started = time.perf_counter()
        ok, image = self._capture.read()
        self.last_read_ms = (time.perf_counter() - started) * 1000.0
        self.last_convert_ms = 0.0
        self.last_pts_ns = None
        if not ok or image is None:
            return False, None
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise RuntimeError(f"DirectShow 摄像头 {self._device} 返回的不是 BGR uint8 图像")

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError(f"DirectShow 摄像头 {self._device} 返回无效图像尺寸")
        if self.info is None:
            fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            source_format = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
            self.info = CameraInfo(
                device=self._device,
                backend=self.backend_name,
                width=width,
                height=height,
                requested_fps=self.config.requested_fps,
                negotiated_fps=fps if math.isfinite(fps) and fps > 0.0 else 0.0,
                source_format=source_format if fourcc and source_format.isprintable() else "unknown",
                output_format="BGR",
            )
        return True, image

    def release(self):
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
