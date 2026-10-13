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

        # config.fourcc 为空时保持原来的行为：先打开，再 set 各项。
        #
        # 注意（实测）：这台球台相机在同一分辨率下同时提供
        #     mjpeg   1280x720 fps=200
        #     yuyv422 1280x720 fps=10
        # 而"打开之后再 set(CAP_PROP_FOURCC)"是**无声失败**的 —— 驱动不会重新协商，
        # 于是永远停在 yuyv422 的 10 fps（实测 read 中位 97ms）。
        # 只有在 VideoCapture() 构造时把 FOURCC 一起传进去，才能真正切到 MJPG
        # （实测 read 中位约 6ms，约 100 FPS 进管线）。
        #
        # 但该格式的实际视野/清晰度需要在现场核对后再决定是否默认启用，
        # 所以这里做成显式开关（main.py --camera-fast），默认不变。
        fast_fourcc = getattr(self.config, "fourcc", None)
        if fast_fourcc:
            params = [
                cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width),
                cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height),
                cv2.CAP_PROP_FPS, int(round(self.config.requested_fps)),
                cv2.CAP_PROP_FOURCC, int(cv2.VideoWriter_fourcc(*fast_fourcc)),
            ]
            try:
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW, params)
            except (cv2.error, TypeError):
                # 旧版 OpenCV 没有三参重载；退回下面的"打开后再设"
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fast_fourcc))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                capture.set(cv2.CAP_PROP_FPS, self.config.requested_fps)
        else:
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"DirectShow 无法打开摄像头 {index}；检查设备编号和占用情况")
            if not fast_fourcc:
                # This camera needs MJPG for its high-speed capture mode.
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
