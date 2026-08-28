"""OpenCV V4L2 摄像头后端。"""

import cv2
import numpy as np

from .backend import CameraBackend
from .types import CameraInfo

SUPPORTED_FOURCC = frozenset({"MJPG", "YUYV", "YUY2"})


class V4L2Backend(CameraBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.capture = None

    def open(self, device: str) -> CameraInfo:
        c = self.config
        fourcc = c.pixel_format.upper()
        if fourcc not in SUPPORTED_FOURCC:
            supported = ", ".join(sorted(SUPPORTED_FOURCC))
            raise ValueError(f"V4L2 不支持 FOURCC {c.pixel_format!r}，支持：{supported}")
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"无法打开 V4L2 设备: {device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.requested_fps)
        # 缓冲区只留一帧，降低延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = cap
        self.info = CameraInfo(
            device=device,
            backend="V4L2",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            requested_fps=c.requested_fps,
            negotiated_fps=cap.get(cv2.CAP_PROP_FPS),
            source_format=c.pixel_format,
            output_format="BGR",
        )
        return self.info

    def read(self):
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        return True, self._as_bgr(frame)

    @staticmethod
    def _as_bgr(frame):
        if frame.dtype != np.uint8:
            raise RuntimeError("V4L2 backend returned a non-uint8 frame")
        if frame.ndim != 3:
            raise RuntimeError("V4L2 backend returned a non-color frame")
        if frame.shape[2] == 3:
            return frame
        if frame.shape[2] == 4:
            # 有的驱动会多给一个 alpha 通道
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        raise RuntimeError("V4L2 backend returned an unsupported channel count")

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
