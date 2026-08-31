"""摄像头层数据结构。"""

from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class Frame:
    """一帧图像，timestamp 用单调时钟。"""

    image: Any
    timestamp: float = field(default_factory=time.perf_counter)
    sequence: int = 0


@dataclass
class CameraConfig:
    """摄像头的请求参数，实际开成什么模式要看 CameraInfo。"""

    device: Optional[str] = None
    width: int = 1280
    height: int = 720
    requested_fps: float = 200.0
    pixel_format: str = "MJPG"
    # Jetson 视觉主链路固定使用硬件 GStreamer，避免 auto 静默回退到低性能 V4L2。
    backend: str = "gstreamer"


@dataclass
class CameraInfo:
    """后端协商出的实际模式。

    negotiated_fps 是驱动报的值，不代表真实采集速度，真实速度看 CameraStats。
    """

    device: str
    backend: str
    width: int
    height: int
    requested_fps: float
    negotiated_fps: float
    source_format: str
    output_format: str
