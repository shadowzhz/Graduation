"""Public camera data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time


# 不使用 slots=True，以兼容 Jetson 上常见的 Python 3.8/3.9。
@dataclass(frozen=True)
class Frame:
    """A captured image with monotonic timestamp and buffer sequence."""

    # timestamp 使用单调时钟，适合计算帧间隔；sequence 用于识别新帧。

    image: Any
    timestamp: float = field(default_factory=time.perf_counter)
    sequence: int = 0


@dataclass(frozen=True)
class CameraConfig:
    """Requested camera mode and backend preferences."""

    # 这里的参数只是请求值，最终模式以 CameraInfo 中后端协商结果为准。

    device: Optional[str] = None
    width: int = 1280
    height: int = 720
    requested_fps: float = 200.0
    pixel_format: str = "MJPG"
    backend: str = "auto"


@dataclass(frozen=True)
class CameraInfo:
    """Camera mode negotiated by the backend.

    ``negotiated_fps`` is the value reported by the driver/pipeline. It is
    not the measured capture rate; that value is ``CameraStats.capture_fps``.
    """

    # negotiated_fps 是驱动报告值，真实采集速度需要查看 CameraStats。

    device: str
    backend: str
    width: int
    height: int
    requested_fps: float
    negotiated_fps: float
    source_format: str
    output_format: str
