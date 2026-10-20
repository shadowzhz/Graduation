"""摄像头层数据结构，定义数据格式"""

from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class Frame:
    """一帧图像；timestamp 是采样读取和颜色转换完成后的 host 单调时钟。"""

    image: Any
    timestamp: float = field(default_factory=time.perf_counter)
    sequence: int = 0       # 帧编号
    capture_read_ms: float = 0.0  # pull + map + conversion (host time)
    color_convert_ms: float = 0.0
    gst_pts_ns: Optional[int] = None  # Gst clock domain, not exposure latency


@dataclass
class CameraConfig:
    """摄像头的请求参数，实际开成什么模式要看 CameraInfo。"""

    device: Optional[str] = None

    # 请求分辨率
    width: int = 1280
    height: int = 720

    # 请求帧率
    requested_fps: float = 200.0

    # 可选：在 open 阶段就把像素格式协商成这个 FOURCC（如 "MJPG"）。
    #
    # 为 None 时保持原行为：先打开再 set —— 对这台球台相机是无声失败的，
    # 会停在 yuyv422 的 10 fps。设成 "MJPG" 才真的切到 MJPG 高速模式
    # （实测约 100 FPS 进管线）。
    # 主程序默认传 "MJPG"（2026-10-20 起），--camera-slow 时传 None 退回旧行为。
    fourcc: Optional[str] = None

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
