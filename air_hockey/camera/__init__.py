"""摄像头采集层。"""

from .types import Frame, CameraInfo, CameraConfig
from .gst_backend import GStreamerBackend
from .buffer import FrameBuffer
from .stats import FPSStats, CameraStats
from .manager import CameraManager
