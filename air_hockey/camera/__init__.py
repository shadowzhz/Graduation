"""摄像头采集层。"""

from .types import Frame, CameraInfo, CameraConfig
from .backend import CameraBackend
from .gst_backend import GStreamerBackend
from .v4l2_backend import V4L2Backend
from .buffer import FrameBuffer
from .stats import FPSStats, CameraStats
from .manager import CameraManager
