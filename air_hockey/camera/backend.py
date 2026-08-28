"""摄像头后端公共接口。"""

from abc import ABC, abstractmethod

from .types import CameraConfig, CameraInfo


class CameraBackend(ABC):
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.info = None

    @abstractmethod
    def open(self, device: str) -> CameraInfo:
        """打开设备，返回实际模式。"""

    @abstractmethod
    def read(self):
        """读一帧，返回 (是否成功, BGR uint8 图像)。"""

    @abstractmethod
    def release(self) -> None:
        """释放资源。"""
