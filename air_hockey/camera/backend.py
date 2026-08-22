"""Backend abstraction used by :class:`CameraManager`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .types import CameraConfig, CameraInfo


class CameraBackend(ABC):
    """Small common interface for camera implementations."""

    # CameraManager 只依赖这个接口，因此可以在 GStreamer 和 V4L2 间回退。

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.info: Optional[CameraInfo] = None

    @abstractmethod
    def open(self, device: str) -> CameraInfo:
        """Open ``device`` and return its actual mode."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one BGR ``uint8`` image with shape ``H x W x 3``."""

    @abstractmethod
    def release(self) -> None:
        """Release all backend resources."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
