"""First-stage stone detection pipeline."""

from .detector import StoneDetector
from .types import Detection, Frame, ROI

__all__ = ["Detection", "Frame", "ROI", "StoneDetector"]
