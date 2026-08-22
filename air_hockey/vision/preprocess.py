"""Small, independently testable image preprocessing operations."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import ROI

ColorRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


def validate_bgr(image: np.ndarray) -> np.ndarray:
    """Validate the camera contract without copying the image."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR uint8 HxWx3 ndarray")
    return image


def crop_roi(image: np.ndarray, roi: Optional[ROI]) -> Tuple[np.ndarray, int, int]:
    """Crop ``image`` and return ``(crop, offset_x, offset_y)``."""

    validate_bgr(image)
    if roi is None:
        return image, 0, 0
    bounded = roi.clamp(image.shape[1], image.shape[0])
    if bounded is None:
        return image[0:0, 0:0], 0, 0
    # 返回偏移量，检测器可把 ROI 内坐标还原为原图坐标。
    return (
        image[bounded.y:bounded.y + bounded.height, bounded.x:bounded.x + bounded.width],
        bounded.x,
        bounded.y,
    )


def convert_color(image: np.ndarray, color_space: str) -> np.ndarray:
    """Convert BGR input to HSV or Lab."""

    validate_bgr(image)
    normalized = color_space.strip().lower()
    if normalized == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if normalized in {"lab", "l*a*b*"}:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    raise ValueError("color_space must be 'hsv' or 'lab'")


def threshold(image: np.ndarray, lower: Sequence[int], upper: Sequence[int], color_space: str = "hsv") -> np.ndarray:
    """Convert BGR image and apply an inclusive color range threshold."""

    converted = convert_color(image, color_space)
    lower_array = np.asarray(tuple(lower), dtype=np.uint8)
    upper_array = np.asarray(tuple(upper), dtype=np.uint8)
    if lower_array.shape != (3,) or upper_array.shape != (3,):
        raise ValueError("lower and upper thresholds must contain three values")
    # inRange 输出单通道二值图：目标像素为 255，其余为 0。
    return cv2.inRange(converted, lower_array, upper_array)


def morphology(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Remove speckles and close small gaps in a binary mask."""

    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("mask must be a uint8 single-channel image")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if iterations == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 开运算去除散点，闭运算填补目标区域的小孔和断裂。
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def external_contours(mask: np.ndarray):
    """Find external contours in a binary mask."""

    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("mask must be a uint8 single-channel image")
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


__all__ = [
    "ColorRange",
    "convert_color",
    "crop_roi",
    "external_contours",
    "morphology",
    "threshold",
    "validate_bgr",
]
