"""检测前的图像预处理。"""

import cv2
import numpy as np

from .types import ROI


def validate_bgr(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR uint8 HxWx3 ndarray")
    return image


def crop_roi(image, roi):
    """裁出 ROI，返回 (裁剪图, x 偏移, y 偏移)。"""
    validate_bgr(image)
    if roi is None:
        return image, 0, 0
    bounded = roi.clamp(image.shape[1], image.shape[0])
    if bounded is None:
        return image[0:0, 0:0], 0, 0
    return (
        image[bounded.y:bounded.y + bounded.height, bounded.x:bounded.x + bounded.width],
        bounded.x,
        bounded.y,
    )


def convert_color(image, color_space):
    if color_space == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space in ("lab", "l*a*b*"):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    raise ValueError("color_space must be 'hsv' or 'lab'")


def threshold(image, lower, upper, color_space="hsv"):
    """按颜色上下限出二值图。"""
    converted = convert_color(image, color_space)
    lower_array = np.asarray(tuple(lower), dtype=np.uint8)
    upper_array = np.asarray(tuple(upper), dtype=np.uint8)
    if lower_array.shape != (3,) or upper_array.shape != (3,):
        raise ValueError("lower and upper thresholds must contain three values")
    return cv2.inRange(converted, lower_array, upper_array)


def morphology(mask, kernel_size=5, iterations=1):
    """开运算去散点，闭运算补小孔。"""
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("mask must be a uint8 single-channel image")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if iterations == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def external_contours(mask):
    # [-2] 是为了同时兼容新旧 OpenCV 的 findContours 返回值
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
