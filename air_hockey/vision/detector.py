"""Contour-based stone detector."""

from __future__ import annotations

from math import inf, pi
from typing import Optional, Sequence, Tuple, Union

import cv2

from camera.types import Frame

from .preprocess import crop_roi, external_contours, morphology, threshold, validate_bgr
from .types import Detection, ROI


class StoneDetector:
    """使用颜色阈值和轮廓几何约束检测近似圆形目标。"""

    def __init__(
        self,
        *,
        roi: Optional[Union[ROI, Tuple[int, int, int, int]]] = None,
        color_space: str = "hsv",
        lower: Sequence[int] = (0, 40, 30),
        upper: Sequence[int] = (180, 255, 255),
        morphology_kernel: int = 5,
        morphology_iterations: int = 1,
        min_area: float = 100.0,
        max_area: float = inf,
        min_radius: float = 3.0,
        max_radius: float = inf,
        min_circularity: float = 0.55,
    ) -> None:
        if isinstance(roi, tuple):
            roi = ROI(*roi)
        if min_area < 0 or max_area < min_area:
            raise ValueError("invalid area limits")
        if min_radius < 0 or max_radius < min_radius:
            raise ValueError("invalid radius limits")
        if not 0.0 <= min_circularity <= 1.0:
            raise ValueError("min_circularity must be between 0 and 1")
        self.roi = roi
        self.color_space = color_space
        self.lower = tuple(lower)
        self.upper = tuple(upper)
        self.morphology_kernel = morphology_kernel
        self.morphology_iterations = morphology_iterations
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.min_circularity = float(min_circularity)

    def detect(self, frame: Frame) -> Optional[Detection]:
        """Run the pipeline and return the highest-scoring candidate."""

        if not isinstance(frame, Frame):
            raise TypeError("detect expects a camera.types.Frame")
        validate_bgr(frame.image)
        # 检测在 ROI 内完成，但最终坐标仍返回原始图像坐标。
        cropped, offset_x, offset_y = crop_roi(frame.image, self.roi)
        if cropped.size == 0:
            return None
        mask = threshold(cropped, self.lower, self.upper, self.color_space)
        mask = morphology(mask, self.morphology_kernel, self.morphology_iterations)
        candidates = []
        for contour in external_contours(mask):
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            # 圆形度为 1 时最接近圆，周长退化的轮廓已在上面排除。
            circularity = 4.0 * pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue
            detection = Detection(
                center_x=float(center_x + offset_x),
                center_y=float(center_y + offset_y),
                radius=float(radius),
                area=area,
                timestamp=frame.timestamp,
                circularity=float(circularity),
            )
            score = self._score_candidate(detection)
            candidates.append((score, Detection(
                center_x=detection.center_x,
                center_y=detection.center_y,
                radius=detection.radius,
                area=detection.area,
                timestamp=detection.timestamp,
                circularity=detection.circularity,
                score=score,
            )))
        if not candidates:
            return None
        # 先比较综合评分，评分相同时优先保留面积更大的候选。
        return max(candidates, key=lambda item: (item[0], item[1].area))[1]

    def _score_candidate(self, detection: Detection) -> float:
        """Score valid candidates by circularity and useful contour size."""

        # Once a contour is comfortably above the noise floor, circularity
        # dominates. The area term prevents tiny barely-valid blobs winning.
        area_score = min(1.0, detection.area / max(self.min_area * 4.0, 1.0))
        return 0.8 * detection.circularity + 0.2 * area_score


__all__ = ["StoneDetector"]
