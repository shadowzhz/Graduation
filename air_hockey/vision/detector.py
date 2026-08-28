"""颜色阈值 + 轮廓几何约束的冰壶检测。"""

from dataclasses import replace
from math import inf, pi

import cv2

from camera.types import Frame

from .preprocess import crop_roi, external_contours, morphology, threshold, validate_bgr
from .types import Detection, ROI


class StoneDetector:
    def __init__(self, roi=None, color_space="hsv", lower=(0, 40, 30), upper=(180, 255, 255),
                 morphology_kernel=5, morphology_iterations=1, min_area=100.0, max_area=inf,
                 min_radius=3.0, max_radius=inf, min_circularity=0.55) -> None:
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

    def detect(self, frame):
        """跑一遍检测管线，返回得分最高的候选，找不到返回 None。"""
        if not isinstance(frame, Frame):
            raise TypeError("detect expects a camera.types.Frame")
        validate_bgr(frame.image)
        # 在 ROI 里检测，返回坐标时把偏移加回去
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
            candidates.append((score, replace(detection, score=score)))
        if not candidates:
            return None
        # 分数相同取面积大的
        return max(candidates, key=lambda item: (item[0], item[1].area))[1]

    def _score_candidate(self, detection):
        # 圆形度为主，面积项防止刚好过线的小噪点拿最高分
        area_score = min(1.0, detection.area / max(self.min_area * 4.0, 1.0))
        return 0.8 * detection.circularity + 0.2 * area_score
