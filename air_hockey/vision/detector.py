"""颜色阈值 + 轮廓几何约束的冰壶检测，检测这一帧里冰壶在哪里"""

from dataclasses import replace
from math import inf, pi        # inf 代表无穷大

import cv2

from camera.types import Frame

from .preprocess import crop_roi, external_contours, morphology, threshold, validate_bgr
from .types import Detection, ROI

# 定义冰壶检测器类，输出冰壶的位置、大小、圆形度、时间戳和分数
class StoneDetector:
    def __init__(self, roi=None, color_space="hsv", lower=(0, 40, 30), upper=(180, 255, 255),
                 morphology_kernel=5, morphology_iterations=1, min_area=100.0, max_area=inf,
                 min_radius=3.0, max_radius=inf, min_circularity=0.55) -> None:     # None 代表这一帧没有找到符合要求的冰壶
        if isinstance(roi, tuple):      # 在什么区域找
            roi = ROI(*roi)
        if min_area < 0 or max_area < min_area:             # 面积过滤
            raise ValueError("invalid area limits")
        if min_radius < 0 or max_radius < min_radius:       # 半径限定
            raise ValueError("invalid radius limits")
        if not 0.0 <= min_circularity <= 1.0:               # 圆形度过滤
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

    # 拿一帧图像尝试找冰壶，新增 dynamic_roi 参数
    def detect(self, frame, dynamic_roi=None):
        """跑一遍检测管线，返回得分最高的候选，找不到返回 None。"""
        # 优先使用动态 ROI，如果没有传动态 ROI（比如刚启动或跟丢了），则使用全局自定的 self.roi
        current_roi = dynamic_roi if dynamic_roi is not None else self.roi
        
        cropped, offset_x, offset_y = crop_roi(frame.image, current_roi)       # 将图像按 current_roi 处理
        # 裁剪完没有数据则返回
        if cropped.size == 0:
            return None
        
        mask = threshold(cropped, self.lower, self.upper, self.color_space)             # 颜色筛选
        mask = morphology(mask, self.morphology_kernel, self.morphology_iterations)     # 形态学处理

        candidates = []

        # 轮廓提取
        for contour in external_contours(mask):
            area = float(cv2.contourArea(contour))                              # 计算这个轮廓有多大
            if area < self.min_area or area > self.max_area:                    # 面积过滤
                continue
            perimeter = float(cv2.arcLength(contour, True))                     # 计算轮廓周长
            if perimeter <= 0.0:                                                # 周长检查
                continue
            circularity = 4.0 * pi * area / (perimeter * perimeter)             # 计算圆形度
            if circularity < self.min_circularity:                              # 圆形度过滤
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)      # 最小外接圆
            if radius < self.min_radius or radius > self.max_radius:            # 外接圆半径过滤
                continue

            # 开始生成真正的检测结果 (此时的 x,y 是相对于原畸变图像的)
            detection = Detection(
                # 坐标偏移
                center_x=float(center_x + offset_x),
                center_y=float(center_y + offset_y),

                radius=float(radius),
                area=area,
                timestamp=frame.timestamp,              # 时间戳，用于计算轨迹预测
                circularity=float(circularity),
            )
            score = self._score_candidate(detection)    # 计算检测分数
            candidates.append((score, replace(detection, score=score)))         # 保存候选
        # 没有分数直接返回
        if not candidates:
            return None
        # 分数相同取面积大的
        return max(candidates, key=lambda item: (item[0], item[1].area))[1]

    # 给候选冰壶评分
    def _score_candidate(self, detection):
        # 圆形度为主，面积项防止刚好过线的小噪点拿最高分
        area_score = min(1.0, detection.area / max(self.min_area * 4.0, 1.0))
        return 0.8 * detection.circularity + 0.2 * area_score
