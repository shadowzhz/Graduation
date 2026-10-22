"""颜色阈值 + 轮廓几何约束的冰壶检测，检测这一帧里冰壶在哪里"""

from dataclasses import replace
from math import hypot, inf, pi        # inf 代表无穷大

import cv2

from ..game_state import CurlingState

from .preprocess import crop_roi, external_contours, morphology, threshold
from .types import Detection, ROI

# 全幅搜索降采样：像素数超过这个值才启用（小 ROI 不受影响）
SEARCH_DOWNSCALE_AREA = 250_000
SEARCH_DOWNSCALE_SCALE = 0.5
SEARCH_DOWNSCALE_INTERP = cv2.INTER_NEAREST   # 最近邻：缩放开销最小，阈值不受影响

# 冰壶检测器：对外返回统一 CurlingState（raw 像素坐标 + 置信度），找不到返回 None
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

    @staticmethod
    def select_candidate(candidates, *, predicted_position=None, previous_position=None,
                         previous_radius=None, max_distance=80.0):
        """根据原轨迹预测/历史位置与候选质量做关联，不匹配回退最高分候选。"""
        if not candidates:
            return None
        if predicted_position is None and previous_position is None:
            return candidates[0]

        max_dist = max(float(max_distance), 1e-6)
        matching = []

        for c in candidates:
            d_pred = hypot(c.x - predicted_position[0], c.y - predicted_position[1]) if predicted_position else None
            d_prev = hypot(c.x - previous_position[0], c.y - previous_position[1]) if previous_position else None
            dists = [d for d in (d_pred, d_prev) if d is not None]
            if not dists:
                continue
            gate_dist = min(dists)
            if gate_dist > max_dist:
                continue

            match_dist = d_pred if (d_pred is not None and d_pred <= max_dist) else gate_dist
            dist_score = max(0.0, 1.0 - (match_dist / max_dist))

            if previous_radius is not None and previous_radius > 0 and c.radius > 0:
                radius_sim = min(c.radius, previous_radius) / max(c.radius, previous_radius)
                score = 0.6 * c.confidence + 0.3 * dist_score + 0.1 * radius_sim
            else:
                score = 0.65 * c.confidence + 0.35 * dist_score

            matching.append((score, c))

        if not matching:
            return candidates[0]
        return max(matching, key=lambda item: item[0])[1]

    def detect_candidates(self, frame, dynamic_roi=None):
        """跑一遍检测管线，返回按原 (score, area) 稳定降序排序的 CurlingState 候选列表。"""
        current_roi = dynamic_roi if dynamic_roi is not None else self.roi

        cropped, offset_x, offset_y = crop_roi(frame.image, current_roi)       # 将图像按 current_roi 处理
        if cropped.size == 0:
            return []

        # ★ 丢检时会退回全幅 ROI，阈值+形态学+轮廓是这时掉帧率的主因。
        #   大区域先降采样搜索，坐标/面积/半径按 scale 换算回原分辨率；
        #   小 ROI（正常追踪）面积远达不到阈值，走原路径，行为完全不变。
        scale = 1.0
        if cropped.shape[0] * cropped.shape[1] > SEARCH_DOWNSCALE_AREA:
            scale = SEARCH_DOWNSCALE_SCALE
            cropped = cv2.resize(cropped, None, fx=scale, fy=scale,
                                 interpolation=SEARCH_DOWNSCALE_INTERP)

        mask = threshold(cropped, self.lower, self.upper, self.color_space)             # 颜色筛选
        kernel = self.morphology_kernel if scale >= 1.0 else max(3, int(round(self.morphology_kernel * scale)))
        mask = morphology(mask, kernel, self.morphology_iterations)                     # 形态学处理

        raw = []
        for contour in external_contours(mask):
            area = float(cv2.contourArea(contour))                              # 计算这个轮廓有多大
            if area < self.min_area * scale * scale or area > self.max_area * scale * scale:
                continue
            perimeter = float(cv2.arcLength(contour, True))                     # 计算轮廓周长
            if perimeter <= 0.0:                                                # 周长检查
                continue
            circularity = 4.0 * pi * area / (perimeter * perimeter)             # 计算圆形度
            if circularity < self.min_circularity:                              # 圆形度过滤
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)      # 最小外接圆
            if radius < self.min_radius * scale or radius > self.max_radius * scale:
                continue

            detection = Detection(
                center_x=float(center_x / scale + offset_x),
                center_y=float(center_y / scale + offset_y),
                radius=float(radius / scale),
                area=float(area / (scale * scale)),
                timestamp=frame.timestamp,              # 时间戳，用于计算轨迹预测
                circularity=float(circularity),
            )
            score = self._score_candidate(detection)    # 计算检测分数
            raw.append((score, area, replace(detection, score=score)))

        if not raw:
            return []
        raw.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [CurlingState.from_detection(item[2]) for item in raw]

    def detect(self, frame, dynamic_roi=None):
        """跑一遍检测管线，返回得分最高候选对应的 CurlingState，找不到返回 None。"""
        return self.select_candidate(self.detect_candidates(frame, dynamic_roi=dynamic_roi))
    # 给候选冰壶评分
    def _score_candidate(self, detection):
        # 圆形度为主，面积项防止刚好过线的小噪点拿最高分
        area_score = min(1.0, detection.area / max(self.min_area * 4.0, 1.0))
        return 0.8 * detection.circularity + 0.2 * area_score
