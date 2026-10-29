"""单目标轨迹跟踪。"""

from __future__ import annotations

from math import hypot

from .types import Track, TrackState


def _detection_confidence(detection) -> float:
    """检测结果可能是 CurlingState(confidence) 或 Detection(score)。"""
    return float(getattr(detection, "confidence", 0.0) or getattr(detection, "score", 0.0))


class StoneTracker:
    """常速模型预测 + 速度指数平滑的单目标跟踪器。"""

    # max_distance      最大允许移动距离
    # max_missed_frames 最大允许漏检帧数
    # velocity_alpha    平滑系数
    # min_confirmed_hits 最少硬检测命中次数
    # replacement_hits  已有轨迹时，远处新目标接管轨迹所需的连续命中次数（1 = 立即接管）
    # max_speed         候选确认时允许的最大像素速度（px/s），冰壶最高约 1100 单位/s ≈ 2500px/s
    def __init__(self, max_distance=80.0, max_missed_frames=5, velocity_alpha=0.2,
                 min_confirmed_hits=2, replacement_hits=1, max_speed=3000.0) -> None:
        if max_distance <= 0.0:
            raise ValueError("最大允许移动距离必须有效")
        if max_missed_frames < 0:
            raise ValueError("最大允许漏检帧数必须是非负数")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("平滑系数必须在 (0, 1]")
        if int(min_confirmed_hits) < 1 or int(min_confirmed_hits) != min_confirmed_hits:
            raise ValueError("目标确认命中次数必须是正整数")
        if int(replacement_hits) < 1 or int(replacement_hits) != replacement_hits:
            raise ValueError("轨迹接管命中次数必须是正整数")
        self.max_distance = float(max_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.velocity_alpha = float(velocity_alpha)
        self.min_confirmed_hits = int(min_confirmed_hits)
        self.replacement_hits = int(replacement_hits)
        self.max_speed = float(max_speed)
        self._track = None
        self._candidate = None          # 等待接管的远处候选 (detection, 连续命中次数)
        self._next_track_id = 1     # 准备分配下一个唯一的 ID,用于区分不同的目标，但是目前只考虑使用一个冰壶

    # 外部调用函数，外部可以直接查看当前轨迹
    @property
    def track(self):
        return self._track

    # Detector 提供一个新的检测结果，用来更新 Tracker
    def update(self, detection):
        """用新的检测结果校正轨迹。"""
        if detection is None:                           # 如果当前帧没检测到目标
            return self._handle_missing_detection()     # 漏检计数
        if self._track is None:                         # 如果没有轨迹
            self._candidate = None
            self._track = self._create_track(detection) # 创建对象
            return [self._track]

        predicted_x, predicted_y = self.predict_position(detection.timestamp)

        predicted_distance = hypot(detection.center_x - predicted_x, detection.center_y - predicted_y)                  # 新检测点与预测位置的距离
        actual_distance = hypot(detection.center_x - self._track.center_x, detection.center_y - self._track.center_y)   # 新检测点与实际位置的距离
        # 如果距离过大，丢弃该检测
        if min(predicted_distance, actual_distance) > self.max_distance:        # min 是为了容忍一定的误差
            return self._handle_unmatched_detection(detection)
        self._candidate = None
        self._update_track(detection)
        return [self._track]

    def predict_position(self, timestamp: float) -> tuple[float, float]:
        """根据最后已知速度外推指定时间戳的位置。"""
        dt = max(0.0, float(timestamp) - self._track.last_timestamp)
        return (self._track.center_x + self._track.vx * dt, self._track.center_y + self._track.vy * dt)

    def _create_track(self, detection) -> Track:
        track = Track(
            track_id=self._next_track_id,
            center_x=detection.center_x,
            center_y=detection.center_y,
            radius=detection.radius,
            vx=0.0,
            vy=0.0,
            last_timestamp=detection.timestamp,
            confidence=_detection_confidence(detection),
            state=(TrackState.ACTIVE if self.min_confirmed_hits == 1 else TrackState.TENTATIVE),
        )
        self._next_track_id += 1
        return track

    def _update_track(self, detection) -> None:
        track = self._track
        dt = detection.timestamp - track.last_timestamp     # 计算两次检测之间经过了多长时间
        if dt > 0.0:
            measured_vx = (detection.center_x - track.center_x) / dt
            measured_vy = (detection.center_y - track.center_y) / dt
            alpha = self.velocity_alpha
            track.vx = (1.0 - alpha) * track.vx + alpha * measured_vx
            track.vy = (1.0 - alpha) * track.vy + alpha * measured_vy
        track.center_x = detection.center_x
        track.center_y = detection.center_y
        track.radius = detection.radius
        track.last_timestamp = detection.timestamp
        track.confidence = _detection_confidence(detection)
        track.age += 1
        track.missed_frames = 0
        track.hits += 1
        track.state = (TrackState.ACTIVE if track.hits >= self.min_confirmed_hits
                       else TrackState.TENTATIVE)

    def _handle_missing_detection(self):
        if self._track is None:
            return []
        if self._track.state == TrackState.TENTATIVE:
            self._track.hits = 0
        self._track.missed_frames += 1
        if self._track.missed_frames > self.max_missed_frames:
            self._track.state = TrackState.LOST
            self._track = None
            return []
        self._track.state = TrackState.LOST
        return [self._track]

    def _handle_unmatched_detection(self, detection):
        hits = 1
        if self._candidate is not None:
            previous, previous_hits = self._candidate
            # 两次候选之间允许的位移随时间间隔放宽：低帧率下快球一帧能走很远
            dt = max(0.0, float(detection.timestamp) - float(previous.timestamp))
            gate = self.max_distance + self.max_speed * dt
            if hypot(detection.center_x - previous.center_x,
                     detection.center_y - previous.center_y) <= gate:
                hits = previous_hits + 1
        if hits < self.replacement_hits:
            # 还没确认：旧轨迹按漏检处理（宽限期内继续外推），候选留待下一帧
            self._candidate = (detection, hits)
            return self._handle_missing_detection()
        self._candidate = None
        self._track = self._create_track(detection)
        return [self._track]
