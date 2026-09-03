"""单目标轨迹跟踪。"""

from dataclasses import replace
from math import hypot

from .types import Detection, Track, TrackState


class StoneTracker:
    """常速模型预测 + 速度指数平滑的单目标跟踪器。"""

    # max_distance      最大允许移动距离
    # max_missed_frames 最大允许漏检帧数
    # velocity_alpha    平滑系数
    def __init__(self, max_distance=80.0, max_missed_frames=5, velocity_alpha=0.2) -> None:
        if max_distance <= 0.0:
            raise ValueError("最大允许移动距离必须有效")
        if max_missed_frames < 0:
            raise ValueError("最大允许漏检帧数必须是非负数")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("平滑系数必须在 (0, 1]")
        self.max_distance = float(max_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.velocity_alpha = float(velocity_alpha)
        self._track = None
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
            self._track = self._create_track(detection) # 创建对象
            return [self._track]

        predicted_x, predicted_y = self._predict_position(detection.timestamp)

        predicted_distance = hypot(detection.center_x - predicted_x, detection.center_y - predicted_y)                  # 新检测点与预测位置的距离
        actual_distance = hypot(detection.center_x - self._track.center_x, detection.center_y - self._track.center_y)   # 新检测点与实际位置的距离
        # 如果距离过大，丢弃该检测
        if min(predicted_distance, actual_distance) > self.max_distance:        # min 是为了容忍一定的误差
            return self._handle_unmatched_detection(detection)
        self._update_track(detection)
        return [self._track]

    def predict(self, timestamp):
        """在检测间隔帧上按常速模型返回预测轨迹，不修改内部观测状态。"""
        # 如果没有跟踪的目标，直接返回空列表
        if self._track is None:
            return []
        timestamp = float(timestamp)                    # 当前帧的时间戳
        dt = timestamp - self._track.last_timestamp     # 预测时间差
        missed = self._track.missed_frames + 1          # 漏检计数 
        # 如果超过最大允许漏检帧数，返回空列表，否则创建新对象
        if missed > self.max_missed_frames:
            return []
        return [replace(
            self._track,

            # 匀速运动模型
            center_x=self._track.center_x + self._track.vx * dt,
            center_y=self._track.center_y + self._track.vy * dt,

            last_timestamp=timestamp,
            age=self._track.age + 1,
            missed_frames=missed,
            state=TrackState.LOST,
        )]

    def reset(self) -> None:
        self._track = None

    def _predict_position(self, timestamp):
        dt = max(0.0, timestamp - self._track.last_timestamp)
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
        track.age += 1
        track.missed_frames = 0
        track.state = TrackState.ACTIVE

    def _handle_missing_detection(self):
        if self._track is None:
            return []
        self._track.missed_frames += 1
        if self._track.missed_frames > self.max_missed_frames:
            self._track.state = TrackState.LOST
            self._track = None
            return []
        self._track.state = TrackState.LOST
        return [self._track]

    def _handle_unmatched_detection(self, detection):
        self._track = self._create_track(detection)
        return [self._track]
