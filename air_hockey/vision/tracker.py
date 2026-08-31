"""单目标轨迹跟踪。"""

from math import hypot

from .types import Detection, Track, TrackState


class StoneTracker:
    """常速模型预测 + 速度指数平滑的单目标跟踪器。"""

    def __init__(self, max_distance=80.0, max_missed_frames=5, velocity_alpha=0.2) -> None:
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        self.max_distance = float(max_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.velocity_alpha = float(velocity_alpha)
        self._track = None
        self._next_track_id = 1

    @property
    def track(self):
        return self._track

    def update(self, detection):
        """喂入检测结果，校正当前轨迹。"""
        if detection is None:
            return self._handle_missing_detection()

        if self._track is None:
            self._track = self._create_track(detection)
            return [self._track]

        predicted_x, predicted_y = self._predict_position(detection.timestamp)
        predicted_distance = hypot(
            detection.center_x - predicted_x,
            detection.center_y - predicted_y,
        )
        actual_distance = hypot(
            detection.center_x - self._track.center_x,
            detection.center_y - self._track.center_y,
        )
        if min(predicted_distance, actual_distance) > self.max_distance:
            return self._handle_unmatched_detection(detection)

        self._update_track(detection)
        return [self._track]

    def predict(self, timestamp):
        """在没有新检测结果的帧上按常速模型推进轨迹。"""
        if self._track is None:
            return []

        dt = max(0.0, float(timestamp) - self._track.last_timestamp)
        self._track.center_x += self._track.vx * dt
        self._track.center_y += self._track.vy * dt
        self._track.last_timestamp = float(timestamp)
        self._track.age += 1
        self._track.missed_frames += 1

        if self._track.missed_frames > self.max_missed_frames:
            self._track.state = TrackState.LOST
            self._track = None
            return []

        self._track.state = TrackState.LOST
        return [self._track]

    def reset(self) -> None:
        self._track = None

    def _predict_position(self, timestamp):
        dt = max(0.0, timestamp - self._track.last_timestamp)
        return (
            self._track.center_x + self._track.vx * dt,
            self._track.center_y + self._track.vy * dt,
        )

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
        dt = detection.timestamp - track.last_timestamp

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
