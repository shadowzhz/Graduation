"""Single-object tracker for vision detections."""

from __future__ import annotations

from math import hypot
from typing import List, Optional

from .types import Detection, Track, TrackState


class StoneTracker:
    """Track one detected stone across consecutive frames.

    The first version intentionally uses a simple nearest-distance model.
    Kalman filtering and multi-object association can be added later.
    """

    def __init__(
        self,
        *,
        max_distance: float = 80.0,
        max_missed_frames: int = 5,
    ) -> None:
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")

        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")

        self.max_distance = float(max_distance)
        self.max_missed_frames = int(max_missed_frames)

        self._track: Optional[Track] = None
        self._next_track_id = 1

    @property
    def track(self) -> Optional[Track]:
        """Return the current track, if one exists."""
        return self._track

    def update(self, detection: Optional[Detection]) -> List[Track]:
        """Update the tracker with the latest detection.

        Returns a list for future compatibility with multi-object tracking.
        The current implementation contains at most one track.
        """
        if detection is None:
            return self._handle_missing_detection()

        if self._track is None:
            self._track = self._create_track(detection)
            return [self._track]

        distance = hypot(
            detection.center_x - self._track.center_x,
            detection.center_y - self._track.center_y,
        )

        if distance > self.max_distance:
            return self._handle_unmatched_detection(detection)

        self._update_track(detection)
        return [self._track]

    def reset(self) -> None:
        """Remove the current track and reset tracker state."""
        self._track = None

    def _create_track(self, detection: Detection) -> Track:
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

    def _update_track(self, detection: Detection) -> None:
        assert self._track is not None

        previous_timestamp = self._track.last_timestamp
        dt = detection.timestamp - previous_timestamp

        if dt > 0.0:
            self._track.vx = (
                detection.center_x - self._track.center_x
            ) / dt

            self._track.vy = (
                detection.center_y - self._track.center_y
            ) / dt

        self._track.center_x = detection.center_x
        self._track.center_y = detection.center_y
        self._track.radius = detection.radius
        self._track.last_timestamp = detection.timestamp

        self._track.age += 1
        self._track.missed_frames = 0
        self._track.state = TrackState.ACTIVE

    def _handle_missing_detection(self) -> List[Track]:
        if self._track is None:
            return []

        self._track.missed_frames += 1

        if self._track.missed_frames > self.max_missed_frames:
            self._track.state = TrackState.LOST
            self._track = None
            return []

        self._track.state = TrackState.LOST
        return [self._track]

    def _handle_unmatched_detection(
        self,
        detection: Detection,
    ) -> List[Track]:
        """Start a new track when the detection is too far away."""
        self._track = self._create_track(detection)
        return [self._track]


__all__ = ["StoneTracker"]
