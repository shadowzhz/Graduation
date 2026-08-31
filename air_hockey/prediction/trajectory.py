"""Trajectory prediction module."""

from dataclasses import dataclass


@dataclass
class PredictedPoint:
    x: float
    y: float
    timestamp: float


class TrajectoryPredictor:
    def predict(self, x, y, vx, vy, future_time):
        """Constant velocity prediction."""
        return PredictedPoint(
            x=x + vx * future_time,
            y=y + vy * future_time,
            timestamp=future_time,
        )
