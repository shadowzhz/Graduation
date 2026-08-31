"""Camera distortion correction layer."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


class Undistorter:
    def __init__(self, calibration_file=None, enabled=True):
        self.enabled = bool(enabled)
        self.loaded = False
        self.camera_matrix = None
        self.dist_coeffs = None

        if calibration_file:
            path = Path(calibration_file)
            if path.exists():
                data = np.load(path)
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
                self.loaded = True

    def apply(self, frame):
        if not self.enabled or not self.loaded:
            return frame

        image = cv2.undistort(
            frame.image,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return replace(frame, image=image)
