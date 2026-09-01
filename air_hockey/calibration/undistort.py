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
        self.new_camera_matrix = None
        self.roi = None
        self.map1 = None
        self.map2 = None
        self._map_size = None

        if calibration_file:
            path = Path(calibration_file)
            if path.exists():
                data = np.load(path)
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
                self.loaded = True
                image_size = data.get("image_size")
                if image_size is not None:
                    self._init_maps(tuple(int(v) for v in image_size))

    def _init_maps(self, image_size):
        width, height = image_size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (width, height),
            0,
            (width, height),
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )
        self._map_size = (width, height)

    def apply(self, frame):
        if not self.enabled or not self.loaded:
            return frame

        height, width = frame.image.shape[:2]
        if self._map_size != (width, height):
            self._init_maps((width, height))

        image = cv2.remap(frame.image, self.map1, self.map2, cv2.INTER_LINEAR)
        x, y, roi_w, roi_h = self.roi
        if roi_w > 0 and roi_h > 0:
            image = image[y:y + roi_h, x:x + roi_w]
        return replace(frame, image=image)

    def apply_image(self, image):
        if not self.enabled or not self.loaded:
            return image

        height, width = image.shape[:2]
        if self._map_size != (width, height):
            self._init_maps((width, height))

        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        x, y, roi_w, roi_h = self.roi
        if roi_w > 0 and roi_h > 0:
            undistorted = undistorted[y:y + roi_h, x:x + roi_w]
        return undistorted
