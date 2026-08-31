"""视觉帧处理管线。"""

from pathlib import Path

import cv2
import numpy as np

from camera.types import Frame


class VisionPipeline:
    """对相机帧执行可选的畸变校正。"""

    def __init__(self, calibration_file=None, enabled=True):
        self.enabled = bool(enabled)
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None

        if not self.enabled or calibration_file is None:
            return

        calibration_path = Path(calibration_file)
        if not calibration_path.is_file():
            return

        with np.load(calibration_path) as data:
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

    def process(self, frame):
        """校正一帧图像，并保留原帧的时间戳和序号。"""
        if not self.enabled or self.camera_matrix is None:
            return frame

        image = cv2.undistort(frame.image, self.camera_matrix, self.dist_coeffs)
        return Frame(image=image, timestamp=frame.timestamp, sequence=frame.sequence)
