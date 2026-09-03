"""视觉帧处理管线。"""

from pathlib import Path

import cv2
import numpy as np

from camera.types import Frame


class VisionPipeline:
    """对相机帧执行可选的畸变校正。"""

    def __init__(self, calibration_file=None, enabled=True):
        self.enabled = bool(enabled)        # 强制将输入参数转换为布尔值
        self.calibration_file = calibration_file
        self.camera_matrix = None       # 相机矩阵
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.map1 = None
        self.map2 = None
        self.roi = None
        self._map_size = None

        # 如果没有启动或者没有修正文件就返回
        if not self.enabled or calibration_file is None:
            return

        calibration_path = Path(calibration_file)
        if not calibration_path.is_file():
            return

        with np.load(calibration_path) as data:
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            image_size = data.get("image_size")

        if image_size is not None:
            self._init_maps(tuple(int(value) for value in image_size))

    def _init_maps(self, image_size):
        """按给定的 (width, height) 预计算畸变校正映射表。"""
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

    def process(self, frame):
        """校正一帧图像，并保留原帧的时间戳和序号。"""
        # 如果没有启动或者相机矩阵为空就直接返回当前帧
        if not self.enabled or self.camera_matrix is None:
            return frame

        height, width = frame.image.shape[:2]
        # 如果尺寸不一致就重新初始化
        if self._map_size != (width, height):
            self._init_maps((width, height))

        # 使用生成的的映射表对图像进行畸变
        image = cv2.remap(frame.image, self.map1, self.map2, cv2.INTER_LINEAR)      # 图像校正
        # 区域剪裁
        x, y, roi_width, roi_height = self.roi
        if roi_width > 0 and roi_height > 0:
            image = image[y:y + roi_height, x:x + roi_width]
        return Frame(image=image, timestamp=frame.timestamp, sequence=frame.sequence)
