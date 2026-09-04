"""视觉帧处理管线。"""

from pathlib import Path

import cv2
import numpy as np

from camera.types import Frame


class VisionPipeline:
    """对相机帧和坐标执行畸变校正。"""

    def __init__(self, calibration_file=None, enabled=True):
        self.enabled = bool(enabled)        # 强制将输入参数转换为布尔值
        self.calibration_file = calibration_file
        self.camera_matrix = None       # 相机矩阵
        self.dist_coeffs = None
        self.new_camera_matrix = None
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
        """预计算最优相机矩阵，删除了原本生成大映射表的代码。"""
        width, height = image_size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (width, height),
            0,
            (width, height),
        )
        self._map_size = (width, height)

    def process(self, frame):
        """直接返回原始帧，将畸变校正推迟到最终的坐标点上（巨幅降低CPU延迟）。"""
        # 确保初始化过了新相机矩阵
        if self.enabled and self.camera_matrix is not None:
            height, width = frame.image.shape[:2]
            if self._map_size != (width, height):
                self._init_maps((width, height))
        
        # 直接返回原图，不再执行极其耗时的 cv2.remap
        return frame

    def undistort_point(self, x, y):
        """
        对检测出的坐标单个像素点执行逆畸变计算。
        由于取消了全图裁剪，为了保证坐标系和旧版本行为一致，这里会自动扣除裁剪偏移。
        """
        if not self.enabled or self.camera_matrix is None:
            return x, y

        # OpenCV 的 undistortPoints 需要特定 shape 的 3D numpy 数组
        pts = np.array([[[x, y]]], dtype=np.float32)

        # 核心：点级别数学逆向畸变，耗时接近 0ms
        undistorted_pts = cv2.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.new_camera_matrix
        )

        ux = float(undistorted_pts[0][0][0])
        uy = float(undistorted_pts[0][0][1])

        # 扣除原版 remap 逻辑中裁剪所带来的像素偏移，保证跟原有的物理映射完全一致
        if self.roi is not None:
            cx, cy, roi_width, roi_height = self.roi
            if roi_width > 0 and roi_height > 0:
                ux -= cx
                uy -= cy

        return ux, uy
