"""基于棋盘格的单目相机标定。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    """相机标定结果。"""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float
    rms: float


class CameraCalibrator:
    """使用棋盘格角点计算相机内参和畸变参数。"""

    def __init__(self, pattern_size: tuple[int, int], square_size: float = 1.0):
        cols, rows = pattern_size
        if cols < 2 or rows < 2:
            raise ValueError("pattern_size 必须是至少 2x2 的内角点")
        if square_size <= 0:
            raise ValueError("square_size 必须大于 0")

        self.pattern_size = (cols, rows)
        self.square_size = float(square_size)
        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size
        self._template_object_points = objp

    @property
    def sample_count(self) -> int:
        return len(self._image_points)

    def detect(self, image: np.ndarray):
        """检测棋盘格，返回可供绘制的角点；未检测到时返回 None。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            return None

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return refined

    def add_sample(self, image: np.ndarray, corners=None) -> bool:
        """加入一组标定样本；返回是否成功加入。"""
        if corners is None:
            corners = self.detect(image)
        if corners is None:
            return False

        height, width = image.shape[:2]
        image_size = (width, height)
        if self._image_size is None:
            self._image_size = image_size
        elif self._image_size != image_size:
            raise ValueError("所有标定图像必须保持相同分辨率")

        self._object_points.append(self._template_object_points.copy())
        self._image_points.append(corners)
        return True

    def calibrate(self) -> CalibrationResult:
        """执行标定并计算平均重投影误差。"""
        if self.sample_count < 5:
            raise ValueError("至少采集 5 组有效棋盘格图像，建议 15~25 组")
        if self._image_size is None:
            raise ValueError("没有有效标定样本")

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points,
            self._image_points,
            self._image_size,
            None,
            None,
        )

        total_error = 0.0
        total_points = 0
        for objp, corners, rvec, tvec in zip(
            self._object_points, self._image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                objp, rvec, tvec, camera_matrix, dist_coeffs
            )
            total_error += cv2.norm(corners, projected, cv2.NORM_L2)
            total_points += len(corners)

        reprojection_error = total_error / total_points
        return CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=self._image_size,
            reprojection_error=float(reprojection_error),
            rms=float(rms),
        )

    @staticmethod
    def save(result: CalibrationResult, path: str | Path) -> None:
        """保存为 NumPy npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix=result.camera_matrix,
            dist_coeffs=result.dist_coeffs,
            image_size=np.asarray(result.image_size, dtype=np.int32),
            reprojection_error=result.reprojection_error,
            rms=result.rms,
        )

    @staticmethod
    def load(path: str | Path) -> CalibrationResult:
        """加载已保存的标定结果。"""
        with np.load(path) as data:
            return CalibrationResult(
                camera_matrix=data["camera_matrix"],
                dist_coeffs=data["dist_coeffs"],
                image_size=tuple(int(x) for x in data["image_size"]),
                reprojection_error=float(data["reprojection_error"]),
                rms=float(data["rms"]),
            )

    @staticmethod
    def undistort(image: np.ndarray, result: CalibrationResult) -> np.ndarray:
        """使用标定结果进行畸变校正。"""
        return cv2.undistort(
            image,
            result.camera_matrix,
            result.dist_coeffs,
        )
