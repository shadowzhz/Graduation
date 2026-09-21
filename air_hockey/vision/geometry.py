"""相机几何与坐标变换模块。

负责相机原始像素坐标 (raw pixel)、去畸变像素坐标 (undistorted pixel)
以及桌面/球台坐标 (table/rink coordinate) 之间的单向清晰转换。

坐标流定义：
  raw pixel -> undistorted pixel -> (Homography) -> table coordinate
  table coordinate -> undistorted pixel -> raw pixel

undistorted -> table 的映射有两种模式：
  1. 配置了 homography_matrix 时，使用 undistorted pixel -> table 的 3x3 单应矩阵；
  2. 未配置时，回退为基于 table_roi 的轴对齐线性缩放（历史行为）。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class CameraGeometry:
    """相机几何与多坐标系变换器。"""

    def __init__(
        self,
        rink_bounds: Tuple[float, float, float, float],
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
        table_roi: Optional[Tuple[float, float, float, float]] = None,
        homography_matrix: Optional[np.ndarray] = None,
        homography_image_size: Optional[Tuple[int, int]] = None,
        enabled: bool = True,
    ) -> None:
        if rink_bounds is None:
            raise ValueError("rink_bounds must be provided")
        rink_bounds_tuple = tuple(float(v) for v in rink_bounds)
        if len(rink_bounds_tuple) != 4:
            raise ValueError(f"rink_bounds must contain exactly 4 values (min_x, max_x, min_y, max_y), got {rink_bounds}")
        self.rink_bounds = rink_bounds_tuple

        self.enabled = bool(enabled)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64) if camera_matrix is not None else None
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else None
        self.image_size = None
        self.new_camera_matrix = None
        self.table_roi = None
        self.homography_matrix = None
        self.homography_inverse = None
        self.homography_image_size = None

        if image_size is not None:
            self.set_image_size(image_size)

        if table_roi is not None:
            self.set_table_roi(table_roi)

        if homography_matrix is not None:
            self.set_homography(homography_matrix, homography_image_size)

    @classmethod
    def from_calibration_file(
        cls,
        calibration_file: Optional[Union[str, Path]],
        rink_bounds: Tuple[float, float, float, float],
        table_roi: Optional[Tuple[float, float, float, float]] = None,
        table_calibration_file: Optional[Union[str, Path]] = None,
        enabled: bool = True,
    ) -> "CameraGeometry":
        """从标定 .npz 文件构建 CameraGeometry。

        calibration_file 提供相机内参 + 畸变；table_calibration_file 提供
        undistorted -> table 的球台四点单应矩阵。

        table_calibration_file 为 None 时明确使用旧 ROI 线性映射；
        非 None 时文件必须存在，否则直接抛 FileNotFoundError（不做静默 fallback）。
        """
        if rink_bounds is None:
            raise ValueError("rink_bounds must be provided")

        homography = cls._load_homography(table_calibration_file)
        homography_matrix = homography[0] if homography is not None else None
        homography_image_size = homography[1] if homography is not None else None

        if not enabled:
            return cls(
                rink_bounds=rink_bounds,
                table_roi=table_roi,
                homography_matrix=homography_matrix,
                homography_image_size=homography_image_size,
                enabled=False,
            )

        if calibration_file is None:
            raise FileNotFoundError("calibration_file must be provided when distortion correction is enabled")

        path = Path(calibration_file)
        if not path.is_file():
            raise FileNotFoundError(f"calibration file not found: {calibration_file}")

        with np.load(path) as data:
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            raw_size = data.get("image_size")
            image_size = tuple(int(v) for v in raw_size) if raw_size is not None else None

        return cls(
            rink_bounds=rink_bounds,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            table_roi=table_roi,
            homography_matrix=homography_matrix,
            homography_image_size=homography_image_size,
            enabled=True,
        )

    @staticmethod
    def _load_homography(table_calibration_file: Optional[Union[str, Path]]):
        """读取 undistorted -> table 的单应矩阵与标定分辨率。

        path 为 None 时返回 None（走旧 ROI 线性映射）；path 非 None 但文件不存在
        时直接抛 FileNotFoundError，不做静默 fallback。

        image_size 为必填项：旧版 Homography 文件缺少该字段会让 homography_image_size
        变成 None，从而静默绕过运行分辨率检查，因此这里直接拒绝。
        """
        if table_calibration_file is None:
            return None
        path = Path(table_calibration_file)
        if not path.is_file():
            raise FileNotFoundError(f"table calibration file not found: {table_calibration_file}")
        with np.load(path) as data:
            if "image_size" not in data.files:
                raise ValueError(
                    f"table calibration file {table_calibration_file} is missing required 'image_size'; "
                    "regenerate it with air_hockey/tools/calibrate_table.py"
                )
            homography_matrix = data["homography_matrix"]
            raw_size = data["image_size"]
        image_size = tuple(int(v) for v in np.asarray(raw_size).ravel())
        if len(image_size) != 2:
            raise ValueError(f"image_size must contain exactly 2 values (width, height), got {raw_size}")
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size width and height must be positive, got {raw_size}")
        return homography_matrix, image_size

    def set_homography(
        self,
        homography_matrix: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """设置 undistorted pixel -> table 的 3x3 单应矩阵，并立即求解逆矩阵。

        Homography 基于去畸变坐标标定，因此要求相机畸变校正处于启用状态
        （enabled=True）；disable_undistort + Homography 属于坐标定义冲突，直接报错。
        """
        if not self.enabled:
            raise ValueError(
                "undistorted -> table Homography requires distortion correction "
                "(enabled=True); disable_undistort + Homography is not allowed"
            )
        matrix = np.asarray(homography_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"homography_matrix must be a 3x3 matrix, got shape {matrix.shape}")
        self.homography_matrix = matrix
        self.homography_inverse = np.linalg.inv(matrix)
        if image_size is not None:
            self.homography_image_size = (int(image_size[0]), int(image_size[1]))
        self._validate_homography_image_size()

    def _validate_homography_image_size(self) -> None:
        """Homography 标定分辨率必须与当前实际图像分辨率一致，不做自动缩放。"""
        if self.homography_image_size is None or self.image_size is None:
            return
        if tuple(self.image_size) != tuple(self.homography_image_size):
            raise ValueError(
                f"Homography calibrated for "
                f"{self.homography_image_size[0]}x{self.homography_image_size[1]}, "
                f"but runtime image is {self.image_size[0]}x{self.image_size[1]}"
            )

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        """更新图像分辨率并计算最优新相机矩阵（不保存/不使用 calibration ROI 裁剪偏移）。"""
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image_size: {image_size}")
        self.image_size = (width, height)

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            self.new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (width, height),
                0,
                (width, height),
            )

        self._validate_homography_image_size()

    def set_table_roi(self, table_roi: Tuple[float, float, float, float]) -> None:
        """设置定义在原始相机图像像素坐标系下的球台区域 (x, y, w, h)。"""
        x, y, w, h = (float(v) for v in table_roi)
        if w <= 0 or h <= 0:
            raise ValueError(f"table_roi width and height must be positive: {table_roi}")
        self.table_roi = (x, y, w, h)

    def _get_undistorted_table_bounds(self) -> Tuple[float, float, float, float]:
        """获取去畸变坐标系下球台矩形的 (ux0, uy0, ux1, uy1)。"""
        if self.table_roi is None:
            raise ValueError("table_roi is not set")
        rx, ry, rw, rh = self.table_roi
        ux0, uy0 = self.raw_to_undistorted(rx, ry)
        ux1, uy1 = self.raw_to_undistorted(rx + rw, ry + rh)
        return ux0, uy0, ux1, uy1

    def _get_rink_scales(self) -> Tuple[float, float]:
        """计算去畸变像素到球台坐标的缩放比例 (scale_x, scale_y)。"""
        ux0, uy0, ux1, uy1 = self._get_undistorted_table_bounds()
        span_w = max(1.0, ux1 - ux0)
        span_h = max(1.0, uy1 - uy0)
        rink_left, rink_right, rink_top, rink_bottom = self.rink_bounds
        return (
            (rink_right - rink_left) / span_w,
            (rink_bottom - rink_top) / span_h,
        )

    def raw_to_undistorted(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """原始相机像素坐标 -> 去畸变相机像素坐标。

        无任何人工 calibration ROI 偏移扣除。
        """
        if not self.enabled or self.camera_matrix is None:
            return float(raw_x), float(raw_y)

        pts = np.array([[[float(raw_x), float(raw_y)]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.new_camera_matrix,
        )
        return float(undistorted[0][0][0]), float(undistorted[0][0][1])

    def undistorted_to_raw(self, undist_x: float, undist_y: float) -> Tuple[float, float]:
        """去畸变相机像素坐标 -> 原始相机像素坐标（逆投影）。"""
        if not self.enabled or self.camera_matrix is None or self.new_camera_matrix is None:
            return float(undist_x), float(undist_y)

        fx = self.new_camera_matrix[0, 0]
        fy = self.new_camera_matrix[1, 1]
        cx = self.new_camera_matrix[0, 2]
        cy = self.new_camera_matrix[1, 2]

        pts_3d = np.array(
            [[[(float(undist_x) - cx) / fx, (float(undist_y) - cy) / fy, 1.0]]],
            dtype=np.float32,
        )
        projected, _ = cv2.projectPoints(
            pts_3d,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.camera_matrix,
            self.dist_coeffs,
        )
        return float(projected[0][0][0]), float(projected[0][0][1])

    def undistorted_to_table(self, undist_x: float, undist_y: float) -> Tuple[float, float]:
        """去畸变相机像素坐标 -> 球台/场地坐标。

        配置了 Homography 时使用 undistorted -> table 的透视变换；
        否则回退为基于 table_roi 的轴对齐线性缩放。
        """
        if self.homography_matrix is not None:
            return self._apply_homography(self.homography_matrix, undist_x, undist_y)
        ux0, uy0, _ux1, _uy1 = self._get_undistorted_table_bounds()
        scale_x, scale_y = self._get_rink_scales()
        rink_left, _rink_right, rink_top, _rink_bottom = self.rink_bounds
        table_x = rink_left + (float(undist_x) - ux0) * scale_x
        table_y = rink_top + (float(undist_y) - uy0) * scale_y
        return table_x, table_y

    def table_to_undistorted(self, table_x: float, table_y: float) -> Tuple[float, float]:
        """球台/场地坐标 -> 去畸变相机像素坐标。

        配置了 Homography 时使用其逆矩阵进行透视变换；否则回退为 ROI 线性缩放。
        """
        if self.homography_inverse is not None:
            return self._apply_homography(self.homography_inverse, table_x, table_y)
        ux0, uy0, _ux1, _uy1 = self._get_undistorted_table_bounds()
        scale_x, scale_y = self._get_rink_scales()
        rink_left, _rink_right, rink_top, _rink_bottom = self.rink_bounds
        undist_x = ux0 + (float(table_x) - rink_left) / scale_x
        undist_y = uy0 + (float(table_y) - rink_top) / scale_y
        return undist_x, undist_y

    @staticmethod
    def _apply_homography(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """对单个点应用 3x3 单应矩阵。"""
        pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def raw_to_table(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """原始相机像素坐标 -> 球台/场地坐标。

        链式调用: raw pixel -> undistorted pixel -> table coordinate
        """
        undist_x, undist_y = self.raw_to_undistorted(raw_x, raw_y)
        return self.undistorted_to_table(undist_x, undist_y)

    def table_to_raw(self, table_x: float, table_y: float) -> Tuple[float, float]:
        """球台/场地坐标 -> 原始相机像素坐标。

        链式调用: table coordinate -> undistorted pixel -> raw pixel
        """
        undist_x, undist_y = self.table_to_undistorted(table_x, table_y)
        return self.undistorted_to_raw(undist_x, undist_y)

    def raw_velocity_to_table(
        self,
        raw_x: float,
        raw_y: float,
        raw_vx: float,
        raw_vy: float,
        dt: float = 0.01,
    ) -> Tuple[float, float]:
        """通过位置微小位移差分计算球台坐标系速度，保证与位置使用同一套坐标转换逻辑。"""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        tx0, ty0 = self.raw_to_table(raw_x, raw_y)
        tx1, ty1 = self.raw_to_table(raw_x + raw_vx * dt, raw_y + raw_vy * dt)
        return (tx1 - tx0) / dt, (ty1 - ty0) / dt
