"""相机几何与坐标变换模块。

负责相机原始像素坐标 (raw pixel)、去畸变像素坐标 (undistorted pixel)
以及桌面/球台坐标 (table/rink coordinate) 之间的单向清晰转换。

坐标流定义：
  raw pixel -> undistorted pixel -> table coordinate
  table coordinate -> undistorted pixel -> raw pixel
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

        if image_size is not None:
            self.set_image_size(image_size)

        if table_roi is not None:
            self.set_table_roi(table_roi)

    @classmethod
    def from_calibration_file(
        cls,
        calibration_file: Optional[Union[str, Path]],
        rink_bounds: Tuple[float, float, float, float],
        table_roi: Optional[Tuple[float, float, float, float]] = None,
        enabled: bool = True,
    ) -> "CameraGeometry":
        """从标定 .npz 文件构建 CameraGeometry。"""
        if rink_bounds is None:
            raise ValueError("rink_bounds must be provided")

        if not enabled:
            return cls(
                rink_bounds=rink_bounds,
                table_roi=table_roi,
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
            enabled=True,
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
        """去畸变相机像素坐标 -> 球台/场地坐标。"""
        ux0, uy0, _ux1, _uy1 = self._get_undistorted_table_bounds()
        scale_x, scale_y = self._get_rink_scales()
        rink_left, _rink_right, rink_top, _rink_bottom = self.rink_bounds
        table_x = rink_left + (float(undist_x) - ux0) * scale_x
        table_y = rink_top + (float(undist_y) - uy0) * scale_y
        return table_x, table_y

    def table_to_undistorted(self, table_x: float, table_y: float) -> Tuple[float, float]:
        """球台/场地坐标 -> 去畸变相机像素坐标。"""
        ux0, uy0, _ux1, _uy1 = self._get_undistorted_table_bounds()
        scale_x, scale_y = self._get_rink_scales()
        rink_left, _rink_right, rink_top, _rink_bottom = self.rink_bounds
        undist_x = ux0 + (float(table_x) - rink_left) / scale_x
        undist_y = uy0 + (float(table_y) - rink_top) / scale_y
        return undist_x, undist_y

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
