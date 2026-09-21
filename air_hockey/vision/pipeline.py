"""视觉帧处理管线。"""

from pathlib import Path
from typing import Optional, Tuple, Union

from ..camera.types import Frame
from .geometry import CameraGeometry


class VisionPipeline:
    """管理相机标定参数与几何坐标转换。

    不再修改检测结果坐标，只负责相机标定参数和坐标转换。
    """

    def __init__(
        self,
        calibration_file: Optional[Union[str, Path]] = None,
        enabled: bool = True,
        table_roi: Optional[Tuple[float, float, float, float]] = None,
        rink_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.geometry = CameraGeometry.from_calibration_file(
            calibration_file=calibration_file,
            table_roi=table_roi,
            rink_bounds=rink_bounds,
            enabled=enabled,
        )

    @property
    def enabled(self) -> bool:
        return self.geometry.enabled

    @property
    def camera_matrix(self):
        return self.geometry.camera_matrix

    def process(self, frame: Frame) -> Frame:
        """接收原始帧并确保相机分辨率同步到 CameraGeometry，直接返回原图帧。"""
        if self.geometry.enabled and self.geometry.camera_matrix is not None:
            height, width = frame.image.shape[:2]
            if self.geometry.image_size != (width, height):
                self.geometry.set_image_size((width, height))
        return frame

    def raw_to_undistorted(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """原始相机像素 -> 去畸变相机像素。"""
        return self.geometry.raw_to_undistorted(raw_x, raw_y)

    def raw_to_table(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """原始相机像素 -> 球台坐标。"""
        return self.geometry.raw_to_table(raw_x, raw_y)
