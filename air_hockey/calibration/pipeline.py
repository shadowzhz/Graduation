"""视觉输入预处理流水线。

负责连接：
摄像头帧 -> 畸变校正 -> 视觉算法

保持 detector/tracker 对标定流程无感知。
"""

from calibration.undistort import Undistorter


class VisionPipeline:
    def __init__(self, calibration_file=None, enabled=True):
        self.undistorter = Undistorter(
            calibration_file=calibration_file,
            enabled=enabled,
        )

    def process(self, frame):
        """返回供检测器使用的校正后 Frame。"""
        return self.undistorter.apply(frame)
