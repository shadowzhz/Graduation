"""针对 renderer 的单元测试。"""

import numpy as np

from app.renderer import format_status, render
from app.vision_runtime import VisionResult
from camera.types import Frame
from vision.types import Detection, Track, TrackState


class MockGeometry:
    """用于测试 renderer 调用坐标转换的 Mock 对象。"""

    def __init__(self, offset_x=100.0, offset_y=50.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.calls = []

    def table_to_raw(self, tx, ty):
        self.calls.append((float(tx), float(ty)))
        return float(tx + self.offset_x), float(ty + self.offset_y)


def test_render_preserves_shape_and_does_not_modify_original():
    """验证输入 VisionResult 时，输出尺寸不变、原图未被修改，且 table 轨迹经 geometry 转换后绘制。"""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    original_copy = img.copy()

    frame = Frame(image=img, timestamp=1.0, sequence=1)
    detection = Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)
    track = Track(
        track_id=1,
        center_x=500.0,
        center_y=300.0,
        radius=25.0,
        vx=100.0,
        vy=50.0,
        last_timestamp=1.0,
        state=TrackState.ACTIVE,
    )
    result = VisionResult(
        frame=frame,
        detection=detection,
        track=track,
        fps=60.0,
        trajectory=[(10.0, 20.0), (30.0, 40.0)],
        ai_target=(50.0, 60.0),
    )
    table_roi = (350, 0, 580, 650)
    geometry = MockGeometry(offset_x=200.0, offset_y=100.0)

    output = render(result, table_roi, geometry)

    # 验证输出尺寸和类型不变
    assert output.shape == img.shape
    assert output.dtype == img.dtype

    # 验证不修改原图且不是同一个内存视图
    assert np.array_equal(img, original_copy)
    assert not np.shares_memory(output, img)

    # 验证确实进行了绘制
    assert np.any(output != 0)

    # 验证 table trajectory 和 ai_target 经过 geometry 转换
    assert (10.0, 20.0) in geometry.calls
    assert (30.0, 40.0) in geometry.calls
    assert (50.0, 60.0) in geometry.calls


def test_format_status():
    """验证 GUI 状态栏字符串格式化输出。"""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=1.0, sequence=1)
    result_empty = VisionResult(frame=frame, fps=30.0)
    status_empty = format_status(result_empty, correction_enabled=True)
    assert "未检测到冰壶" in status_empty
    assert "校正 ON" in status_empty
