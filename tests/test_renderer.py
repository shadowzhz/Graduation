"""针对 renderer 的单元测试。"""

import numpy as np

from app.renderer import render
from app.vision_runtime import VisionResult
from camera.types import Frame
from vision.types import Detection, Track, TrackState


def test_render_preserves_shape_and_does_not_modify_original():
    """验证输入 VisionResult + 图像时，输出尺寸不变且不修改原图。"""
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
        pixel_trajectory=[(500.0, 300.0), (510.0, 305.0)],
        ai_target_pixel=(520, 310),
    )
    table_roi = (350, 0, 580, 650)

    # 方式一：默认从 result.frame.image 获取图像
    output = render(result, table_roi)

    # 验证输出尺寸和类型不变
    assert output.shape == img.shape
    assert output.dtype == img.dtype

    # 验证不修改原图且不是同一个内存视图
    assert np.array_equal(img, original_copy)
    assert not np.shares_memory(output, img)

    # 验证确实进行了绘制
    assert np.any(output != 0)

    # 方式二：显式传入图像
    output_explicit = render(result, table_roi, image=img)
    assert output_explicit.shape == img.shape
    assert np.array_equal(img, original_copy)
    assert not np.shares_memory(output_explicit, img)
