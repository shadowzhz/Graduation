"""针对 renderer 的单元测试。"""
from pathlib import Path
import numpy as np

from air_hockey import core_config as core
from air_hockey.app.rally import RallyDecision, RallyState
from air_hockey.app.renderer import COLOR_TARGET, format_status, render, render_preview
from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
from air_hockey.vision.types import Detection, Track, TrackState
from air_hockey.vision.geometry import CameraGeometry


class MockGeometry:
    """用于测试 renderer 调用坐标转换的 Mock 对象。"""

    def __init__(self, offset_x=100.0, offset_y=50.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.calls = []

    def table_to_raw(self, tx, ty):
        self.calls.append((float(tx), float(ty)))
        return float(tx + self.offset_x), float(ty + self.offset_y)

    def table_to_raw_batch(self, points):
        return [self.table_to_raw(tx, ty) for tx, ty in points]


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


def test_render_draws_requested_target_not_raw_ai_target():
    for preview_width in (None, 300):
        for rally_present, rally_target in ((False, None), (True, (150.0, 180.0)), (True, None)):
            _assert_requested_target(preview_width, rally_present, rally_target)


def _assert_requested_target(preview_width, rally_present, rally_target):
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    result = VisionResult(
        frame=Frame(image=image, timestamp=1.0, sequence=1),
        ai_target=(350.0, 280.0),
    )
    if rally_present:
        result.rally = RallyDecision(
            state=RallyState.DEFENDING,
            target=rally_target,
            reason="incoming puck",
            direction="incoming",
            mallet_position=(300.0, 100.0),
            mallet_source="software",
        )
    geometry = MockGeometry()
    roi = (5, 50, 580, 340)
    if preview_width is None:
        output = render(result, roi, geometry)
        scale = 1.0
    else:
        output = render_preview(result, roi, geometry, preview_width)
        scale = preview_width / image.shape[1]

    target = rally_target if rally_present else result.ai_target
    if target is not None:
        x = round((target[0] + geometry.offset_x) * scale)
        y = round((target[1] + geometry.offset_y) * scale)
        assert np.array_equal(output[y, x], COLOR_TARGET)
        assert geometry.calls == [target]
    else:
        assert geometry.calls == []
        assert not np.any(np.all(output == COLOR_TARGET, axis=2))

    if rally_present:
        old_x = round((result.ai_target[0] + geometry.offset_x) * scale)
        old_y = round((result.ai_target[1] + geometry.offset_y) * scale)
        assert np.array_equal(output[old_y, old_x], [0, 0, 0])
    assert not np.any(image)


def test_render_batch_trajectory_matches_pointwise_exact_pixels():
    """回归测试：验证使用批量轨迹转换与逐点转换绘制的像素图像绝对一致（像素差 0）。"""
    calib_file = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"
    table_calib = Path(__file__).resolve().parents[1] / "calibration" / "table_homography.npz"
    rink_bounds = (core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM)
    table_roi = (350, 0, 580, 650)

    geom = CameraGeometry.from_calibration_file(
        calib_file,
        rink_bounds=rink_bounds,
        table_calibration_file=table_calib if table_calib.is_file() else None,
        table_roi=table_roi,
        enabled=True,
    )

    # 205 点轨迹
    xs = np.linspace(core.RINK_LEFT + 20, core.RINK_RIGHT - 20, 205)
    ys = np.linspace(core.RINK_TOP + 50, core.RINK_BOTTOM - 50, 205)
    trajectory = list(zip(xs.tolist(), ys.tolist()))

    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=1.0, sequence=1)
    result = VisionResult(frame=frame, fps=60.0, trajectory=trajectory)

    # 1. 批量与逐点转换点坐标完全一致
    pw_pts = [geom.table_to_raw(x, y) for x, y in trajectory]
    batch_pts = geom.table_to_raw_batch(trajectory)
    assert pw_pts == batch_pts

    # 2. 模拟逐点转换验证渲染像素一致
    class PointwiseBaselineGeometry:
        def table_to_raw(self, tx, ty):
            return geom.table_to_raw(tx, ty)

        def table_to_raw_batch(self, points):
            return [geom.table_to_raw(tx, ty) for tx, ty in points]

    pw_geom = PointwiseBaselineGeometry()
    output_batch = render(result, table_roi, geom)
    output_pw = render(result, table_roi, pw_geom)

    # 验证输出图像像素绝对一致（最大像素差为 0）
    diff = np.max(np.abs(output_batch.astype(int) - output_pw.astype(int)))
    assert diff == 0, f"Expected pixel diff 0, got {diff}"

    # 验证预览模式下的像素一致性
    prev_batch = render_preview(result, table_roi, geom, 640)
    prev_pw = render_preview(result, table_roi, pw_geom, 640)
    diff_prev = np.max(np.abs(prev_batch.astype(int) - prev_pw.astype(int)))
    assert diff_prev == 0, f"Expected preview pixel diff 0, got {diff_prev}"
