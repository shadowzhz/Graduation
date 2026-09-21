"""视觉几何与坐标转换测试。

验证：
1. raw 坐标经过转换后结果正确。
2. Tracker 坐标不受相机校正 ROI 影响。
3. 动态 ROI 中心与 Tracker raw 坐标一致。
4. 开启/关闭畸变校正时，不产生坐标系错位。
5. 开启畸变校正后速度转换通过差分正确计算，位置与速度使用同一套逻辑。
6. 缺少 rink_bounds 时抛错。
7. 启用校正但标定文件不存在时抛错。
"""

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from air_hockey import core_config as core
from game_state import StoneState
from air_hockey.vision.detector import StoneDetector
from air_hockey.vision.geometry import CameraGeometry
from air_hockey.vision.pipeline import VisionPipeline
from air_hockey.vision.tracker import StoneTracker, TrackState
from air_hockey.vision.types import Detection, ROI

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"

RINK_BOUNDS = (core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM)
# 固定的 table 目标点顺序：左上 -> 右上 -> 右下 -> 左下
RINK_CORNERS = np.array(
    [
        [core.RINK_LEFT, core.RINK_TOP],
        [core.RINK_RIGHT, core.RINK_TOP],
        [core.RINK_RIGHT, core.RINK_BOTTOM],
        [core.RINK_LEFT, core.RINK_BOTTOM],
    ],
    dtype=np.float32,
)
# 透视梯形：去畸变像素平面上的任意四边形，用于验证 Homography 的透视非线性
TRAPEZOID_SOURCE = np.array(
    [[120.0, 60.0], [780.0, 90.0], [830.0, 640.0], [90.0, 610.0]],
    dtype=np.float32,
)


def _make_trapezoid_geometry():
    """构造启用 Homography 的 CameraGeometry（无相机内参，直接验证 undistorted -> table）。"""
    homography = cv2.getPerspectiveTransform(TRAPEZOID_SOURCE, RINK_CORNERS)
    geometry = CameraGeometry(
        rink_bounds=RINK_BOUNDS,
        homography_matrix=homography,
        enabled=True,
    )
    return geometry, homography


def _write_homography_file(path, homography, image_size):
    """写入一个符合 calibrate_table.py 输出格式的球台 Homography 文件。"""
    np.savez(
        path,
        homography_matrix=np.asarray(homography, dtype=np.float64),
        raw_points=np.zeros((4, 2), dtype=np.float64),
        undistorted_points=np.zeros((4, 2), dtype=np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
    )


def _diagonal_intersection(quad):
    """四边形对角线 TL-BR 与 TR-BL 的交点。"""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
    denominator = (x1 - x3) * (y2 - y4) - (y1 - y3) * (x2 - x4)
    px = ((x1 * y3 - y1 * x3) * (x2 - x4) - (x1 - x3) * (x2 * y4 - y2 * x4)) / denominator
    py = ((x1 * y3 - y1 * x3) * (y2 - y4) - (y1 - y3) * (x2 * y4 - y2 * x4)) / denominator
    return px, py



def test_raw_to_undistorted_and_table():
    """测试 raw 坐标经 CameraGeometry 转换后结果正确，且具备可逆性。"""
    table_roi = (100.0, 50.0, 400.0, 600.0)
    rink_bounds = (0.0, 1000.0, 0.0, 2000.0)
    geom = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=True,
    )
    assert geom.enabled
    assert geom.camera_matrix is not None

    # 1. 验证 raw -> undistorted -> raw 可逆性
    test_raw_x, test_raw_y = 300.0, 350.0
    ux, uy = geom.raw_to_undistorted(test_raw_x, test_raw_y)
    rec_raw_x, rec_raw_y = geom.undistorted_to_raw(ux, uy)
    assert abs(rec_raw_x - test_raw_x) < 0.01
    assert abs(rec_raw_y - test_raw_y) < 0.01

    # 2. 验证 table_roi 边界点映射至 rink_bounds
    tx_tl, ty_tl = geom.raw_to_table(100.0, 50.0)
    assert abs(tx_tl - 0.0) < 1e-4
    assert abs(ty_tl - 0.0) < 1e-4

    tx_br, ty_br = geom.raw_to_table(500.0, 650.0)
    assert abs(tx_br - 1000.0) < 1e-4
    assert abs(ty_br - 2000.0) < 1e-4

    # 3. 验证 raw -> table -> raw 可逆性
    tx, ty = geom.raw_to_table(test_raw_x, test_raw_y)
    back_raw_x, back_raw_y = geom.table_to_raw(tx, ty)
    assert abs(back_raw_x - test_raw_x) < 0.05
    assert abs(back_raw_y - test_raw_y) < 0.05


def test_raw_velocity_with_distortion():
    """验证开启畸变校正后速度通过差分计算正确，保证位置和速度使用同一套坐标转换逻辑。"""
    table_roi = (350.0, 0.0, 580.0, 650.0)
    rink_bounds = (36.0, 564.0, 46.0, 714.0)
    geom = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=True,
    )

    raw_x, raw_y = 400.0, 300.0
    raw_vx, raw_vy = 120.0, -80.0
    dt = 0.01

    # 执行速度转换
    tvx, tvy = geom.raw_velocity_to_table(raw_x, raw_y, raw_vx, raw_vy, dt=dt)

    # 独立用 raw_to_table 计算差分进行断言，验证位置和速度转换逻辑严格一致
    tx0, ty0 = geom.raw_to_table(raw_x, raw_y)
    tx1, ty1 = geom.raw_to_table(raw_x + raw_vx * dt, raw_y + raw_vy * dt)
    expected_tvx = (tx1 - tx0) / dt
    expected_tvy = (ty1 - ty0) / dt

    assert abs(tvx - expected_tvx) < 1e-9
    assert abs(tvy - expected_tvy) < 1e-9

    # 速度为 0 时结果为 0
    zero_vx, zero_vy = geom.raw_velocity_to_table(raw_x, raw_y, 0.0, 0.0)
    assert zero_vx == 0.0 and zero_vy == 0.0

    # 关闭畸变校正时，差分计算结果应与线性比例严格一致
    geom_off = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=False,
    )
    scale_x, scale_y = geom_off._get_rink_scales()
    off_vx, off_vy = geom_off.raw_velocity_to_table(raw_x, raw_y, raw_vx, raw_vy, dt=dt)
    assert abs(off_vx - raw_vx * scale_x) < 1e-5
    assert abs(off_vy - raw_vy * scale_y) < 1e-5


def test_missing_rink_bounds_raises_error():
    """验证缺少 rink_bounds 时直接抛错，不保留任何宽泛 fallback。"""
    # 直接实例化 CameraGeometry 时不给 rink_bounds
    try:
        CameraGeometry(rink_bounds=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rink_bounds must be provided" in str(e)

    # 通过 from_calibration_file 调用时不给 rink_bounds
    try:
        CameraGeometry.from_calibration_file(CALIB_FILE, rink_bounds=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rink_bounds must be provided" in str(e)

    # 通过 VisionPipeline 调用时不给 rink_bounds
    try:
        VisionPipeline(calibration_file=CALIB_FILE, rink_bounds=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rink_bounds must be provided" in str(e)

    # 传入不符合 4 元素长度的 rink_bounds
    try:
        CameraGeometry(rink_bounds=(0.0, 100.0))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exactly 4 values" in str(e)


def test_enabled_calibration_missing_file_raises_error():
    """验证启用校正但标定文件不存在时直接抛错。"""
    rink_bounds = (0.0, 1000.0, 0.0, 2000.0)

    # calibration_file 为 None 且 enabled=True
    try:
        CameraGeometry.from_calibration_file(None, rink_bounds=rink_bounds, enabled=True)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "calibration_file must be provided" in str(e)

    # calibration_file 不存在 且 enabled=True
    fake_path = "this_calibration_file_does_not_exist_12345.npz"
    try:
        CameraGeometry.from_calibration_file(fake_path, rink_bounds=rink_bounds, enabled=True)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "calibration file not found" in str(e)

    # 如果 enabled=False，则不要求标定文件存在
    geom = CameraGeometry.from_calibration_file(fake_path, rink_bounds=rink_bounds, enabled=False)
    assert not geom.enabled
    assert geom.camera_matrix is None


def test_tracker_independent_of_calibration_roi():
    """测试 Tracker 坐标完全运行在 raw pixel，不受相机校正 ROI 影响。"""
    tracker1 = StoneTracker(max_distance=100.0)
    tracker2 = StoneTracker(max_distance=100.0)

    # 模拟真实原始相机检测点
    det1 = Detection(center_x=450.0, center_y=320.0, radius=25.0, area=1960.0, timestamp=0.0)
    det2 = Detection(center_x=465.0, center_y=325.0, radius=25.0, area=1960.0, timestamp=0.05)

    # Tracker 1: 直接以 raw 检测结果更新
    tracker1.update(det1)
    track1 = tracker1.update(det2)[0]

    # Tracker 2: 即便存在 VisionPipeline (包含标定参数)，检测结果不被修改，Tracker 2 也接收相同的 raw 坐标
    pipeline = VisionPipeline(
        CALIB_FILE,
        rink_bounds=(0.0, 1000.0, 0.0, 2000.0),
        table_calibration_file=None,
        enabled=True,
    )
    assert pipeline.enabled
    # 按照新架构，pipeline 不会改变检测结果
    tracker2.update(det1)
    track2 = tracker2.update(det2)[0]

    # 验证两个 Tracker 的内部状态完全一致且均为原始像素
    assert track1.center_x == 465.0
    assert track1.center_y == 325.0
    assert track2.center_x == 465.0
    assert track2.center_y == 325.0
    assert track1.vx == track2.vx
    assert track1.vy == track2.vy


def test_dynamic_roi_centers_on_tracker_raw():
    """测试动态 ROI 中心与 Tracker raw 预测坐标严格一致，没有任何人工偏移。"""
    tracker = StoneTracker()
    tracker.update(Detection(center_x=400.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0))
    tracker.update(Detection(center_x=420.0, center_y=310.0, radius=25.0, area=1960.0, timestamp=1.1))

    next_time = 1.2
    pred_raw_x, pred_raw_y = tracker.predict_position(next_time)

    # 按照 main.py 中的动态 ROI 生成逻辑（全在 raw pixel 上操作）
    box_size = 140
    img_w, img_h = 1280, 720
    rx = max(0, min(int(pred_raw_x - box_size / 2), img_w - 1))
    ry = max(0, min(int(pred_raw_y - box_size / 2), img_h - 1))
    rw = max(1, min(box_size, img_w - rx))
    rh = max(1, min(box_size, img_h - ry))
    dynamic_roi = ROI(x=rx, y=ry, width=rw, height=rh)

    roi_center_x = dynamic_roi.x + dynamic_roi.width / 2.0
    roi_center_y = dynamic_roi.y + dynamic_roi.height / 2.0

    # 验证中心点与预测 raw 坐标之差仅为整数下取整误差（<= 0.5 像素），不存在 calibration ROI 偏移
    assert abs(roi_center_x - pred_raw_x) <= 0.5
    assert abs(roi_center_y - pred_raw_y) <= 0.5


def test_distortion_toggle_no_offset():
    """测试开启/关闭畸变校正时，球台边界与场地坐标映射不产生错位。"""
    table_roi = (350.0, 0.0, 580.0, 650.0)
    rink_bounds = (36.0, 564.0, 46.0, 714.0)

    geom_enabled = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=True,
    )
    geom_disabled = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=False,
    )

    # 开启校正与关闭校正时，球台四个角在球台坐标系下的位置必须严格一致（即 rink_bounds 定义的四个角）
    top_left_raw = (350.0, 0.0)
    bottom_right_raw = (930.0, 650.0)

    tx_on_tl, ty_on_tl = geom_enabled.raw_to_table(*top_left_raw)
    tx_off_tl, ty_off_tl = geom_disabled.raw_to_table(*top_left_raw)

    tx_on_br, ty_on_br = geom_enabled.raw_to_table(*bottom_right_raw)
    tx_off_br, ty_off_br = geom_disabled.raw_to_table(*bottom_right_raw)

    assert abs(tx_on_tl - 36.0) < 1e-4
    assert abs(ty_on_tl - 46.0) < 1e-4
    assert abs(tx_off_tl - 36.0) < 1e-4
    assert abs(ty_off_tl - 46.0) < 1e-4

    assert abs(tx_on_br - 564.0) < 1e-4
    assert abs(ty_on_br - 714.0) < 1e-4
    assert abs(tx_off_br - 564.0) < 1e-4
    assert abs(ty_off_br - 714.0) < 1e-4

    # 验证关闭校正时 raw_to_undistorted 为恒等映射
    rand_x, rand_y = 480.0, 270.0
    ux_off, uy_off = geom_disabled.raw_to_undistorted(rand_x, rand_y)
    assert ux_off == rand_x and uy_off == rand_y


def test_homography_corners_map_to_rink_corners():
    """四个 Homography 角点（undistorted）必须精确映射到 rink 四角（TL -> TR -> BR -> BL）。"""
    geometry, _homography = _make_trapezoid_geometry()
    for (sx, sy), (tx, ty) in zip(TRAPEZOID_SOURCE, RINK_CORNERS):
        mapped_x, mapped_y = geometry.undistorted_to_table(float(sx), float(sy))
        assert abs(mapped_x - float(tx)) < 1e-3
        assert abs(mapped_y - float(ty)) < 1e-3


def test_homography_center_maps_correctly():
    """四边形对角线交点（即 rink 中心的原像）必须映射到 rink 中心。"""
    geometry, _homography = _make_trapezoid_geometry()
    center_undist_x, center_undist_y = _diagonal_intersection(TRAPEZOID_SOURCE)
    table_x, table_y = geometry.undistorted_to_table(center_undist_x, center_undist_y)
    assert abs(table_x - core.RINK_CENTER_X) < 1e-3
    assert abs(table_y - core.RINK_CENTER_Y) < 1e-3

    # 反方向：table 中心经逆 Homography 后仍回到 table 中心
    back_x, back_y = geometry.table_to_undistorted(core.RINK_CENTER_X, core.RINK_CENTER_Y)
    assert abs(back_x - center_undist_x) < 1e-3
    assert abs(back_y - center_undist_y) < 1e-3


def test_homography_undistorted_table_round_trip():
    """undistorted -> table -> undistorted 往返一致。"""
    geometry, _homography = _make_trapezoid_geometry()
    for undist_x, undist_y in ((450.0, 150.0), (300.0, 380.0), (700.0, 500.0)):
        table_x, table_y = geometry.undistorted_to_table(undist_x, undist_y)
        back_x, back_y = geometry.table_to_undistorted(table_x, table_y)
        assert abs(back_x - undist_x) < 1e-6
        assert abs(back_y - undist_y) < 1e-6


def test_homography_raw_table_round_trip():
    """含畸变校正时，raw -> table -> raw 往返一致。"""
    base = CameraGeometry.from_calibration_file(
        CALIB_FILE, rink_bounds=RINK_BOUNDS, table_calibration_file=None, enabled=True
    )
    raw_quad = np.array([[350.0, 50.0], [930.0, 60.0], [950.0, 620.0], [330.0, 600.0]], dtype=np.float64)
    undistorted = np.array([base.raw_to_undistorted(px, py) for px, py in raw_quad], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(undistorted, RINK_CORNERS)

    geometry = CameraGeometry(
        rink_bounds=RINK_BOUNDS,
        camera_matrix=base.camera_matrix,
        dist_coeffs=base.dist_coeffs,
        image_size=base.image_size,
        homography_matrix=homography,
        enabled=True,
    )

    for raw_x, raw_y in ((500.0, 200.0), (700.0, 400.0), (400.0, 550.0)):
        table_x, table_y = geometry.raw_to_table(raw_x, raw_y)
        back_x, back_y = geometry.table_to_raw(table_x, table_y)
        assert abs(back_x - raw_x) < 0.05
        assert abs(back_y - raw_y) < 0.05


def test_homography_local_scale_varies_across_trapezoid():
    """透视梯形下，不同位置的局部比例尺必须不同（非线性透视映射）。"""
    geometry, _homography = _make_trapezoid_geometry()

    def local_scale(undist_x, undist_y, step=1.0):
        base_x, base_y = geometry.undistorted_to_table(undist_x, undist_y)
        next_x, next_y = geometry.undistorted_to_table(undist_x + step, undist_y)
        down_x, down_y = geometry.undistorted_to_table(undist_x, undist_y + step)
        return (
            math.hypot(next_x - base_x, next_y - base_y) / step,
            math.hypot(down_x - base_x, down_y - base_y) / step,
        )

    top_scale_x, top_scale_y = local_scale(450.0, 110.0)
    bottom_scale_x, bottom_scale_y = local_scale(460.0, 590.0)

    assert abs(top_scale_x - bottom_scale_x) > 1e-3
    assert abs(top_scale_y - bottom_scale_y) > 1e-3


def test_homography_velocity_matches_finite_difference():
    """Homography 下速度转换仍与位置有限差分严格一致。"""
    geometry, _homography = _make_trapezoid_geometry()
    raw_x, raw_y = 460.0, 320.0
    raw_vx, raw_vy = 120.0, -80.0
    dt = 0.01

    table_vx, table_vy = geometry.raw_velocity_to_table(raw_x, raw_y, raw_vx, raw_vy, dt=dt)
    tx0, ty0 = geometry.raw_to_table(raw_x, raw_y)
    tx1, ty1 = geometry.raw_to_table(raw_x + raw_vx * dt, raw_y + raw_vy * dt)

    assert abs(table_vx - (tx1 - tx0) / dt) < 1e-9
    assert abs(table_vy - (ty1 - ty0) / dt) < 1e-9
    # 非零速度，且已进入透视局部比例（不是简单的线性缩放）
    assert abs(table_vx) > 1e-6
    assert abs(table_vy) > 1e-6


def test_homography_ignores_table_roi():
    """Homography 启用后，table_roi 不再参与 undistorted <-> table 的数学映射。"""
    homography = cv2.getPerspectiveTransform(TRAPEZOID_SOURCE, RINK_CORNERS)
    without_roi = CameraGeometry(rink_bounds=RINK_BOUNDS, homography_matrix=homography, enabled=True)
    with_roi = CameraGeometry(
        rink_bounds=RINK_BOUNDS,
        table_roi=(0.0, 0.0, 100.0, 100.0),
        homography_matrix=homography,
        enabled=True,
    )
    # undistorted -> table 不受 table_roi 影响
    for undist_x, undist_y in ((450.0, 150.0), (300.0, 380.0), (700.0, 500.0)):
        assert without_roi.undistorted_to_table(undist_x, undist_y) == with_roi.undistorted_to_table(undist_x, undist_y)
    # table -> undistorted 在真正的 table 坐标上同样不受 table_roi 影响
    for table_x, table_y in (
        (core.RINK_CENTER_X, core.RINK_CENTER_Y),
        (core.RINK_LEFT, core.RINK_TOP),
        (core.RINK_RIGHT, core.RINK_BOTTOM),
    ):
        assert without_roi.table_to_undistorted(table_x, table_y) == with_roi.table_to_undistorted(table_x, table_y)


def test_missing_table_calibration_file_raises_error():
    """table_calibration_file 非 None 但文件不存在时直接 FileNotFoundError，不静默 fallback。"""
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "no_table_homography.npz"
        try:
            CameraGeometry.from_calibration_file(
                CALIB_FILE,
                rink_bounds=RINK_BOUNDS,
                table_calibration_file=missing,
                enabled=True,
            )
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as exc:
            assert "table calibration file not found" in str(exc)


def test_none_table_calibration_uses_linear_roi():
    """table_calibration_file=None 时明确使用旧 ROI 线性映射。"""
    rink_bounds = (0.0, 1000.0, 0.0, 2000.0)
    geom = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=(100.0, 50.0, 400.0, 600.0),
        rink_bounds=rink_bounds,
        table_calibration_file=None,
        enabled=True,
    )
    assert geom.homography_matrix is None

    tx_tl, ty_tl = geom.raw_to_table(100.0, 50.0)
    assert abs(tx_tl - 0.0) < 1e-4
    assert abs(ty_tl - 0.0) < 1e-4
    tx_br, ty_br = geom.raw_to_table(500.0, 650.0)
    assert abs(tx_br - 1000.0) < 1e-4
    assert abs(ty_br - 2000.0) < 1e-4


def test_homography_resolution_mismatch_raises_error():
    """Homography 标定分辨率与运行分辨率不同时抛 ValueError，不自动缩放。"""
    homography = cv2.getPerspectiveTransform(TRAPEZOID_SOURCE, RINK_CORNERS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table_homography.npz"
        _write_homography_file(path, homography, (1280, 720))
        geom = CameraGeometry.from_calibration_file(
            CALIB_FILE,
            rink_bounds=RINK_BOUNDS,
            table_calibration_file=path,
            enabled=True,
        )
        assert geom.homography_image_size == (1280, 720)
        try:
            geom.set_image_size((1920, 1080))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "Homography calibrated for 1280x720" in str(exc)
            assert "runtime image is 1920x1080" in str(exc)


def test_disable_undistort_with_homography_rejected():
    """disable_undistort 与 Homography 同时启用时直接报错，不偷偷改变坐标定义。"""
    homography = cv2.getPerspectiveTransform(TRAPEZOID_SOURCE, RINK_CORNERS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table_homography.npz"
        _write_homography_file(path, homography, (1280, 720))
        try:
            CameraGeometry.from_calibration_file(
                CALIB_FILE,
                rink_bounds=RINK_BOUNDS,
                table_calibration_file=path,
                enabled=False,
            )
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "Homography" in str(exc)

    try:
        CameraGeometry(rink_bounds=RINK_BOUNDS, homography_matrix=homography, enabled=False)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "Homography" in str(exc)


def test_homography_file_loads_matrix_and_image_size():
    """Homography 文件正确加载后，matrix 与 image_size 应正确。"""
    homography = cv2.getPerspectiveTransform(TRAPEZOID_SOURCE, RINK_CORNERS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table_homography.npz"
        _write_homography_file(path, homography, (1280, 720))
        geom = CameraGeometry.from_calibration_file(
            CALIB_FILE,
            rink_bounds=RINK_BOUNDS,
            table_calibration_file=path,
            enabled=True,
        )
        assert geom.homography_matrix is not None
        assert geom.homography_image_size == (1280, 720)
        assert np.max(np.abs(geom.homography_matrix - homography)) < 1e-9
        for (sx, sy), (tx, ty) in zip(TRAPEZOID_SOURCE, RINK_CORNERS):
            gx, gy = geom.undistorted_to_table(float(sx), float(sy))
            assert abs(gx - tx) < 1e-3
            assert abs(gy - ty) < 1e-3
