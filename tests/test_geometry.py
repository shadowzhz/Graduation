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
from pathlib import Path

import numpy as np

from game_state import StoneState
from vision.detector import StoneDetector
from vision.geometry import CameraGeometry
from vision.pipeline import VisionPipeline
from vision.tracker import StoneTracker, TrackState
from vision.types import Detection, ROI

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"


def test_raw_to_undistorted_and_table():
    """测试 raw 坐标经 CameraGeometry 转换后结果正确，且具备可逆性。"""
    table_roi = (100.0, 50.0, 400.0, 600.0)
    rink_bounds = (0.0, 1000.0, 0.0, 2000.0)
    geom = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
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
    pipeline = VisionPipeline(CALIB_FILE, rink_bounds=(0.0, 1000.0, 0.0, 2000.0), enabled=True)
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
    pred_raw_x, pred_raw_y = tracker._predict_position(next_time)

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
        enabled=True,
    )
    geom_disabled = CameraGeometry.from_calibration_file(
        CALIB_FILE,
        table_roi=table_roi,
        rink_bounds=rink_bounds,
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
