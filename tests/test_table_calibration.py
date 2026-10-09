import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from air_hockey.tools import calibrate_table

IMAGE_SIZE = (1280, 720)
VALID_QUAD = np.array(
    [[100.0, 100.0], [1100.0, 120.0], [1050.0, 620.0], [150.0, 600.0]],
    dtype=np.float64,
)


def test_valid_trapezoid_passes_corner_validation():
    area = calibrate_table.validate_corner_points(VALID_QUAD, IMAGE_SIZE)
    assert area > IMAGE_SIZE[0] * IMAGE_SIZE[1] * 0.05


def test_corner_validation_rejects_wrong_order():
    wrong_order = np.array(
        [[100.0, 100.0], [1100.0, 620.0], [1100.0, 100.0], [100.0, 620.0]],
    )
    try:
        calibrate_table.validate_corner_points(wrong_order, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "TL -> TR -> BR -> BL" in str(exc)


def test_corner_validation_rejects_repeated_points():
    repeated = VALID_QUAD.copy()
    repeated[1] = repeated[0]
    try:
        calibrate_table.validate_corner_points(repeated, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "重复" in str(exc)


def test_corner_validation_rejects_nearly_collinear_points():
    nearly_collinear = np.array(
        [[100.0, 100.0], [1200.0, 101.0], [1100.0, 102.0], [200.0, 101.0]],
    )
    try:
        calibrate_table.validate_corner_points(nearly_collinear, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_corner_validation_rejects_tiny_quadrilateral():
    tiny = np.array([[100.0, 100.0], [150.0, 100.0], [150.0, 150.0], [100.0, 150.0]])
    try:
        calibrate_table.validate_corner_points(tiny, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "面积过小" in str(exc)


def test_corner_validation_rejects_nan_and_out_of_bounds_points():
    invalid = VALID_QUAD.copy()
    invalid[0, 0] = np.nan
    try:
        calibrate_table.validate_corner_points(invalid, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "有限坐标" in str(exc)

    invalid = VALID_QUAD.copy()
    invalid[0, 0] = IMAGE_SIZE[0]
    try:
        calibrate_table.validate_corner_points(invalid, IMAGE_SIZE)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "图像范围内" in str(exc)


def test_save_homography_rejects_bad_order_without_writing():
    geometry = SimpleNamespace(raw_to_undistorted=lambda x, y: (x, y))
    wrong_order = np.array(
        [[100.0, 100.0], [1100.0, 620.0], [1100.0, 100.0], [100.0, 620.0]],
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "table.npz"
        try:
            calibrate_table.save_homography(geometry, wrong_order, IMAGE_SIZE, output)
            assert False, "Should have raised ValueError"
        except ValueError:
            assert not output.exists()


def test_save_homography_accepts_valid_trapezoid():
    geometry = SimpleNamespace(raw_to_undistorted=lambda x, y: (x, y))
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "table.npz"
        calibrate_table.save_homography(geometry, VALID_QUAD, IMAGE_SIZE, output)
        with np.load(output) as calibration:
            assert np.isfinite(calibration["homography_matrix"]).all()
            assert calibration["raw_points"].shape == (4, 2)
            assert tuple(calibration["image_size"]) == IMAGE_SIZE


def test_static_image_resolution_must_match_camera_calibration():
    calibration_file = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"
    with np.load(calibration_file) as calibration:
        calibrated_width, calibrated_height = (int(value) for value in calibration["image_size"])
    image = np.zeros((calibrated_height, calibrated_width + 1, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "table.npz"
        args = SimpleNamespace(
            image="static.png", calibration=str(calibration_file), output=str(output), device=None,
        )
        with patch.object(calibrate_table, "parse_args", return_value=args), \
                patch.object(calibrate_table.cv2, "imread", return_value=image):
            try:
                calibrate_table.main()
                assert False, "Should have raised SystemExit"
            except SystemExit as exc:
                assert f"当前图像：{calibrated_width + 1}x{calibrated_height}" in str(exc)
                assert f"camera calibration：{calibrated_width}x{calibrated_height}" in str(exc)
        assert not output.exists()
