"""预览的缩放、坐标标注和 RGB 像素输出回归。"""

import numpy as np

from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
from air_hockey.vision.types import Detection
from main import DISPLAY_WIDTH, encode_preview_ppm

TABLE_ROI = (350, 0, 580, 650)


class MockGeometry:
    def table_to_raw(self, tx, ty):
        return float(tx), float(ty)


def split_ppm(ppm):
    header, pixels = ppm.split(b"\n", 1)
    magic, width, height, maxval = header.split(b" ")
    assert magic == b"P6"
    return int(width), int(height), int(maxval), pixels


def make_result(fps):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return VisionResult(frame=Frame(image=img, timestamp=1.0, sequence=1), fps=fps)


def test_encode_preview_ppm_draws_in_scaled_raw_pixel_coordinates():
    result = make_result(42.0)
    result.frame.image[:] = (10, 20, 30)
    result.detection = Detection(center_x=500.0, center_y=300.0, radius=25.0, area=1960.0, timestamp=1.0)
    result.ai_target = (1000.0, 400.0)
    result.trajectory = [(1050.0, 500.0), (1150.0, 600.0)]

    width, height, maxval, pixels = split_ppm(encode_preview_ppm(result, TABLE_ROI, MockGeometry()))
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

    assert (width, height, maxval) == (DISPLAY_WIDTH, 360, 255)
    assert tuple(rgb[330, 500]) == (30, 20, 10)  # BGR 输入被转换为 RGB，不在标注区
    assert tuple(rgb[120, 175]) == (255, 255, 0)  # 球台 ROI 左边框缩放到一半
    assert tuple(rgb[150, 250]) == (0, 255, 0)  # 冰壶观测中心
    assert tuple(rgb[200, 500]) == (255, 0, 255)  # table->raw 后的 AI 十字中心
    assert tuple(rgb[275, 550]) != (30, 20, 10)  # 预测轨迹连线
    assert np.all(result.frame.image == (10, 20, 30))  # 绘制不污染处理帧


def test_encode_preview_ppm_preserves_non_default_aspect_ratio():
    result = make_result(60.0)
    result.frame.image = np.zeros((400, 800, 3), dtype=np.uint8)
    result.detection = Detection(center_x=400.0, center_y=200.0, radius=15.0, area=700.0, timestamp=1.0)
    width, height, _, pixels = split_ppm(encode_preview_ppm(result, TABLE_ROI, MockGeometry()))
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
    assert (width, height) == (640, 320)
    assert tuple(rgb[160, 320]) == (0, 255, 0)
