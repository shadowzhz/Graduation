"""预览的缩放、坐标标注和 RGB 像素输出回归。"""

import numpy as np

from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
from air_hockey.vision.types import Detection
from air_hockey.app.preview import DISPLAY_WIDTH, PreviewBrightness, encode_preview_ppm
from air_hockey.app.renderer import COLOR_PUCK, COLOR_TABLE, COLOR_TARGET


def rgb_of(bgr):
    return tuple(reversed(bgr))


TABLE_ROI = (350, 0, 580, 650)


class MockGeometry:
    def table_to_raw(self, tx, ty):
        return float(tx), float(ty)

    def table_to_raw_batch(self, points):
        return [(float(x), float(y)) for x, y in points]


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
    assert tuple(rgb[120, 175]) == rgb_of(COLOR_TABLE)  # 球台 ROI 左边框缩放到一半
    assert tuple(rgb[150, 250]) == rgb_of(COLOR_PUCK)  # 冰壶观测中心
    assert tuple(rgb[200, 500]) == rgb_of(COLOR_TARGET)  # table->raw 后的 AI 十字中心
    assert tuple(rgb[275, 550]) != (30, 20, 10)  # 预测轨迹连线
    assert np.all(result.frame.image == (10, 20, 30))  # 绘制不污染处理帧


def test_encode_preview_ppm_preserves_non_default_aspect_ratio():
    result = make_result(60.0)
    result.frame.image = np.zeros((400, 800, 3), dtype=np.uint8)
    result.detection = Detection(center_x=400.0, center_y=200.0, radius=15.0, area=700.0, timestamp=1.0)
    width, height, _, pixels = split_ppm(encode_preview_ppm(result, TABLE_ROI, MockGeometry()))
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
    assert (width, height) == (640, 320)
    assert tuple(rgb[160, 320]) == rgb_of(COLOR_PUCK)


def test_encode_preview_brightness_gain_does_not_change_camera_frame():
    result = make_result(30.0)
    result.frame.image[:] = (100, 80, 60)

    width, height, _, pixels = split_ppm(
        encode_preview_ppm(result, TABLE_ROI, MockGeometry(), brightness_gain=0.5)
    )
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

    assert tuple(rgb[330, 500]) == (30, 40, 50)
    assert np.all(result.frame.image == (100, 80, 60))


def test_preview_brightness_compensates_flicker_and_follows_slowly():
    brightness = PreviewBrightness()
    bright = np.full((80, 80, 3), 120, dtype=np.uint8)
    dark = np.full((80, 80, 3), 60, dtype=np.uint8)

    assert brightness.gain(bright) == 1.0
    # 突然变暗一半：放大补偿，但不超过 1.33 上限
    assert brightness.gain(dark) == 1.33
    # 参考亮度缓慢跟随（每帧 0.5%），不会一帧就跳到新亮度
    assert 119.0 < brightness.reference < 120.0
