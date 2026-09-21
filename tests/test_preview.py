"""针对预览编码路径的单元测试。

处理线程只保存最新 VisionResult，render 标注移到按预览帧率限速的编码线程。
这里锁定编码线程的输出顺序：render -> resize -> BGR->RGB -> PPM。
"""

import cv2
import numpy as np

from air_hockey.app.renderer import render
from air_hockey.app.vision_runtime import VisionResult
from air_hockey.camera.types import Frame
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


def test_encode_preview_ppm_matches_render_then_resize():
    """PPM 必须等于 render 后再 resize、BGR->RGB 的像素，保证显示效果不变。"""
    result = make_result(42.0)
    geometry = MockGeometry()

    ppm = encode_preview_ppm(result, TABLE_ROI, geometry)

    width, height, maxval, pixels = split_ppm(ppm)

    marked = render(result, TABLE_ROI, geometry)
    scale = DISPLAY_WIDTH / marked.shape[1]
    expected = cv2.resize(
        marked, (DISPLAY_WIDTH, round(marked.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )
    expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)

    assert (width, height) == (expected.shape[1], expected.shape[0])
    assert maxval == 255
    assert len(pixels) == width * height * 3
    assert pixels == expected.tobytes()


def test_encode_preview_ppm_includes_render_annotations():
    """render 的标注（ROI 框、FPS 文字）必须出现在最终 PPM 中，而不是未标注的原始帧。"""
    result = make_result(99.0)
    ppm = encode_preview_ppm(result, TABLE_ROI, MockGeometry())
    _, _, _, pixels = split_ppm(ppm)
    assert any(pixels)
