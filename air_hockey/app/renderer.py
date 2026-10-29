"""实时视觉结果渲染模块。

负责 OpenCV 可视化绘制与 GUI 状态文字格式化，不包含线程、Tkinter 或业务逻辑。
"""

from typing import Sequence
import math
import cv2
import numpy as np

from .vision_runtime import VisionResult

# 叠加层配色（BGR）。白色球台、红色冰壶上都要醒目：冰壶相关统一用青色，AI 目标用琥珀色。
COLOR_TABLE = (150, 150, 150)       # 球台 ROI：细灰线，只做参照
COLOR_PUCK = (255, 200, 0)          # 冰壶观测 / 速度 / 预测轨迹
COLOR_TARGET = (0, 170, 255)        # 球槌目标


def render_preview(result: VisionResult, table_roi: Sequence[int], camera_geometry, width: int) -> np.ndarray:
    """先把原始画面缩到 width 宽，再按同一比例把 raw 像素标注画上去；不复制全分辨率画面。"""
    image = result.frame.image
    source_height, source_width = image.shape[:2]
    height = max(1, round(source_height * width / source_width))
    output = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = width / source_width, height / source_height
    scale = min(scale_x, scale_y)
    canvas_h, canvas_w = output.shape[:2]

    def point(x, y):
        """投影到画布坐标，夹到 ±2 倍画布并转成 int。

        估计偶尔给出离谱位置，超大或非有限的坐标会让 cv2.line 抛 "Can't parse 'pt1'"，
        把整个预览线程打死；夹住后不会崩，也还能看出线指向哪边。
        """
        px, py = x * scale_x, y * scale_y
        if not math.isfinite(px):
            px = 0.0
        if not math.isfinite(py):
            py = 0.0
        return (max(-2 * canvas_w, min(2 * canvas_w, int(px))),
                max(-2 * canvas_h, min(2 * canvas_h, int(py))))

    def size(value):
        return max(1, round(value * scale))

    rx, ry, rw, rh = table_roi
    cv2.rectangle(output, point(rx, ry), point(rx + rw, ry + rh), COLOR_TABLE, 1)

    if result.detection is not None:
        center = point(result.detection.center_x, result.detection.center_y)
        cv2.circle(output, center, size(result.detection.radius), COLOR_PUCK, size(2), cv2.LINE_AA)
        cv2.circle(output, center, size(3), COLOR_PUCK, -1, cv2.LINE_AA)

    if result.track is not None:
        center = point(result.track.center_x, result.track.center_y)
        end = point(result.track.center_x + result.track.vx * 0.1, result.track.center_y + result.track.vy * 0.1)
        cv2.arrowedLine(output, center, end, COLOR_PUCK, size(2), cv2.LINE_AA, tipLength=0.2)

    prediction = result.prediction
    if prediction is not None:
        if len(prediction.trajectory) > 1:
            pixel_trajectory = camera_geometry.table_to_raw_batch(prediction.trajectory)
            for i in range(len(pixel_trajectory) - 1):
                cv2.line(output, point(*pixel_trajectory[i]), point(*pixel_trajectory[i + 1]),
                         COLOR_PUCK, size(1), cv2.LINE_AA)
            for px, py in pixel_trajectory[1:]:
                cv2.circle(output, point(px, py), size(3), COLOR_PUCK, -1, cv2.LINE_AA)
        # 预测终点：空心圆，与 AI 目标的十字区分
        end_px, end_py = camera_geometry.table_to_raw(*prediction.endpoint)
        cv2.circle(output, point(end_px, end_py), size(10), COLOR_PUCK, size(2), cv2.LINE_AA)

    target = result.rally.target if result.rally is not None else None
    if target is not None:
        tx, ty = camera_geometry.table_to_raw(target[0], target[1])
        center = point(tx, ty)
        cv2.circle(output, center, size(18), COLOR_TARGET, size(2), cv2.LINE_AA)
        cv2.drawMarker(output, center, COLOR_TARGET, cv2.MARKER_CROSS, size(24), size(2))
    return output


def format_status(result: VisionResult, correction_enabled: bool = False) -> str:
    """生成状态栏文本；result.rally 由实时回合在调用前填好。"""
    rally = result.rally
    stone = result.curling_state
    velocity = f"({stone.vx:.0f}, {stone.vy:.0f})" if stone is not None else "none"
    target = f"({rally.target[0]:.0f}, {rally.target[1]:.0f})" if rally.target is not None else "none"
    phase = result.ai_decision.stalled_stone_phase if result.ai_decision is not None else "idle"
    return (
        f"FPS {result.fps:.1f} | 校正 {'ON' if correction_enabled else 'OFF'} | {rally.state.name}: {rally.reason} "
        f"| track_valid={result.track_confirmed} | {rally.direction} | 速度 {velocity} "
        f"| mallet ({rally.mallet_position[0]:.0f}, {rally.mallet_position[1]:.0f}) {rally.mallet_source} "
        f"| target {target} | hit phase {phase}"
    )
