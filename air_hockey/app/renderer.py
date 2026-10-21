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


def render(result: VisionResult, table_roi: Sequence[int], camera_geometry) -> np.ndarray:
    """可视化绘制函数。

    全程统一使用 raw pixel，无需人工添加任何偏移。
    接收 VisionResult、table_roi 与 camera_geometry，返回标注后的图像副本，不修改原图。
    """
    return _draw_annotations(result.frame.image.copy(), result, table_roi, camera_geometry, 1.0, 1.0)


def render_preview(result: VisionResult, table_roi: Sequence[int], camera_geometry, width: int) -> np.ndarray:
    """先缩小原始画面，再在预览分辨率绘制标注；不复制全分辨率画面。"""
    image = result.frame.image
    source_height, source_width = image.shape[:2]
    height = max(1, round(source_height * width / source_width))
    output = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return _draw_annotations(output, result, table_roi, camera_geometry, width / source_width, height / source_height)


def _draw_annotations(
    output: np.ndarray, result: VisionResult, table_roi: Sequence[int], camera_geometry,
    scale_x: float, scale_y: float,
) -> np.ndarray:
    """把 raw 像素标注投影到目标画布；render 与预览使用相同的几何数据。"""
    scale = min(scale_x, scale_y)
    canvas_h, canvas_w = output.shape[:2]

    def point(x, y):
        """投影到画布坐标。

        ⚠️ 必须**夹取范围**并把类型收敛成 Python int —— 检测偶尔会给出完全离谱的
        位置（现场实测出现过 puck=(-1095.4, 4225.9)，球台只有 600x760），
        算出来的画线坐标会让 cv2.line 抛
            "Can't parse 'pt1'. Sequence item with index 0 has a wrong type"
        而把整个预览渲染搞崩（表现为"实时处理失败"）。
        夹到 ±2 倍画布范围足够：既不会崩，也还是能看到线指向哪边。
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

    if result.trajectory and len(result.trajectory) > 1:
        pixel_trajectory = camera_geometry.table_to_raw_batch(result.trajectory)
        for i in range(len(pixel_trajectory) - 1):
            cv2.line(output, point(*pixel_trajectory[i]), point(*pixel_trajectory[i + 1]),
                     COLOR_PUCK, size(1), cv2.LINE_AA)
        for px, py in pixel_trajectory[1:]:
            cv2.circle(output, point(px, py), size(3), COLOR_PUCK, -1, cv2.LINE_AA)

    # 预测终点：空心圆，与 AI 目标的十字区分
    if result.trajectory_debug is not None:
        ex, ey = result.trajectory_debug.predicted_endpoint
        end_px, end_py = camera_geometry.table_to_raw(ex, ey)
        cv2.circle(output, point(end_px, end_py), size(10), COLOR_PUCK, size(2), cv2.LINE_AA)

    target = result.rally.target if result.rally is not None else result.ai_target
    if target is not None:
        tx, ty = camera_geometry.table_to_raw(target[0], target[1])
        center = point(tx, ty)
        cv2.circle(output, center, size(18), COLOR_TARGET, size(2), cv2.LINE_AA)
        cv2.drawMarker(output, center, COLOR_TARGET, cv2.MARKER_CROSS, size(24), size(2))
    return output


def format_status(result: VisionResult, correction_enabled: bool = False) -> str:
    """生成 GUI 状态栏显示的简要文本。"""
    corr_status = "ON" if correction_enabled else "OFF"
    if result.rally is not None:
        rally = result.rally
        velocity = (
            f"({result.stone_state.vx:.0f}, {result.stone_state.vy:.0f})"
            if result.stone_state is not None else "none"
        )
        target = f"({rally.target[0]:.0f}, {rally.target[1]:.0f})" if rally.target is not None else "none"
        phase = result.ai_decision.stalled_stone_phase if result.ai_decision is not None else "idle"
        return (
            f"FPS {result.fps:.1f} | 校正 {corr_status} | {rally.state.name}: {rally.reason} "
            f"| track_valid={result.track_confirmed} | {rally.direction} | 速度 {velocity} "
            f"| mallet ({rally.mallet_position[0]:.0f}, {rally.mallet_position[1]:.0f}) {rally.mallet_source} "
            f"| target {target} | hit phase {phase}"
        )
    if result.track is not None and result.stone_state is not None:
        target_str = (
            f"({result.ai_target[0]:.0f}, {result.ai_target[1]:.0f})"
            if result.ai_target is not None
            else "none"
        )
        # 轨迹调试信息：当前坐标 / 速度方向 / 预测终点
        debug_str = ""
        if result.trajectory_debug is not None:
            direction_x, direction_y = result.trajectory_debug.direction
            end_x, end_y = result.trajectory_debug.predicted_endpoint
            debug_str = f"| 方向 ({direction_x:.2f}, {direction_y:.2f}) | 预测终点 ({end_x:.0f}, {end_y:.0f}) "
        return (
            f"FPS {result.fps:.1f} | 校正 {corr_status} | "
            f"追踪 {result.track.state.value} | 位置 ({result.stone_state.x:.0f}, {result.stone_state.y:.0f}) "
            f"| 速度 ({result.stone_state.vx:.0f}, {result.stone_state.vy:.0f}) | "
            f"{debug_str}AI 目标 {target_str}"
        )
    return f"FPS {result.fps:.1f} | 校正 {corr_status} | 未检测到冰壶"
