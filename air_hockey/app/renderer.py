"""实时视觉结果渲染模块。

负责 OpenCV 可视化绘制与 GUI 状态文字格式化，不包含线程、Tkinter 或业务逻辑。
"""

from typing import Sequence
import cv2
import numpy as np

from .vision_runtime import VisionResult


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

    def point(x, y):
        return round(x * scale_x), round(y * scale_y)

    def size(value):
        return max(1, round(value * scale))

    rx, ry, rw, rh = table_roi
    cv2.rectangle(output, point(rx, ry), point(rx + rw, ry + rh), (0, 255, 255), size(2))

    if result.detection is not None:
        center = point(result.detection.center_x, result.detection.center_y)
        cv2.circle(output, center, size(result.detection.radius), (0, 255, 0), size(2))
        cv2.circle(output, center, size(3), (0, 255, 0), -1)

    if result.track is not None:
        center = point(result.track.center_x, result.track.center_y)
        end = point(result.track.center_x + result.track.vx * 0.1, result.track.center_y + result.track.vy * 0.1)
        cv2.arrowedLine(output, center, end, (0, 0, 255), size(2), tipLength=0.2)

    if result.trajectory and len(result.trajectory) > 1:
        pixel_trajectory = [camera_geometry.table_to_raw(px, py) for px, py in result.trajectory]
        for i in range(len(pixel_trajectory) - 1):
            pt1 = point(*pixel_trajectory[i])
            pt2 = point(*pixel_trajectory[i + 1])
            fade = max(50, 255 - i * 6)
            line_color = (fade, int(fade * 0.75), 50)
            cv2.line(output, pt1, pt2, line_color, size(2), cv2.LINE_AA)
        for i, (px, py) in enumerate(pixel_trajectory):
            alpha = max(60, 255 - i * 6)
            cv2.circle(output, point(px, py), size(2), (alpha, 170, 70), -1)

    # 轨迹调试：在预测终点画一个三角标记，与 AI 目标十字区分
    if result.trajectory_debug is not None:
        ex, ey = result.trajectory_debug.predicted_endpoint
        end_px, end_py = camera_geometry.table_to_raw(ex, ey)
        cv2.drawMarker(
            output,
            point(end_px, end_py),
            (0, 165, 255),
            cv2.MARKER_TRIANGLE_UP,
            size(18),
            size(2),
        )

    if result.ai_target is not None:
        tx, ty = camera_geometry.table_to_raw(result.ai_target[0], result.ai_target[1])
        cv2.drawMarker(
            output,
            point(tx, ty),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            size(28),
            size(3),
        )

    cv2.putText(output, f"FPS {result.fps:.1f}", point(14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * scale, (255, 255, 255), size(2))
    return output


def format_status(result: VisionResult, correction_enabled: bool = False) -> str:
    """生成 GUI 状态栏显示的简要文本。"""
    corr_status = "ON" if correction_enabled else "OFF"
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
