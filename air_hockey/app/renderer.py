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
    output = result.frame.image.copy()

    rx, ry, rw, rh = table_roi
    cv2.rectangle(output, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), (0, 255, 255), 2)

    if result.detection is not None:
        center = (round(result.detection.center_x), round(result.detection.center_y))
        cv2.circle(output, center, max(1, round(result.detection.radius)), (0, 255, 0), 2)
        cv2.circle(output, center, 3, (0, 255, 0), -1)

    if result.track is not None:
        center = (round(result.track.center_x), round(result.track.center_y))
        end = (
            round(result.track.center_x + result.track.vx * 0.1),
            round(result.track.center_y + result.track.vy * 0.1),
        )
        cv2.arrowedLine(output, center, end, (0, 0, 255), 2, tipLength=0.2)

    if result.trajectory and len(result.trajectory) > 1:
        pixel_trajectory = [camera_geometry.table_to_raw(px, py) for px, py in result.trajectory]
        for i in range(len(pixel_trajectory) - 1):
            pt1 = (round(pixel_trajectory[i][0]), round(pixel_trajectory[i][1]))
            pt2 = (round(pixel_trajectory[i + 1][0]), round(pixel_trajectory[i + 1][1]))
            fade = max(50, 255 - i * 6)
            line_color = (fade, int(fade * 0.75), 50)
            cv2.line(output, pt1, pt2, line_color, 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pixel_trajectory):
            alpha = max(60, 255 - i * 6)
            cv2.circle(output, (round(px), round(py)), 2, (alpha, 170, 70), -1)

    if result.ai_target is not None:
        tx, ty = camera_geometry.table_to_raw(result.ai_target[0], result.ai_target[1])
        cv2.drawMarker(
            output,
            (round(tx), round(ty)),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            28,
            3,
        )

    cv2.putText(output, f"FPS {result.fps:.1f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
        return (
            f"FPS {result.fps:.1f} | 校正 {corr_status} | "
            f"追踪 {result.track.state.value} | 位置 ({result.stone_state.x:.0f}, {result.stone_state.y:.0f}) "
            f"| 速度 ({result.stone_state.vx:.0f}, {result.stone_state.vy:.0f}) | "
            f"AI 目标 {target_str}"
        )
    return f"FPS {result.fps:.1f} | 校正 {corr_status} | 未检测到冰壶"
