"""实时视觉结果渲染模块。

负责 OpenCV 可视化绘制，不包含线程、Tkinter 或业务逻辑。
"""

from typing import Optional, Sequence
import cv2
import numpy as np


def render(result, table_roi: Sequence[int], image: Optional[np.ndarray] = None) -> np.ndarray:
    """可视化绘制函数。

    全程统一使用 raw pixel，无需人工添加任何偏移。
    接收 VisionResult 与 table_roi（以及可选的 image 覆盖），返回标注后的图像副本，不修改原图。
    """
    if image is None:
        image = getattr(result.frame, "image", result.frame)

    output = image.copy()

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

    trajectory = result.pixel_trajectory
    if trajectory and len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            pt1 = (round(trajectory[i][0]), round(trajectory[i][1]))
            pt2 = (round(trajectory[i + 1][0]), round(trajectory[i + 1][1]))
            fade = max(50, 255 - i * 6)
            line_color = (fade, int(fade * 0.75), 50)
            cv2.line(output, pt1, pt2, line_color, 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(trajectory):
            alpha = max(60, 255 - i * 6)
            cv2.circle(output, (round(px), round(py)), 2, (alpha, 170, 70), -1)

    if result.ai_target_pixel is not None:
        cv2.drawMarker(
            output,
            (result.ai_target_pixel[0], result.ai_target_pixel[1]),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            28,
            3,
        )

    cv2.putText(output, f"FPS {result.fps:.1f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output
