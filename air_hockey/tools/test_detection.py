"""实时 StoneDetector 测试 GUI。

Examples::

    python3 tools/test_detection.py
    python3 tools/test_detection.py --roi 250 100 780 520
    python3 tools/test_detection.py --lower 170 100 80 --upper 179 255 255
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import cv2

# Keep direct execution (``python3 tools/test_detection.py``) convenient.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera import CameraConfig, CameraManager  # noqa: E402
from camera.types import Frame  # noqa: E402
from vision import Detection, ROI, StoneDetector  # noqa: E402


PREVIEW_WIDTH, PREVIEW_HEIGHT = 960, 540
FPS_UPDATE_MS = 500
VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT = 320, 180
CAMERA_CONFIG = CameraConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时 StoneDetector 测试")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--color-space", choices=("hsv", "lab"), default="hsv")
    parser.add_argument("--lower", nargs=3, type=int, default=(170, 80, 50), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", nargs=3, type=int, default=(179, 255, 255), metavar=("C1", "C2", "C3"))
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--min-circularity", type=float, default=0.55)
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=30.0,
        help="检测、标注和 GUI 预览的最大刷新率，默认 30 FPS",
    )
    parser.add_argument(
        "--hardware-preview",
        action="store_true",
        help="在 Jetson 上使用 nveglglessink 直接渲染预览，避免 Tk/PNG 开销",
    )
    return parser


args = build_parser().parse_args()
if args.preview_fps <= 0.0:
    raise SystemExit("--preview-fps 必须大于 0")
PREVIEW_INTERVAL_SECONDS = 1.0 / args.preview_fps
VIDEO_UPDATE_MS = max(1, round(PREVIEW_INTERVAL_SECONDS * 1000.0))
detector = StoneDetector(
    roi=ROI(*args.roi) if args.roi else None,
    color_space=args.color_space,
    lower=args.lower,
    upper=args.upper,
    min_area=args.min_area,
    min_radius=args.min_radius,
    min_circularity=args.min_circularity,
)

camera = None
running = False
preview_thread = None
preview_stop = threading.Event()
preview_lock = threading.Lock()
latest_preview_data = None
latest_preview_sequence = 0
latest_detection: Optional[Detection] = None
display_image = None
displayed_preview_sequence = -1


def show_tk_preview():
    """确保预览标签可见，并清除启动前的占位内容。"""
    if not video_label.winfo_ismapped():
        video_label.pack(fill="both", expand=True)
    video_label.config(text="", image="")


def annotate(frame_image, detection: Optional[Detection]):
    """Draw detector diagnostics on a BGR frame before preview encoding."""

    # 标注使用副本，不能修改 CameraManager 缓存中的原始帧。
    output = frame_image.copy()
    height, width = output.shape[:2]
    if detector.roi is not None:
        bounded = detector.roi.clamp(width, height)
        if bounded is not None:
            cv2.rectangle(
                output,
                (bounded.x, bounded.y),
                (bounded.x + bounded.width - 1, bounded.y + bounded.height - 1),
                (255, 180, 0),
                2,
            )
    if detection is None:
        cv2.putText(output, "No detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return output

    center = (round(detection.center_x), round(detection.center_y))
    cv2.circle(output, center, round(detection.radius), (0, 255, 0), 3)
    cv2.drawMarker(output, center, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    label = f"r={detection.radius:.1f} score={detection.score:.3f}"
    cv2.putText(
        output,
        label,
        (max(0, center[0] - round(detection.radius)), max(28, center[1] - round(detection.radius) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return output


def preview_loop():
    """按预览帧率处理最新帧，避免显示负载拖慢高速采集。"""

    global latest_preview_data, latest_preview_sequence, latest_detection
    seen_sequence = -1
    while not preview_stop.wait(PREVIEW_INTERVAL_SECONDS):
        if camera is None:
            continue
        frame = camera.get_latest_frame()
        # CameraManager 只保留最新帧；sequence 可避免重复处理同一帧。
        if frame is None or frame.sequence == seen_sequence:
            continue
        seen_sequence = frame.sequence
        detection = detector.detect(frame)
        # 检测结果和编码后的预览由 GUI 定时器读取，写入时必须加锁。
        with preview_lock:
            latest_detection = detection
        # 硬件预览由 GStreamer 直接绘制，Python 线程只保留检测工作。
        if camera is not None and camera.hardware_preview:
            continue
        annotated = annotate(frame.image, detection)
        preview = cv2.resize(
            annotated,
            (VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".png", preview, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            with preview_lock:
                latest_preview_data = base64.b64encode(encoded.tobytes())
                latest_preview_sequence += 1


def start_camera():
    """启动摄像头和检测线程，并重置界面上的旧统计值。"""
    global camera, preview_thread, running
    if running:
        return
    for label, value in (
        (fps_value, "0.0"), (avg_value, "0.0"), (max_value, "0.0"),
        (ratio_value, "0.0%"), (frame_value, "0"), (time_value, "0.0 s"),
        (detection_value, "无"), (center_value, "-"), (radius_value, "-"),
        (score_value, "-"),
    ):
        label.config(text=value)
    status_value.config(text="启动中")
    if args.hardware_preview:
        video_label.pack_forget()
        video_panel.update_idletasks()
        preview_window_handle = int(video_panel.winfo_id())
    else:
        preview_window_handle = None
    camera = CameraManager(CAMERA_CONFIG, preview_window_handle=preview_window_handle)
    try:
        camera.start(timeout=2.0)
    except Exception as exc:
        camera.stop()
        camera = None
        show_tk_preview()
        status_value.config(text="启动失败")
        print(f"启动失败：{exc!r}")
        messagebox.showerror("摄像头启动失败", str(exc))
        return
    preview_stop.clear()
    if not camera.hardware_preview:
        show_tk_preview()
    print(f"检测和预览刷新率上限：{args.preview_fps:g} FPS")
    preview_thread = threading.Thread(target=preview_loop, name="vision-preview", daemon=True)
    preview_thread.start()
    running = True
    status_value.config(text="检测中")


def stop_camera():
    """停止采集和预览线程，同时清理线程共享数据。"""
    global camera, preview_thread, latest_preview_data, latest_preview_sequence, latest_detection, running
    if camera is None and not running:
        return
    running = False
    preview_stop.set()
    if camera is not None:
        camera.stop()
    camera = None
    if preview_thread is not None and preview_thread.is_alive():
        preview_thread.join(timeout=1.0)
    preview_thread = None
    with preview_lock:
        latest_preview_data = None
        latest_preview_sequence = 0
        latest_detection = None
    status_value.config(text="已停止")
    show_tk_preview()
    print("采集已停止")


def update_video():
    """仅在出现新预览帧时，由 Tk 主线程更新视频控件。"""
    global display_image, displayed_preview_sequence
    if running and camera is not None and not camera.hardware_preview:
        with preview_lock:
            preview_data = latest_preview_data
            preview_sequence = latest_preview_sequence
        if preview_data is not None and preview_sequence != displayed_preview_sequence:
            display_image = tk.PhotoImage(data=preview_data)
            video_label.config(image=display_image)
            displayed_preview_sequence = preview_sequence
    root.after(VIDEO_UPDATE_MS, update_video)


def update_statistics():
    """定时读取采集统计和最新检测结果并刷新右侧面板。"""
    if running and camera is not None:
        stats = camera.get_stats()
        fps_value.config(text=f"{stats.current_fps:.1f}")
        avg_value.config(text=f"{stats.average_fps:.1f}")
        max_value.config(text=f"{stats.max_fps:.1f}")
        ratio_value.config(text=f"{stats.current_fps / CAMERA_CONFIG.requested_fps * 100:.1f}%")
        frame_value.config(text=f"{stats.frame_count:,}")
        time_value.config(text=f"{stats.elapsed:.1f} s")
        with preview_lock:
            detection = latest_detection
        if detection is None:
            detection_value.config(text="未找到")
            center_value.config(text="-")
            radius_value.config(text="-")
            score_value.config(text="-")
        else:
            detection_value.config(text="已找到")
            center_value.config(text=f"({detection.center_x:.1f}, {detection.center_y:.1f})")
            radius_value.config(text=f"{detection.radius:.1f}")
            score_value.config(text=f"{detection.score:.3f}")
    root.after(FPS_UPDATE_MS, update_statistics)


def create_row(parent, name, default="0"):
    """创建一行标签和值，并返回可更新的值标签。"""
    row = tk.Frame(parent)
    row.pack(fill="x", pady=6)
    tk.Label(row, text=name, font=("Microsoft YaHei", 10)).pack(side="left")
    value = tk.Label(row, text=default, font=("Arial", 12, "bold"))
    value.pack(side="right")
    return value


def close_app():
    """窗口关闭时先释放摄像头，再销毁 Tk 根窗口。"""
    stop_camera()
    root.destroy()


print(f"程序路径：{os.path.realpath(__file__)}")
root = tk.Tk()
root.title("Machine Camera 冰壶实时检测测试（Linux）")
root.geometry("1280x700")
root.resizable(False, False)

left_panel = tk.Frame(root)
left_panel.pack(side="left", padx=15, pady=15)
tk.Label(left_panel, text="Machine Camera 实时检测", font=("Microsoft YaHei", 16, "bold")).pack(pady=(0, 10))
video_panel = tk.Frame(left_panel, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg="black", bd=1, relief="solid")
video_panel.pack()
video_panel.pack_propagate(False)
video_label = tk.Label(video_panel, bg="black")
video_label.pack(fill="both", expand=True)
tk.Label(
    left_panel,
    text=(f"{detector.color_space.upper()} | {CAMERA_CONFIG.width} × {CAMERA_CONFIG.height} | "
          f"请求 {CAMERA_CONFIG.requested_fps:g} FPS"),
    font=("Microsoft YaHei", 10),
).pack(pady=8)

right_panel = tk.Frame(root, width=260)
right_panel.pack(side="right", fill="y", padx=20, pady=20)
tk.Label(right_panel, text="采集性能", font=("Microsoft YaHei", 18, "bold")).pack(pady=(10, 20))
tk.Label(right_panel, text="真实输入 FPS", font=("Microsoft YaHei", 12)).pack()
fps_value = tk.Label(right_panel, text="0.0", font=("Arial", 48, "bold"))
fps_value.pack(pady=(5, 20))
avg_value = create_row(right_panel, "平均 FPS", "0.0")
max_value = create_row(right_panel, "最高 FPS", "0.0")
ratio_value = create_row(right_panel, f"{CAMERA_CONFIG.requested_fps:g}FPS 达成率", "0.0%")
frame_value = create_row(right_panel, "累计帧数", "0")
time_value = create_row(right_panel, "运行时间", "0.0 s")
status_value = create_row(right_panel, "状态", "等待")
tk.Label(right_panel, text="检测结果", font=("Microsoft YaHei", 15, "bold")).pack(pady=(22, 8))
detection_value = create_row(right_panel, "目标", "无")
center_value = create_row(right_panel, "中心", "-")
radius_value = create_row(right_panel, "半径", "-")
score_value = create_row(right_panel, "Score", "-")

button_frame = tk.Frame(right_panel)
button_frame.pack(pady=(22, 0))
tk.Button(button_frame, text="开始检测", width=16, height=2, command=start_camera).pack(pady=6)
tk.Button(button_frame, text="停止检测", width=16, height=2, command=stop_camera).pack(pady=6)
root.protocol("WM_DELETE_WINDOW", close_app)
root.after(FPS_UPDATE_MS, update_statistics)
root.after(VIDEO_UPDATE_MS, update_video)
root.mainloop()
