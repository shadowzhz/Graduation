"""Jetson 摄像头 FPS 测试工具的 Tk GUI。"""

import base64
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

import cv2

from camera import CameraConfig, CameraManager

CAMERA_CONFIG = CameraConfig()
BENCHMARK_MODE = "--benchmark" in sys.argv
PREVIEW_WIDTH, PREVIEW_HEIGHT = 960, 540
FPS_UPDATE_MS = 500
# GUI 只需平滑预览，不应以显示刷新率抢占高速采集线程。
VIDEO_UPDATE_MS = 33
VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT = 320, 180

camera = None
preview_stop = threading.Event()
preview_lock = threading.Lock()
latest_preview_data = None
display_image = None
running = False
preview_thread = None


def show_tk_preview():
    """确保视频标签已显示，并清除启动前的占位内容。"""
    if not video_label.winfo_ismapped():
        video_label.pack(fill="both", expand=True)
    video_label.config(text="", image="")


def preview_loop():
    """后台线程读取最新帧并编码预览图，避免阻塞 Tk 主线程。"""

    global latest_preview_data
    seen_sequence = -1
    while not preview_stop.wait(VIDEO_UPDATE_MS / 1000.0):
        frame = camera.get_latest_frame()
        # 只处理 sequence 变化后的新帧，避免重复编码同一图像。
        if frame is None or frame.sequence == seen_sequence:
            continue
        seen_sequence = frame.sequence
        # 预览缩放只影响显示，不改变 CameraManager 中的原始分辨率帧。
        preview = cv2.resize(frame.image, (VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", preview, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            with preview_lock:
                latest_preview_data = base64.b64encode(encoded.tobytes())


def start_camera():
    """启动摄像头，并按需启动预览线程。"""
    global camera, preview_thread, running
    if running:
        return
    for label, value in ((fps_value, "0.0"), (avg_value, "0.0"), (max_value, "0.0"),
                         (ratio_value, "0.0%"), (frame_value, "0"), (time_value, "0.0 s")):
        label.config(text=value)
    status_value.config(text="启动中")
    camera = CameraManager(CAMERA_CONFIG)
    try:
        camera.start(timeout=2.0)
    except Exception as exc:
        camera.stop()
        camera = None
        status_value.config(text="启动失败")
        print(f"启动失败：{exc!r}")
        messagebox.showerror("摄像头启动失败", str(exc))
        return
    if not BENCHMARK_MODE:
        preview_stop.clear()
        show_tk_preview()
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
    running = True
    status_value.config(text="采集中")


def stop_camera():
    """停止摄像头和预览线程，清空待显示的图像。"""
    global camera, preview_thread, latest_preview_data, running
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
    status_value.config(text="已停止")
    print("采集已停止")


def update_video():
    """以约 30 FPS 由 Tk 定时器在主线程中更新视频控件。"""
    global display_image
    if running and not BENCHMARK_MODE:
        with preview_lock:
            preview_data = latest_preview_data
        if preview_data is not None:
            display_image = tk.PhotoImage(data=preview_data)
            video_label.config(image=display_image)
    root.after(VIDEO_UPDATE_MS, update_video)


def update_statistics():
    """定时刷新实时 FPS、平均 FPS 和峰值 FPS。"""
    if running and camera is not None:
        stats = camera.get_stats()
        fps_value.config(text=f"{stats.current_fps:.1f}")
        avg_value.config(text=f"{stats.average_fps:.1f}")
        max_value.config(text=f"{stats.max_fps:.1f}")
        ratio_value.config(text=f"{stats.current_fps / CAMERA_CONFIG.requested_fps * 100:.1f}%")
        frame_value.config(text=f"{stats.frame_count:,}")
        time_value.config(text=f"{stats.elapsed:.1f} s")
    root.after(FPS_UPDATE_MS, update_statistics)


def close_app():
    """关闭窗口前释放后台采集资源。"""
    stop_camera()
    root.destroy()


print(f"程序路径：{os.path.realpath(__file__)}")
root = tk.Tk()
root.title("Machine Camera 高速相机 FPS 测试（Linux）")
root.geometry("1280x650")
root.resizable(False, False)
left_panel = tk.Frame(root)
left_panel.pack(side="left", padx=15, pady=15)
tk.Label(left_panel, text="Machine Camera 实时画面", font=("Microsoft YaHei", 16, "bold")).pack(pady=(0, 10))
video_panel = tk.Frame(left_panel, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg="black", bd=1, relief="solid")
video_panel.pack()
video_panel.pack_propagate(False)
video_label = tk.Label(video_panel, bg="black")
video_label.pack(fill="both", expand=True)
tk.Label(
    left_panel,
    text=(f"GStreamer/V4L2 | 请求 {CAMERA_CONFIG.width} × "
          f"{CAMERA_CONFIG.height} | 请求 {CAMERA_CONFIG.requested_fps:g} FPS"),
    font=("Microsoft YaHei", 10),
).pack(pady=8)
right_panel = tk.Frame(root, width=260)
right_panel.pack(side="right", fill="y", padx=20, pady=20)
tk.Label(right_panel, text="采集性能", font=("Microsoft YaHei", 18, "bold")).pack(pady=(10, 20))
tk.Label(right_panel, text="真实输入 FPS", font=("Microsoft YaHei", 12)).pack()
fps_value = tk.Label(right_panel, text="0.0", font=("Arial", 48, "bold"))
fps_value.pack(pady=(5, 25))


def create_row(name, default="0"):
    """创建一个统计信息行，并返回用于更新数值的标签。"""
    row = tk.Frame(right_panel)
    row.pack(fill="x", pady=8)
    tk.Label(row, text=name, font=("Microsoft YaHei", 10)).pack(side="left")
    value = tk.Label(row, text=default, font=("Arial", 13, "bold"))
    value.pack(side="right")
    return value


avg_value = create_row("平均 FPS", "0.0")
max_value = create_row("最高 FPS", "0.0")
ratio_value = create_row(f"{CAMERA_CONFIG.requested_fps:g}FPS 达成率", "0.0%")
frame_value = create_row("累计帧数", "0")
time_value = create_row("运行时间", "0.0 s")
status_value = create_row("状态", "等待")
button_frame = tk.Frame(right_panel)
button_frame.pack(pady=(35, 0))
tk.Button(button_frame, text="开始测试", width=16, height=2, command=start_camera).pack(pady=6)
tk.Button(button_frame, text="停止测试", width=16, height=2, command=stop_camera).pack(pady=6)
root.protocol("WM_DELETE_WINDOW", close_app)
root.after(FPS_UPDATE_MS, update_statistics)
root.after(VIDEO_UPDATE_MS, update_video)
root.mainloop()
