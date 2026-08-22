"""Jetson 摄像头 FPS 测试工具（GStreamer 优先，V4L2 回退）。"""

import glob
import base64
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo

    Gst.init(None)
except (ImportError, ValueError):
    Gst = GstVideo = None

cv2.setNumThreads(1)

CAMERA_DEVICE = None       # None=自动选择，也可指定为 "/dev/video0"
WIDTH, HEIGHT = 1280, 720
TARGET_FPS = 200
VIDEO_WIDTH, VIDEO_HEIGHT = 960, 540
FPS_UPDATE_MS = 500
# 优先使用 Jetson 的硬件视频显示；硬件显示不可用时回退到 Tk 预览。
USE_HARDWARE_PREVIEW = True
# 仅用于 Tk 回退预览。硬件预览不受这个定时器影响。
VIDEO_UPDATE_MS = 16
VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT = 320, 180

capture = None
capture_thread = None
capture_stop = threading.Event()
capture_ready = threading.Event()
preview_stop = threading.Event()
frame_lock = threading.Lock()
preview_lock = threading.Lock()
latest_frame = None
latest_preview_data = None
running = False
frame_count = 0
last_frame_count = 0
current_fps = average_fps = max_fps = 0.0
start_time = last_fps_time = 0.0
display_image = None
preview_thread = None
hardware_preview = False


class GstCameraCapture:
    """用 appsink 提供 read() 接口，同时让视频 sink 嵌入 Tk 窗口。"""

    def __init__(self, pipeline, appsink):
        self.pipeline = pipeline
        self.appsink = appsink
        self.width = WIDTH
        self.height = HEIGHT
        self.fps = TARGET_FPS
        self.released = False

    def read(self):
        if self.released:
            return False, None
        # 短超时便于 stop_camera() 及时释放管道。
        sample = self.appsink.emit("try-pull-sample", 100 * Gst.MSECOND)
        if sample is None:
            return False, None
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        pixel_format = structure.get_string("format")
        channels = 4 if pixel_format in {"BGRx", "BGRA", "RGBA", "RGBx"} else 3
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # 正常情况下每行没有额外填充；若驱动提供 stride，则按整帧
            # buffer 长度推导行跨度，再裁剪到有效宽度。
            raw = np.frombuffer(mapped.data, dtype=np.uint8)
            row_bytes = int(width) * channels
            if raw.size == int(height) * row_bytes:
                frame = raw.reshape((int(height), int(width), channels)).copy()
            else:
                stride = raw.size // int(height)
                frame = raw.reshape((int(height), stride))[:, :row_bytes]
                frame = frame.reshape((int(height), int(width), channels)).copy()
        finally:
            buffer.unmap(mapped)
        return True, frame

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if property_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def release(self):
        if not self.released:
            self.released = True
            self.pipeline.set_state(Gst.State.NULL)


def camera_candidates():
    if CAMERA_DEVICE:
        return [CAMERA_DEVICE]
    return sorted(
        glob.glob("/dev/video*"),
        key=lambda path: int(path.rsplit("video", 1)[-1]),
    )


def open_camera():
    """打开摄像头并请求 1280x720、MJPG、200 FPS。"""
    for device in camera_candidates():
        gstreamer_result = open_gstreamer_camera(device)
        if gstreamer_result is not None:
            gstreamer_cap, uses_hardware_preview = gstreamer_result
            actual_width = int(gstreamer_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(gstreamer_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = gstreamer_cap.get(cv2.CAP_PROP_FPS)
            print(
                f"摄像头：{device}，后端：GStreamer，"
                f"实际模式：{actual_width}x{actual_height} @ {actual_fps:.2f} FPS，"
                f"预览：{'Jetson 硬件' if uses_hardware_preview else 'Tk'}"
            )
            return gstreamer_cap, device, uses_hardware_preview

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(
            f"摄像头：{device}，后端：V4L2，"
            f"实际模式：{actual_width}x{actual_height} @ {actual_fps:.2f} FPS"
        )
        return cap, device, False
    raise RuntimeError("找不到可用摄像头，请检查 /dev/video* 和 video 组权限")


def open_gstreamer_camera(device):
    """用明确的帧率和最新帧策略打开 GStreamer 管道。"""
    hardware_result = open_embedded_hardware_camera(device)
    if hardware_result is not None:
        return hardware_result

    if cv2.CAP_GSTREAMER <= 0:
        return None

    pipelines = []
    source = (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={TARGET_FPS}/1 ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! "
    )
    pipelines.extend(
        (
            (
                source
                + "nvvidconv ! video/x-raw,format=BGRx ! "
                "appsink drop=true max-buffers=1 sync=false",
                False,
            ),
            (
                f"v4l2src device={device} io-mode=2 ! "
                f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={TARGET_FPS}/1 ! "
                "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false",
                False,
            ),
        )
    )
    for pipeline, uses_hardware_preview in pipelines:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, uses_hardware_preview
        cap.release()
    return None


def open_embedded_hardware_camera(device):
    """将 Jetson 硬件视频 sink 绑定到 Tk 的 video_panel。"""
    if not USE_HARDWARE_PREVIEW or Gst is None or GstVideo is None:
        return None

    pipeline_description = (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={TARGET_FPS}/1 ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! tee name=split "
        "split. ! queue max-size-buffers=2 leaky=downstream ! "
        # nveglglessink 是面向 GstVideoOverlay 的 EGL sink，适合嵌入 Tk
        # 的 X11 子窗口；nv3dsink 在部分 Jetson 镜像上嵌入子窗口会直接崩溃。
        "nvvidconv ! nvegltransform ! "
        "nveglglessink name=preview_sink create-window=false sync=false "
        f"window-width={VIDEO_WIDTH} window-height={VIDEO_HEIGHT} "
        "split. ! queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "appsink name=capture_sink drop=true max-buffers=1 sync=false"
    )
    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipeline_description)
        preview_sink = pipeline.get_by_name("preview_sink")
        capture_sink = pipeline.get_by_name("capture_sink")
        if preview_sink is None or capture_sink is None:
            raise RuntimeError("GStreamer 管道缺少视频 sink")

        # Tk 的窗口必须先完成创建，才能拿到可嵌入的 X11 window ID。
        video_panel.update_idletasks()
        window_handle = int(video_panel.winfo_id())
        GstVideo.VideoOverlay.set_window_handle(preview_sink, window_handle)
        pipeline.set_state(Gst.State.PLAYING)
        return GstCameraCapture(pipeline, capture_sink), True
    except Exception as exc:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        print(f"嵌入式 Jetson 硬件预览不可用，回退普通预览：{exc}")
        return None


def capture_loop():
    global latest_frame, frame_count
    while not capture_stop.is_set():
        cap = capture
        if cap is None:
            break
        try:
            ok, frame = cap.read()
        except cv2.error:
            if not capture_stop.is_set():
                print("读取摄像头画面失败")
            break
        if not ok:
            if capture_stop.is_set():
                break
            # GstCameraCapture 的短超时不是摄像头错误，继续等待下一帧。
            time.sleep(0.001)
            continue
        with frame_lock:
            latest_frame = frame
            frame_count += 1
        capture_ready.set()


def build_camera_graph():
    """启动采集线程，并将硬件预览嵌入原来的 video_panel。"""
    global capture, capture_thread, preview_thread, hardware_preview
    if USE_HARDWARE_PREVIEW:
        # 硬件 sink 要渲染到父窗口；先移除覆盖在上面的 Tk Label。
        video_label.pack_forget()
        video_panel.update_idletasks()
    try:
        capture, device, hardware_preview = open_camera()
    except Exception:
        show_tk_preview()
        raise
    capture_stop.clear()
    capture_ready.clear()
    preview_stop.clear()
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    if not hardware_preview:
        show_tk_preview()
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
    print(f"已开始采集：{device}")


def show_tk_preview():
    """显示 Tk 回退预览，并清除硬件预览时的占位内容。"""
    if not video_label.winfo_ismapped():
        video_label.pack(fill="both", expand=True)
    video_label.config(text="", image="")


def preview_loop():
    """在后台生成低分辨率预览，避免 Tk 主线程阻塞摄像头采集。"""
    global latest_preview_data
    seen_frame_count = -1
    while not preview_stop.wait(VIDEO_UPDATE_MS / 1000.0):
        with frame_lock:
            frame = latest_frame
            current_frame_count = frame_count
        if frame is None or current_frame_count == seen_frame_count:
            continue
        seen_frame_count = current_frame_count
        preview = cv2.resize(
            frame,
            (VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(
            ".png",
            preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if ok:
            data = base64.b64encode(encoded.tobytes())
            with preview_lock:
                latest_preview_data = data


def start_camera():
    global running, frame_count, last_frame_count
    global current_fps, average_fps, max_fps, start_time, last_fps_time
    if running:
        return
    frame_count = last_frame_count = 0
    current_fps = average_fps = max_fps = 0.0
    for label, value in ((fps_value, "0.0"), (avg_value, "0.0"),
                         (max_value, "0.0"), (ratio_value, "0.0%"),
                         (frame_value, "0"), (time_value, "0.0 s")):
        label.config(text=value)
    status_value.config(text="启动中")
    try:
        build_camera_graph()
        if not capture_ready.wait(timeout=2.0):
            raise RuntimeError("摄像头未返回首帧")
    except Exception as exc:
        stop_camera()
        status_value.config(text="启动失败")
        print(f"启动失败：{exc!r}")
        messagebox.showerror("摄像头启动失败", str(exc))
        return
    with frame_lock:
        frame_count = 0
    last_frame_count = 0
    start_time = last_fps_time = time.perf_counter()
    running = True
    status_value.config(text="采集中")


def stop_camera():
    global running, capture, capture_thread, preview_thread, hardware_preview
    global latest_frame, latest_preview_data
    if not running and capture is None:
        return
    running = False
    capture_stop.set()
    preview_stop.set()
    if capture is not None:
        capture.release()
        capture = None
    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
    capture_thread = None
    if preview_thread is not None and preview_thread.is_alive():
        preview_thread.join(timeout=1.0)
    preview_thread = None
    with frame_lock:
        latest_frame = None
    with preview_lock:
        latest_preview_data = None
    if hardware_preview:
        show_tk_preview()
        hardware_preview = False
    status_value.config(text="已停止")
    print("采集已停止")


def update_video():
    global display_image
    if running and not hardware_preview:
        with preview_lock:
            preview_data = latest_preview_data
        if preview_data is not None:
            display_image = tk.PhotoImage(data=preview_data)
            video_label.config(image=display_image)
    root.after(VIDEO_UPDATE_MS, update_video)


def update_statistics():
    global last_frame_count, last_fps_time, current_fps, average_fps, max_fps
    if running:
        now = time.perf_counter()
        elapsed = now - start_time
        interval = now - last_fps_time
        with frame_lock:
            count = frame_count
        if interval > 0:
            current_fps = (count - last_frame_count) / interval
            last_frame_count, last_fps_time = count, now
        if elapsed > 0:
            average_fps = count / elapsed
        max_fps = max(max_fps, current_fps)
        fps_value.config(text=f"{current_fps:.1f}")
        avg_value.config(text=f"{average_fps:.1f}")
        max_value.config(text=f"{max_fps:.1f}")
        ratio_value.config(text=f"{current_fps / TARGET_FPS * 100:.1f}%")
        frame_value.config(text=f"{count:,}")
        time_value.config(text=f"{elapsed:.1f} s")
    root.after(FPS_UPDATE_MS, update_statistics)


def close_app():
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
video_panel = tk.Frame(left_panel, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg="black", bd=1, relief="solid")
video_panel.pack()
video_panel.pack_propagate(False)
video_label = tk.Label(video_panel, bg="black")
video_label.pack(fill="both", expand=True)
tk.Label(left_panel, text="GStreamer/V4L2 | 请求 1280 × 720 | 请求 200 FPS", font=("Microsoft YaHei", 10)).pack(pady=8)

right_panel = tk.Frame(root, width=260)
right_panel.pack(side="right", fill="y", padx=20, pady=20)
tk.Label(right_panel, text="采集性能", font=("Microsoft YaHei", 18, "bold")).pack(pady=(10, 20))
tk.Label(right_panel, text="真实输入 FPS", font=("Microsoft YaHei", 12)).pack()
fps_value = tk.Label(right_panel, text="0.0", font=("Arial", 48, "bold"))
fps_value.pack(pady=(5, 25))


def create_row(name, default="0"):
    row = tk.Frame(right_panel)
    row.pack(fill="x", pady=8)
    tk.Label(row, text=name, font=("Microsoft YaHei", 10)).pack(side="left")
    value = tk.Label(row, text=default, font=("Arial", 13, "bold"))
    value.pack(side="right")
    return value


avg_value = create_row("平均 FPS", "0.0")
max_value = create_row("最高 FPS", "0.0")
ratio_value = create_row("200FPS 达成率", "0.0%")
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
