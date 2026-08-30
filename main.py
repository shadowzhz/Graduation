"""项目统一入口。

用法：
    python main.py               # 启动冰壶仿真游戏
    python main.py --vision      # 实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI
"""

import argparse
import base64
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VISION_ROOT = PROJECT_ROOT / "air_hockey"
SIM_ROOT = PROJECT_ROOT / "冰壶仿真"
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(VISION_ROOT))

import cv2
import tkinter as tk

import air_hockey_config as layout
from air_hockey_ai import AirHockeyAI
from camera import CameraConfig, CameraManager
from game_state import GameState, StoneState
from vision import StoneDetector
from vision.tracker import StoneTracker

DISPLAY_WIDTH = 640
STATS_INTERVAL = 5.0
AI_HOME_Y = layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28


def run_game():
    subprocess.run([sys.executable, str(SIM_ROOT / "air_hockey.py")])


def _rink_scales(roi):
    """ROI 的像素尺度和场地尺度的比值，坐标换算用。"""
    _x, _y, w, h = roi
    return (
        (layout.RINK_RIGHT - layout.RINK_LEFT) / w,
        (layout.RINK_BOTTOM - layout.RINK_TOP) / h,
    )


def pixel_to_rink(x, y, roi):
    """临时的线性映射：把检测 ROI 拉伸到虚拟场地。

    这只是占位实现，实体台子搭好后要换成四点 Homography 标定。
    """
    roi_x, roi_y, _w, _h = roi
    scale_x, scale_y = _rink_scales(roi)
    return (
        layout.RINK_LEFT + (x - roi_x) * scale_x,
        layout.RINK_TOP + (y - roi_y) * scale_y,
    )


def rink_to_pixel(x, y, roi):
    roi_x, roi_y, _w, _h = roi
    scale_x, scale_y = _rink_scales(roi)
    return (
        roi_x + (x - layout.RINK_LEFT) / scale_x,
        roi_y + (y - layout.RINK_TOP) / scale_y,
    )


def track_to_rink_state(track, roi):
    """视觉链路 -> 游戏侧的桥：像素坐标的追踪结果转成场地坐标的 StoneState。"""
    stone = StoneState.from_tracker(track)
    x, y = pixel_to_rink(stone.x, stone.y, roi)
    scale_x, scale_y = _rink_scales(roi)
    return replace(stone, x=x, y=y, vx=stone.vx * scale_x, vy=stone.vy * scale_y)


def annotate(image, roi, detection, track, ai_target_pixel, display_fps):
    """画在副本上：ROI 框、检测圆、追踪速度箭头、AI 目标十字。"""
    output = image.copy()
    x, y, w, h = roi
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
    if detection is not None:
        center = (round(detection.center_x), round(detection.center_y))
        cv2.circle(output, center, max(1, round(detection.radius)), (0, 255, 0), 2)
        cv2.circle(output, center, 3, (0, 255, 0), -1)
    if track is not None:
        center = (round(track.center_x), round(track.center_y))
        end = (round(track.center_x + track.vx * 0.1), round(track.center_y + track.vy * 0.1))
        cv2.arrowedLine(output, center, end, (0, 0, 255), 2, tipLength=0.2)
    if ai_target_pixel is not None:
        cv2.drawMarker(output, ai_target_pixel, (255, 0, 255), cv2.MARKER_CROSS, 28, 3)
    cv2.putText(output, f"FPS {display_fps:.1f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


class VisionWindow:
    """窗口只负责显示；图像编码在独立预览线程完成，不阻塞处理主循环。"""

    def __init__(self):
        self.closed = False
        self.root = tk.Tk()
        self.root.title("冰壶视觉演示：检测 -> 追踪 -> AI")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.StringVar(value="等待画面")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")
        self._photo = None
        self._seen_seq = -1
        self._lock = None
        self._data = None

    def _close(self):
        self.closed = True

    def start_timer(self, lock, data):
        """Tk 主线程定时器：把预览线程编码好的 PNG 贴到界面。"""
        self._lock = lock
        self._data = data
        self._update_image()

    def _update_image(self):
        if self.closed:
            return
        with self._lock:
            png = self._data["png"]
            seq = self._data["seq"]
        if png is not None and seq != self._seen_seq:
            try:
                self._photo = tk.PhotoImage(data=png)
                self.label.configure(image=self._photo)
                self._seen_seq = seq
            except tk.TclError:
                self.closed = True
        try:
            self.root.after(33, self._update_image)
        except tk.TclError:
            self.closed = True

    def set_status(self, text):
        self.status.set(text)

    def close(self):
        if self.root.winfo_exists():
            self.root.destroy()


def run_vision(args):
    """实时演示：处理主循环只做检测/追踪/决策，显示由预览线程独立完成。"""
    roi = tuple(args.roi)
    try:
        window = VisionWindow()
    except tk.TclError as exc:
        raise SystemExit(f"无法打开窗口：{exc}")

    detector = StoneDetector(
        roi=roi,
        lower=tuple(args.lower),
        upper=tuple(args.upper),
        min_area=500.0,
        min_radius=25.0,
        min_circularity=0.65,
    )
    tracker = StoneTracker()
    ai = AirHockeyAI()
    camera = CameraManager(CameraConfig())
    try:
        camera.start()
    except Exception as exc:
        window.close()
        raise SystemExit(f"摄像头启动失败：{exc}")

    print("实时视觉演示开始，Q / ESC 或关闭窗口退出")

    # 预览线程：把最新标注帧编码成 PNG，按预览帧率限速
    preview_lock = threading.Lock()
    preview_data = {"png": None, "seq": 0}
    annotated = {"img": None, "seq": -1}
    display_stop = threading.Event()

    def preview_loop():
        seen = -1
        while not display_stop.wait(1.0 / args.preview_fps):
            with preview_lock:
                img = annotated["img"]
                seq = annotated["seq"]
            if img is None or seq == seen:
                continue
            seen = seq
            scale = DISPLAY_WIDTH / img.shape[1]
            small = cv2.resize(
                img, (DISPLAY_WIDTH, round(img.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            ok, encoded = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                with preview_lock:
                    preview_data["png"] = base64.b64encode(encoded.tobytes())
                    preview_data["seq"] += 1

    preview_thread = threading.Thread(target=preview_loop, name="preview", daemon=True)
    preview_thread.start()
    window.start_timer(preview_lock, preview_data)

    target = (layout.RINK_CENTER_X, AI_HOME_Y)
    reaction_timer = 0.0
    stalled_phase = "idle"
    last_timestamp = None
    last_sequence = -1
    display_fps = 0.0
    stats_timer = time.perf_counter()
    detect_ms = 0.0
    frame_ms = 0.0
    processed = 0

    try:
        while not window.closed:
            frame = camera.get_latest_frame()
            if frame is None or frame.sequence == last_sequence:
                time.sleep(0.001)
                continue
            last_sequence = frame.sequence

            now = frame.timestamp
            dt = 0.0 if last_timestamp is None else max(0.0, now - last_timestamp)
            last_timestamp = now
            if dt > 0:
                instant = 1.0 / dt
                display_fps = instant if display_fps == 0.0 else display_fps * 0.9 + instant * 0.1

            t0 = time.perf_counter()
            detection = detector.detect(frame)
            detect_ms = detect_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

            tracks = tracker.update(detection)
            track = tracks[0] if tracks else None

            status_text = "未检测到冰壶"
            ai_target_pixel = None
            if track is not None:
                stone = track_to_rink_state(track, roi)
                state = GameState(
                    ai_x=layout.RINK_CENTER_X,
                    ai_y=AI_HOME_Y,
                    ai_home_y=AI_HOME_Y,
                    target_x=target[0],
                    target_y=target[1],
                    stone=stone,
                    awaiting_serve=False,
                    current_server="player",
                    serve_phase="idle",
                    stalled_stone_phase=stalled_phase,
                    reaction_timer=reaction_timer,
                    difficulty=layout.DIFFICULTIES["普通"],
                )
                decision = ai.update(state, dt)
                target = (decision.target_x, decision.target_y)
                reaction_timer = decision.reaction_timer
                stalled_phase = decision.stalled_stone_phase
                tx, ty = rink_to_pixel(decision.target_x, decision.target_y, roi)
                ai_target_pixel = (round(tx), round(ty))
                status_text = (
                    f"追踪 {track.state.value} | 场地坐标 ({stone.x:.0f}, {stone.y:.0f}) "
                    f"v=({stone.vx:.0f}, {stone.vy:.0f}) | AI 目标 ({target[0]:.0f}, {target[1]:.0f})"
                )
            window.set_status(status_text + "    Q / ESC 退出")

            # 标注本身很便宜，每帧都做；缩放和 PNG 编码留给预览线程
            marked = annotate(frame.image, roi, detection, track, ai_target_pixel, display_fps)
            with preview_lock:
                annotated["img"] = marked
                annotated["seq"] = frame.sequence

            frame_ms = frame_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
            processed += 1
            if time.perf_counter() - stats_timer >= STATS_INTERVAL:
                stats_timer = time.perf_counter()
                capture = camera.get_stats()
                print(
                    f"[stats] 处理 {display_fps:.1f} FPS | 单帧 {frame_ms:.1f} ms "
                    f"(检测 {detect_ms:.1f} ms) | 采集 {capture.current_fps:.1f} FPS | 累计 {processed} 帧"
                )
    finally:
        display_stop.set()
        camera.stop()
        window.close()


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")
    parser.add_argument("--vision", action="store_true", help="实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--roi", type=int, nargs=4, default=(350, 0, 580, 650), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(179, 255, 255), metavar=("C1", "C2", "C3"))
    args = parser.parse_args()
    if args.vision:
        run_vision(args)
    else:
        run_game()


if __name__ == "__main__":
    main()
