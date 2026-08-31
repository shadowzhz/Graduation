"""项目统一入口。

用法：
    python main.py               # 启动冰壶仿真游戏
    python main.py --vision      # 实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI

线程分工：
    主线程    Tk mainloop + 显示定时器
    处理线程  取最新帧 -> 间隔检测/追踪 -> AI -> 标注
    编码线程  最新标注帧 -> 缩放 -> PNG（按预览帧率限速）
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
from vision import StoneDetector, VisionPipeline
from vision.tracker import StoneTracker

DISPLAY_WIDTH = 640
STATS_INTERVAL = 5.0
DETECTION_INTERVAL = 3
AI_HOME_Y = layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28


def run_game():
    subprocess.run([sys.executable, str(SIM_ROOT / "air_hockey.py")])


def _rink_scales(roi):
    _x, _y, w, h = roi
    return (
        (layout.RINK_RIGHT - layout.RINK_LEFT) / w,
        (layout.RINK_BOTTOM - layout.RINK_TOP) / h,
    )


def pixel_to_rink(x, y, roi):
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
    stone = StoneState.from_tracker(track)
    x, y = pixel_to_rink(stone.x, stone.y, roi)
    scale_x, scale_y = _rink_scales(roi)
    return replace(stone, x=x, y=y, vx=stone.vx * scale_x, vy=stone.vy * scale_y)


def annotate(image, roi, detection, track, ai_target_pixel, display_fps):
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
    """Tk 窗口，只在主线程使用；定时器从共享区取编码好的 PNG 显示。"""

    def __init__(self):
        self.closed = False
        self.root = tk.Tk()
        self.root.title("冰壶视觉演示：检测 -> 追踪 -> AI")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.status = tk.StringVar(value="等待画面")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")
        self._photo = None
        self._seen_seq = -1
        self._lock = None
        self._shared = None

    def start(self, lock, shared):
        self._lock = lock
        self._shared = shared
        self.root.after(33, self._tick)

    def _tick(self):
        if self.closed:
            return
        try:
            with self._lock:
                png = self._shared["png"]
                seq = self._shared["png_seq"]
                status = self._shared["status"]
                fatal = self._shared["fatal"]
            if png is not None and seq != self._seen_seq:
                self._photo = tk.PhotoImage(data=png)
                self.label.configure(image=self._photo)
                self._seen_seq = seq
            self.status.set(fatal or status)
            if fatal:
                self.closed = True
                self.root.after(2500, self.root.destroy)
                return
            self.root.after(33, self._tick)
        except tk.TclError:
            self.closed = True

    def _close(self):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def close(self):
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass


def run_vision(args):
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
    tracker = StoneTracker(max_missed_frames=DETECTION_INTERVAL + 1)
    ai = AirHockeyAI()
    vision_pipeline = VisionPipeline(
        calibration_file=args.calibration,
        enabled=not args.disable_undistort,
    )
    camera = CameraManager(CameraConfig())
    try:
        camera.start()
    except Exception as exc:
        window.close()
        raise SystemExit(f"摄像头启动失败：{exc}")

    print("实时视觉演示开始，Q / ESC 或关闭窗口退出")
    print(f"视觉管线：最新帧 + 每 {DETECTION_INTERVAL} 帧检测，其余帧使用 tracker 预测")
    print("视觉校正:")
    print(f"  calibration: {args.calibration}")
    print(f"  undistort: {'enabled' if not args.disable_undistort else 'disabled'}")

    preview_lock = threading.Lock()
    shared = {"img": None, "img_seq": -1, "png": None, "png_seq": 0,
              "status": "等待画面", "fatal": None}
    stop = threading.Event()

    def processing_loop():
        """只取最新帧；检测按固定间隔运行，中间帧由 tracker 预测。"""
        last_sequence = -1
        last_timestamp = None
        frame_index = 0
        display_fps = 0.0
        stats_timer = time.perf_counter()
        detect_ms = 0.0
        frame_ms = 0.0
        target = [layout.RINK_CENTER_X, AI_HOME_Y]
        reaction_timer = 0.0
        stalled_phase = "idle"
        try:
            while not stop.is_set() and not window.closed:
                frame = camera.get_latest_frame()
                if frame is None or frame.sequence == last_sequence:
                    time.sleep(0.001)
                    continue
                last_sequence = frame.sequence
                frame_index += 1
                frame = vision_pipeline.process(frame)

                now = frame.timestamp
                dt = 0.0 if last_timestamp is None else max(0.0, now - last_timestamp)
                last_timestamp = now
                if dt > 0:
                    instant = 1.0 / dt
                    display_fps = instant if display_fps == 0.0 else display_fps * 0.9 + instant * 0.1

                t0 = time.perf_counter()
                detection = None
                if frame_index == 1 or frame_index % DETECTION_INTERVAL == 0:
                    detection = detector.detect(frame)
                    detect_ms = detect_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
                    tracks = tracker.update(detection)
                else:
                    tracks = tracker.predict(frame.timestamp)
                track = tracks[0] if tracks else None

                correction_status = "ON" if vision_pipeline.camera_matrix is not None else "OFF"
                status_text = f"FPS {display_fps:.1f} | 校正 {correction_status} | 未检测到冰壶"
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
                    target[0], target[1] = decision.target_x, decision.target_y
                    reaction_timer = decision.reaction_timer
                    stalled_phase = decision.stalled_stone_phase
                    tx, ty = rink_to_pixel(decision.target_x, decision.target_y, roi)
                    ai_target_pixel = (round(tx), round(ty))
                    status_text = (
                        f"FPS {display_fps:.1f} | 校正 {correction_status} | "
                        f"追踪 {track.state.value} | 位置 ({stone.x:.0f}, {stone.y:.0f}) "
                        f"| 速度 ({stone.vx:.0f}, {stone.vy:.0f}) | "
                        f"AI 目标 ({target[0]:.0f}, {target[1]:.0f})"
                    )

                marked = annotate(frame.image, roi, detection, track, ai_target_pixel, display_fps)
                with preview_lock:
                    shared["img"] = marked
                    shared["img_seq"] = frame.sequence
                    shared["status"] = status_text + "    Q / ESC 退出"

                frame_ms = frame_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
                if time.perf_counter() - stats_timer >= STATS_INTERVAL:
                    stats_timer = time.perf_counter()
                    capture = camera.get_stats()
                    print(
                        f"[stats] 处理 {display_fps:.1f} FPS | 单帧 {frame_ms:.1f} ms "
                        f"(检测 {detect_ms:.1f} ms) | 采集 {capture.current_fps:.1f} FPS"
                    )
        except Exception as exc:
            with preview_lock:
                shared["fatal"] = f"处理线程异常退出：{exc!r}"
        finally:
            stop.set()

    def encoding_loop():
        """最新标注帧 -> 缩放 -> PNG，按预览帧率限速。"""
        seen = -1
        while not stop.wait(1.0 / args.preview_fps):
            with preview_lock:
                img = shared["img"]
                seq = shared["img_seq"]
            if img is None or seq == seen:
                continue
            seen = seq
            scale = DISPLAY_WIDTH / img.shape[1]
            small = cv2.resize(img, (DISPLAY_WIDTH, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                with preview_lock:
                    shared["png"] = base64.b64encode(encoded.tobytes())
                    shared["png_seq"] += 1

    threading.Thread(target=processing_loop, name="processing", daemon=True).start()
    threading.Thread(target=encoding_loop, name="encoder", daemon=True).start()
    window.start(preview_lock, shared)

    try:
        window.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        camera.stop()
        window.close()


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")
    parser.add_argument("--vision", action="store_true", help="实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--calibration", default="camera_calibration.npz", help="相机标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正")
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
