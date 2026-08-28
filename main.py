"""项目统一入口。

用法：
    python main.py               # 启动冰壶仿真游戏
    python main.py --vision      # 实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI
"""

import argparse
import base64
import subprocess
import sys
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

DISPLAY_WIDTH = 800
PREVIEW_FPS = 30
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
    """单窗口预览，主循环直接调 draw() 刷新。"""

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

    def _close(self):
        self.closed = True

    def draw(self, annotated, status_text):
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".ppm", rgb)
        if ok:
            self._photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()), format="PPM")
            self.label.configure(image=self._photo)
        self.status.set(status_text)
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        if self.root.winfo_exists():
            self.root.destroy()


def run_vision(args):
    """实时演示的主循环，也是将来正式应用的主循环雏形。"""
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
    target = (layout.RINK_CENTER_X, AI_HOME_Y)
    reaction_timer = 0.0
    stalled_phase = "idle"
    last_timestamp = None
    last_sequence = -1
    last_draw = 0.0
    display_fps = 0.0

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

            detection = detector.detect(frame)
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

            if now - last_draw >= 1.0 / PREVIEW_FPS:
                last_draw = now
                annotated = annotate(frame.image, roi, detection, track, ai_target_pixel, display_fps)
                scale = DISPLAY_WIDTH / annotated.shape[1]
                resized = cv2.resize(annotated, (DISPLAY_WIDTH, round(annotated.shape[0] * scale)))
                window.draw(resized, status_text + "    Q / ESC 退出")
    finally:
        camera.stop()
        window.close()


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")
    parser.add_argument("--vision", action="store_true", help="实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI")
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
