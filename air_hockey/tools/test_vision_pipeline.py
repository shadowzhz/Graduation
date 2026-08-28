"""实时验证 Camera -> Detection -> Tracker -> GameState 管线。

Tracker 的像素坐标直接映射成 StoneState / PuckState，先不接 AI 和 Homography。
用法: python3 air_hockey/tools/test_vision_pipeline.py --device /dev/video0
"""


import argparse
import base64
from pathlib import Path
import sys
import time
import tkinter as tk

import cv2


# 往上找包含 game_state.py 的仿真工程目录
SCRIPT_DIR = Path(__file__).resolve().parent
VISION_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = VISION_ROOT.parent
SIMULATION_ROOT = next(
    (
        directory
        for directory in PROJECT_ROOT.iterdir()
        if (directory / "game_state.py").is_file()
        and (directory / "air_hockey_ai.py").is_file()
    ),
    None,
)
if SIMULATION_ROOT is None:
    raise RuntimeError("未找到包含 game_state.py 的游戏工程")
sys.path.insert(0, str(VISION_ROOT))
sys.path.insert(0, str(SIMULATION_ROOT))

from camera import CameraConfig, CameraManager
from game_state import GameState, StoneState
from vision import StoneDetector
from vision.tracker import StoneTracker
from vision.types import Detection, Track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时运行 Camera -> Detector -> Tracker -> GameState。"
    )
    parser.add_argument("--backend", default="auto", choices=("auto", "gstreamer", "v4l2"))
    parser.add_argument("--device", default=None, help="例如 /dev/video0")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=(350, 0, 580, 650))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80))
    parser.add_argument("--upper", type=int, nargs=3, default=(179, 255, 255))
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--min-radius", type=float, default=25.0)
    parser.add_argument("--min-circularity", type=float, default=0.65)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-missed-frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=200.0, help="请求摄像头帧率")
    parser.add_argument("--duration", type=float, default=0.0, help="秒数；0 表示直到退出")
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--no-display", action="store_true", help="不打开 Tk 预览窗口")
    return parser


def annotate(image, roi, detection, track, stone, game_state, display_fps):
    """画 Detection、Tracker 和统一状态，画在副本上。"""
    output = image.copy()
    x, y, width, height = roi
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    if detection is not None:
        center = (round(detection.center_x), round(detection.center_y))
        cv2.circle(output, center, max(1, round(detection.radius)), (0, 255, 0), 2)
        cv2.circle(output, center, 3, (0, 255, 0), -1)
        cv2.putText(output, "Detection", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if track is not None:
        center = (round(track.center_x), round(track.center_y))
        cv2.circle(output, center, 6, (0, 0, 255), -1)
        cv2.arrowedLine(output, center, (round(track.center_x + track.vx * 0.10), round(track.center_y + track.vy * 0.10)), (0, 0, 255), 2, tipLength=0.20)
    if stone is None:
        state_text = "StoneState: none"
    else:
        state_text = (
            f"StoneState x={stone.x:.1f} y={stone.y:.1f} "
            f"v=({stone.vx:.1f}, {stone.vy:.1f}) {stone.tracking_state.value}"
        )
    cv2.putText(output, f"Tracker: {track.state.value if track else 'none'}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(output, state_text, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    cv2.putText(output, state_text, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1)
    cv2.putText(output, f"GameState puck: {'ready' if game_state else 'none'}   Display FPS: {display_fps:.1f}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (35, 35, 35), 1)
    return output


class TkPreview:
    """Tk 双画面预览，避开 OpenCV HighGUI 冲突。"""

    def __init__(self) -> None:
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Vision -> Tracker -> GameState")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.StringVar(value="等待摄像头画面")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0, 8))
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self._photo = None

    def _close(self) -> None:
        self.closed = True

    def draw(self, raw, annotated, stone) -> None:
        target_width = 640
        scale = min(1.0, target_width / raw.shape[1])
        size = (round(raw.shape[1] * scale), round(raw.shape[0] * scale))
        raw_preview = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        annotated_preview = cv2.resize(annotated, size, interpolation=cv2.INTER_AREA)
        combined = cv2.hconcat((raw_preview, annotated_preview))
        cv2.putText(combined, "RAW", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined, "DETECTION / TRACKER / STATE", (size[0] + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".ppm", rgb)
        if not ok:
            return
        self._photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()), format="PPM")
        self.label.configure(image=self._photo)
        self.status.set(
            "StoneState: none"
            if stone is None
            else f"StoneState ({stone.x:.1f}, {stone.y:.1f})  v=({stone.vx:.1f}, {stone.vy:.1f})  {stone.tracking_state.value}    Q / ESC: quit"
        )
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self) -> None:
        if self.root.winfo_exists():
            self.root.destroy()


def main():
    args = build_parser().parse_args()
    if args.print_interval <= 0 or args.preview_fps <= 0 or args.fps <= 0:
        raise SystemExit("--print-interval、--preview-fps 和 --fps 必须大于 0")
    preview = None
    if not args.no_display:
        try:
            preview = TkPreview()
        except tk.TclError as exc:
            raise SystemExit(f"无法启动 Tk 预览，请使用 --no-display：{exc}") from exc

    camera = CameraManager(CameraConfig(device=args.device, width=args.width, height=args.height, requested_fps=args.fps, pixel_format="MJPG", backend=args.backend))
    detector = StoneDetector(roi=tuple(args.roi), color_space="hsv", lower=tuple(args.lower), upper=tuple(args.upper), min_area=args.min_area, min_radius=args.min_radius, min_circularity=args.min_circularity)
    tracker = StoneTracker(max_distance=args.max_distance, max_missed_frames=args.max_missed_frames)
    last_sequence = -1
    last_display_time = 0.0
    last_print_time = 0.0
    last_frame_time = time.monotonic()
    display_fps = 0.0
    started = time.monotonic()

    try:
        camera.start()
        print("Vision pipeline started: Camera -> StoneDetector -> StoneTracker -> StoneState -> PuckState/GameState")
        print("Press Ctrl+C to stop" + (", or Q / ESC in the preview window." if preview else "."))
        while True:
            frame = camera.get_latest_frame()
            if frame is None or frame.sequence == last_sequence:
                time.sleep(0.001)
                continue
            last_sequence = frame.sequence
            now = time.monotonic()
            frame_dt = now - last_frame_time
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                display_fps = instant_fps if display_fps == 0.0 else display_fps * 0.9 + instant_fps * 0.1
            last_frame_time = now

            detection = detector.detect(frame)
            tracks = tracker.update(detection)
            track = tracks[0] if tracks else None
            stone = StoneState.from_tracker(track) if track is not None else None
            game_state = GameState.from_vision(stone) if stone is not None else None

            if stone is not None and now - last_print_time >= args.print_interval:
                last_print_time = now
                print(
                    f"StoneState(x={stone.x:.1f}, y={stone.y:.1f}, vx={stone.vx:.1f}, "
                    f"vy={stone.vy:.1f}, tracking_state={stone.tracking_state.value}) | "
                    f"PuckState(x={game_state.puck.x:.1f}, y={game_state.puck.y:.1f})"
                )
            elif stone is None and now - last_print_time >= args.print_interval:
                last_print_time = now
                print("StoneState: none")

            if preview is not None and now - last_display_time >= 1.0 / args.preview_fps:
                last_display_time = now
                preview.draw(frame.image, annotate(frame.image, tuple(args.roi), detection, track, stone, game_state, display_fps), stone)
                if preview.closed:
                    break
            if args.duration > 0 and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    main()
