"""离线验证：仿真冰壶 -> Detection -> StoneTracker -> AirHockeyAI。

本脚本不启动 CameraManager，也不会调用 StoneDetector；它直接由仿真位置构造
Detection，以便单独验证从视觉坐标到 AI 决策的闭环。

Examples::

    python3 air_hockey/tools/test_ai_pipeline.py
    python3 air_hockey/tools/test_ai_pipeline.py --visualize --duration 0
    python3 air_hockey/tools/test_ai_pipeline.py --drop-rate 0.08 --realtime
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import random
import sys
import time
import tkinter as tk
from typing import Optional


# ``tools`` 目录直接运行时，加入视觉工程和相邻仿真工程的模块根目录。
SCRIPT_DIR = Path(__file__).resolve().parent
VISION_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = VISION_ROOT.parent
SIMULATION_ROOT = next(
    (
        directory
        for directory in PROJECT_ROOT.iterdir()
        if (directory / "air_hockey_ai.py").is_file()
        and (directory / "air_hockey_physics.py").is_file()
    ),
    None,
)
if SIMULATION_ROOT is None:
    raise RuntimeError("未找到包含 air_hockey_ai.py 的仿真工程")
sys.path.insert(0, str(VISION_ROOT))
sys.path.insert(0, str(SIMULATION_ROOT))

import air_hockey_config as layout
from air_hockey_ai import AIDecision, AirHockeyAI
from air_hockey_physics import PuckMotion, goal_scorer
from game_state import GameState, PuckState, StoneState
from vision.tracker import StoneTracker
from vision.types import Detection, Track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线运行仿真位置 -> Detection -> StoneTracker -> AirHockeyAI。"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="仿真帧率，默认 60")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="运行秒数；0 表示持续运行至 Ctrl+C / Q，默认 15",
    )
    parser.add_argument(
        "--print-interval", type=float, default=0.25, help="终端输出间隔（秒）"
    )
    parser.add_argument("--drop-rate", type=float, default=0.0, help="模拟漏检概率 [0, 1)")
    parser.add_argument("--noise", type=float, default=1.2, help="Detection 坐标高斯噪声（像素）")
    parser.add_argument(
        "--ai-aim-error",
        type=float,
        default=0.0,
        help="注入到现有 AI 的瞄准随机误差（像素）；默认 0，便于观察闭环稳定性",
    )
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--realtime", action="store_true", help="按真实时间播放，而非尽快执行")
    parser.add_argument("--visualize", action="store_true", help="显示轨迹窗口（默认使用 Tk）")
    parser.add_argument(
        "--visualizer",
        choices=("tk", "opencv"),
        default="tk",
        help="可视化后端，默认 tk；opencv 使用 OpenCV HighGUI",
    )
    return parser


def reset_puck(puck: PuckMotion) -> None:
    """模拟得分后的发球，防止冰球穿过球门后持续跑到场地外。"""
    puck.x = layout.RINK_CENTER_X - 90.0
    puck.y = layout.RINK_BOTTOM - 120.0
    puck.set_immediate_velocity(320.0, -570.0)


def simulate_puck_step(puck: PuckMotion, dt: float) -> bool:
    """推进已有 PuckMotion；返回本帧是否因穿过球门而重新发球。"""
    puck.advance_velocity(dt)
    puck.x += puck.vx * dt
    puck.y += puck.vy * dt
    puck.resolve_walls()

    # PuckMotion 会允许球从球门口离场，真实游戏控制器随后会结算比分。
    # 离线脚本没有比分系统，因此在同一位置重新发球。
    if goal_scorer(puck) is not None:
        reset_puck(puck)
        return True

    # 速度衰减为零后再次发球，让离线演示一直产生有意义的 AI 输入。
    if math.hypot(puck.vx, puck.vy) <= layout.PUCK_STOP_SPEED:
        puck.set_immediate_velocity(260.0, -520.0)
    return False


def simulate_detection(puck: PuckMotion, timestamp: float, rng: random.Random, noise: float, drop_rate: float) -> Optional[Detection]:
    """用仿真真值构造单帧 Detection，模拟定位抖动和可选漏检。"""
    if rng.random() < drop_rate:
        return None
    center_x = puck.x + rng.gauss(0.0, noise)
    center_y = puck.y + rng.gauss(0.0, noise)
    radius = max(1.0, layout.PUCK_RADIUS + rng.gauss(0.0, noise * 0.1))
    return Detection(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        area=math.pi * radius * radius,
        timestamp=timestamp,
        circularity=0.97,
        score=0.95,
    )


def game_state_from_track(track: Track, target_x: float, target_y: float, reaction_timer: float, stalled_puck_phase: str, difficulty: layout.Difficulty) -> GameState:
    """唯一的 Vision → Game 适配层：Tracker 输出先转为 StoneState。"""
    ai_home_y = layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28
    stone = StoneState.from_tracker(track)
    return GameState(
        ai_x=layout.RINK_CENTER_X,
        ai_y=ai_home_y,
        ai_home_y=ai_home_y,
        target_x=target_x,
        target_y=target_y,
        puck=PuckState.from_stone(stone),
        awaiting_serve=False,
        current_server="player",
        serve_phase="idle",
        stalled_puck_phase=stalled_puck_phase,
        reaction_timer=reaction_timer,
        difficulty=difficulty,
        stone=stone,
    )


class TkVisualizer:
    """不依赖 OpenCV HighGUI 的轻量可视化，适用于 Snap/Qt 冲突环境。"""

    def __init__(self) -> None:
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Offline AI pipeline")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.canvas = tk.Canvas(
            self.root,
            width=round(layout.CANVAS_WIDTH),
            height=round(layout.CANVAS_HEIGHT),
            background="#f2faff",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())

    def _close(self) -> None:
        self.closed = True

    def draw(self, puck: PuckMotion, track: Optional[Track], target: tuple[float, float], path: deque[tuple[int, int]]) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.create_rectangle(layout.RINK_LEFT, layout.RINK_TOP, layout.RINK_RIGHT, layout.RINK_BOTTOM, outline="#2882b4", width=2)
        canvas.create_line(layout.RINK_LEFT, layout.RINK_CENTER_Y, layout.RINK_RIGHT, layout.RINK_CENTER_Y, fill="#dc5565", width=2)
        if len(path) > 1:
            canvas.create_line(*[coordinate for point in path for coordinate in point], fill="#00a850", width=2)
        canvas.create_oval(puck.x - layout.PUCK_RADIUS, puck.y - layout.PUCK_RADIUS, puck.x + layout.PUCK_RADIUS, puck.y + layout.PUCK_RADIUS, fill="#232323", outline="")
        canvas.create_text(puck.x + 20, puck.y - 18, text="simulated puck", anchor="w", fill="#232323")
        if track is not None:
            canvas.create_oval(track.center_x - track.radius - 4, track.center_y - track.radius - 4, track.center_x + track.radius + 4, track.center_y + track.radius + 4, outline="#e22b2b", width=2)
            canvas.create_line(track.center_x, track.center_y, track.center_x + track.vx * 0.10, track.center_y + track.vy * 0.10, fill="#e22b2b", width=2, arrow=tk.LAST)
        tx, ty = target
        canvas.create_line(tx - 14, ty, tx + 14, ty, fill="#b000b5", width=3)
        canvas.create_line(tx, ty - 14, tx, ty + 14, fill="#b000b5", width=3)
        canvas.create_text(tx + 18, ty - 14, text="AI target", anchor="w", fill="#b000b5")
        canvas.create_text(16, 18, text="green: tracker path   red: tracker   Q / ESC: quit", anchor="w", fill="#555555")
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self) -> None:
        if self.root.winfo_exists():
            self.root.destroy()


class OpenCVVisualizer:
    """可选 OpenCV HighGUI 后端；仅在用户显式选择时加载。"""

    def __init__(self) -> None:
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.closed = False

    def draw(self, puck: PuckMotion, track: Optional[Track], target: tuple[float, float], path: deque[tuple[int, int]]) -> None:
        image = self.np.full((int(layout.CANVAS_HEIGHT), int(layout.CANVAS_WIDTH), 3), (242, 250, 255), dtype=self.np.uint8)
        self.cv2.rectangle(image, (int(layout.RINK_LEFT), int(layout.RINK_TOP)), (int(layout.RINK_RIGHT), int(layout.RINK_BOTTOM)), (180, 130, 40), 2)
        if len(path) > 1:
            self.cv2.polylines(image, [self.np.asarray(path, dtype=self.np.int32)], False, (0, 190, 0), 2)
        self.cv2.circle(image, (round(puck.x), round(puck.y)), round(layout.PUCK_RADIUS), (35, 35, 35), -1)
        if track is not None:
            self.cv2.circle(image, (round(track.center_x), round(track.center_y)), round(track.radius + 4), (0, 0, 255), 2)
        tx, ty = round(target[0]), round(target[1])
        self.cv2.drawMarker(image, (tx, ty), (255, 0, 255), self.cv2.MARKER_CROSS, 28, 3)
        self.cv2.imshow("Offline AI pipeline", image)
        if self.cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            self.closed = True

    def close(self) -> None:
        self.cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.print_interval <= 0:
        raise SystemExit("--fps 和 --print-interval 必须大于 0")
    if not 0.0 <= args.drop_rate < 1.0:
        raise SystemExit("--drop-rate 必须在 [0, 1) 范围内")
    if args.noise < 0:
        raise SystemExit("--noise 不能为负数")
    if args.ai_aim_error < 0:
        raise SystemExit("--ai-aim-error 不能为负数")
    visualizer: TkVisualizer | OpenCVVisualizer | None = None
    if args.visualize:
        try:
            visualizer = TkVisualizer() if args.visualizer == "tk" else OpenCVVisualizer()
        except (ImportError, tk.TclError) as exc:
            raise SystemExit(f"无法启动 {args.visualizer} 可视化：{exc}") from exc

    dt = 1.0 / args.fps
    rng = random.Random(args.seed)
    tracker = StoneTracker(max_distance=80.0, max_missed_frames=5, velocity_alpha=0.2)
    ai = AirHockeyAI()
    difficulty = replace(layout.DIFFICULTIES["普通"], aim_error=args.ai_aim_error)
    puck = PuckMotion(
        x=layout.RINK_CENTER_X - 90.0,
        y=layout.RINK_BOTTOM - 120.0,
        vx=320.0,
        vy=-570.0,
    )
    target = (layout.RINK_CENTER_X, layout.RINK_TOP + 90.0)
    reaction_timer = 0.0
    stalled_puck_phase = "idle"
    path: deque[tuple[int, int]] = deque(maxlen=180)
    elapsed = 0.0
    last_print = -args.print_interval
    started = time.monotonic()

    print("Offline pipeline started: simulation -> Detection -> StoneTracker -> GameState -> AirHockeyAI")
    print(
        "Press Ctrl+C to stop"
        + (f", or Q / ESC in the {args.visualizer} window." if visualizer else ".")
    )
    try:
        while args.duration <= 0.0 or elapsed < args.duration:
            frame_started = time.monotonic()
            restarted = simulate_puck_step(puck, dt)
            if restarted:
                # 让下一帧从新球开始建轨，避免跨球门的速度污染 Tracker。
                tracker.reset()
                path.clear()
                target = (layout.RINK_CENTER_X, layout.RINK_TOP + 90.0)
                reaction_timer = 0.0
                stalled_puck_phase = "idle"
            detection = simulate_detection(puck, elapsed, rng, args.noise, args.drop_rate)
            tracks = tracker.update(detection)
            track = tracks[0] if tracks else None
            if track is not None:
                path.append((round(track.center_x), round(track.center_y)))
                state = game_state_from_track(
                    track,
                    target[0],
                    target[1],
                    reaction_timer,
                    stalled_puck_phase,
                    difficulty,
                )
                decision: AIDecision = ai.update(state, dt)
                target = (decision.target_x, decision.target_y)
                reaction_timer = decision.reaction_timer
                stalled_puck_phase = decision.stalled_puck_phase
                if elapsed >= last_print + args.print_interval:
                    last_print = elapsed
                    print(
                        f"t={elapsed:5.2f}s | simulated puck=({puck.x:6.1f}, {puck.y:6.1f}) "
                        f"v=({puck.vx:6.1f}, {puck.vy:6.1f}) | "
                        f"tracker=({track.center_x:6.1f}, {track.center_y:6.1f}) "
                        f"v=({track.vx:6.1f}, {track.vy:6.1f}) | "
                        f"AI decision=phase:{decision.stalled_puck_phase} | "
                        f"AI target=({decision.target_x:6.1f}, {decision.target_y:6.1f})"
                    )
            if visualizer is not None:
                visualizer.draw(puck, track, target, path)
                if visualizer.closed:
                    break
            elapsed += dt
            # 有窗口时默认按帧率播放，保证轨迹肉眼可观察；纯终端模式可快速跑完。
            if args.realtime or visualizer is not None:
                time.sleep(max(0.0, dt - (time.monotonic() - frame_started)))
    except KeyboardInterrupt:
        pass
    finally:
        if visualizer is not None:
            visualizer.close()
    print(f"Offline pipeline stopped after {elapsed:.2f}s ({time.monotonic() - started:.2f}s wall time).")


if __name__ == "__main__":
    main()
