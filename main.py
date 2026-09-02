"""项目统一入口。

用法：
    python main.py               # 启动冰壶仿真游戏
    python main.py --vision      # 实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI

线程分工：
    主线程    Tk mainloop + 显示定时器
    处理线程  取最新帧 -> 间隔检测/追踪 -> AI -> 标注
    编码线程  最新标注帧 -> 缩放 -> PNG（按预览帧率限速）
"""

import argparse     # 命令行参数解析模块
import base64       # 图片编码模块
import subprocess   # 启动其他程序
import sys
import threading    # 多线程
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # 找到 main.py 所在文件夹
VISION_ROOT = PROJECT_ROOT / "air_hockey"       # 视觉代码
SIM_ROOT = PROJECT_ROOT / "冰壶仿真"             # 仿真代码
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(VISION_ROOT))

import cv2
import tkinter as tk

import air_hockey_config as layout
from air_hockey_ai import AirHockeyAI
from camera import CameraConfig, CameraManager
from game_state import GameState, StoneState
from vision import StoneDetector, VisionPipeline
from vision.predictor import predict_position, predict_trajectory
from vision.tracker import StoneTracker

DISPLAY_WIDTH = 640         # 窗口图片最大宽度 640
STATS_INTERVAL = 5.0        # 统计间隔
DETECTION_INTERVAL = 3      # 检测间隔

AI_HOME_Y = layout.RINK_TOP + (layout.RINK_CENTER_Y - layout.RINK_TOP) * 0.28   # AI 初始位置


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


def annotate(image, roi, detection, track, ai_target_pixel, display_fps, trajectory=None):
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
    if trajectory:
        for i, (px, py) in enumerate(trajectory):
            alpha = max(40, 255 - i * 12)
            cv2.circle(output, (round(px), round(py)), 2, (alpha, 160, 90), -1)
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
    # 初始化 ROI 区域
    roi = tuple(args.roi)   # tuple() 是一个转换工具，可以把一些数据转换成“元组”

    # 读取是否有性能测试模式
    headless = args.headless
    
    # 如果没有加 -- headless 启动性能测试模式
    if not headless:
        window = VisionWindow()     # 打开窗口

    # 创建 detector 对象
    detector = StoneDetector(
        roi=roi,
        lower=tuple(args.lower),
        upper=tuple(args.upper),
        min_area=500.0,
        min_radius=25.0,
        min_circularity=0.65,
    )

    # 创建 tracker 对象
    tracker = StoneTracker(max_missed_frames=DETECTION_INTERVAL + 1)    # 每 3 帧检测一次，中间是 tracker 预测，用于提高速度

    # 创建 ai 对象，目前用的仿真参数，后续真机修改
    ai = AirHockeyAI()

    # 创建 vision_pipeline 对象，相机畸变校正
    vision_pipeline = VisionPipeline(
        calibration_file=args.calibration,
        enabled=not args.disable_undistort,     # 不加参数默认启动校正
    )

    # 创建 camera 对象
    camera = CameraManager()
    try:
        camera.start()
    except Exception as exc:
        if window is not None:
            window.close()
        raise SystemExit(f"摄像头启动失败：{exc}")

    # 终端打印信息
    mode_label = "headless 性能测试" if headless else "实时视觉演示"
    print(f"{mode_label}开始{'，Ctrl+C 退出' if headless else '，Q / ESC 或关闭窗口退出'}")
    print(f"视觉管线：最新帧 + 每 {DETECTION_INTERVAL} 帧检测，其余帧使用 tracker 预测")
    print("视觉校正:")
    print(f"  calibration: {args.calibration}")
    print(f"  undistort: {'校正开启' if not args.disable_undistort else '校正关闭'}")

    preview_lock = threading.Lock()     # 创建一个互斥锁，防止线程冲突

    # 创建一个字典作为线程间传递数据的公共缓冲区
    shared = {"img": None,              # 最新图片数据（Numpy 数组）
              "img_seq": -1,            # 图片帧号
              "png": None,              # 存储压缩后的图片数据
              "png_seq": 0,
              "status": "等待画面", 
              "fatal": None}            # 错误信息缓冲区

    # 创建一个事件对象，用于停止线程
    stop = threading.Event()

    # 视觉主循环，核心部分
    def processing_loop():
        """启动处理线程，只取最新帧；检测按固定间隔运行，中间帧由 tracker 预测。"""
        # 初始化与变量准备
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
        window_closed = lambda: window.closed if window is not None else False

        try:
            while not stop.is_set() and not window_closed():
                # 只处理最新的一帧
                frame = camera.get_latest_frame()
                if frame is None or frame.sequence == last_sequence:
                    time.sleep(0.001)       # 稍微休眠，防止 CPU 空转满载
                    continue
                last_sequence = frame.sequence
                frame_index += 1

                frame = vision_pipeline.process(frame)      # 畸变校正

                now = frame.timestamp
                dt = 0.0 if last_timestamp is None else max(0.0, now - last_timestamp)
                last_timestamp = now
                if dt > 0:
                    instant = 1.0 / dt
                    display_fps = instant if display_fps == 0.0 else display_fps * 0.9 + instant * 0.1

                t0 = time.perf_counter()
                detection = None

                # 第 1 帧以及后面每 3 帧检测一次，降低 CPU 负担
                if frame_index == 1 or frame_index % DETECTION_INTERVAL == 0:
                    detection = detector.detect(frame)
                    detect_ms = detect_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
                    tracks = tracker.update(detection)
                # 否则预测
                else:
                    tracks = tracker.predict(frame.timestamp)

                # 拿到第 1 个目标
                track = tracks[0] if tracks else None

                correction_status = "ON" if vision_pipeline.camera_matrix is not None else "OFF"
                status_text = f"FPS {display_fps:.1f} | 校正 {correction_status} | 未检测到冰壶"
                ai_target_pixel = None
                trajectory = None
                if track is not None:
                    # 坐标转换
                    stone = track_to_rink_state(track, roi)

                    # 预测冰壶 0.5s 后的位置
                    pred_x, pred_y = predict_position(stone.x, stone.y, stone.vx, stone.vy, 0.5)

                    # 生成未来轨迹线
                    trajectory = predict_trajectory(stone.x, stone.y, stone.vx, stone.vy, duration=2.0, step=0.15)

                    # 如果检测到冰壶，就创建一个 state 对象
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

                    # 输入当前冰壶状态，得到 ai 目标位置
                    decision = ai.update(state, dt)
                    target[0], target[1] = decision.target_x, decision.target_y
                    reaction_timer = decision.reaction_timer
                    stalled_phase = decision.stalled_stone_phase
                    tx, ty = rink_to_pixel(decision.target_x, decision.target_y, roi)

                    # 转换回来
                    ai_target_pixel = (round(tx), round(ty))
                    status_text = (
                        f"FPS {display_fps:.1f} | 校正 {correction_status} | "
                        f"追踪 {track.state.value} | 位置 ({stone.x:.0f}, {stone.y:.0f}) "
                        f"| 速度 ({stone.vx:.0f}, {stone.vy:.0f}) | "
                        f"AI 目标 ({target[0]:.0f}, {target[1]:.0f})"
                    )

                # 把信息画到图片上
                marked = annotate(frame.image, roi, detection, track, ai_target_pixel, display_fps, trajectory)

                # 加锁
                with preview_lock:
                    shared["img"] = marked
                    shared["img_seq"] = frame.sequence
                    shared["status"] = status_text + "    Q / ESC 退出"

                # 性能统计打印
                frame_ms = frame_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
                stats_interval = 1.0 if headless else STATS_INTERVAL
                if time.perf_counter() - stats_timer >= stats_interval:
                    stats_timer = time.perf_counter()
                    capture = camera.get_stats()
                    track_state = track.state.value if track else "none"
                    if headless:
                        print(
                            f"[perf] FPS {display_fps:.1f} | 处理 {frame_ms:.1f} ms "
                            f"| 检测 {detect_ms:.1f} ms | 采集 {capture.current_fps:.1f} FPS "
                            f"| Tracker {track_state}"
                        )
                    else:
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
        """启动编码线程，最新标注帧 -> 缩放 -> PNG，按预览帧率限速。"""
        seen = -1       # 记录最后处理过的序列号，防止重复编码

        while not stop.wait(1.0 / args.preview_fps):
            # 加锁读取
            with preview_lock:
                img = shared["img"]
                seq = shared["img_seq"]
            # 去重检查
            if img is None or seq == seen:
                continue
            seen = seq
            scale = DISPLAY_WIDTH / img.shape[1]

            # 缩放
            small = cv2.resize(img, (DISPLAY_WIDTH, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            # 编码
            ok, encoded = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                with preview_lock:
                    shared["png"] = base64.b64encode(encoded.tobytes())
                    shared["png_seq"] += 1

    # 启动子线程
    threading.Thread(target=processing_loop, name="processing", daemon=True).start()
    if not headless:
        threading.Thread(target=encoding_loop, name="encoder", daemon=True).start()
        window.start(preview_lock, shared)

    # 主线程
    try:
        if headless:
            stop.wait()
        else:
            window.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        camera.stop()
        if window is not None:
            window.close()


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")    # 创建 parser 对象

    # 添加参数，均是 bool 值
    parser.add_argument("--vision", action="store_true", help="实时视觉演示：摄像头 -> 检测 -> 追踪 -> AI")
    parser.add_argument("--headless", action="store_true", help="无显示性能测试模式（需配合 --vision）")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正")
    parser.add_argument("--roi", type=int, nargs=4, default=(350, 0, 580, 650), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(179, 255, 255), metavar=("C1", "C2", "C3"))

    args = parser.parse_args()  # 执行解析，将上面的字符串转换成 Python 对象

    if args.vision:
        run_vision(args)
    else:
        run_game()


if __name__ == "__main__":
    main()
