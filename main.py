"""项目统一入口。

用法：
    python main.py                      # 默认：实时视觉演示（摄像头 -> 检测 -> 追踪 -> AI）
    python main.py --headless           # 视觉模式：无显示性能基准测试
    python main.py --sim                # 仿真模式：启动冰壶仿真游戏（也可写作 --game）

线程分工：
    主线程    Tk mainloop + 显示定时器（GUI 模式）/ 等待退出（headless 模式）
    处理线程  取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态（不限速）
    编码线程  最新 VisionResult -> render 标注 -> 缩放 -> PPM（按预览帧率限速，仅 GUI 模式）
"""

import argparse
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # 找到 main.py 所在文件夹
SIM_ROOT = PROJECT_ROOT / "冰壶仿真"             # 仿真代码（仅 --sim 子进程使用）

import cv2
import tkinter as tk

from air_hockey.ai import AIDecision
from air_hockey.app.renderer import format_status, render
from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera import CameraManager
from air_hockey.control import PlcControlAdapter, PlcOutputWorker
from air_hockey.recording import RuntimeRecorder

DISPLAY_WIDTH = 640         # 窗口图片最大宽度 640
STATS_INTERVAL = 5.0        # 统计间隔


def run_game():
    """启动仿真子进程，显式设置 cwd 避免相对路径资源加载报错。"""
    subprocess.run([sys.executable, str(SIM_ROOT / "air_hockey.py")], cwd=str(SIM_ROOT))


def load_plc_interface():
    """按文件路径加载仿真目录中的 PLCInterface。

    不把 冰壶仿真/ 加入 sys.path：该目录含 air_hockey.py，会把 air_hockey 包遮盖掉。
    """
    import importlib.util

    path = SIM_ROOT / "plc_interface.py"
    spec = importlib.util.spec_from_file_location("air_hockey_plc_interface", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLCInterface


def encode_preview_ppm(result, table_roi, camera_geometry):
    """将最新 VisionResult 渲染并编码为 Tk 可显示的 PPM 原始像素。"""
    marked = render(result, table_roi, camera_geometry)
    scale = DISPLAY_WIDTH / marked.shape[1]
    small = cv2.resize(marked, (DISPLAY_WIDTH, round(marked.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return f"P6 {w} {h} 255\n".encode() + rgb.tobytes()


class VisionWindow:
    """Tk 窗口，只在主线程使用；定时器从共享区取 PPM 原始像素显示。"""

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
                ppm = self._shared["ppm"]
                seq = self._shared["ppm_seq"]
                status = self._shared["status"]
                fatal = self._shared["fatal"]
            if ppm is not None and seq != self._seen_seq:
                self._photo = tk.PhotoImage(data=ppm, format="PPM")
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
    headless = args.headless

    window = None
    preview_lock = None
    shared = None

    recorder = None
    if args.record:
        recorder = RuntimeRecorder(
            args.record,
            source="runtime",
            meta={"mode": "headless" if headless else "display"},
        )

    plc = None
    plc_worker = None
    plc_adapter = PlcControlAdapter() if args.plc else None
    if args.plc:
        try:
            plc = load_plc_interface()(args.plc)
        except Exception as exc:
            print(f"[PLC] 初始化失败，跳过 PLC 输出：{exc!r}")
            plc = None
        if plc is not None:
            if plc.connect():
                plc_worker = PlcOutputWorker(plc, interval=1.0 / args.plc_rate)
                plc_worker.start()
                print(f"PLC 输出已启用：{args.plc} @ {args.plc_rate:g} Hz")
            else:
                print("[PLC] 连接失败，跳过 PLC 输出")

    if not headless:
        window = VisionWindow()
        preview_lock = threading.Lock()
        shared = {
            "result": None,
            "result_seq": -1,
            "ppm": None,
            "ppm_seq": 0,
            "status": "等待画面",
            "fatal": None,
        }

    table_calibration_file = None if args.disable_homography else args.table_calibration
    runtime = VisionRuntime(
        table_roi=tuple(args.roi),
        lower=tuple(args.lower),
        upper=tuple(args.upper),
        calibration_file=args.calibration,
        table_calibration_file=table_calibration_file,
        disable_undistort=args.disable_undistort,
    )

    camera = CameraManager()
    try:
        camera.start()
    except Exception as exc:
        if window is not None:
            window.close()
        raise SystemExit(f"摄像头启动失败：{exc}")

    mode_label = "headless 性能测试" if headless else "实时视觉演示"
    print(f"{mode_label}开始{'，Ctrl+C 退出' if headless else '，Q / ESC 或关闭窗口退出'}")
    print(f"视觉管线：最新帧 + 每 {runtime.detection_interval} 帧检测，其余帧使用 tracker 预测")
    print("视觉校正:")
    print(f"  camera calibration: {args.calibration}")
    print(f"  undistort: {'OFF' if args.disable_undistort else 'ON'}")
    print(f"  table mapping: {'linear ROI' if args.disable_homography else 'Homography'}")
    print(f"  table calibration: {table_calibration_file if table_calibration_file is not None else 'None (linear ROI)'}")
    if recorder is not None:
        print(f"运行数据记录: {args.record}")

    stop = threading.Event()

    def processing_loop():
        """处理线程：取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态。"""
        last_sequence = -1
        stats_timer = time.perf_counter()

        window_closed = (lambda: window.closed) if window is not None else (lambda: False)

        try:
            while not stop.is_set() and not window_closed():
                frame = camera.get_latest_frame()
                if frame is None or frame.sequence == last_sequence:
                    time.sleep(0.001)
                    continue
                last_sequence = frame.sequence

                result = runtime.process_frame(frame)

                # AI 决策 -> ControlCommand -> PLC 请求（同时也作为运行日志的一环）
                decision = None
                plc_request = None
                if result.ai_target is not None and result.curling_state is not None:
                    decision = AIDecision(result.ai_target[0], result.ai_target[1], "")
                    if plc_worker is not None and result.prediction is not None:
                        try:
                            command = plc_adapter.build_command(
                                decision,
                                result.curling_state,
                                result.prediction,
                                timestamp=result.frame.timestamp,
                            )
                            plc_request = plc_adapter.to_write_request(
                                command,
                                ai_position=runtime.ai_current_pos,
                                curling_state=result.curling_state,
                                timestamp=result.frame.timestamp,
                            )
                            # 非阻塞：只投递最新请求，实际写入在后台线程完成
                            plc_worker.submit(plc_request)
                        except ValueError:
                            plc_request = None  # 目标非法(非有限值)时跳过本帧 PLC 输出

                if recorder is not None:
                    recorder.record(result, ai_decision=decision, plc_request=plc_request)

                if not headless:
                    status_text = format_status(
                        result,
                        runtime.camera_geometry.enabled and runtime.camera_geometry.camera_matrix is not None,
                    )
                    with preview_lock:
                        shared["result"] = result
                        shared["result_seq"] = result.frame.sequence
                        shared["status"] = status_text + "    Q / ESC 退出"

                stats_interval = 1.0 if headless else STATS_INTERVAL
                if time.perf_counter() - stats_timer >= stats_interval:
                    stats_timer = time.perf_counter()
                    capture = camera.get_stats()
                    track_state = result.track.state.value if result.track else "none"
                    if headless:
                        print(
                            f"[perf] FPS {result.fps:.1f} | 处理 {runtime.frame_ms:.1f} ms "
                            f"| 检测 {runtime.detect_ms:.1f} ms | 采集 {capture.current_fps:.1f} FPS "
                            f"| Tracker {track_state}"
                        )
                    else:
                        print(
                            f"[stats] 处理 {result.fps:.1f} FPS | 单帧 {runtime.frame_ms:.1f} ms "
                            f"(检测 {runtime.detect_ms:.1f} ms) | 采集 {capture.current_fps:.1f} FPS"
                        )
        except Exception as exc:
            traceback.print_exc()
            if preview_lock is not None and shared is not None:
                with preview_lock:
                    shared["fatal"] = f"处理线程异常退出：{exc!r}"
        finally:
            stop.set()

    def encoding_loop():
        """预览转换线程：最新 VisionResult -> render 标注 -> 缩放 -> PPM（按预览帧率限速）。"""
        seen = -1
        while not stop.wait(1.0 / args.preview_fps):
            with preview_lock:
                result = shared["result"]
                seq = shared["result_seq"]
            if result is None or seq == seen:
                continue
            seen = seq

            ppm_data = encode_preview_ppm(result, runtime.table_roi, runtime.camera_geometry)
            with preview_lock:
                shared["ppm"] = ppm_data
                shared["ppm_seq"] += 1

    processing_thread = threading.Thread(target=processing_loop, name="processing", daemon=True)
    processing_thread.start()

    if not headless:
        threading.Thread(target=encoding_loop, name="encoder", daemon=True).start()
        window.start(preview_lock, shared)

    try:
        if headless:
            stop.wait()
        else:
            window.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if processing_thread.is_alive():
            processing_thread.join(timeout=0.6)
        if recorder is not None:
            recorded_path = recorder.close()
            if recorded_path is not None:
                print(f"运行数据已记录 {recorder.frame_count} 帧 -> {recorded_path}")
        if plc_worker is not None:
            plc_worker.stop()
            print(f"PLC 输出已停止（写入 {plc_worker.write_count} 次，失败 {plc_worker.error_count} 次）")
        if plc is not None:
            plc.disconnect()
        camera.stop()
        if window is not None:
            window.close()


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")

    # 模式开关：默认运行视觉模式；传入 --sim / --game 运行仿真游戏
    parser.add_argument("--sim", "--game", dest="sim", action="store_true", help="启动冰壶仿真游戏（默认运行实时视觉演示）")
    parser.add_argument("--vision", action="store_true", help="（兼容保留）实时视觉演示模式")

    # 视觉模式相关参数
    parser.add_argument("--headless", action="store_true", help="无显示性能测试模式")
    parser.add_argument("--record", default=None, metavar="PATH", help="记录运行数据到 JSON 文件（CurlingState/PredictionState/timestamp/FPS）")
    parser.add_argument("--plc", default=None, metavar="IP", help="启用 AI->PLC 输出并连接该 S7 PLC（例如 192.168.0.1）")
    parser.add_argument("--plc-rate", type=float, default=30.0, help="PLC 写入频率上限（Hz）")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机内参标定文件")
    parser.add_argument("--table-calibration", default="calibration/table_homography.npz", help="球台四点 Homography 标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正（必须同时 --disable-homography）")
    parser.add_argument("--disable-homography", action="store_true", help="关闭 Homography，回退到旧 ROI 线性映射（调试用）")
    parser.add_argument("--roi", type=int, nargs=4, default=(350, 0, 580, 650), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(179, 255, 255), metavar=("C1", "C2", "C3"))

    args = parser.parse_args()

    if args.plc and args.plc_rate <= 0.0:
        parser.error("--plc-rate 必须大于 0")

    if not args.sim and args.disable_undistort and not args.disable_homography:
        parser.error(
            "--disable-undistort 必须与 --disable-homography 一起使用："
            "Homography 基于去畸变坐标标定，二者不能只关一半"
        )

    if args.sim:
        run_game()
    else:
        run_vision(args)


if __name__ == "__main__":
    main()
