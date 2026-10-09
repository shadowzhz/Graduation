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
import json
import math
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # 找到 main.py 所在文件夹
SIM_ROOT = PROJECT_ROOT / "冰壶仿真"             # 仿真代码（仅 --sim 子进程使用）

import cv2
import tkinter as tk

from air_hockey.app.renderer import format_status, render_preview
from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.control import PLCLink, PlcControlAdapter
from air_hockey.recording import RuntimeRecorder

DISPLAY_WIDTH = 640         # 窗口图片最大宽度 640
STATS_INTERVAL = 5.0        # 统计间隔


class LatencySamples:
    """保留最近的阶段样本；全部时间均为主机单调时钟，不代表曝光到执行器延迟。"""

    def __init__(self, window=4096):
        self._lock = threading.Lock()
        self._values = {}
        self.window = window

    def add(self, stage, elapsed_ms):
        if elapsed_ms is None:
            return
        with self._lock:
            self._values.setdefault(stage, deque(maxlen=self.window)).append(float(elapsed_ms))

    @staticmethod
    def distribution(values):
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "max": ordered[-1],
        }

    def snapshot(self):
        with self._lock:
            return {key: self.distribution(values) for key, values in self._values.items()}


def run_game(plc_ip=None, plc_rate=30.0):
    """启动仿真子进程，显式设置 cwd 避免相对路径资源加载报错。"""
    command = [sys.executable, str(SIM_ROOT / "air_hockey.py")]
    if plc_ip:
        command.extend(("--plc", plc_ip, "--plc-rate", str(plc_rate)))
    raise SystemExit(subprocess.run(command, cwd=str(SIM_ROOT)).returncode)


def encode_preview_ppm(result, table_roi, camera_geometry, brightness_gain=1.0):
    """将最新 VisionResult 渲染并编码为 Tk 可显示的 PPM 原始像素。"""
    small = render_preview(result, table_roi, camera_geometry, DISPLAY_WIDTH)
    if brightness_gain != 1.0:
        cv2.convertScaleAbs(small, small, alpha=brightness_gain)
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
        self.root.bind("Q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.status = tk.StringVar(value="等待画面")
        status_panel = tk.Frame(self.root, width=DISPLAY_WIDTH, height=64)
        status_panel.pack(fill="x")
        status_panel.pack_propagate(False)
        tk.Label(
            status_panel, textvariable=self.status, anchor="nw", justify="left",
            wraplength=DISPLAY_WIDTH - 12,
        ).pack(fill="both", expand=True)
        self._photo = None
        self._seen_seq = -1
        self._lock = None
        self._shared = None

    def attach_plc(self, link):
        """轴命令由现场操作员主动触发；连接成功不自动使能。"""
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        for label, command in (("E 轴使能", link.enable_axes),
                               ("H 轴回零", link.home_axes),
                               ("C 轴复位", link.axes_reset)):
            tk.Button(controls, text=label, command=command).pack(side="left", expand=True)
        for key, command in (("e", link.enable_axes), ("h", link.home_axes),
                             ("c", link.axes_reset)):
            self.root.bind(key, lambda _event, callback=command: callback())

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
    samples = LatencySamples() if headless or args.perf_json else None
    processing_failure = [None]

    recorder = None

    table_calibration_file = None if args.disable_homography else args.table_calibration
    if table_calibration_file is not None and not Path(table_calibration_file).is_file():
        raise SystemExit(
            f"缺少球台标定文件: {table_calibration_file}。先运行 "
            "python3 air_hockey/tools/calibrate_table.py 完成实台标定；"
            "仅调试线性映射时使用 --disable-homography。"
        )
    runtime = VisionRuntime(
        table_roi=tuple(args.roi),
        lower=tuple(args.lower),
        upper=tuple(args.upper),
        calibration_file=args.calibration,
        table_calibration_file=table_calibration_file,
        disable_undistort=args.disable_undistort,
        profile=samples is not None,
    )

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

    camera = CameraManager(CameraConfig(device=args.camera_device))
    try:
        camera.start()
        if args.record:
            recorder = RuntimeRecorder(
                args.record,
                source="runtime",
                meta={"mode": "headless" if headless else "display"},
            )
    except Exception as exc:
        try:
            camera.stop()
        finally:
            if window is not None:
                window.close()
        raise SystemExit(f"实时运行启动失败：{exc}") from exc

    plc = None
    plc_adapter = PlcControlAdapter() if args.plc else None
    if args.plc:
        plc = PLCLink(args.plc, period=1.0 / args.plc_rate)
        plc.start()
        if window is not None:
            window.attach_plc(plc)
        print(f"PLC 状态读取与运动控制已启动：{args.plc} @ {args.plc_rate:g} Hz；轴需现场主动使能")

    mode_label = "headless 性能测试" if headless else "实时视觉演示"
    print(f"{mode_label}开始{'，Ctrl+C 退出' if headless else '，Q / ESC 或关闭窗口退出'}")
    if camera.info is not None:
        info = camera.info
        print(
            f"{info.backend}: {info.device} | 协商 {info.width}x{info.height} "
            f"@ {info.negotiated_fps:g} FPS ({info.source_format} -> {info.output_format}); "
            "实际采集速度以帧计数为准"
        )
    print(f"视觉管线：最新帧 + 每 {runtime.detection_interval} 帧检测，其余帧使用 tracker 预测")
    print("视觉校正:")
    print(f"  camera calibration: {args.calibration}")
    print(f"  undistort: {'OFF' if args.disable_undistort else 'ON'}")
    print(f"  table mapping: {'linear ROI' if args.disable_homography else 'Homography'}")
    print(f"  table calibration: {table_calibration_file if table_calibration_file is not None else 'None (linear ROI)'}")
    if recorder is not None:
        print(f"运行数据记录: {args.record}")

    stop = threading.Event()
    skipped_frames = [0]
    gst_pts_count = [0]

    def processing_loop():
        """处理线程：取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态。"""
        last_sequence = -1
        stats_timer = time.perf_counter()

        window_closed = (lambda: window.closed) if window is not None else (lambda: False)

        try:
            while not stop.is_set() and not window_closed():
                frame = camera.get_latest_frame()
                if frame is None or frame.sequence == last_sequence:
                    if camera.error is not None:
                        raise RuntimeError(f"摄像头采集已停止：{camera.error}") from camera.error
                    time.sleep(0.001)
                    continue
                skipped_frames[0] += max(0, frame.sequence - (last_sequence if last_sequence >= 0 else 0) - 1)
                last_sequence = frame.sequence

                feedback = plc.feedback if plc is not None else None
                if plc is not None:
                    for message in plc.drain_messages():
                        print(f"[PLC] {message}")
                runtime.set_ai_feedback(feedback)
                process_start = time.perf_counter() if samples is not None else 0.0
                result = runtime.process_frame(frame)
                if samples is not None:
                    samples.add("capture_read_ms", frame.capture_read_ms)
                    samples.add("color_convert_ms", frame.color_convert_ms)
                    samples.add("post_read_wait_ms", max(0.0, (process_start - frame.timestamp) * 1000.0))
                    samples.add("vision_ms", (time.perf_counter() - process_start) * 1000.0)
                    samples.add("detect_ms", runtime.last_detect_ms)
                    samples.add("prediction_ms", runtime.last_predict_ms)
                    samples.add("ai_ms", runtime.last_ai_ms)
                    if frame.gst_pts_ns is not None:
                        gst_pts_count[0] += 1

                # PLC 只接收 AI 半场目标；轴反馈在本帧 AI 计算前更新。
                decision = result.ai_decision
                plc_request = None
                if (plc is not None and result.track_confirmed
                        and decision is not None and result.curling_state is not None):
                    plc_prepare_start = time.perf_counter() if samples is not None else 0.0
                    try:
                        plc_request = plc_adapter.build_target(decision, timestamp=result.frame.timestamp)
                        plc.set_target(plc_request.target_x, plc_request.target_y)
                    except ValueError:
                        plc.clear_target()
                        plc_request = None
                    if samples is not None:
                        samples.add("plc_prepare_submit_ms", (time.perf_counter() - plc_prepare_start) * 1000.0)
                elif plc is not None:
                    plc.clear_target()

                if recorder is not None:
                    recorder.record(result, ai_decision=decision, plc_request=plc_request)

                if not headless:
                    status_text = format_status(
                        result,
                        runtime.camera_geometry.enabled and runtime.camera_geometry.camera_matrix is not None,
                    )
                    if plc is not None:
                        if (not plc.connected or feedback is None or not feedback.valid
                                or time.monotonic() - feedback.stamp > 0.25):
                            status_text += " | PLC 未连接/反馈失效：禁止运动"
                        elif feedback.comm_lost_latch:
                            status_text += " | PLC 通信丢失锁存：请现场检查并复位"
                        elif feedback.kinematics_error or feedback.axis_fault or feedback.group_stop:
                            status_text += f" | PLC 轴报警 {feedback.kinematics_error} / 停止位 {feedback.group_stop}：禁止运动"
                        elif not feedback.axes_ready or not feedback.plc_echo_ok or feedback.comm_lost:
                            status_text += " | PLC 轴未就绪或心跳异常：禁止运动"
                        elif not result.track_confirmed:
                            status_text += " | 冰壶追踪丢失：PLC 目标已清除"
                        elif not plc.armed:
                            status_text += " | PLC 未授权：请现场按 E 使能"
                        else:
                            status_text += f" | PLC 实际位置 ({feedback.x:.0f},{feedback.y:.0f})"
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
                        stage = samples.snapshot() if samples is not None else {}
                        timing = " | ".join(
                            f"{name} p50/p95 {stage[name]['p50']:.2f}/{stage[name]['p95']:.2f} ms"
                            for name in ("capture_read_ms", "vision_ms", "prediction_ms", "post_read_wait_ms")
                            if name in stage
                        )
                        print(
                            f"[perf] FPS {result.fps:.1f} | 处理 {runtime.frame_ms:.1f} ms "
                            f"| 检测 {runtime.detect_ms:.1f} ms | 采集 {capture.current_fps:.1f} FPS "
                            f"| Tracker {track_state} | {timing}"
                        )
                    else:
                        print(
                            f"[stats] 处理 {result.fps:.1f} FPS | 单帧 {runtime.frame_ms:.1f} ms "
                            f"(检测 {runtime.detect_ms:.1f} ms) | 采集 {capture.current_fps:.1f} FPS"
                        )
        except Exception as exc:
            processing_failure[0] = exc
            traceback.print_exc()
            if preview_lock is not None and shared is not None:
                with preview_lock:
                    shared["fatal"] = f"处理线程异常退出：{exc!r}"
        finally:
            stop.set()

    def encoding_loop():
        """预览转换线程：最新 VisionResult -> render 标注 -> 缩放 -> PPM（按预览帧率限速）。"""
        seen = -1
        reference_brightness = None
        try:
            while not stop.wait(1.0 / args.preview_fps):
                with preview_lock:
                    result = shared["result"]
                    seq = shared["result_seq"]
                if result is None or seq == seen:
                    continue
                seen = seq

                encode_start = time.perf_counter() if samples is not None else 0.0
                blue, green, red = cv2.mean(result.frame.image[::8, ::8])[:3]
                brightness = 0.114 * blue + 0.587 * green + 0.299 * red
                if reference_brightness is None:
                    reference_brightness = brightness
                brightness_gain = max(0.75, min(1.33, reference_brightness / max(brightness, 1.0)))
                # ponytail: global preview correction cannot remove local light bands; use flicker-free lighting for those.
                reference_brightness += 0.005 * (brightness - reference_brightness)
                ppm_data = encode_preview_ppm(
                    result, runtime.table_roi, runtime.camera_geometry, brightness_gain
                )
                if samples is not None:
                    samples.add("preview_encode_ms", (time.perf_counter() - encode_start) * 1000.0)
                with preview_lock:
                    shared["ppm"] = ppm_data
                    shared["ppm_seq"] += 1
        except Exception as exc:
            processing_failure[0] = exc
            traceback.print_exc()
            with preview_lock:
                shared["fatal"] = f"预览线程异常退出：{exc!r}"
            stop.set()

    processing_thread = threading.Thread(target=processing_loop, name="processing", daemon=True)
    processing_thread.start()

    if not headless:
        threading.Thread(target=encoding_loop, name="encoder", daemon=True).start()
        window.start(preview_lock, shared)

    try:
        if headless:
            stop.wait(args.benchmark_seconds or None)
        else:
            window.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if plc is not None:
            plc.clear_target()
            if not plc.stop():
                print("[PLC] 仍在等待通信线程撤销使能；请留意实机，必要时使用硬件急停")
                while not plc.stop(timeout=1.0):
                    pass
            for message in plc.drain_messages():
                print(f"[PLC] {message}")
        camera.stop()
        processing_thread.join()
        if recorder is not None:
            recorded_path = recorder.close()
            if recorded_path is not None:
                print(f"运行数据已记录 {recorder.frame_count} 帧 -> {recorded_path}")
        if samples is not None:
            capture = camera.get_stats()
            stages = samples.snapshot()
            info = camera.info
            report = {
                "time_reference": "Host perf_counter: frame timestamp is after backend read, not sensor exposure; raw Gst PTS (when present) has a different clock domain",
                "sample_scope": "Stage distributions cover the latest processed frames only; capture FPS counts all reads. PLC write completion is not motor acknowledgement.",
                "camera": None if info is None else {
                    "device": info.device,
                    "backend": info.backend,
                    "width": info.width,
                    "height": info.height,
                    "requested_fps": info.requested_fps,
                    "negotiated_fps": info.negotiated_fps,
                    "source_format": info.source_format,
                    "output_format": info.output_format,
                },
                "capture": vars(capture),
                "processed_frames": runtime.frame_index,
                "skipped_sequence_frames": skipped_frames[0],
                "raw_gst_pts_samples": gst_pts_count[0],
                "sample_window": samples.window,
                "stages_ms": stages,
            }
            print(f"[perf] 采集 {capture.frame_count} 帧 | 处理 {runtime.frame_index} 帧 | 跳过 {skipped_frames[0]} 帧")
            if args.perf_json:
                output = Path(args.perf_json)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"性能报告已写入: {output}")
        if window is not None:
            window.close()

    if processing_failure[0] is not None:
        raise SystemExit(f"实时处理失败：{processing_failure[0]}")


def main():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")

    # 模式开关：默认运行视觉模式；传入 --sim / --game 运行仿真游戏
    parser.add_argument("--sim", "--game", dest="sim", action="store_true", help="启动冰壶仿真游戏（默认运行实时视觉演示）")
    parser.add_argument("--vision", action="store_true", help="（兼容保留）实时视觉演示模式")

    # 视觉模式相关参数
    parser.add_argument("--headless", action="store_true", help="无显示性能测试模式")
    parser.add_argument("--benchmark-seconds", type=float, default=0.0, metavar="SEC", help="headless 模式自动停止时间；默认持续运行")
    parser.add_argument("--perf-json", default=None, metavar="PATH", help="退出时保存最近 4096 样本的阶段耗时分布")
    parser.add_argument("--record", default=None, metavar="PATH", help="记录运行数据到 JSON 文件（CurlingState/PredictionState/timestamp/FPS）")
    parser.add_argument("--plc", default=None, metavar="IP", help="连接已配置学弟版 DB1/DB18 的 S7-1500T；轴需现场主动使能")
    parser.add_argument("--plc-rate", type=float, default=30.0, help="PLC 通信周期频率（Hz，默认 30）")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--camera-device", default=None, help="Linux 相机路径（如 /dev/video2）；Windows 摄像头编号（默认 0）")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机内参标定文件")
    parser.add_argument("--table-calibration", default="calibration/table_homography.npz", help="球台四点 Homography 标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正（必须同时 --disable-homography）")
    parser.add_argument("--disable-homography", action="store_true", help="关闭 Homography，回退到旧 ROI 线性映射（调试用）")
    parser.add_argument("--roi", type=int, nargs=4, default=(4, 10, 1216, 710), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(10, 255, 255), metavar=("C1", "C2", "C3"))

    args = parser.parse_args()

    if args.plc and (not math.isfinite(args.plc_rate) or args.plc_rate <= 0.0):
        parser.error("--plc-rate 必须为有限正数")

    if args.plc and not args.sim and (args.disable_undistort or args.disable_homography):
        parser.error("实机 --plc 必须启用相机内参及球台 Homography 标定")

    if not math.isfinite(args.benchmark_seconds) or args.benchmark_seconds < 0 or (args.benchmark_seconds and not args.headless):
        parser.error("--benchmark-seconds 必须为非负数，且只适用于 --headless")

    if args.sim and (args.benchmark_seconds or args.perf_json):
        parser.error("--benchmark-seconds 和 --perf-json 只适用于视觉模式")

    if not args.sim and args.disable_undistort and not args.disable_homography:
        parser.error(
            "--disable-undistort 必须与 --disable-homography 一起使用："
            "Homography 基于去畸变坐标标定，二者不能只关一半"
        )

    if args.sim:
        run_game(args.plc, args.plc_rate)
    else:
        run_vision(args)


if __name__ == "__main__":
    main()
