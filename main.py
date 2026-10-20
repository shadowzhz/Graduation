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

# ★★★ Windows 控制台默认是 GBK(936)，打印一个它编不出的字符（² ³ [!] OK …）
#     就会抛 UnicodeEncodeError 把整个程序打断 —— 现场踩过：
#         UnicodeEncodeError: 'gbk' codec can't encode character '\xb2'
#     这里把 stdout/stderr 的编码错误策略改成 replace：编不出的字符显示成 "?"，
#     但**永远不会再因为一句日志而崩**。改任何提示文案都不用再担心这件事。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  # 老解释器 / 被重定向到不支持的对象：忽略，不影响运行

import cv2
import tkinter as tk

from air_hockey.app.renderer import format_status, render_preview
from air_hockey.app.vision_runtime import MAX_DETECTION_GAP, VisionRuntime
from air_hockey.app.rally import (
    DEFAULT_DEFENSE_FORWARD_LIMIT,
    READY_POSITION,
    RallyController,
    RallyState,
)
from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.control import PLCLink, PlcControlAdapter
from air_hockey.control.adapter import PlcTarget
from air_hockey.control.plc import TARGET_MAX_AGE
from air_hockey.control import plc as plc_module
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


def print_latency_report(decision_ages, plc):
    """[延迟] 分段延迟汇总：视觉段 + PC→PLC→轴 段（见 air_hockey/control/latency.py）。"""
    ages = sorted(decision_ages)
    decision_ages.clear()
    if ages:
        vision = (f"视觉 帧读出→目标交给PLC线程 p50 {ages[len(ages) // 2]:.1f}ms "
                  f"p95 {ages[min(len(ages) - 1, int(len(ages) * 0.95))]:.1f}ms")
    else:
        vision = "视觉 本窗口无目标"
    if plc is None:
        print(f"[延迟] {vision}")
        return
    text, hints = plc.latency_report()
    print(f"[延迟] {vision} | {text}")
    for hint in hints:
        print(f"[延迟] [!] {hint}")


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
        """轴命令由现场操作员主动触发；连接成功不自动使能。

        ★ 单轴按钮：某个轴报 801 / 撞限位之后可以**只把那个轴**撤下来，
          另一个继续用；也能单独回零 / 单独复位。
        ★ 单轴使能**不会**让程序开始追球 —— 运动需要 X、Y 同时使能
          （PLCLink._motion_safe 要求 x_en and y_en）。
        [!] 回零会让该轴动（去找参考点），是刻意的人工动作。
        """
        self._plc = link
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        self._axis_buttons = {}
        for axis, name in (("x", "X"), ("y", "Y")):
            button = tk.Button(
                controls, text=f"{name} 轴使能 ( {name.lower()} )",
                command=lambda a=axis: self._toggle_axis(a))
            button.pack(side="left", expand=True)
            self._axis_buttons[axis] = button
            tk.Button(controls, text=f"{name} 回零 ( {name} )",
                      command=lambda a=axis: self._home_axis(a)).pack(side="left", expand=True)
            tk.Button(controls, text=f"{name} 复位",
                      command=lambda a=axis: self._reset_axis(a)).pack(side="left", expand=True)
        second = tk.Frame(self.root)
        second.pack(fill="x")
        for label, command in (("E 双轴使能", link.enable_axes),
                               ("H 双轴回零", link.home_axes),
                               ("C 全部复位", link.axes_reset)):
            tk.Button(second, text=label, command=command).pack(side="left", expand=True)
        for key, command in (("e", link.enable_axes), ("h", link.home_axes),
                             ("c", link.axes_reset)):
            self.root.bind(key, lambda _event, callback=command: callback())
        # ★ 大小写区分：小写 = 单轴使能/撤销，大写(Shift) = 单轴回零
        self.root.bind("x", lambda _event: self._toggle_axis("x"))
        self.root.bind("y", lambda _event: self._toggle_axis("y"))
        self.root.bind("X", lambda _event: self._home_axis("x"))
        self.root.bind("Y", lambda _event: self._home_axis("y"))

    def _toggle_axis(self, axis):
        """单轴使能 / 撤销。切换后立刻刷新按钮文字。"""
        link = getattr(self, "_plc", None)
        if link is None:
            return
        sent, wanted = link.toggle_axis_enable(axis)
        if not sent:
            return
        button = (self._axis_buttons or {}).get(axis)
        if button is not None:
            name = axis.upper()
            button.config(text=(f"{name} 轴撤销 ( {axis} )" if wanted
                                else f"{name} 轴使能 ( {axis} )"))
        if not wanted:
            # 撤下一个轴 → 程序立刻停止发运动（_motion_safe 需要两个都使能）
            print(f"[PLC] {axis.upper()} 轴已撤销使能 —— 在它重新使能前不会下发运动")

    def _home_axis(self, axis):
        link = getattr(self, "_plc", None)
        if link is not None:
            print(f"[PLC] {axis.upper()} 轴回零 —— 该轴会移动，注意手别在行程里")
            link.home_axis(axis)

    def _reset_axis(self, axis):
        link = getattr(self, "_plc", None)
        if link is not None:
            link.reset_axis(axis)

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


def submit_rally_target(result, adapter, plc, *, dry_run=False):
    """直接发布防守/击球意图，PLCLink 仍负责人工授权和实机互锁。"""
    rally = result.rally
    if (rally is None or rally.target is None or
            (rally.state in (RallyState.DEFENDING, RallyState.ATTACKING) and
             not rally.held_target and
             (not result.track_confirmed or result.ai_decision is None or result.curling_state is None))):
        if plc is not None:
            plc.clear_target()
        return None
    try:
        request = adapter.build_target(PlcTarget(*rally.target, result.frame.timestamp),
                                       timestamp=result.frame.timestamp)
    except ValueError:
        if plc is not None:
            plc.clear_target()
        raise
    if plc is not None:
        if dry_run:
            plc.clear_target()
        elif not plc.set_target(request.target_x, request.target_y):
            plc.clear_target()
            return None
    return request


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
        # 球槌速度估计，用来做"提前量"（补偿轴的反应延迟）。
        # 0 或负数 = 不做提前量，等于原来"直接瞄冰壶当前位置"的行为。
        mallet_speed=args.mallet_speed if args.mallet_speed > 0 else None,
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

    camera = CameraManager(CameraConfig(
        device=args.camera_device,
        # --camera-fast 才会在 open 阶段真的把格式协商成 MJPG。
        # 不加这个开关时行为与原来完全一致（yu yv422 10 fps）。
        fourcc="MJPG" if args.camera_fast else None,
    ))
    try:
        camera.start()
        if args.record:
            recorder = RuntimeRecorder(
                args.record,
                source="runtime",
                meta={"mode": "headless" if headless else "display", "control": "direct_play",
                      "plc_mode": "dry_run" if args.dry_run else "live" if args.plc else "no_plc"},
            )
    except Exception as exc:
        try:
            camera.stop()
        finally:
            if window is not None:
                window.close()
        raise SystemExit(f"实时运行启动失败：{exc}") from exc

    plc = None
    plc_adapter = PlcControlAdapter()
    rally = RallyController(defense_forward_limit=args.defense_forward_limit)
    if args.plc:
        # ★ 命令余量必须在建 PLCLink 之前设好 —— 夹取用的 COMMAND_* 是
        #   模块常量，_prepare_move 每次下发时直接读它。
        if args.motion_margin_mm is not None:
            plc_module.set_motion_margin(args.motion_margin_mm)
        # ★ TIA 的动力学预设值（改了 TIA 就用这三个参数同步，否则自检是拿旧数字算的）
        plc_module.set_axis_dynamics(accel_mm_s2=args.tia_accel_mm_s2,
                                    decel_mm_s2=args.tia_decel_mm_s2,
                                    speed_mm_s=args.tia_speed_mm_s)
        print(f"[PLC] 命令余量 {plc_module.MOTION_MARGIN_MM:.0f}mm → "
              f"X {plc_module.COMMAND_X_MIN:.0f}..{plc_module.COMMAND_X_MAX:.0f} "
              f"Y {plc_module.COMMAND_Y_MIN:.0f}..{plc_module.COMMAND_Y_MAX:.0f}"
              f"（机械行程 {plc_module.TRAVEL_X_MIN:.0f}..{plc_module.TRAVEL_X_MAX:.0f} / "
              f"{plc_module.TRAVEL_Y_MIN:.0f}..{plc_module.TRAVEL_Y_MAX:.0f}，两端为限位开关）")
        plc = PLCLink(args.plc, period=1.0 / args.plc_rate,
                      trigger_low_time=args.trigger_low_ms / 1000.0,
                      blend_threshold_mm=args.blend_threshold_mm,
                      target_slew_mm=args.target_slew_mm,
                      target_accel_mm_s2=args.target_accel_mm_s2,
                      sequence_trigger=(args.trigger_mode == "sequence"))
        # ★ 把轴的**实际可达范围**交给 AI：它据此限制"可覆盖范围"
        #   （recovery_bounds），免得老是瞄到轴到不了的边角。
        runtime.ai.recovery_bounds = plc.reachable_bounds
        # 回收按相同命令范围判断能否接触；不可达时清目标，不能直穿冰壶回位。
        rally.reachable_bounds = plc.reachable_bounds
        # ★★ 启动就自检"余量够不够刹得住"：目标斜坡自己会提前减速停在包络内，
        #    冲出包络的只可能是轴按 TIA 减速度刹车的那一段，必须小于包络到限位开关的余量。
        #    按 **--plc-rate** 估算；名义值和实测可能差很多，所以只作提示。
        if args.target_slew_mm and args.target_slew_mm > 0.0:
            brake, speed = plc_module.braking_distance_mm(args.target_slew_mm, args.plc_rate)
            print(f"[PLC] 目标限速 {args.target_slew_mm:g}mm/周期 × {args.plc_rate:g}Hz "
                  f"→ 指令速度约 {speed:.0f}mm/s，按 TIA 减速度 "
                  f"{plc_module.AXIS_DECEL_LIMIT_MM_S2:.0f}mm/s^2 估算超调 {brake:.1f}mm")
            for warning in plc_module.safety_warnings(args.target_slew_mm, args.plc_rate):
                print(f"[PLC] [!] 安全自检：{warning}")
        print("[PLC] 行程保护：命令夹在机械行程内缩余量之内；越界由硬件限位开关负责")
        plc.start()
        if window is not None and not args.dry_run:
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
    print(f"视觉管线：最新帧 + 每 {runtime.detection_interval} 帧检测；"
          f"距上次检测超过 {MAX_DETECTION_GAP * 1000:g} ms 时在下一可用帧补检测，其余帧使用 tracker 预测")
    print(f"Direct play: READY={READY_POSITION}; 检测到就防守/击球，不等二次确认；"
          f"{'DRY-RUN：仅观察，不发送运动/使能/回零/复位' if args.dry_run else '轴运动仍需人工 E 授权'}")
    print("机器半场主动击球/回收，真人半场准备防守；无 PLC 时球槌为 software fallback。")
    print("视觉校正:")
    print(f"  camera calibration: {args.calibration}")
    print(f"  undistort: {'OFF' if args.disable_undistort else 'ON'}")
    print(f"  table mapping: {'linear ROI' if args.disable_homography else 'Homography'}")
    print(f"  table calibration: {table_calibration_file if table_calibration_file is not None else 'None (linear ROI)'}")
    if recorder is not None:
        print(f"运行数据记录: {args.record}")

    stop = threading.Event()
    skipped_frames = [0]
    # 延迟排查：帧读出（frame.timestamp）到目标交给 PLC 线程的耗时（ms）
    decision_ages = deque(maxlen=4096)
    gst_pts_count = [0]

    def processing_loop():
        """处理线程：取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态。"""
        last_sequence = -1
        last_gate = None
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
                result.rally = rally.update(result, now=time.perf_counter(), feedback=feedback,
                                            mallet_position=runtime.ai_current_pos)
                plc_prepare_start = time.perf_counter() if samples is not None else 0.0
                plc_request = submit_rally_target(result, plc_adapter, plc, dry_run=args.dry_run)
                if plc_request is not None:
                    decision_ages.append((time.perf_counter() - frame.timestamp) * 1000.0)
                if samples is not None:
                    samples.add("plc_prepare_submit_ms", (time.perf_counter() - plc_prepare_start) * 1000.0)
                if recorder is not None:
                    recorder.record(result, ai_decision=decision, plc_request=plc_request,
                                    plc_state=None if plc is None else plc.motion_snapshot())

                status_text = format_status(
                    result,
                    runtime.camera_geometry.enabled and runtime.camera_geometry.camera_matrix is not None,
                )
                if args.dry_run:
                    status_text += " | DRY-RUN: motion blocked"
                plc_gate = None
                if plc is not None:
                    safe = (feedback is not None and PLCLink._motion_safe(feedback)
                            and 0 <= time.monotonic() - feedback.stamp <= TARGET_MAX_AGE
                            and plc.connected)
                    armed = plc.armed
                    plc_gate = armed, safe
                    status_text += f" | PLC armed={armed} safe={safe}"
                    if (not plc.connected or feedback is None or not feedback.valid
                            or time.monotonic() - feedback.stamp > TARGET_MAX_AGE):
                        status_text += " | PLC 未连接/反馈失效：禁止运动"
                    elif feedback.comm_lost_latch:
                        status_text += " | PLC 通信丢失锁存：请现场检查并复位"
                    elif feedback.kinematics_error or feedback.axis_fault or feedback.group_stop:
                        # ★ 三个来源的错误码全部打出来，不要再靠猜：
                        #   轴自己的码（DB18.2 / DB18.6）
                        #   运动指令 MC_MoveLinearAbsolute 的码（DB18.58.3 / DB18.60）
                        #   运动学的 ErrorDetail.Number（DB18.56，就是那个 801）
                        which = ("X" if feedback.x_err else "") + ("Y" if feedback.y_err else "")
                        parts = [f"轴故障位={which or '无'}", feedback.fault_detail()]
                        if feedback.group_stop:
                            parts.append("停止位=1")
                        status_text += " | PLC " + " / ".join(parts) + "：禁止运动"
                    elif not feedback.axes_ready or not feedback.plc_echo_ok or feedback.comm_lost:
                        status_text += " | PLC 轴未就绪或心跳异常：禁止运动"
                    elif not armed:
                        status_text += " | PLC 未授权：请现场按 E 使能"
                    else:
                        status_text += f" | PLC 实际位置 ({feedback.x:.0f},{feedback.y:.0f})"
                track = result.track
                track_key = None if track is None else (track.track_id, track.state)
                gate = result.rally.state, result.rally.reason, track_key, plc_gate
                if gate != last_gate:
                    track_text = "none" if track is None else f"{track.track_id}/{track.state.value}"
                    position = None if result.curling_state is None else tuple(round(v, 1) for v in result.curling_state.position)
                    prefix = "; ".join(result.rally.transitions) or "gate changed"
                    print(f"[rally] {prefix} | {status_text} | track={track_text} "
                          f"hard_hit={result.detection is not None} puck={position} "
                          f"frame_age_ms={(time.perf_counter() - frame.timestamp) * 1000:.1f} "
                          f"plc_busy={None if feedback is None else feedback.busy}")
                    last_gate = gate
                if not headless:
                    with preview_lock:
                        shared["result"] = result
                        shared["result_seq"] = result.frame.sequence
                        shared["status"] = status_text + "    Q / ESC 退出"

                stats_interval = 1.0 if headless else STATS_INTERVAL
                if time.perf_counter() - stats_timer >= stats_interval:
                    stats_timer = time.perf_counter()
                    print(f"[rally] {status_text}")
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
                        plc_line = f" | [plc] {plc.diagnostics()}" if plc is not None else ""
                        print(
                            f"[stats] 处理 {result.fps:.1f} FPS | 单帧 {runtime.frame_ms:.1f} ms "
                            f"(检测 {runtime.detect_ms:.1f} ms) | 采集 {capture.current_fps:.1f} FPS"
                            f"{plc_line}"
                        )
                    print_latency_report(decision_ages, plc)
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
    parser.add_argument("--dry-run", action="store_true", help="完整回合影子运行：显示/记录请求目标，不发送 PLC 运动或现场轴命令")
    parser.add_argument("--plc", default=None, metavar="IP", help="连接已配置学弟版 DB1/DB18 的 S7-1500T；轴需现场主动使能")
    parser.add_argument("--plc-rate", type=float, default=30.0, help="PLC 通信周期频率（Hz，默认 30）")
    parser.add_argument("--trigger-low-ms", type=float, default=10.0, metavar="MS",
                        help="运动触发高、低电平均保持的最小脉宽（毫秒，默认 10）。"
                             "实际通信频率受 S7 往返和两相脉宽限制；"
                             "未测 PLC 扫描周期不要调小，否则 PLC 可能漏掉上升沿")
    parser.add_argument("--mallet-speed", type=float, default=500.0, metavar="UNITS_PER_SEC",
                        help="球槌可达速度估计（球台单位/秒，默认 500）。用来算提前量："
                             "目标瞄冰壶再走 lead_time 之后的位置，而不是它现在的位置 —— "
                             "不然轴总是慢一拍、球走了才追到。设 0 关闭提前量。"
                             "这不是 PLC 轴速度设置，需要在现场实测后调准")
    parser.add_argument("--defense-forward-limit", type=float,
                        default=DEFAULT_DEFENSE_FORWARD_LIMIT, metavar="UNITS",
                        help="冰壶在机器半场还在运动时，机械臂最多离开防守线往前多少"
                             "（球台单位，1 单位≈1.317mm）；冰壶停住（死球）时不受此限。"
                             "**默认不限制**（= ai.py 自身的上限，现场选的是「全部追球」）。"
                             "想保守再显式给值：0 = 钉死在防守线（Y 完全不动），"
                             "50 ≈ 往前 66mm，100 ≈ 往前 132mm（数值越大越主动）")
    parser.add_argument("--trigger-mode", choices=("pulse", "sequence"), default="sequence",
                        help="运动触发方式，默认 sequence（序号握手）。"
                             "sequence = 写 DB1.DBD142 序号，PLC 比较后执行，"
                             "**不需要脉宽** → 不受 OB1 扫描周期限制（实测扫描最大 31.7ms，"
                             "脉冲模式会静默丢指令）。"
                             "pulse = 抬 DB1.56.0 的 P 沿，脉宽必须跨过扫描周期，"
                             "周期被锁在约 28ms（35Hz），快球时只能中止重开（一顿一顿）。"
                             "[!] 用 sequence 之前 PLC 里必须有序号比较逻辑（现场已加）")
    parser.add_argument("--blend-threshold-mm", type=float, default=12.0, metavar="MM",
                        help="相邻两条目标之间步距超过它就改用 BufferMode=1（附加停稳），"
                             "否则用 5（混合）。默认 12，与 --target-slew-mm 默认值相同 —— "
                             "步距不超阈值时保持混合。"
                             "★ 太小会频繁附加停稳；不修改 PLC 的实际速度或加速度")
    parser.add_argument("--target-slew-mm", type=float, default=12.0, metavar="MM",
                        help="目标限速（mm/周期），默认 12。"
                             "★★★ 这个治的是「顿」：rally 状态机切换时目标会**瞬间跳变**"
                             "（现场实测 p95 步距 192.8mm），大步会使用 "
                             "BufferMode=1，完成当前运动并停稳后再执行下一条。"
                             "★★★ 但它决定**指令速度**（= slew × 实测周期），"
                             "所以直接决定刹不刹得住：超调 = v^2/(2×TIA 减速度) 必须小于 "
                             "--motion-margin-mm，否则全速冲向边缘时可能碰到限位开关。"
                             "默认 12 @30Hz = 360mm/s，超调约 2mm，配默认余量 5mm 是安全的；"
                             "调大 slew 时看启动自检，必要时同步加大余量")
    parser.add_argument("--motion-margin-mm", type=float, default=5.0, metavar="MM",
                        help="命令余量：下发目标离机械行程端点（硬件限位开关）再内缩多少毫米，"
                             "默认 5。机械行程 X -192..224 / Y -129..187，"
                             "包络 = 行程内缩它。程序不另设软件行程保护；"
                             "★ 它应大于启动自检打印的超调距离，否则全速冲向边缘时可能碰到限位开关")
    parser.add_argument("--target-accel-mm-s2", type=float, default=12000.0, metavar="MM_S2",
                        help="目标加速度限幅（mm/s^2），默认 12000；0 = 关闭。"
                             "★★★ 治的是**跟随误差 801**：只限步长（--target-slew-mm）"
                             "管不住「目标突然反向」——限速器只看步长大小、不看方向。"
                             "★★ 必须**低于 TIA「运动机构运动 → 加速度」的预设值**"
                             "（现在是 15000），否则等于没限。"
                             "现场 TIA 工艺对象 Y轴「状态位和错误位」实测："
                             "错误=跟随误差(红)、警告=动态限值(黄)、已归位=绿、驱动装置=灰，"
                             "快照 physical Y=110.49（行程正中间、prev_busy=True）——"
                             "**轴在正常位置自由运动途中跟不上指令，不是撞限位**。"
                             "算下来 40mm/周期 @65Hz 需要约 169,000 mm/s^2，"
                             "而轴的能力只有 106N/3kg ≈ 35,000 mm/s^2，差约 5 倍。"
                             "限加速度后急转时目标自己拐大弯、直线时仍可全速。"
                             "★ 调大更跟手但更容易报跟随误差；调小更稳但急转变慢")
    parser.add_argument("--tia-decel-mm-s2", type=float, default=None, metavar="MM_S2",
                        help="同步 TIA「运动机构运动 → 减速度」的预设值（默认按 35000）。"
                             "★★★ 它决定**制动距离** = v^2/(2x减速度)，也就决定"
                             "「这个 slew 会不会撞限位」。默认值是按现场已改好的 35000；"
                             "**如果你其实没改减速度，就加 --tia-decel-mm-s2 15000** —— "
                             "否则自检会说安全而实际会撞")
    parser.add_argument("--tia-accel-mm-s2", type=float, default=None, metavar="MM_S2",
                        help="同步 TIA「运动机构运动 → 加速度」的预设值（默认按 35000）。"
                             "用来判断 --target-accel-mm-s2 有没有超过机器能力")
    parser.add_argument("--tia-speed-mm-s", type=float, default=None, metavar="MM_S",
                        help="同步 TIA「运动机构运动 → 速度」的预设值（默认按 3750）。"
                             "用来判断指令速度会不会被 TO 截断")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--camera-device", default=None, help="Linux 相机路径（如 /dev/video2）；Windows 摄像头编号（默认 0）")
    parser.add_argument("--camera-fast", action="store_true",
                        help="在打开设备时就把像素格式协商成 MJPG。这台相机不加该开关会停在 yuyv422 的 10 fps，"
                             "加了约为 100 FPS；但 MJPG 模式的视野/清晰度需现场核对，故默认关闭")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机内参标定文件")
    parser.add_argument("--table-calibration", default="calibration/table_homography.npz", help="球台四点 Homography 标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正（必须同时 --disable-homography）")
    parser.add_argument("--disable-homography", action="store_true", help="关闭 Homography，回退到旧 ROI 线性映射（调试用）")
    parser.add_argument("--roi", type=int, nargs=4, default=(4, 10, 1216, 710), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(10, 255, 255), metavar=("C1", "C2", "C3"))

    args = parser.parse_args()

    if args.sim and args.dry_run:
        parser.error("--dry-run 只适用于实时视觉回合；仿真不接受该安全开关")

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
