"""实时视觉回合：相机 -> VisionRuntime -> RallyController -> PLCLink。

线程分工：
    主线程    Tk mainloop + 显示定时器（GUI 模式）/ 等待退出（headless 模式）
    处理线程  取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态（不限速）
    编码线程  最新 VisionResult -> render 标注 -> 缩放 -> PPM（按预览帧率限速，仅 GUI 模式）
"""

import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from ..camera import CameraConfig, CameraManager
from ..control import MotionLimits, PLCLink, PlcControlAdapter
from ..control.adapter import PlcTarget
from ..control.limits import TRAVEL_X_MAX, TRAVEL_X_MIN, TRAVEL_Y_MAX, TRAVEL_Y_MIN
from ..control.link import TARGET_MAX_AGE
from ..recording import RuntimeRecorder
from .perf import LatencySamples, build_perf_report, print_latency_report
from .preview import PreviewBrightness, VisionWindow, encode_preview_ppm
from .rally import READY_POSITION, RallyController, RallyState
from .renderer import format_status
from .vision_runtime import MAX_DETECTION_GAP, VisionRuntime

STATS_INTERVAL = 5.0        # 统计间隔


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


def motion_limits_from_args(args):
    """--motion-margin-mm 与 TIA 动力学预设值（--tia-*，未给的按默认值）。"""
    overrides = {
        "margin_mm": args.motion_margin_mm,
        # ★ TIA 的动力学预设值（改了 TIA 就用这三个参数同步，否则自检是拿旧数字算的）
        "accel_mm_s2": args.tia_accel_mm_s2,
        "decel_mm_s2": args.tia_decel_mm_s2,
        "speed_mm_s": args.tia_speed_mm_s,
    }
    return MotionLimits(**{name: value for name, value in overrides.items() if value is not None})


def start_plc(args, runtime, rally):
    """按命令行参数配置并启动 PLCLink，打印行程包络与制动自检。"""
    limits = motion_limits_from_args(args)
    print(f"[PLC] 命令余量 {limits.margin_mm:.0f}mm → "
          f"X {limits.command_x_min:.0f}..{limits.command_x_max:.0f} "
          f"Y {limits.command_y_min:.0f}..{limits.command_y_max:.0f}"
          f"（机械行程 {TRAVEL_X_MIN:.0f}..{TRAVEL_X_MAX:.0f} / "
          f"{TRAVEL_Y_MIN:.0f}..{TRAVEL_Y_MAX:.0f}，两端为限位开关）")
    plc = PLCLink(args.plc, period=1.0 / args.plc_rate,
                  blend_threshold_mm=args.blend_threshold_mm,
                  target_slew_mm=args.target_slew_mm,
                  target_accel_mm_s2=args.target_accel_mm_s2,
                  limits=limits)
    # ★ 把轴的**实际可达范围**交给 AI：它据此限制"可覆盖范围"
    #   （recovery_bounds），免得老是瞄到轴到不了的边角。
    runtime.ai.recovery_bounds = plc.reachable_bounds
    # 回收按相同命令范围判断能否接触；不可达时清目标，不能直穿冰壶回位。
    rally.reachable_bounds = plc.reachable_bounds
    # ★★ 启动就自检"余量够不够刹得住"：目标斜坡自己会提前减速停在包络内，
    #    冲出包络的只可能是轴按 TIA 减速度刹车的那一段，必须小于包络到限位开关的余量。
    #    按 **--plc-rate** 估算；名义值和实测可能差很多，所以只作提示。
    if not args.target_slew_mm or args.target_slew_mm <= 0.0:
        print("[PLC] 终点模式：只下发 AI 终点；速度/加速度由 TIA 运动机构设置决定")
    else:
        brake, speed = limits.braking_distance_mm(args.target_slew_mm, args.plc_rate)
        print(f"[PLC] 目标限速 {args.target_slew_mm:g}mm/周期 × {args.plc_rate:g}Hz "
              f"→ 指令速度约 {speed:.0f}mm/s，按 TIA 减速度 "
              f"{limits.decel_mm_s2:.0f}mm/s^2 估算超调 {brake:.1f}mm")
        for warning in limits.safety_warnings(args.target_slew_mm, args.plc_rate):
            print(f"[PLC] [!] 安全自检：{warning}")
    print("[PLC] 行程保护：命令夹在机械行程内缩余量之内；越界由硬件限位开关负责")
    plc.start()
    return plc


def plc_status_text(plc, feedback, now):
    """状态栏的 PLC 部分；返回 (文字, (armed, safe))。"""
    safe = (feedback is not None and PLCLink._motion_safe(feedback)
            and 0 <= now - feedback.stamp <= TARGET_MAX_AGE
            and plc.connected)
    armed = plc.armed
    text = f" | PLC armed={armed} safe={safe}"
    if (not plc.connected or feedback is None or not feedback.valid
            or now - feedback.stamp > TARGET_MAX_AGE):
        text += " | PLC 未连接/反馈失效：禁止运动"
    elif feedback.comm_lost_latch:
        text += " | PLC 通信丢失锁存：请现场检查并复位"
    elif feedback.kinematics_error or feedback.axis_fault or feedback.group_stop:
        # ★ 三个来源的错误码全部打出来，不要再靠猜：
        #   轴自己的码（DB18.2 / DB18.6）
        #   运动指令 MC_MoveLinearAbsolute 的码（DB18.58.3 / DB18.60）
        #   运动学的 ErrorDetail.Number（DB18.56，就是那个 801）
        which = ("X" if feedback.x_err else "") + ("Y" if feedback.y_err else "")
        parts = [f"轴故障位={which or '无'}", feedback.fault_detail()]
        if feedback.group_stop:
            parts.append("停止位=1")
        text += " | PLC " + " / ".join(parts) + "：禁止运动"
    elif not feedback.axes_ready or not feedback.plc_echo_ok or feedback.comm_lost:
        text += " | PLC 轴未就绪或心跳异常：禁止运动"
    elif not armed:
        text += " | PLC 未授权：请现场按 E 使能"
    else:
        text += f" | PLC 实际位置 ({feedback.x:.0f},{feedback.y:.0f})"
    return text, (armed, safe)


def run_vision(args):
    LiveSession(args).run()


class LiveSession:
    """一次实时运行：构造时打开相机 / 记录器 / PLC，run() 启动线程直到退出并收尾。"""

    def __init__(self, args):
        self.args = args
        self.headless = args.headless
        self.window = None
        self.preview_lock = None
        self.shared = None
        self.samples = LatencySamples() if self.headless or args.perf_json else None
        self.failure = None
        self.recorder = None
        self.plc = None

        self.stop = threading.Event()
        self.skipped_frames = 0
        # 延迟排查：帧读出（frame.timestamp）到目标交给 PLC 线程的耗时（ms）
        self.decision_ages = deque(maxlen=4096)
        self.gst_pts_count = 0

        self.table_calibration_file = None if args.disable_homography else args.table_calibration
        if self.table_calibration_file is not None and not Path(self.table_calibration_file).is_file():
            raise SystemExit(
                f"缺少球台标定文件: {self.table_calibration_file}。先运行 "
                "python3 air_hockey/tools/calibrate_table.py 完成实台标定；"
                "仅调试线性映射时使用 --disable-homography。"
            )
        self.runtime = VisionRuntime(
            table_roi=tuple(args.roi),
            lower=tuple(args.lower),
            upper=tuple(args.upper),
            calibration_file=args.calibration,
            table_calibration_file=self.table_calibration_file,
            disable_undistort=args.disable_undistort,
            # 球槌速度估计，用来做"提前量"（补偿轴的反应延迟）。
            # 0 或负数 = 不做提前量，等于原来"直接瞄冰壶当前位置"的行为。
            mallet_speed=args.mallet_speed if args.mallet_speed > 0 else None,
            profile=self.samples is not None,
        )

        if not self.headless:
            self.window = VisionWindow(dry_run=args.dry_run)
            self.preview_lock = threading.Lock()
            self.shared = {
                "result": None,
                "result_seq": -1,
                "ppm": None,
                "ppm_seq": 0,
                "plc": None,
                "fatal": None,
            }

        self.camera = CameraManager(CameraConfig(
            device=args.camera_device,
            # 默认在 open 阶段就把格式协商成 MJPG（Windows 上不这样做会停在 YUY2 10 FPS）。
            # 2026-10-20 现场：漏了这个开关时采集只有 9.9 FPS，两种格式下冰壶像素半径相同（约 31px），
            # 视野一致，所以改为默认开启；--camera-slow 退回旧行为。
            fourcc=None if args.camera_slow else "MJPG",
        ))
        try:
            self.camera.start()
            if args.record:
                self.recorder = RuntimeRecorder(
                    args.record,
                    source="runtime",
                    meta={"mode": "headless" if self.headless else "display", "control": "direct_play",
                          "plc_mode": "dry_run" if args.dry_run else "live" if args.plc else "no_plc"},
                )
        except Exception as exc:
            try:
                self.camera.stop()
            finally:
                if self.window is not None:
                    self.window.close()
            raise SystemExit(f"实时运行启动失败：{exc}") from exc

        self.plc_adapter = PlcControlAdapter()
        self.rally = RallyController(defense_forward_limit=args.defense_forward_limit)
        if args.plc:
            self.plc = start_plc(args, self.runtime, self.rally)
            if self.window is not None and not args.dry_run:
                self.window.attach_plc(self.plc)
            print(f"PLC 状态读取与运动控制已启动：{args.plc} @ {args.plc_rate:g} Hz；轴需现场主动使能")

        self._print_banner()

    def _print_banner(self):
        args = self.args
        mode_label = "headless 性能测试" if self.headless else "实时视觉演示"
        print(f"{mode_label}开始{'，Ctrl+C 退出' if self.headless else '，Q / ESC 或关闭窗口退出'}")
        if self.camera.info is not None:
            info = self.camera.info
            print(
                f"{info.backend}: {info.device} | 协商 {info.width}x{info.height} "
                f"@ {info.negotiated_fps:g} FPS ({info.source_format} -> {info.output_format}); "
                "实际采集速度以帧计数为准"
            )
        print(f"视觉管线：最新帧 + 每 {self.runtime.detection_interval} 帧检测；"
              f"距上次检测超过 {MAX_DETECTION_GAP * 1000:g} ms 时在下一可用帧补检测，其余帧使用 tracker 预测")
        print(f"Direct play: READY={READY_POSITION}; 检测到就防守/击球，不等二次确认；"
              f"{'DRY-RUN：仅观察，不发送运动/使能/回零/复位' if args.dry_run else '轴运动仍需人工 E 授权'}")
        print("机器半场主动击球/回收，真人半场准备防守；无 PLC 时球槌为 software fallback。")
        print("视觉校正:")
        print(f"  camera calibration: {args.calibration}")
        print(f"  undistort: {'OFF' if args.disable_undistort else 'ON'}")
        print(f"  table mapping: {'linear ROI' if args.disable_homography else 'Homography'}")
        print(f"  table calibration: {self.table_calibration_file if self.table_calibration_file is not None else 'None (linear ROI)'}")
        if self.recorder is not None:
            print(f"运行数据记录: {args.record}")

    def run(self):
        processing_thread = threading.Thread(target=self._processing_loop, name="processing", daemon=True)
        processing_thread.start()

        if not self.headless:
            threading.Thread(target=self._encoding_loop, name="encoder", daemon=True).start()
            self.window.start(self.preview_lock, self.shared)

        try:
            if self.headless:
                self.stop.wait(self.args.benchmark_seconds or None)
            else:
                self.window.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            self._shutdown(processing_thread)

        if self.failure is not None:
            raise SystemExit(f"实时处理失败：{self.failure}")

    # ------------------------------------------------------------------
    # 处理线程
    # ------------------------------------------------------------------
    def _processing_loop(self):
        """处理线程：取最新帧 -> VisionRuntime 处理 -> 保存最新 VisionResult 与状态。"""
        last_sequence = -1
        self._last_gate = None
        self._stats_timer = time.perf_counter()

        window = self.window
        window_closed = (lambda: window.closed) if window is not None else (lambda: False)

        try:
            while not self.stop.is_set() and not window_closed():
                frame = self.camera.get_latest_frame()
                if frame is None or frame.sequence == last_sequence:
                    if self.camera.error is not None:
                        raise RuntimeError(f"摄像头采集已停止：{self.camera.error}") from self.camera.error
                    time.sleep(0.001)
                    continue
                self.skipped_frames += max(0, frame.sequence - (last_sequence if last_sequence >= 0 else 0) - 1)
                last_sequence = frame.sequence
                self._handle_frame(frame)
        except Exception as exc:
            self.failure = exc
            traceback.print_exc()
            if self.preview_lock is not None and self.shared is not None:
                with self.preview_lock:
                    self.shared["fatal"] = f"处理线程异常退出：{exc!r}"
        finally:
            self.stop.set()

    def _handle_frame(self, frame):
        plc, samples, runtime = self.plc, self.samples, self.runtime

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
                self.gst_pts_count += 1

        # PLC 只接收 AI 半场目标；轴反馈在本帧 AI 计算前更新。
        decision = result.ai_decision
        result.rally = self.rally.update(result, now=time.perf_counter(), feedback=feedback,
                                         mallet_position=runtime.ai_current_pos)
        plc_prepare_start = time.perf_counter() if samples is not None else 0.0
        plc_request = submit_rally_target(result, self.plc_adapter, plc, dry_run=self.args.dry_run)
        if plc_request is not None:
            self.decision_ages.append((time.perf_counter() - frame.timestamp) * 1000.0)
        if samples is not None:
            samples.add("plc_prepare_submit_ms", (time.perf_counter() - plc_prepare_start) * 1000.0)
        if self.recorder is not None:
            self.recorder.record(result, ai_decision=decision, plc_request=plc_request,
                                 plc_state=None if plc is None else plc.motion_snapshot())

        status_text = format_status(
            result,
            runtime.camera_geometry.enabled and runtime.camera_geometry.camera_matrix is not None,
        )
        if self.args.dry_run:
            status_text += " | DRY-RUN: motion blocked"
        plc_gate = plc_text = None
        if plc is not None:
            plc_text, plc_gate = plc_status_text(plc, feedback, time.perf_counter())
            status_text += plc_text
        self._log_gate_change(result, frame, feedback, status_text, plc_gate)
        if not self.headless:
            with self.preview_lock:
                self.shared["result"] = result
                self.shared["result_seq"] = result.frame.sequence
                self.shared["plc"] = None if plc is None else (plc_text, plc_gate)

        stats_interval = 1.0 if self.headless else STATS_INTERVAL
        if time.perf_counter() - self._stats_timer >= stats_interval:
            self._stats_timer = time.perf_counter()
            self._print_stats(result, status_text)

    def _log_gate_change(self, result, frame, feedback, status_text, plc_gate):
        """回合状态 / 追踪 / PLC 门控任一变化时打印一行 [rally]。"""
        track = result.track
        track_key = None if track is None else (track.track_id, track.state)
        gate = result.rally.state, result.rally.reason, track_key, plc_gate
        if gate == self._last_gate:
            return
        track_text = "none" if track is None else f"{track.track_id}/{track.state.value}"
        position = None if result.curling_state is None else tuple(round(v, 1) for v in result.curling_state.position)
        prefix = "; ".join(result.rally.transitions) or "gate changed"
        print(f"[rally] {prefix} | {status_text} | track={track_text} "
              f"hard_hit={result.detection is not None} puck={position} "
              f"frame_age_ms={(time.perf_counter() - frame.timestamp) * 1000:.1f} "
              f"plc_busy={None if feedback is None else feedback.busy}")
        self._last_gate = gate

    def _print_stats(self, result, status_text):
        runtime, plc = self.runtime, self.plc
        print(f"[rally] {status_text}")
        capture = self.camera.get_stats()
        track_state = result.track.state.value if result.track else "none"
        if self.headless:
            stage = self.samples.snapshot() if self.samples is not None else {}
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
        if capture.current_fps and capture.current_fps < 30.0:
            print(f"[stats] [!] 相机采集只有 {capture.current_fps:.1f} FPS：目标每 "
                  f"{1000.0 / capture.current_fps:.0f} ms 才更新一次，反应必然慢。"
                  "检查像素格式是否为 MJPG（启动行会打印协商格式），不要加 --camera-slow")
        print_latency_report(self.decision_ages, plc)

    # ------------------------------------------------------------------
    # 编码线程（仅 GUI 模式）
    # ------------------------------------------------------------------
    def _encoding_loop(self):
        """预览转换线程：最新 VisionResult -> render 标注 -> 缩放 -> PPM（按预览帧率限速）。"""
        seen = -1
        brightness = PreviewBrightness()
        try:
            while not self.stop.wait(1.0 / self.args.preview_fps):
                with self.preview_lock:
                    result = self.shared["result"]
                    seq = self.shared["result_seq"]
                if result is None or seq == seen:
                    continue
                seen = seq

                encode_start = time.perf_counter() if self.samples is not None else 0.0
                ppm_data = encode_preview_ppm(
                    result, self.runtime.table_roi, self.runtime.camera_geometry,
                    brightness.gain(result.frame.image),
                )
                if self.samples is not None:
                    self.samples.add("preview_encode_ms", (time.perf_counter() - encode_start) * 1000.0)
                with self.preview_lock:
                    self.shared["ppm"] = ppm_data
                    self.shared["ppm_seq"] += 1
        except Exception as exc:
            self.failure = exc
            traceback.print_exc()
            with self.preview_lock:
                self.shared["fatal"] = f"预览线程异常退出：{exc!r}"
            self.stop.set()

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _shutdown(self, processing_thread):
        plc = self.plc
        if plc is not None:
            plc.clear_target()
            if not plc.stop():
                print("[PLC] 仍在等待通信线程撤销使能；请留意实机，必要时使用硬件急停")
                while not plc.stop(timeout=1.0):
                    pass
            for message in plc.drain_messages():
                print(f"[PLC] {message}")
        self.camera.stop()
        processing_thread.join()
        if self.recorder is not None:
            recorded_path = self.recorder.close()
            if recorded_path is not None:
                print(f"运行数据已记录 {self.recorder.frame_count} 帧 -> {recorded_path}")
        if self.samples is not None:
            capture = self.camera.get_stats()
            print(f"[perf] 采集 {capture.frame_count} 帧 | 处理 {self.runtime.frame_index} 帧 | 跳过 {self.skipped_frames} 帧")
            if self.args.perf_json:
                report = build_perf_report(self.camera, self.runtime, self.samples,
                                           self.skipped_frames, self.gst_pts_count)
                output = Path(self.args.perf_json)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"性能报告已写入: {output}")
        if self.window is not None:
            self.window.close()
