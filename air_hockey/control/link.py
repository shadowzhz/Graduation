"""PLCLink：唯一拥有 S7 传输的后台工作线程（反馈 / 心跳 / 授权互锁 / 运动下发 / 诊断）。

坐标映射按 link 各自的 zone；反馈与目标在同一游戏坐标区间，与显示缩放无关。
start/stop 与 set_target 不做任何网络 I/O。
"""

from __future__ import annotations

import math
import queue
import threading
import time

from . import protocol
from .feedback import PLCFeedback
from .interface import PLCInterface
from .latency import LatencyProbe
from .limits import (
    DEFAULT_LIMITS, DEFAULT_ZONE, TRAVEL_X_MAX, TRAVEL_X_MIN, TRAVEL_Y_MAX, TRAVEL_Y_MIN,
    game_to_physical, physical_to_game, validate_game_point, validated_zone,
)
from .protocol import (
    AXIS_NAMES,
)

DEFAULT_PERIOD = 0.03
RECONNECT_INTERVAL = 2.0
TARGET_MAX_AGE = 0.25
ECHO_TIMEOUT = 1.0


class PLCLink:
    """Single-owner worker; start/stop and target publication do no network I/O."""

    def __init__(self, ip, rack=0, slot=1, period=DEFAULT_PERIOD,
                 client_factory=None, zone=None, limits=None):
        self.ip = ip
        self.zone = validated_zone(DEFAULT_ZONE if zone is None else zone)
        # 命令包络：AI 的可达范围和下发夹取都用这一份
        self.limits = DEFAULT_LIMITS if limits is None else limits
        self.period = max(0.005, float(period))
        if not math.isfinite(self.period):
            raise ValueError("PLC period must be finite")
        self._sequence_sent = 0
        self._requires_snap7 = client_factory is None
        self._client = (PLCInterface(ip, rack, slot, zone=self.zone, limits=self.limits)
                        if client_factory is None else client_factory(ip, rack, slot))
        if isinstance(self._client, PLCInterface):
            self._client.zone = self.zone
            self._client.limits = self.limits
        self._lock = threading.Lock()
        self._target = None
        self._target_at = 0.0
        self._feedback = PLCFeedback()
        self._commands = queue.Queue()
        self._messages = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._disable_on_stop = False
        self._armed = False
        self._arm_pending = False
        self._ever_connected = False
        self._snap7_warned = False
        self._last_retry = 0.0
        self._last_plc_heartbeat = None
        self._echo_change_at = 0.0
        self._echo_seen = False
        self.cycle_count = 0
        # 统计实际频率、同点跳过和目标步距；不把它们直接当作 801 的原因。
        self._stats_at = time.perf_counter()
        self._stats_cycles = 0
        self._stats_counters = {"sent": 0, "deadzone": 0, "pending": 0, "other": 0}
        self._stats_dists = []
        # 现场延迟排查：PC → PLC → 轴 分段统计（见 control/latency.py）
        self.latency = LatencyProbe(self.period)
        self._sent_snapshot = None      # (perf_counter 时刻, 物理 x, 物理 y) 最近一次成功下发

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._disarm_locked()
            self._feedback = PLCFeedback()
            while not self._commands.empty():
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
            self._last_retry = 0.0
            self._stop_event.clear()
            self._disable_on_stop = False
            self._thread = threading.Thread(target=self._run, name="plc-link", daemon=True)
            self._thread.start()

    def stop(self, disable_axes=True, timeout=1.5):
        """Request worker cleanup; False means timeout, NEVER perform caller-thread I/O.

        A timed-out worker retains ownership and cleans up when its blocking I/O returns.
        A successful return confirms cleanup attempts completed, not that an offline
        PLC physically accepted the disable command; inspect drain_messages().
        """
        with self._lock:
            self._disarm_locked()
            self._stop_event.set()
            self._disable_on_stop |= disable_axes
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        if thread.is_alive():
            return False
        with self._lock:
            if self._thread is thread:
                self._thread = None
        return True

    @property
    def connected(self):
        return self._client.connected

    def diagnostics(self):
        """返回并重置通信频率、跳过次数和相邻目标步距，不代表电机执行确认。"""
        with self._lock:
            now = time.perf_counter()
            elapsed = max(1e-9, now - self._stats_at)
            counters = dict(self._stats_counters)
            dists = self._stats_dists
            cycles = self.cycle_count - self._stats_cycles
            self._stats_at = now
            self._stats_cycles = self.cycle_count
            self._stats_counters = {"sent": 0, "deadzone": 0, "pending": 0, "other": 0}
            self._stats_dists = []
            # 序号模式的诊断：PLC 已处理到哪
            seq_ack = self._feedback.seq_ack
            seq_now = getattr(self._client, "sequence", None)
            seq_sent = self._sequence_sent

        dists.sort()
        median = dists[len(dists) // 2] if dists else 0.0
        p95 = dists[min(len(dists) - 1, int(len(dists) * 0.95))] if dists else 0.0

        text = (f"周期 {cycles / elapsed:5.1f}Hz | 下发 {counters['sent'] / elapsed:5.1f}Hz | "
                f"死区跳过 {counters['deadzone']} | 执行中同点跳过 {counters['pending']}")
        text += f" | 步距中位 {median:.1f}mm p95 {p95:.1f}mm"
        client = self._client
        if counters["other"]:
            text += f" | 其它状态 {counters['other']}"
        # 夹取比例：目标被命令包络截住的频率，高说明经常要求够不到的位置
        clamp_stats = getattr(client, "clamp_stats", None)
        counts = clamp_stats.snapshot() if clamp_stats is not None else {"total": 0}
        if counts["total"]:
            pct_x = 100.0 * counts["x"] / counts["total"]
            pct_y = 100.0 * counts["y"] / counts["total"]
            text += f" | 目标被夹取 X {pct_x:.0f}% Y {pct_y:.0f}%"
            if pct_x > 30.0 or pct_y > 30.0:
                text += "  ← 频繁要求够不到的位置，机械臂长期顶在命令限位上"
        if clamp_stats is not None:
            clamp_stats.reset()
        # 序号握手：PLC 落后几条指令
        if seq_ack is None or seq_now is None:
            text += f" | 序号 已发 {seq_sent} 条（PLC 未回报 ack，检查 DB1.94）"
        else:
            # seq_now 是序号值（从时间戳起算），seq_sent 是发送条数，两者不能混着显示
            lag = (seq_now - seq_ack) & 0x7FFFFFFF
            text += f" | 序号 已发 {seq_sent} 条 / PLC 落后 {lag}"
            if lag > 3:
                text += "  ← PLC 跟不上，指令在积压"
        return text

    @property
    def feedback(self):
        with self._lock:
            return self._feedback

    @property
    def armed(self):
        """True only after explicit enable_axes and a healthy fresh PLC report."""
        with self._lock:
            return (self._armed and self._client.connected and not self._stop_event.is_set()
                    and self._motion_safe(self._feedback)
                    and time.perf_counter() - self._feedback.stamp <= TARGET_MAX_AGE)

    def game_to_physical(self, x, y):
        return game_to_physical(self.zone, self.limits, x, y)

    def physical_to_game(self, x, y):
        return physical_to_game(self.zone, x, y)

    @property
    def reachable_bounds(self):
        """AI 可请求的命令范围（游戏坐标），由命令包络 limits.command_* 决定。

        必须和下发时夹取用的是同一份 limits：范围更宽的话 AI 会一直瞄够不到的点，
        下发时又被默默夹回包络，机械臂长期顶在包络角点上。
        """
        limits = self.limits
        left, top = self.physical_to_game(limits.command_x_min, limits.command_y_min)
        right, bottom = self.physical_to_game(limits.command_x_max, limits.command_y_max)
        return left, right, top, bottom

    def latency_report(self):
        """延迟排查汇总（一行文本, 结论提示列表），调用后清零统计窗口。"""
        return self.latency.report()

    def motion_snapshot(self):
        """给运行记录用的轴状态快照（游戏坐标）；时间用"距今多久"表示，与时钟源无关。"""
        now = time.perf_counter()
        with self._lock:
            feedback, sent, armed = self._feedback, self._sent_snapshot, self._armed
        if not feedback.valid:
            return None
        snapshot = {
            "age": now - feedback.stamp, "x": feedback.x, "y": feedback.y,
            "busy": bool(feedback.busy), "armed": bool(armed),
            "seq_ack": feedback.seq_ack, "remaining": feedback.mc_remaining, "sent": None,
        }
        if sent is not None:
            try:
                sent_x, sent_y = self.physical_to_game(sent[1], sent[2])
                snapshot["sent"] = {"age": now - sent[0], "x": sent_x, "y": sent_y}
            except (ValueError, OverflowError):
                pass
        return snapshot

    def set_target(self, game_x, game_y):
        try:
            x, y = validate_game_point(self.zone, game_x, game_y)
        except (ValueError, OverflowError):
            return False
        with self._lock:
            if self._stop_event.is_set():
                return False
            self._target = x, y
            self._target_at = time.perf_counter()
        return True

    def clear_target(self):
        with self._lock:
            self._target = None
            self._target_at = 0.0

    def reset_tracking(self):
        return self._run_on_worker("清死区", self._client.reset_tracking)

    def enable_axes(self):
        return self._run_on_worker("使能", self._enable_on_worker)

    # ---- 单轴操作：某个轴报故障 / 撞限位后只撤下那个轴，或单独回零 / 复位 ----
    # 单轴使能不会让程序开始发运动：_motion_safe 要求 x_en 和 y_en 同时为真。
    def set_axis_enable(self, axis, enabled):
        """axis 为 "x" / "y"。使能走 _run_on_worker（要靠新状态才 arm）。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        if enabled:
            return self._run_on_worker(f"{axis.upper()} 轴使能",
                                       lambda: self._enable_on_worker(axis))
        return self._queue_disarmed_command(f"{axis.upper()} 轴撤销",
                                           lambda: self._client.set_axis_enable(axis, False))

    def _enable_on_worker(self, axis=None):
        """使能双轴（axis=None，E 键 / 按钮）或单轴；授权要等下一份健康的新状态。"""
        ok = self._client.enable_axes() if axis is None else self._client.set_axis_enable(axis, True)
        if not ok:
            return False
        # 报警/回零可能中断旧目标：重新授权后不能把未完成的目标当作已到达，
        # 使能前留下的目标也不能成为第一条运动指令。
        self._client.reset_tracking()
        with self._lock:
            if not self._stop_event.is_set():
                self._target = None
                self._target_at = 0.0
                self._arm_pending = True
        return True

    def toggle_axis_enable(self, axis):
        """按当前反馈状态取反。返回 (是否已发出, 切换后的期望状态)。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False, None
        feedback = self.feedback
        currently = feedback.x_en if axis == "x" else feedback.y_en
        wanted = not bool(currently)
        return self.set_axis_enable(axis, wanted), wanted

    def home_axis(self, axis):
        """单轴回零。当前 MC_Home Mode=0：不移动，直接把当前位置设为原点（见 PLCInterface.home_axis）。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        return self._queue_disarmed_command(f"{axis.upper()} 轴回零",
                                           lambda: self._client.home_axis(axis))

    def reset_axis(self, axis):
        """单轴复位（清该轴报警）。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        return self._queue_disarmed_command(f"{axis.upper()} 轴复位",
                                           lambda: self._client.reset_axis(axis))

    def home_axes(self):
        return self._queue_disarmed_command("回零", self._client.home_axes)

    def axes_reset(self):
        return self._queue_disarmed_command("轴复位", self._client.axes_reset)

    def set_cylinder(self, on):
        """开关气缸启动（DB1.118.0）。不改变轴授权/目标状态。"""
        return self._run_on_worker("气缸" + ("开" if on else "关"),
                                   lambda: self._client.set_cylinder(on))

    def drain_messages(self):
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def _disarm_locked(self):
        """撤销授权并清掉目标；调用方必须持有 self._lock。"""
        self._target = None
        self._target_at = 0.0
        self._armed = False
        self._arm_pending = False

    def _run_on_worker(self, name, func):
        with self._lock:
            if self._stop_event.is_set() or not self._client.connected:
                return False
            self._commands.put((name, func))
            return True

    def _queue_disarmed_command(self, name, func):
        with self._lock:
            if self._stop_event.is_set() or not self._client.connected:
                return False
            self._disarm_locked()
            self._commands.put((name, lambda: self._execute_disarmed(func)))
            return True

    def _execute_disarmed(self, func):
        try:
            return func()
        finally:
            with self._lock:
                self._disarm_locked()

    def _drain_commands(self):
        while not self._stop_event.is_set():
            try:
                name, func = self._commands.get_nowait()
            except queue.Empty:
                return
            if not self._client.connected:
                continue
            try:
                if not func():
                    self._messages.put(f"{name}指令发送失败")
            except Exception as exc:
                self._messages.put(f"{name}异常：{exc}")

    def _invalidate_feedback(self):
        with self._lock:
            self._disarm_locked()
            if self._feedback.valid:
                self._feedback = PLCFeedback()
            # Commands from the old connection must not be replayed on reconnect.
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break

    def _retry_connect(self, now):
        if not protocol.HAS_SNAP7 and self._requires_snap7:
            if not self._snap7_warned:
                self._snap7_warned = True
                self._messages.put(protocol.SNAP7_ERROR)
            return
        if now - self._last_retry < RECONNECT_INTERVAL or self._stop_event.is_set():
            return
        self._last_retry = now
        was_connected = self._ever_connected
        if self._client.connect(quiet=True):
            self._ever_connected = True
            self._last_plc_heartbeat = None
            self._echo_change_at = now
            self._echo_seen = False
            self._messages.put("PLC 已重新连接" if was_connected else "PLC 已连接")
        elif self._requires_snap7:
            self._messages.put(f"PLC 连接失败：{self._client.last_error}")

    def _publish(self, status):
        now = time.perf_counter()
        try:
            x, y = self.physical_to_game(status["x_pos"], status["y_pos"])
        except (ValueError, OverflowError):
            self._invalidate_feedback()
            return None
        heartbeat = status["plc_heartbeat"]
        if self._last_plc_heartbeat is None:
            self._last_plc_heartbeat = heartbeat
            self._echo_change_at = now
        elif heartbeat != self._last_plc_heartbeat:
            self._last_plc_heartbeat = heartbeat
            self._echo_change_at = now
            self._echo_seen = True
        echo_ok = self._echo_seen and now - self._echo_change_at < ECHO_TIMEOUT
        with self._lock:
            previous = self._feedback
            dt = now - previous.stamp
            vx = (x - previous.x) / dt if previous.valid and dt > 1e-6 else 0.0
            vy = (y - previous.y) / dt if previous.valid and dt > 1e-6 else 0.0
            self._feedback = PLCFeedback(
                stamp=now, valid=True, x=x, y=y, vx=vx, vy=vy,
                x_en=status["x_en"], y_en=status["y_en"],
                x_err=status["x_err"], y_err=status["y_err"],
                x_errid=status["x_errid"], y_errid=status["y_errid"],
                mc_error=status["mc_error"], mc_errid=status["mc_errid"],
                busy=status["busy"], comm_lost=status["comm_lost"],
                axes_ready=status["axes_ready"], plc_echo_ok=echo_ok,
                group_stop=status["group_stop"],
                comm_lost_latch=status["comm_lost_latch"],
                kinematics_error=status["kinematics_error"],
                kin_reaction=status["kin_reaction"],
                seq_ack=status["seq_ack"],
                mc_remaining=status["mc_remaining"],
            )
            return self._feedback

    @staticmethod
    def _motion_safe(feedback):
        return (feedback.valid and feedback.plc_echo_ok and feedback.axes_ready
                and feedback.x_en and feedback.y_en
                and not feedback.axis_fault and not feedback.comm_lost
                and not feedback.comm_lost_latch and not feedback.group_stop
                and feedback.kinematics_error == 0)

    def _cycle(self):
        client = self._client
        cycle_start = time.perf_counter()
        status = client.read_status_and_position()
        if status is None or not client.connected:
            self._invalidate_feedback()
            return
        client.keep_alive()
        status_done = time.perf_counter()
        if not client.connected or self._stop_event.is_set():
            self._invalidate_feedback()
            return
        previous = self.feedback
        feedback = self._publish(status)
        if feedback is None:
            return
        # 死区判据要用轴的实际位置判断"到没到"，不更新会把没走到的目标当成已到达、一直跳过
        try:
            client.actual_game = (feedback.x, feedback.y)
        except Exception:
            pass
        if isinstance(client, PLCInterface):
            client.actual_busy = feedback.busy
        safe = self._motion_safe(feedback)
        revoked = False
        interrupted_target = None
        with self._lock:
            # Serious faults revoke even a pending arm request. During the initial
            # heartbeat/enable ramp, hold pending without allowing any movement.
            # 行程越界由硬件限位开关负责（触发后 TIA 报轴故障，走 axis_fault 撤销）。
            fault = (feedback.axis_fault or feedback.comm_lost or feedback.comm_lost_latch
                     or feedback.group_stop or feedback.kinematics_error != 0
                     or (not feedback.plc_echo_ok
                         and time.perf_counter() - self._echo_change_at >= ECHO_TIMEOUT)
                     or (self._armed and not safe))
            if fault:
                revoked = self._armed or self._arm_pending
                interrupted_target = self._target
                self._disarm_locked()
            elif safe and self._arm_pending:
                self._armed = True
                self._arm_pending = False
            target = (self._target if safe and self._armed
                      and not self._stop_event.is_set()
                      and time.perf_counter() - self._target_at <= TARGET_MAX_AGE else None)
            armed_now = self._armed
        # 延迟排查统计（非必要功能，任何异常都不能影响通信）
        try:
            self.latency.on_status(
                status_done, (status_done - cycle_start) * 1000.0,
                (float(status["x_pos"]), float(status["y_pos"])), armed=armed_now,
                busy=feedback.busy, seq_ack=feedback.seq_ack,
                remaining=feedback.mc_remaining)
        except Exception:
            pass
        if revoked:
            def _limit_note(px, py):
                """把原始位置和行程比一下 —— 故障出现时一眼看出是不是碰到了限位开关。"""
                notes = []
                for axis, value, low, high, margin_low, margin_high in (
                        ("X", px, TRAVEL_X_MIN, TRAVEL_X_MAX, self.limits.command_x_min, self.limits.command_x_max),
                        ("Y", py, TRAVEL_Y_MIN, TRAVEL_Y_MAX, self.limits.command_y_min, self.limits.command_y_max)):
                    if value < low or value > high:
                        notes.append(f"{axis}={value:.2f} 已超出机械行程[{low},{high}] ← 限位开关")
                    elif value < margin_low or value > margin_high:
                        notes.append(f"{axis}={value:.2f} 在命令范围外（行程内），离限位开关 "
                                     f"{min(abs(value - low), abs(value - high)):.2f}mm")
                    else:
                        notes.append(f"{axis}={value:.2f} 在命令范围内")
                return " | ".join(notes)

            self._messages.put(
                f"运动撤销快照: kin={feedback.kinematics_error} "
                f"en=({feedback.x_en},{feedback.y_en}) ready={feedback.axes_ready} "
                f"err=({feedback.x_err},{feedback.y_err}) busy={feedback.busy} "
                f"echo={feedback.plc_echo_ok} comm_lost={feedback.comm_lost} "
                f"stop={feedback.group_stop} latch={feedback.comm_lost_latch} "
                f"physical=({status['x_pos']:.3f},{status['y_pos']:.3f}) "
                + _limit_note(status["x_pos"], status["y_pos"]) + " "
                + feedback.fault_detail() + " "
                f"prev_valid={previous.valid} prev_kin={previous.kinematics_error} "
                f"prev_en=({previous.x_en},{previous.y_en}) prev_ready={previous.axes_ready} "
                f"prev_err=({previous.x_err},{previous.y_err}) prev_busy={previous.busy} "
                f"intent={interrupted_target} "
                f"last_sent=({getattr(client, '_last_phys_x', None)},{getattr(client, '_last_phys_y', None)}) "
                f"step_mm={getattr(client, 'last_send_dist_mm', None)}"
            )
        # Network I/O is never under the lock. Cancellation of an already executing
        # db_write cannot be guaranteed; no subsequent target remains queued.
        if target is not None:
            # 状态读取期间生产者可能更新、清除目标或撤销授权，发送前取最新意图。
            with self._lock:
                target = (self._target if self._armed and not self._stop_event.is_set()
                          and time.perf_counter() - self._target_at <= TARGET_MAX_AGE
                          and time.perf_counter() - feedback.stamp <= TARGET_MAX_AGE else None)
                target_at = self._target_at
            if target is None:
                self.cycle_count += 1
                return
            send_start = time.perf_counter()
            sent_sequence = None
            # 序号只在真正下发成功时推进（interface 内部），死区/执行中跳过不推进
            client.send_linear_move(*target)
            if client.connected and client.last_send_state == "sent":
                self._sequence_sent += 1
                sent_sequence = getattr(client, "sequence", None)
            send_done = time.perf_counter()
            try:
                sent_x = getattr(client, "_last_phys_x", None)
                sent_y = getattr(client, "_last_phys_y", None)
                if client.connected and client.last_send_state == "sent" and sent_x is not None:
                    with self._lock:
                        self._sent_snapshot = (send_done, sent_x, sent_y)
                    self.latency.on_send(send_done, (send_done - send_start) * 1000.0,
                                         send_start - target_at, sent_x, sent_y, seq=sent_sequence)
            except Exception:
                pass
            # 诊断累计必须容错：任何异常都不能打死 worker 线程
            try:
                state = client.last_send_state
                with self._lock:
                    # deadzone-resend 也真发出去了（目标没变但轴没到位），要算进下发频率
                    if state in ("sent", "deadzone-resend"):
                        self._stats_counters["sent"] += 1
                        self._stats_dists.append(
                            float(getattr(client, "last_send_dist_mm", 0.0) or 0.0))
                    elif state == "deadzone":
                        self._stats_counters["deadzone"] += 1
                    elif state == "pending":
                        self._stats_counters["pending"] += 1
                    else:
                        self._stats_counters["other"] += 1
            except Exception:
                pass
        if not client.connected:
            self._invalidate_feedback()
        self.cycle_count += 1

    def _run(self):
        client = self._client
        # 按绝对截止时间排程：每轮 wait(period - 耗时) 会把唤醒开销累加成周期误差（30Hz 只到 28.8）
        next_due = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                try:
                    if client.connected:
                        self._drain_commands()
                        if client.connected and not self._stop_event.is_set():
                            self._cycle()
                    else:
                        self._invalidate_feedback()
                        self._retry_connect(cycle_start)
                except Exception as exc:
                    self._messages.put(f"PLC 通信异常：{exc}")
                    self._invalidate_feedback()
                    client.disconnect()
                next_due += self.period
                remaining = next_due - time.perf_counter()
                if remaining > 0:
                    self._stop_event.wait(remaining)
                else:
                    next_due = time.perf_counter()   # 落后太多就重新对齐，避免雪崩
        finally:
            self._invalidate_feedback()
            if client.connected:
                if self._disable_on_stop:
                    try:
                        if not client.disable_axes():
                            self._messages.put("PLC 撤销使能失败，请现场确认轴状态")
                    except Exception as exc:
                        self._messages.put(f"PLC 撤销使能失败，请现场确认轴状态：{exc}")
            elif self._disable_on_stop and self._ever_connected:
                self._messages.put("PLC 已断线，无法确认撤销使能；请现场确认轴状态")
            try:
                client.disconnect()
            except Exception:
                pass
