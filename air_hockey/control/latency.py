"""PC → PLC → 轴 的分段延迟统计，用于现场排查"反应慢"。

PLCLink 的工作线程在每个周期调用 on_status / on_send，主线程定期调用 report()
取一行汇总并清零窗口。所有位置都是物理坐标（mm），时间是 time.perf_counter()（Windows 上 monotonic 只有 15.6ms 分辨率）。

各段含义（按信号流向）：
  S7 读 / S7 写      一次状态读取、一次运动下发占用的 S7 往返时间
  目标等待下发        视觉线程 set_target 到工作线程真正下发之间的等待（≤ 一个周期）
  序号确认            写入序号到读回 PLC 的 seq_ack ≥ 该序号（分辨率 = 一个周期）
  OB1                PLC 主循环时间（RT_INFO），序号逻辑每个 OB1 才看一次
  起步等待            轴静止时下发远处新指令，到轴真正开始移动的时间
  轴落后指令          轴实际位置离最新已下发指令点 > MATCH_MM 的周期占比
  差距 / 时间滞后     轴离最新指令点多远；轴现在所在的位置是多久以前下发的
  实测速度 / 指令速度  轴实际移动速度与指令点移动速度（两者接近才说明轴跟得上）
  TO 剩余            MC 作业的 RemainingDistance：大 = TO 里还排着很长的路
"""

from __future__ import annotations

import math
import threading
from collections import deque

MATCH_MM = 5.0              # 轴离指令点多近算"到了这个指令点"
HISTORY_SECONDS = 3.0       # 指令历史保留多久（时间滞后最多能测到这么长）
START_GAP_MM = 10.0         # 轴静止时，新指令离轴多远才算一次"起步"
START_MOVED_MM = 3.0        # 轴移动多少算"已经起步"
STILL_SPEED_MM_S = 20.0     # 低于此速度算静止

_KEYS = ("read_ms", "send_ms", "target_age_ms", "ack_ms", "ob1_ms", "start_ms",
         "gap_mm", "lag_ms", "act_speed", "cmd_speed", "remaining_mm")


def _seq_reached(seq, ack):
    """序号按 31 位回绕：ack 是否已经到达或越过 seq。"""
    return ((ack - seq) & 0x7FFFFFFF) < 0x40000000


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class LatencyProbe:
    def __init__(self, period):
        self.period = float(period)
        self._lock = threading.Lock()
        self._history = deque()        # (t, x, y) 已下发的指令点
        self._pending_seq = deque()    # (seq, t) 等待 PLC 确认的序号
        self._last_actual = None       # (t, x, y)
        self._last_cmd = None          # (t, x, y)
        self._start = None             # (t, x, y) 起步计时：下发时刻与当时轴位置
        self._actual_speed = 0.0
        self._reset_window()

    def _reset_window(self):
        self._w = {key: [] for key in _KEYS}
        self._armed_cycles = 0
        self._behind = 0
        self._unmatched = 0

    # ---- 工作线程调用 -------------------------------------------------------
    def on_send(self, now, send_ms, target_age_s, x, y, seq=None):
        """一次成功下发：x/y 为发出去的物理指令点。"""
        with self._lock:
            w = self._w
            w["send_ms"].append(send_ms)
            if target_age_s is not None and target_age_s >= 0.0:
                w["target_age_ms"].append(target_age_s * 1000.0)
            if self._last_cmd is not None:
                t0, x0, y0 = self._last_cmd
                step = math.hypot(x - x0, y - y0)
                if now > t0 and step > 0.5:
                    w["cmd_speed"].append(step / (now - t0))
            self._last_cmd = (now, x, y)
            self._history.append((now, x, y))
            while self._history and now - self._history[0][0] > HISTORY_SECONDS:
                self._history.popleft()
            if seq is not None:
                self._pending_seq.append((seq & 0x7FFFFFFF, now))
                while len(self._pending_seq) > 512:
                    self._pending_seq.popleft()
            # 起步计时：轴静止、而新指令离它足够远
            if (self._start is None and self._last_actual is not None
                    and self._actual_speed < STILL_SPEED_MM_S):
                _, ax, ay = self._last_actual
                if math.hypot(x - ax, y - ay) > START_GAP_MM:
                    self._start = (now, ax, ay)

    def on_status(self, now, read_ms, actual, *, armed, busy=False, seq_ack=None,
                  remaining=None, ob1_ms=None):
        """一次状态读取：actual 为轴实际物理位置 (x, y)。"""
        with self._lock:
            w = self._w
            w["read_ms"].append(read_ms)
            if ob1_ms is not None:
                w["ob1_ms"].append(ob1_ms)
            if seq_ack is not None:
                ack = seq_ack & 0x7FFFFFFF
                while self._pending_seq and _seq_reached(self._pending_seq[0][0], ack):
                    seq, sent_at = self._pending_seq.popleft()
                    if seq == ack:
                        w["ack_ms"].append((now - sent_at) * 1000.0)
            if actual is None:
                return
            ax, ay = actual
            if self._last_actual is not None:
                t0, x0, y0 = self._last_actual
                if now > t0:
                    self._actual_speed = math.hypot(ax - x0, ay - y0) / (now - t0)
                    if armed:
                        w["act_speed"].append(self._actual_speed)
            self._last_actual = (now, ax, ay)
            if self._start is not None:
                t_start, sx, sy = self._start
                if math.hypot(ax - sx, ay - sy) >= START_MOVED_MM:
                    w["start_ms"].append((now - t_start) * 1000.0)
                    self._start = None
                elif now - t_start > HISTORY_SECONDS:
                    self._start = None
            if not armed:
                self._start = None
                return
            self._armed_cycles += 1
            if busy and remaining is not None and math.isfinite(remaining):
                w["remaining_mm"].append(abs(remaining))
            if not self._history:
                return
            _, cx, cy = self._history[-1]
            gap = math.hypot(cx - ax, cy - ay)
            w["gap_mm"].append(gap)
            if gap <= MATCH_MM:
                return
            self._behind += 1
            for t_cmd, hx, hy in reversed(self._history):
                if math.hypot(hx - ax, hy - ay) <= MATCH_MM:
                    w["lag_ms"].append((now - t_cmd) * 1000.0)
                    break
            else:
                self._unmatched += 1

    # ---- 主线程调用 ---------------------------------------------------------
    def report(self):
        """返回 (一行汇总文本, 结论提示列表)，并清零统计窗口。"""
        with self._lock:
            w, armed, behind, unmatched = self._w, self._armed_cycles, self._behind, self._unmatched
            self._reset_window()

        def p(key, q, unit="ms", fmt="{:.0f}"):
            value = _percentile(w[key], q)
            return "-" if value is None else fmt.format(value) + unit

        parts = [
            f"S7 读 {p('read_ms', 0.5, fmt='{:.1f}')} 写 {p('send_ms', 0.5, fmt='{:.1f}')}",
            f"目标等待下发 p50 {p('target_age_ms', 0.5)} p95 {p('target_age_ms', 0.95)}",
            f"序号确认 p50 {p('ack_ms', 0.5)} p95 {p('ack_ms', 0.95)}",
            f"OB1 p50 {p('ob1_ms', 0.5, fmt='{:.1f}')} max {p('ob1_ms', 1.0, fmt='{:.1f}')}",
            f"起步等待 p50 {p('start_ms', 0.5)} p95 {p('start_ms', 0.95)}（{len(w['start_ms'])} 次）",
        ]
        if armed:
            parts.append(
                f"轴落后指令 {100.0 * behind / armed:.0f}% 周期，差距 p50 {p('gap_mm', 0.5, 'mm')} "
                f"p95 {p('gap_mm', 0.95, 'mm')}，时间滞后 p50 {p('lag_ms', 0.5)} p95 {p('lag_ms', 0.95)}"
                + (f"（{unmatched} 次不在指令路径上）" if unmatched else ""))
        else:
            parts.append("轴未授权：不统计跟随")
        parts.append(f"速度 实测 p95 {p('act_speed', 0.95, 'mm/s')} / 指令 p95 {p('cmd_speed', 0.95, 'mm/s')}")
        parts.append(f"TO剩余 p50 {p('remaining_mm', 0.5, 'mm')} p95 {p('remaining_mm', 0.95, 'mm')}")

        hints = []
        period_ms = self.period * 1000.0
        io = (_percentile(w["read_ms"], 0.5) or 0.0) + (_percentile(w["send_ms"], 0.5) or 0.0)
        if io > 0.6 * period_ms:
            hints.append(f"S7 通讯占了周期的 {100.0 * io / period_ms:.0f}%：实际下发频率上不去")
        ack = _percentile(w["ack_ms"], 0.5)
        if ack is not None and ack > 2.5 * period_ms:
            hints.append("PLC 取新序号慢（超过 2 个周期）：查 OB1 时间和 PLC 侧序号/Execute 逻辑")
        start = _percentile(w["start_ms"], 0.5)
        if start is not None and start > 150.0:
            hints.append("轴起步等待长：指令到了 PLC 但轴迟迟不动（TO 作业排队/触发逻辑）")
        lag = _percentile(w["lag_ms"], 0.5)
        if armed and behind > 0.3 * armed and lag is not None and lag > 150.0:
            hints.append("轴长期落后于已下发指令：瓶颈在 PLC/TO 执行，不在视觉和上位机；"
                         "此时调大 --target-slew-mm 只会让积压更多")
        act, cmd = _percentile(w["act_speed"], 0.95), _percentile(w["cmd_speed"], 0.95)
        if act is not None and cmd is not None and cmd > 0 and act < 0.6 * cmd:
            hints.append(f"轴实测速度只有指令速度的 {100.0 * act / cmd:.0f}%")
        keeps_up = armed and (behind <= 0.1 * armed or (lag is not None and lag < 100.0))
        if keeps_up and act is not None and cmd is not None and act >= 0.8 * cmd:
            hints.append("轴跟得上指令：瓶颈是指令速度上限（--target-slew-mm × 下发频率），可以逐步调大")
        return " | ".join(parts), hints
