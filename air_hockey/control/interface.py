"""PLCInterface：阻塞式 S7 传输 + 运动指令的坐标换算、限速斜坡、死区与 BufferMode。

只由 PLCLink 的工作线程调用。
"""

from __future__ import annotations

import ctypes
import math
import struct
import time

from . import protocol
from .limits import (
    COORD_EPSILON, DEFAULT_LIMITS, DEFAULT_ZONE, ClampStats,
    game_to_physical, game_to_physical_raw, physical_to_game, validated_zone,
)
from .protocol import (
    AXIS_ENABLE_BITS, AXIS_HOME_BITS, AXIS_NAMES, AXIS_RESET_BITS, COMM_AXES_READY_BIT,
    COMM_BYTE, COMM_HEARTBEAT_BIT, COMM_KIN_ERROR_OFFSET, COMM_KIN_REACTION_OFFSET,
    COMM_LINK_LOST_BIT, COMM_MC_ERRID_OFFSET, COMM_MC_ERR_BIT, COMM_MC_ERR_OFFSET,
    COMM_MC_REMAINING_OFFSET, COMM_OB1_TIME_OFFSET, COMM_PLC_HEARTBEAT_BIT, COMM_READ_SIZES,
    COMM_READ_START, COMM_RTINFO_STATUS_OFFSET, COMM_SEQ_ACK_OFFSET, COMM_SEQ_OFFSET,
    COMM_TRANSITION_OFFSET, COMM_WARNING_WORD_OFFSET, COMM_X_ERRID_OFFSET, COMM_X_ERR_BIT,
    COMM_Y_ERRID_OFFSET, COMM_Y_ERR_BIT, COMM_Y_ERR_OFFSET, DB_COMM, DB_MOTION,
    GROUP_COMM_LOST_LATCH_BIT, GROUP_CTRL_BYTE, GROUP_RESET_BIT, GROUP_STOP_BIT,
    HOMING_BYTE, MOTION_BLOCK_SIZE, MOTION_BLOCK_START, MOTION_BUSY_BYTE, MOTION_CMD_BYTE,
    MOTION_STATUS_BYTE, STATUS_READ_SIZE, STATUS_READ_START, TRIGGER_BYTE,
    TRIGGER_LINEAR_BIT,
)

DEAD_ZONE_MM = 3.0
# Measured on the machine: 50mm with BufferMode=5 stalled streaming motion.
#
# 现场实测的含义：步距 ≤ 这个值时用 BufferMode=5（混合，平滑）；更大就用 0
# （中止重开，会一顿一顿）。50mm 用混合会"卡住"，所以当年把阈值定在 10mm。
#
# ★ 但"卡住"的原因是 TO 的「作业序列 → 最大舍入间距」被设成了 100%：
#   混合半径可以吃掉整个运动，轴一直在拐弯、走不出直线。
#   把舍入间距调小（例如 30%）之后，混合应当可以在更大的步距下正常工作，
#   那时这个阈值就可以放宽 —— 快球才可能真正平滑。
#
# 所以做成可调（--blend-threshold-mm），配合 PLC 侧舍入间距一起实验。
BUFFER_MODE_IMMEDIATE_MM = 10.0

# 精磨距离 = 步距 × 这个比例。必须 **< 1**：等于 1 时 TO 会把整段都圆掉、
# 到不了目标点。0.75 是"最后四分之一才混合"的意思，实测既连续又走得准。
TRANSITION_RATIO = 0.75

# 运动积压上限（单位：步，即 target_slew_mm 的倍数）。
# 每触发一次 MC_MoveLinearAbsolute（BufferMode=5），作业就排进 TO_Kinematics 的作业队列；
# 轴跟不上指令时队列越堆越长，轴会把一串过时的中间点依次走完（现场实测固定落后 300~420ms）。
# 轴忙且"上次下发点"领先实际位置超过这么多步时，本周期不再追加作业，等轴追上来。
# ponytail: 4 步 ≈ 30Hz 下 130ms 的余量（含反馈与序号确认各约一个周期）；实机若出现走走停停就调大。
MAX_BACKLOG_STEPS = 4.0

# 终点模式（target_slew_mm <= 0）：只下发 AI 的终点，不再切成一串中间点。
# MC_MoveLinearAbsolute 没有"中止当前作业"的 BufferMode（只有 1/2/5，见西门子 S7-1500T
# Kinematics 手册 MC_MoveLinearAbsolute 参数表），每个新终点都会排进作业队列。所以轴忙时
# 新终点离上次下发的终点不够远就不发，避免 AI 目标的小抖动把队列塞满。
# ponytail: 10mm 是估计值；实机若反应迟钝就调小，若又出现一顿一顿就调大。
ENDPOINT_RESEND_MM = 10.0


def _braking_step(dist: float, max_delta: float) -> float:
    """按剩余距离算出不超调的最大离散步长。"""
    if dist <= 0.0 or max_delta <= 0.0:
        return 0.0
    if dist <= max_delta:
        return dist
    d = dist - max_delta / 8.0
    disc = max_delta * max_delta + 8.0 * max_delta * d
    return max(0.0, (math.sqrt(disc) - max_delta) / 2.0)


class PLCInterface:
    """Blocking S7 transport, exclusively operated by PLCLink's worker."""

    def __init__(self, ip="192.168.0.64", rack=0, slot=1, zone=None,
                 blend_threshold_mm=BUFFER_MODE_IMMEDIATE_MM, target_slew_mm=0.0,
                 target_accel_mm_s2=0.0, period=0.0, limits=None):
        self.ip, self.rack, self.slot = ip, rack, slot
        # 步距超过它就用 BufferMode=1（附加运动：先走完当前再走新的）而不是 5（混合）。
        # 现场实测 50mm + 混合会"卡住"（TO 舍入间距 100%），所以默认 10mm。
        # 调小 TO 的舍入间距后可以放宽这个值让快球也平滑。
        self.blend_threshold_mm = float(blend_threshold_mm)
        # 目标限速（mm/周期）。0 = 关闭。
        # ★ 建议 = blend_threshold_mm：这样步距永远不超阈值 → 永不中止 → 不顿。
        self.target_slew_mm = float(target_slew_mm)
        # ★★★ 目标加速度限幅（mm/s^2），0 = 关闭；period 是这个限幅换算用的周期。
        # 见 _prepare_move 里的说明 —— TIA 状态位显示 Y 轴报的是"跟随误差"，
        # 光限步长治不了（目标突然反向需要的加速度是轴能力的约 5 倍）。
        self.target_accel_mm_s2 = float(target_accel_mm_s2)
        self.period = float(period)
        # 上一个周期真正发出去的步长（mm），加速度限幅要拿它比。
        self._last_step = None
        # 上一次真正发送的时刻 —— 加速度限幅要用**实测** dt，
        # 不能用 --plc-rate 的名义周期（名义 200Hz、实际常只有 70Hz）。
        self._last_send_at = None
        # ★ 轴的实际位置（游戏坐标），由 PLCLink 每个周期从反馈里更新。
        #   _prepare_move 的死区判据要用它 —— 只看"上次发了什么"会假死，见那里的注释。
        self.actual_game = None
        self.actual_busy = False
        self.zone = validated_zone(DEFAULT_ZONE if zone is None else zone)
        # 命令包络与 TIA 动力学（PLCLink 会用它自己的那份覆盖，和 zone 一样）
        self.limits = DEFAULT_LIMITS if limits is None else limits
        self.clamp_stats = ClampStats()
        self._client = None
        self._connected = False
        self.last_heartbeat_time = 0.0
        self.heartbeat_state = False
        self._last_phys_x = None
        self._last_phys_y = None
        # 序号握手：每成功下发一条 +1。种子取当前时间，避免重启后恰好与 PLC 里记的"上次序号"相同、
        # 第一条指令被当成旧指令丢掉。
        self.sequence = int(time.time()) % 1000000 + 1
        self.last_send_state = "unknown"
        self.last_buffer_mode = None
        # 必须初始化：它原本只在 send_linear_move 里、且从**第二次**下发才开始赋值
        # （第一次下发时 _last_phys_x 还是 None，那个分支不成立）。
        # 读它的诊断代码会在第一次下发时抛 AttributeError。
        self.last_send_dist_mm = 0.0
        # 精磨距离（TransitionParameter[1]）随步距自适应写回 DB1.DBD150。
        # transition_enabled 会在写入失败（DB 里没这个变量）时自动关掉，
        # 之后退回由 TO「最大舍入间距」决定混合距离 —— 不会影响运动本身。
        self.transition_enabled = True
        self._last_transition = None
        self.last_transition_mm = None
        self.last_error = ""
        self._multi_read_failed = False
        # DB18 当前用哪种长度读（66 / 62 / 58），见 COMM_READ_SIZES
        self._comm_size = None
        self.round_trips = 0

    @property
    def connected(self):
        return self._connected

    def connect(self, quiet=False):
        self.last_error = ""
        if not protocol.HAS_SNAP7:
            self.last_error = protocol.SNAP7_ERROR
            if not quiet:
                print(f"[PLC] {protocol.SNAP7_ERROR}")
            return False
        try:
            self._client = protocol.snap7.client.Client()
            self._client.connect(self.ip, self.rack, self.slot)
            self._connected = True
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, False)
            self.clear_trigger()
            self.reset_tracking()
            self.last_heartbeat_time = 0.0
            self.heartbeat_state = False
            return True
        except Exception as exc:
            self.last_error = str(exc)
            if not quiet:
                print(f"[PLC] 连接失败: {exc}")
            self.disconnect()
            return False

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self.reset_tracking()

    def reset_tracking(self):
        self._last_phys_x = None
        self._last_phys_y = None
        self._last_step = None
        self._last_send_at = None
        return True

    def _write_bool(self, db, offset, bit, value):
        self.round_trips += 1
        try:
            library = getattr(self._client, "_lib", None) or getattr(self._client, "_library", None)
            if library is None:
                request = bytearray(self._client.protocol.build_write_request(
                    protocol.Areas.DB, db, offset * 8 + bit, protocol.WordLen.Bit, bytes([bool(value)])))
                # ponytail: 3.0 错把 1 位编码成 8 位；升级协议接口时重验封包与应答。
                data_start = 10 + struct.unpack_from(">H", request, 6)[0]
                struct.pack_into(">H", request, data_start + 2, 1)
                response = self._client._send_receive(request)
                self._client.protocol.check_write_response(response)
            else:
                pointer = getattr(self._client, "_s7_client", None) or getattr(self._client, "_pointer", None)
                if pointer is None:
                    raise RuntimeError("Snap7 native client handle is unavailable")
                data = ctypes.c_uint8(bool(value))
                result = library.Cli_WriteArea(
                    pointer, protocol.Areas.DB.value, db, offset * 8 + bit,
                    1, protocol.WordLen.Bit.value, ctypes.byref(data))
                if result != 0:
                    raise RuntimeError(self._client.error_text(result))
        except Exception:
            self._connected = False
            raise

    def _read_db_blocks(self, specs):
        if protocol.Areas is not None and not self._multi_read_failed:
            try:
                items = [{"area": protocol.Areas.DB, "db_number": db, "start": start, "size": size}
                         for db, start, size in specs]
                self.round_trips += 1
                result, data = self._client.read_multi_vars(items)
                if result == 0 and data is not None and len(data) == len(specs):
                    blocks = [bytearray(block) for block in data]
                    if all(len(block) == spec[2] for block, spec in zip(blocks, specs)):
                        return blocks
                raise RuntimeError(f"read_multi_vars result={result}")
            except Exception as exc:
                self._multi_read_failed = True
                print(f"[PLC] 多变量读取不可用，回退逐块读取: {exc}")
        self.round_trips += len(specs)
        return [self._client.db_read(db, start, size) for db, start, size in specs]

    def _read_status_blocks(self):
        """读 DB1 + DB18，兼容 DB18 的三种长度（58 / 62 / 66）。

        从新的往旧的试，读到哪个能用就记住哪个 —— 现场加减 DB18 字段时
        不该把整个连接判死（那会表现成"PLC 未连接"，很难查）。
        """
        sizes = list(COMM_READ_SIZES)
        if self._comm_size in sizes:                 # 先用上次成功的长度
            sizes.remove(self._comm_size)
            sizes.insert(0, self._comm_size)
        for size in sizes:
            try:
                db1, db18 = self._read_db_blocks(
                    ((DB_MOTION, STATUS_READ_START, STATUS_READ_SIZE),
                     (DB_COMM, COMM_READ_START, size)))
            except Exception:
                continue
            self._comm_size = size
            return db1, db18
        return None, None

    def clear_trigger(self):
        self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, False)

    def set_axis_enable(self, axis, enabled):
        """单轴使能 / 撤销。axis 为 "x" 或 "y"。

        两个轴必须**都**使能程序才允许下发运动（见 PLCLink._motion_safe），
        所以单轴使能只用于现场单独调试或单独撤下一个报故障的轴。
        """
        if not self.connected:
            return False
        bit = AXIS_ENABLE_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, bool(enabled))
            return True
        except Exception:
            self._connected = False
            return False

    def reset_axis(self, axis):
        """单轴复位（清该轴的报警）。"""
        if not self.connected:
            return False
        bit = AXIS_RESET_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, False)
            return True
        except Exception:
            self._connected = False
            return False

    def home_axis(self, axis):
        """单轴回零。[!] 回零**会让该轴动**（去找参考点），是刻意的人工动作。"""
        if not self.connected:
            return False
        bit = AXIS_HOME_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, HOMING_BYTE, bit, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, HOMING_BYTE, bit, False)
            return True
        except Exception:
            self._connected = False
            return False

    def enable_axes(self):
        return all(self.set_axis_enable(axis, True) for axis in AXIS_NAMES)

    def disable_axes(self):
        # 两个都要写：用 all() 会在第一个失败时短路，漏掉另一个。
        return all([self.set_axis_enable(axis, False) for axis in AXIS_NAMES])

    def home_axes(self):
        return all(self.home_axis(axis) for axis in AXIS_NAMES)

    def axes_reset(self):
        # 组复位 + 两个单轴复位
        ok = all([self.reset_axis(axis) for axis in AXIS_NAMES])
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, False)
            return ok
        except Exception:
            self._connected = False
            return False

    def game_to_physical(self, x, y):
        return game_to_physical(self.zone, self.limits, x, y, self.clamp_stats)

    def game_to_physical_raw(self, x, y):
        return game_to_physical_raw(self.zone, x, y)

    def physical_to_game(self, x, y):
        return physical_to_game(self.zone, x, y)

    def _prepare_move(self, game_x, game_y):
        """坐标换算 + 目标限速 + 死区 + BufferMode。

        返回 None（不发）或 (物理x, 物理y, mode, 步距)。步距在**第一次**下发时是 None。

        ★ 目标限速（slew）治的是"顿"：rally 状态机切换时目标会**瞬间跳变**
        （现场实测 p95 步距 192.8mm —— Y 在防守线 140 和追击 286 之间来回跳），
        大步使用 BufferMode=1，完成当前运动并停稳后才执行下一条。
        把跳变削成"每周期最多走 target_slew_mm"，只要 slew ≤ blend_threshold，
        步距就永远不超阈值 → 永远走混合、永不中止 —— 运动从"跳"变成"斜坡"。
        """
        try:
            phys_x, phys_y = self.game_to_physical(game_x, game_y)
        except (ValueError, OverflowError):
            self.last_send_state = "invalid"
            return None
        # 执行中的同点不再排队；在限速前比较，避免挡住仍在推进的斜坡目标。
        # 终点模式下"同点"放宽到 ENDPOINT_RESEND_MM。
        same_point_mm = ENDPOINT_RESEND_MM if self.target_slew_mm <= 0.0 else DEAD_ZONE_MM
        if (self.actual_busy and self._last_phys_x is not None
                and max(abs(phys_x - self._last_phys_x),
                        abs(phys_y - self._last_phys_y)) < same_point_mm):
            self.last_send_state = "pending"
            return None
        # 运动积压：轴忙且上次下发点已远远领先实际位置 → 队列里已有好几条作业在排，先不追加。
        actual = self.actual_game
        if (self.actual_busy and self.target_slew_mm > 0.0
                and self._last_phys_x is not None and actual is not None):
            try:
                ax, ay = self.game_to_physical_raw(actual[0], actual[1])
            except (ValueError, OverflowError):
                ax = None
            if ax is not None and max(abs(self._last_phys_x - ax), abs(self._last_phys_y - ay)) \
                    > MAX_BACKLOG_STEPS * self.target_slew_mm:
                self.last_send_state = "backlog"
                return None
        # ★★★ 起点：优先用"上次真正发出去的位置"；还没有就用轴的**实际位置**打底 ——
        # 这样 arming 后的**第一条**指令也走限速斜坡。
        #
        # 现场（TIA 工艺对象 Y轴 → 状态位和错误位：错误 = 跟随误差）：
        # 按 E 使能时 _enable_on_worker 会 reset_tracking() → _last_phys=None，
        # 于是 arming 后的**第一条**指令**完全不限速**，直接跳到当前目标的
        # 命令包络角点（实测 game(533,349) → phys(200,170)，从轴所在的
        # (-117,-75) 算起约 320mm）。轴以最大加速度扑过去 → 跟随误差 →
        # Y 轴 16#8001 → 整机撤销使能。快照 prev_en=(True,True) prev_busy=True
        # 正是"刚使能、第一条指令正在跑"。前面 X 轴的 801 是同一个原因。
        #
        # [!] 只做**局部变量**，绝不写回 self._last_phys_* —— 那是"下发之前就改
        #    状态"，会让"写失败后必须清空跟踪"的保证失效（有测试守着）。
        #    self._last_phys_* 只在 _commit_move 里更新。
        origin_x, origin_y = self._last_phys_x, self._last_phys_y
        if origin_x is None:
            actual = getattr(self, "actual_game", None)
            if actual is not None:
                try:
                    origin_x, origin_y = self.game_to_physical_raw(actual[0], actual[1])
                except (ValueError, OverflowError):
                    origin_x = origin_y = None
        if origin_x is not None:
            goal_phys_x = phys_x
            goal_phys_y = phys_y
            dx = phys_x - origin_x
            dy = phys_y - origin_y
            span = max(abs(dx), abs(dy))
            now_mono = time.perf_counter()
            dt = (now_mono - self._last_send_at
                  if self._last_send_at is not None else self.period)
            if dt <= 0.0 or not math.isfinite(dt):
                dt = self.period
            max_delta = self.target_accel_mm_s2 * dt * dt if self.target_accel_mm_s2 > 0.0 else 0.0

            # 目标限速与提前减速：步长不超 target_slew_mm 且按剩余距离减速刹停
            v_limit = self.target_slew_mm if self.target_slew_mm > 0.0 else span
            if max_delta > 0.0:
                v_limit = min(v_limit, _braking_step(span, max_delta))

            if span > v_limit:
                ratio = v_limit / span
                phys_x = origin_x + dx * ratio
                phys_y = origin_y + dy * ratio

            dx_step = phys_x - origin_x
            dy_step = phys_y - origin_y
            distance = max(abs(dx_step), abs(dy_step))

            # 加速度限幅：限制相邻周期步长变化量
            if self.target_accel_mm_s2 > 0.0 and self._last_step is not None and max_delta > 0.0:
                ddx = dx_step - self._last_step[0]
                ddy = dy_step - self._last_step[1]
                accel_span = max(abs(ddx), abs(ddy))
                if accel_span > max_delta:
                    ratio = max_delta / accel_span
                    phys_x = origin_x + self._last_step[0] + ddx * ratio
                    phys_y = origin_y + self._last_step[1] + ddy * ratio

            # 包络兜底：加速度限幅可能让中间指令冲过目标，中间指令同样不得越出命令包络。
            # 轴本来就在包络外（行程内）时，允许从起点按斜坡逐步回来、不跳到包络边界，
            # 但绝不比起点更往外 —— 不会被命令往限位开关方向走。
            limits = self.limits
            phys_x = max(min(limits.command_x_min, origin_x), min(max(limits.command_x_max, origin_x), phys_x))
            phys_y = max(min(limits.command_y_min, origin_y), min(max(limits.command_y_max, origin_y), phys_y))

            # ★ 注意：distance 已经在上面（加速度限幅之前）算好了，这里不要
            #   用限幅后的步长重算 —— 那正是把它锁死的写法。见上面的注释。
            # 同点到位则跳过；空闲但未到位时重发，执行中的同点已在限速前跳过。
            # 只有当总目标差距（span）本身已进入死区时，才进入死区到位跳过判定；
            # 只要总目标还在远方（span >= DEAD_ZONE_MM），哪怕这一拍步距只有 1~2mm，也必须连续下发！
            if span < DEAD_ZONE_MM:
                actual_gap = None
                actual = getattr(self, "actual_game", None)
                if actual is not None:
                    try:
                        ax, ay = self.game_to_physical_raw(actual[0], actual[1])
                        actual_gap = max(abs(goal_phys_x - ax), abs(goal_phys_y - ay))
                    except (ValueError, OverflowError):
                        actual_gap = None
                if actual_gap is None or actual_gap < DEAD_ZONE_MM:
                    self.last_send_state = "deadzone"
                    return None
                self.last_send_state = "deadzone-resend"
            # ★★★ TO_Kinematics 的 BufferMode 只有 1/2/5，和轴版不同（**没有 0 = 中止**）：
            #     1 = 附加运动    当前运动走完、停稳后才执行新作业  ← 会停
            #     2 = 混合（较低速度）  5 = 混合（较高速度）
            # 现场踩过：照抄轴版发 0 是非法值，作业被拒绝。
            # 斜坡模式和终点模式都统一用 5：新作业与当前运动混合，不停稳。
            buffer_mode = 5
            # 精磨距离和诊断步距必须用**实际发出去的步长**（加速度限幅、包络兜底之后）。
            # 限幅前的 distance 可能比实际步长大好几倍（目标反向时实测 20mm vs 6.5mm），
            # 精磨距离按它算就会超过整段运动，TO 把整段圆掉、到不了目标点。
            distance = max(abs(phys_x - origin_x), abs(phys_y - origin_y))
        else:
            distance = None
            buffer_mode = 5
        return phys_x, phys_y, buffer_mode, distance

    def _write_payload(self, phys_x, phys_y, buffer_mode):
        data = bytearray(MOTION_BLOCK_SIZE)
        # The machine's TO_Kinematics uses components 1=X, 3=Y, NOT [X,Y,0,0].
        struct.pack_into(">ddddi", data, 0, phys_x, 0.0, phys_y, 0.0, buffer_mode)
        self.round_trips += 1
        self._client.db_write(DB_MOTION, MOTION_BLOCK_START, data)

    def _commit_move(self, phys_x, phys_y, buffer_mode, distance):
        # 记住这一周期真正发出去的步长 —— 下一周期的加速度限幅要拿它比。
        if self._last_phys_x is not None:
            self._last_step = (phys_x - self._last_phys_x, phys_y - self._last_phys_y)
        self._last_phys_x, self._last_phys_y = phys_x, phys_y
        self._last_send_at = time.perf_counter()
        self.last_buffer_mode = buffer_mode
        if distance is not None:
            self.last_send_dist_mm = distance
        self.last_send_state = "sent"

    def _write_transition(self, distance):
        """写 TransitionParameter[1]（精磨距离）＝ 混合起点离目标多远。

        ★★★ 为什么必须每周期跟着步距写，不能留一个固定值：
        它的语义是「在距离目标还有这么远时，就开始和下一段作业混合」。
          · 步距 20mm、精磨 15mm → 只最后 15mm 混合 ✓ 连续
          · 步距  5mm、精磨 15mm → 精磨距离比整段还长 → TO 把整段圆掉，
            **到不了目标**。小步距恰恰是"瞄准到点"的精细修正，被圆掉就瞄不准。

        ★★★ 步距接近 0 时**绝不能写 0**（现场踩过）：
        目标没变时步距就是 0.1mm 量级，写成 0 的含义是「精磨距离 = 0」=
        **完全不许混合**，等于把刚修好的混合又关掉。日志里
        `步距中位 0.1mm | 精磨 0.0mm` 就是这个 bug。
        这种周期本来就没什么可混合的，**保持上一次的值**才对。
        """
        if not self.transition_enabled:
            return
        if self.target_slew_mm <= 0.0:
            # 终点模式：-1 = 由 TO 取"最大可能的舍入间距"（受 TIA「最大舍入间距」设置限制），
            # 新终点一来就尽早拐向它，不必先走到旧终点附近。
            value = -1.0
        elif distance is None or distance <= DEAD_ZONE_MM:
            return          # 步距太小：保持上一次的精磨距离，不要写 0
        else:
            value = min(self.blend_threshold_mm, distance * TRANSITION_RATIO)
        value = round(value, 1)
        # 变化不大就不写 —— 省一次往返，也避免 PLC 读到中间态。
        if (self._last_transition is not None
                and abs(value - self._last_transition) < 0.2):
            return
        try:
            self._client.db_write(DB_MOTION, COMM_TRANSITION_OFFSET,
                                  struct.pack(">d", value))
        except Exception as exc:
            self.transition_enabled = False
            self.last_error = (
                f"PLC 写 TransitionParameter[1] 失败（DB1.DBD{COMM_TRANSITION_OFFSET}）"
                f"：{exc} —— 已停用该写入；混合距离改回由 TO「最大舍入间距」决定")
            print(f"[PLC] {self.last_error}")
            return
        self.round_trips += 1
        self._last_transition = value
        self.last_transition_mm = value

    def send_linear_move(self, game_x, game_y):
        """序号握手：写目标块，再写递增的序号 —— PLC 比较序号后执行。

        [!] 顺序不能反：**先写目标块，再写序号**。这样 PLC 看到新序号时
        目标已经稳定。序号只在整条写成功后才推进。
        """
        if not self.connected:
            self.last_send_state = "disconnected"
            return False
        prepared = self._prepare_move(game_x, game_y)
        if prepared is None:
            return self.last_send_state != "invalid"
        phys_x, phys_y, buffer_mode, distance = prepared
        try:
            self._write_payload(phys_x, phys_y, buffer_mode)
            # ★ 精磨距离必须在触发/序号之前写好 —— PLC 看到新序号时两个值都要是新的。
            self._write_transition(distance)
            candidate = (self.sequence + 1) & 0x7FFFFFFF
            self.round_trips += 1
            self._client.db_write(DB_MOTION, COMM_SEQ_OFFSET, struct.pack(">i", candidate))
            self.sequence = candidate
            self._commit_move(phys_x, phys_y, buffer_mode, distance)
            return True
        except Exception:
            self._connected = False
            self.last_send_state = "disconnected"
            return False

    def read_status_and_position(self):
        if not self.connected:
            return None
        try:
            db1, db18 = self._read_status_blocks()
            if db1 is None:
                self._connected = False
                return None
            status, comm, group = db1[MOTION_STATUS_BYTE], db18[COMM_BYTE], db1[GROUP_CTRL_BYTE]

            def _word(offset):
                """按无符号读一个 Word；DB18 还是旧长度时返回 None。"""
                if offset + 2 > len(db18):
                    return None
                return struct.unpack_from(">H", db18, offset)[0]

            def _bit(offset, bit):
                if offset >= len(db18):
                    return None
                return bool(db18[offset] & (1 << bit))

            def _dword(offset):
                if offset + 4 > len(db18):
                    return None
                return struct.unpack_from(">I", db18, offset)[0]

            def _lreal(offset):
                if offset + 8 > len(db18):
                    return None
                return struct.unpack_from(">d", db18, offset)[0]

            return {
                "x_en": bool(status & 0x04), "y_en": bool(status & 0x08),
                "x_home": bool(status & 0x10), "y_home": bool(status & 0x20),
                "x_pos": struct.unpack_from(">d", db1, 2)[0],
                "y_pos": struct.unpack_from(">d", db1, 10)[0],
                "x_err": bool(comm & (1 << COMM_X_ERR_BIT)),
                "y_err": bool(db18[COMM_Y_ERR_OFFSET] & (1 << COMM_Y_ERR_BIT)),
                # ★ 轴自己的故障码（一直没读）
                "x_errid": _word(COMM_X_ERRID_OFFSET),
                "y_errid": _word(COMM_Y_ERRID_OFFSET),
                # MC_MoveLinearAbsolute 的 Error / ErrorId（现场新接的）
                "mc_error": _bit(COMM_MC_ERR_OFFSET, COMM_MC_ERR_BIT),
                "mc_errid": _word(COMM_MC_ERRID_OFFSET),
                "busy": bool(db1[MOTION_BUSY_BYTE] & 1),
                "plc_heartbeat": bool(comm & (1 << COMM_PLC_HEARTBEAT_BIT)),
                "comm_lost": bool(comm & (1 << COMM_LINK_LOST_BIT)),
                "axes_ready": bool(comm & (1 << COMM_AXES_READY_BIT)),
                "group_stop": bool(group & (1 << GROUP_STOP_BIT)),
                "comm_lost_latch": bool(group & (1 << GROUP_COMM_LOST_LATCH_BIT)),
                # ★ ErrorDetail.Number 在 TO 里是 **UDInt**（DB12.Static.ErrorDetail.Number），
                # 程序用 MOVE 送进 %DB18.DBW56（Word，只保留低 16 位）。
                # 必须按**无符号**读 —— 按有符号读的话，16#8001 这类码会变成 -32767，
                # 和 TIA 里看到的对不上，故障就成了谜。
                "kinematics_error": struct.unpack_from(">H", db18, COMM_KIN_ERROR_OFFSET)[0],
                # ErrorDetail.Reaction（DInt，有符号）—— 读不到就是 None（DB18 短）
                "kin_reaction": (struct.unpack_from(">i", db18, COMM_KIN_REACTION_OFFSET)[0]
                                 if COMM_KIN_REACTION_OFFSET + 4 <= len(db18) else None),
                # PLC 已处理到哪个序号（它写在 DB1.64）。和 Python 的序号一比，
                # 就知道"PLC 落后几条指令" —— 这是判断它跟不跟得上的直接指标。
                "seq_ack": (struct.unpack_from(">i", db1, COMM_SEQ_ACK_OFFSET)[0]
                            if COMM_SEQ_ACK_OFFSET + 4 <= len(db1) else None),
                # ---- 诊断补充字段（PLC 没加就是 None，界面显示"未读到"）----
                # ★ WarningWord：软限位之类的**预警**，能在报故障之前看到征兆
                "warning_word": _dword(COMM_WARNING_WORD_OFFSET),
                # ★ 当前运动还差多远（接的是 MC.RemainingDistance）
                "mc_remaining": _lreal(COMM_MC_REMAINING_OFFSET),
                # ★ OB1 循环时间：infoData 是 LTime（64 位有符号，单位纳秒）
                "ob1_ns": (struct.unpack_from(">q", db18, COMM_OB1_TIME_OFFSET)[0]
                           if COMM_OB1_TIME_OFFSET + 8 <= len(db18) else None),
                "rtinfo_status": (struct.unpack_from(">h", db18, COMM_RTINFO_STATUS_OFFSET)[0]
                                  if COMM_RTINFO_STATUS_OFFSET + 2 <= len(db18) else None),
            }
        except Exception:
            self._connected = False
            return None

    def keep_alive(self):
        if not self.connected:
            return
        now = time.perf_counter()
        if now - self.last_heartbeat_time <= 0.1:
            return
        self.heartbeat_state = not self.heartbeat_state
        try:
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, self.heartbeat_state)
            self.last_heartbeat_time = now
        except Exception:
            self._connected = False
