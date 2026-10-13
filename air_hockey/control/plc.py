"""S7-1500 TO_Kinematics DB1/DB18 protocol and non-blocking game link.

Only the PLCLink worker owns the transport. Coordinate mapping is per-link; feedback
is in the same game-coordinate zone as targets, independent of display scaling.
"""

from __future__ import annotations

import math
import ctypes
import queue
import struct
import threading
import time
from dataclasses import dataclass

from air_hockey import core_config

try:
    import snap7
    try:
        from snap7.type import Areas, WordLen
    except ModuleNotFoundError as exc:
        if exc.name != "snap7.type":
            raise
        from snap7.types import Areas, WordLen
    SNAP7_ERROR = ""
except ImportError as exc:
    snap7 = None
    Areas = None
    SNAP7_ERROR = f"PLC 无法加载 python-snap7：{exc}"

HAS_SNAP7 = snap7 is not None

DB_MOTION = 1
DB_COMM = 18
MOTION_CMD_BYTE = 0
HOMING_BYTE = 1
MOTION_STATUS_BYTE = 1
MOTION_BLOCK_START = 20
MOTION_BLOCK_SIZE = 36
TRIGGER_BYTE = 56
TRIGGER_LINEAR_BIT = 0
MOTION_BUSY_BYTE = 18
GROUP_CTRL_BYTE = 140
GROUP_RESET_BIT = 3
GROUP_STOP_BIT = 2
GROUP_COMM_LOST_LATCH_BIT = 4
STATUS_READ_START = 0
STATUS_READ_SIZE = 141
COMM_READ_START = 0
COMM_READ_SIZE = 58
COMM_KIN_ERROR_OFFSET = 56
COMM_BYTE = 0
COMM_HEARTBEAT_BIT = 0
COMM_PLC_HEARTBEAT_BIT = 1
COMM_LINK_LOST_BIT = 2
COMM_AXES_READY_BIT = 3
COMM_X_ERR_BIT = 4
COMM_Y_ERR_OFFSET = 4
COMM_Y_ERR_BIT = 0
ECHO_TIMEOUT = 1.0
# ⚠️ 两套常量，用途不同，不要混：
#   PHYSICAL_* = 轴的行程安全限位（绝不能命令越过）
#   TABLE_*    = 球台真实边缘在【轴坐标】里的位置（只用于游戏<->机器坐标映射）
#
# 2026-10-13 现场实测（PLC 绝对位置读数 + 卷尺）：
#   球台（墙内）480mm x 880mm，球槌半径实测约 26.5mm
#   在轴 X=-188 / 219 处球槌边缘离左右墙各约 10mm
#   Y 最前处（185）球槌边缘离中线约 30mm
# 由此反推球台边缘（轴坐标）：
#   左墙 -188 - 26.5 - 10 = -224.5    右墙 219 + 26.5 + 10 = 255.5   （差 480 ✓）
#   中线  185 + 26.5 + 30 =  241.5    后墙 241.5 - 880/2   = -198.5   （差 440 ✓）
# TABLE_* 只由这些**几何测量**决定，与轴限位无关，所以改限位时不要动它们。
#
# 2026-10-13 稍后：X 轴冲得太猛撞到左右限位，现场把 X 的限位收到 -180 / 215
# （左右各留 8mm / 4mm 余量）。TABLE_X_* 保持不变 —— 球台边缘没变，
# 只是球槌现在停在离墙 18mm 而不是 10mm 处。
#
# 也就是说：球台比轴的行程大 —— 球槌边缘在 X 上离墙十几毫米，
# 在 Y 上离后墙差 47mm、离中线差 30mm。这是机械事实，软件只能如实反映。
PHYSICAL_X_MIN, PHYSICAL_X_MAX = -180.0, 215.0
PHYSICAL_Y_MIN, PHYSICAL_Y_MAX = -125.0, 185.0
TABLE_X_MIN, TABLE_X_MAX = -224.5, 255.5
TABLE_Y_MIN, TABLE_Y_MAX = -198.5, 241.5
COORD_EPSILON = 1e-6
DEAD_ZONE_MM = 3.0
# Measured on the machine: 50mm with BufferMode=5 stalled streaming motion.
BUFFER_MODE_IMMEDIATE_MM = 10.0
DEFAULT_PERIOD = 0.03
RECONNECT_INTERVAL = 2.0
TRIGGER_LOW_TIME = 0.010
TARGET_MAX_AGE = 0.25
DEFAULT_ZONE = (
    core_config.RINK_LEFT, core_config.RINK_RIGHT,
    core_config.RINK_TOP, core_config.RINK_CENTER_Y,
)


def _validated_zone(zone):
    left, right, top, bottom = map(float, zone)
    if not all(map(math.isfinite, (left, right, top, bottom))):
        raise ValueError("PLC zone must have finite bounds")
    if right <= left or bottom <= top:
        raise ValueError("PLC zone must have positive width and height")
    return left, right, top, bottom


def _finite_pair(x, y):
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("PLC coordinates must be finite")
    return x, y


def _game_to_physical(zone, x, y):
    """游戏场地坐标 -> 机器坐标。

    映射用 TABLE_*（球台真实边缘在轴坐标里的位置），
    夹取用 PHYSICAL_*（轴的行程安全限位）—— 这两件事必须分开：
    球台比轴的行程大，映射到球台边缘时会被安全限位截住，
    这正是"球从边角过去、轴到不了"的物理事实。
    """
    x, y = _validate_game_point(zone, x, y)
    left, right, top, bottom = zone
    px = TABLE_X_MIN + (x - left) * (TABLE_X_MAX - TABLE_X_MIN) / (right - left)
    py = TABLE_Y_MIN + (y - top) * (TABLE_Y_MAX - TABLE_Y_MIN) / (bottom - top)
    return (max(PHYSICAL_X_MIN, min(PHYSICAL_X_MAX, px)),
            max(PHYSICAL_Y_MIN, min(PHYSICAL_Y_MAX, py)))


def _validate_game_point(zone, x, y):
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    if (x < left - COORD_EPSILON or x > right + COORD_EPSILON
            or y < top - COORD_EPSILON or y > bottom + COORD_EPSILON):
        raise ValueError(f"PLC game coordinates ({x}, {y}) are outside zone {zone}")
    return max(left, min(right, x)), max(top, min(bottom, y))


def _physical_to_game(zone, x, y):
    """机器坐标 -> 游戏场地坐标（_game_to_physical 的逆，用同一套 TABLE_*）。"""
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    return (left + (x - TABLE_X_MIN) * (right - left) / (TABLE_X_MAX - TABLE_X_MIN),
            top + (y - TABLE_Y_MIN) * (bottom - top) / (TABLE_Y_MAX - TABLE_Y_MIN))


@dataclass(frozen=True)
class PLCFeedback:
    """Immutable worker snapshot; position in game units, velocity in units/s."""

    stamp: float = 0.0
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    x_en: bool = False
    y_en: bool = False
    x_home: bool = False
    y_home: bool = False
    x_err: bool = False
    y_err: bool = False
    busy: bool = False
    comm_lost: bool = False
    axes_ready: bool = False
    plc_echo_ok: bool = True
    group_stop: bool = False
    comm_lost_latch: bool = False
    kinematics_error: int = 0

    @property
    def axis_fault(self):
        return self.x_err or self.y_err


class PLCInterface:
    """Blocking S7 transport, exclusively operated by PLCLink's worker."""

    def __init__(self, ip="192.168.0.64", rack=0, slot=1, zone=None):
        self.ip, self.rack, self.slot = ip, rack, slot
        self.zone = _validated_zone(DEFAULT_ZONE if zone is None else zone)
        self._client = None
        self._connected = False
        self.last_heartbeat_time = 0.0
        self.heartbeat_state = False
        self._last_phys_x = None
        self._last_phys_y = None
        self.last_send_state = "unknown"
        self.last_buffer_mode = None
        # 必须初始化：它原本只在 send_linear_move 里、且从**第二次**下发才开始赋值
        # （第一次下发时 _last_phys_x 还是 None，那个分支不成立）。
        # 读它的诊断代码会在第一次下发时抛 AttributeError。
        self.last_send_dist_mm = 0.0
        self.last_error = ""
        self._multi_read_failed = False
        self.round_trips = 0

    @property
    def connected(self):
        return self._connected

    def connect(self, quiet=False):
        self.last_error = ""
        if not HAS_SNAP7:
            self.last_error = SNAP7_ERROR
            if not quiet:
                print(f"[PLC] {SNAP7_ERROR}")
            return False
        try:
            self._client = snap7.client.Client()
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
        return True

    def _write_bool(self, db, offset, bit, value):
        self.round_trips += 1
        try:
            library = getattr(self._client, "_lib", None) or getattr(self._client, "_library", None)
            if library is None:
                request = bytearray(self._client.protocol.build_write_request(
                    Areas.DB, db, offset * 8 + bit, WordLen.Bit, bytes([bool(value)])))
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
                    pointer, Areas.DB.value, db, offset * 8 + bit,
                    1, WordLen.Bit.value, ctypes.byref(data))
                if result != 0:
                    raise RuntimeError(self._client.error_text(result))
        except Exception:
            self._connected = False
            raise

    def _read_db_blocks(self, specs):
        if Areas is not None and not self._multi_read_failed:
            try:
                items = [{"area": Areas.DB, "db_number": db, "start": start, "size": size}
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

    def clear_trigger(self):
        self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, False)

    def enable_axes(self):
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 6, True)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 7, True)
            return True
        except Exception:
            self._connected = False
            return False

    def disable_axes(self):
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 6, False)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 7, False)
            return True
        except Exception:
            self._connected = False
            return False

    def home_axes(self):
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, HOMING_BYTE, 0, True)
            self._write_bool(DB_MOTION, HOMING_BYTE, 1, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, HOMING_BYTE, 0, False)
            self._write_bool(DB_MOTION, HOMING_BYTE, 1, False)
            return True
        except Exception:
            self._connected = False
            return False

    def axes_reset(self):
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 4, True)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 5, True)
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 4, False)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, 5, False)
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, False)
            return True
        except Exception:
            self._connected = False
            return False

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    def send_linear_move(self, game_x, game_y):
        """Write 36-byte [X,0,Y,0], mode, then raise DB1.56.0 atomically."""
        if not self.connected:
            self.last_send_state = "disconnected"
            return False
        try:
            phys_x, phys_y = self.game_to_physical(game_x, game_y)
        except (ValueError, OverflowError):
            self.last_send_state = "invalid"
            return False
        if self._last_phys_x is not None:
            distance = max(abs(phys_x - self._last_phys_x),
                           abs(phys_y - self._last_phys_y))
            if distance < DEAD_ZONE_MM:
                self.last_send_state = "deadzone"
                return True
            buffer_mode = 0 if distance > BUFFER_MODE_IMMEDIATE_MM else 5
        else:
            buffer_mode = 5
        try:
            data = bytearray(MOTION_BLOCK_SIZE)
            # The machine's TO_Kinematics uses components 1=X, 3=Y, NOT [X,Y,0,0].
            struct.pack_into(">ddddi", data, 0, phys_x, 0.0, phys_y, 0.0, buffer_mode)
            self.round_trips += 1
            self._client.db_write(DB_MOTION, MOTION_BLOCK_START, data)
            self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, True)
            if self._last_phys_x is not None:
                self.last_send_dist_mm = distance
            self._last_phys_x, self._last_phys_y = phys_x, phys_y
            self.last_buffer_mode = buffer_mode
            self.last_send_state = "sent"
            return True
        except Exception:
            self._connected = False
            self.last_send_state = "disconnected"
            return False

    def read_status_and_position(self):
        if not self.connected:
            return None
        try:
            db1, db18 = self._read_db_blocks(((DB_MOTION, STATUS_READ_START, STATUS_READ_SIZE),
                                                (DB_COMM, COMM_READ_START, COMM_READ_SIZE)))
            status, comm, group = db1[MOTION_STATUS_BYTE], db18[COMM_BYTE], db1[GROUP_CTRL_BYTE]
            return {
                "x_en": bool(status & 0x04), "y_en": bool(status & 0x08),
                "x_home": bool(status & 0x10), "y_home": bool(status & 0x20),
                "x_pos": struct.unpack_from(">d", db1, 2)[0],
                "y_pos": struct.unpack_from(">d", db1, 10)[0],
                "x_err": bool(comm & (1 << COMM_X_ERR_BIT)),
                "y_err": bool(db18[COMM_Y_ERR_OFFSET] & (1 << COMM_Y_ERR_BIT)),
                "busy": bool(db1[MOTION_BUSY_BYTE] & 1),
                "plc_heartbeat": bool(comm & (1 << COMM_PLC_HEARTBEAT_BIT)),
                "comm_lost": bool(comm & (1 << COMM_LINK_LOST_BIT)),
                "axes_ready": bool(comm & (1 << COMM_AXES_READY_BIT)),
                "group_stop": bool(group & (1 << GROUP_STOP_BIT)),
                "comm_lost_latch": bool(group & (1 << GROUP_COMM_LOST_LATCH_BIT)),
                "kinematics_error": struct.unpack_from(">h", db18, COMM_KIN_ERROR_OFFSET)[0],
            }
        except Exception:
            self._connected = False
            return None

    def keep_alive(self):
        if not self.connected:
            return
        now = time.monotonic()
        if now - self.last_heartbeat_time <= 0.1:
            return
        self.heartbeat_state = not self.heartbeat_state
        try:
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, self.heartbeat_state)
            self.last_heartbeat_time = now
        except Exception:
            self._connected = False


class PLCLink:
    """Single-owner worker; start/stop and target publication do no network I/O."""

    def __init__(self, ip, rack=0, slot=1, period=DEFAULT_PERIOD,
                 client_factory=None, zone=None, trigger_low_time=TRIGGER_LOW_TIME):
        self.ip = ip
        self.zone = _validated_zone(DEFAULT_ZONE if zone is None else zone)
        self.period = max(0.005, float(period))
        if not math.isfinite(self.period):
            raise ValueError("PLC period must be finite")
        # 每个周期清掉运动触发之后，至少要压低这么久，PLC 的 P 沿才抓得到。
        #
        # ⚠️ 这个值是**每个周期的固定开销**：它 + 4 次 S7 往返(≈8.3ms) 就是最短周期。
        # 10ms 时实际只能跑到约 55Hz —— 即使 --plc-rate 填 100。
        # 而周期越长，每周期目标步距越大；步距一旦超过 BUFFER_MODE_IMMEDIATE_MM
        # 就会切成 BufferMode=0（中止重开），表现就是"一个点一个点走"。
        # 调小它可以直接提高实际指令频率。下限取决于 PLC 的 OB1 扫描周期，
        # 至少要保证 P 沿能在一低一高之间被扫到（一般 2~3ms 够）。
        self.trigger_low_time = max(0.0, float(trigger_low_time))
        if not math.isfinite(self.trigger_low_time):
            raise ValueError("trigger_low_time must be finite")
        self._requires_snap7 = client_factory is None
        self._client = (PLCInterface(ip, rack, slot, zone=self.zone) if client_factory is None
                        else client_factory(ip, rack, slot))
        if isinstance(self._client, PLCInterface):
            self._client.zone = self.zone
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
        self._connect_attempted = False
        self._ever_connected = False
        self._snap7_warned = False
        self._last_retry = 0.0
        self._last_plc_heartbeat = None
        self._echo_change_at = 0.0
        self._echo_seen = False
        self.cycle_count = 0
        # 现场诊断：实际周期频率、下发频率、BufferMode 分布、步距。
        #
        # "机械臂一个点一个点走"就是 BufferMode=0（中止当前运动、立刻重开）占比高：
        # 每来一条指令轴就被中止一次，速度反复归零。这个统计能把"顿"量出来，
        # 也能验证 --trigger-low-ms / --plc-rate 调了到底有没有用。
        self._stats_at = time.monotonic()
        self._stats_cycles = 0
        self._stats_counters = {"sent": 0, "deadzone": 0, "other": 0,
                                "mode0": 0, "mode5": 0}
        self._stats_dists = []

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._target = None
            self._target_at = 0.0
            self._feedback = PLCFeedback()
            self._armed = False
            self._arm_pending = False
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
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
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

    @property
    def last_send_state(self):
        return self._client.last_send_state

    def diagnostics(self):
        """返回并重置"上一次调用以来"的 PLC 侧实测统计。

        现场判断"顿"还是"顺"用：BufferMode=0 是中止重开，占比高就是
        "一个点一个点走"；同时还能验证 --trigger-low-ms / --plc-rate
        调了到底有没有真的提高周期频率。
        """
        with self._lock:
            now = time.monotonic()
            elapsed = max(1e-9, now - self._stats_at)
            counters = dict(self._stats_counters)
            dists = self._stats_dists
            cycles = self.cycle_count - self._stats_cycles
            self._stats_at = now
            self._stats_cycles = self.cycle_count
            self._stats_counters = {"sent": 0, "deadzone": 0, "other": 0,
                                    "mode0": 0, "mode5": 0}
            self._stats_dists = []

        mode_total = counters["mode0"] + counters["mode5"]
        dists.sort()
        median = dists[len(dists) // 2] if dists else 0.0
        p95 = dists[min(len(dists) - 1, int(len(dists) * 0.95))] if dists else 0.0

        text = (f"周期 {cycles / elapsed:5.1f}Hz | 下发 {counters['sent'] / elapsed:5.1f}Hz | "
                f"死区跳过 {counters['deadzone']}")
        if mode_total:
            abort_pct = 100.0 * counters["mode0"] / mode_total
            text += (f" | BufferMode 混合 {counters['mode5']} / 中止 {counters['mode0']}"
                     f"（中止 {abort_pct:.0f}%）")
        else:
            text += " | BufferMode 无下发"
        text += f" | 步距中位 {median:.1f}mm p95 {p95:.1f}mm"
        if counters["other"]:
            text += f" | 其它状态 {counters['other']}"
        if mode_total and 100.0 * counters["mode0"] / mode_total > 20.0:
            text += "  ← 中止占比偏高，轴会一顿一顿；试试 --trigger-low-ms 3"
        return text


    @property
    def connect_attempted(self):
        return self._connect_attempted

    @property
    def ever_connected(self):
        return self._ever_connected

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
                    and time.monotonic() - self._feedback.stamp <= TARGET_MAX_AGE)

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    def set_target(self, game_x, game_y):
        try:
            x, y = _validate_game_point(self.zone, game_x, game_y)
        except (ValueError, OverflowError):
            return False
        with self._lock:
            if self._stop_event.is_set():
                return False
            self._target = x, y
            self._target_at = time.monotonic()
        return True

    def clear_target(self):
        with self._lock:
            self._target = None
            self._target_at = 0.0

    def reset_tracking(self):
        return self._run_on_worker("清死区", self._client.reset_tracking)

    def enable_axes(self):
        return self._run_on_worker("使能", self._enable_on_worker)

    def _enable_on_worker(self):
        if not self._client.enable_axes():
            return False
        with self._lock:
            if not self._stop_event.is_set():
                # Arming awaits a healthy fresh status, never a cached snapshot.
                # A pre-enable target must not become the first motion command.
                self._target = None
                self._target_at = 0.0
                self._arm_pending = True
        return True

    def home_axes(self):
        return self._queue_disarmed_command("回零", self._client.home_axes)

    def axes_reset(self):
        return self._queue_disarmed_command("轴复位", self._client.axes_reset)

    def drain_messages(self):
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

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
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
            self._commands.put((name, lambda: self._execute_disarmed(func)))
            return True

    def _execute_disarmed(self, func):
        try:
            return func()
        finally:
            with self._lock:
                self._target = None
                self._target_at = 0.0
                self._armed = False
                self._arm_pending = False

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
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
            if self._feedback.valid:
                self._feedback = PLCFeedback()
            # Commands from the old connection must not be replayed on reconnect.
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break

    def _retry_connect(self, now):
        if not HAS_SNAP7 and self._requires_snap7:
            if not self._snap7_warned:
                self._snap7_warned = True
                self._connect_attempted = True
                self._messages.put(SNAP7_ERROR)
            return
        if now - self._last_retry < RECONNECT_INTERVAL or self._stop_event.is_set():
            return
        self._last_retry = now
        was_connected = self._ever_connected
        self._connect_attempted = True
        if self._client.connect(quiet=True):
            self._ever_connected = True
            self._last_plc_heartbeat = None
            self._echo_change_at = now
            self._echo_seen = False
            self._messages.put("PLC 已重新连接" if was_connected else "PLC 已连接")
        elif self._requires_snap7:
            self._messages.put(f"PLC 连接失败：{self._client.last_error}")

    def _publish(self, status):
        now = time.monotonic()
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
                x_home=status["x_home"], y_home=status["y_home"],
                x_err=status["x_err"], y_err=status["y_err"],
                busy=status["busy"], comm_lost=status["comm_lost"],
                axes_ready=status["axes_ready"], plc_echo_ok=echo_ok,
                group_stop=status["group_stop"],
                comm_lost_latch=status["comm_lost_latch"],
                kinematics_error=status["kinematics_error"],
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
        try:
            client.clear_trigger()
        except Exception as exc:
            self._messages.put(f"PLC 清除运动触发失败：{exc}")
            client.disconnect()
            self._invalidate_feedback()
            return
        # 触发必须是"先低够久、再抬起来"，PLC 的 P 沿才抓得到。
        # 但读状态/发心跳本来就要花约 3.3ms，那段时间触发也是低的 —— 所以
        # 只在真正要下发时才把**剩余**的压低时间等满，不要无条件等一整个
        # trigger_low_time（那会让每个周期白白多花掉这段时间）。
        cleared_at = time.monotonic()
        status = client.read_status_and_position()
        if status is None or not client.connected:
            self._invalidate_feedback()
            return
        client.keep_alive()
        if not client.connected or self._stop_event.is_set():
            self._invalidate_feedback()
            return
        feedback = self._publish(status)
        if feedback is None:
            return
        safe = self._motion_safe(feedback)
        with self._lock:
            # Serious faults revoke even a pending arm request. During the initial
            # heartbeat/enable ramp, hold pending without allowing any movement.
            fault = (feedback.axis_fault or feedback.comm_lost or feedback.comm_lost_latch
                     or feedback.group_stop or feedback.kinematics_error != 0
                     or (not feedback.plc_echo_ok
                         and time.monotonic() - self._echo_change_at >= ECHO_TIMEOUT)
                     or (self._armed and not safe))
            if fault:
                self._armed = False
                self._arm_pending = False
                self._target = None
                self._target_at = 0.0
            elif safe and self._arm_pending:
                self._armed = True
                self._arm_pending = False
            target = (self._target if safe and self._armed
                      and not self._stop_event.is_set()
                      and time.monotonic() - self._target_at <= TARGET_MAX_AGE else None)
        # Network I/O is never under the lock. Cancellation of an already executing
        # db_write cannot be guaranteed; no subsequent target remains queued.
        if target is not None:
            # 把"触发低电平"的剩余时间补满，然后才抬起来下发。
            remaining_low = self.trigger_low_time - (time.monotonic() - cleared_at)
            if remaining_low > 0.0 and self._stop_event.wait(remaining_low):
                return
            client.send_linear_move(*target)
            # 现场诊断累计。整个累计块必须容错 —— 诊断是"非必要"的，
            # 绝不能让它的任何异常把 worker 线程打死（那会让后续命令全部失效）。
            try:
                state = client.last_send_state
                if state == "sent":
                    self._stats_counters["sent"] += 1
                    mode = getattr(client, "last_buffer_mode", None)
                    if mode in (0, 5):
                        self._stats_counters["mode0" if mode == 0 else "mode5"] += 1
                        self._stats_dists.append(
                            float(getattr(client, "last_send_dist_mm", 0.0) or 0.0))
                elif state == "deadzone":
                    self._stats_counters["deadzone"] += 1
                else:
                    self._stats_counters["other"] += 1
            except Exception:
                pass
        if not client.connected:
            self._invalidate_feedback()
        self.cycle_count += 1

    def _run(self):
        client = self._client
        try:
            while not self._stop_event.is_set():
                cycle_start = time.monotonic()
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
                remaining = self.period - (time.monotonic() - cycle_start)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        finally:
            self._invalidate_feedback()
            if client.connected:
                try:
                    client.clear_trigger()
                except Exception as exc:
                    self._messages.put(f"PLC 清除运动触发失败，请现场确认轴状态：{exc}")
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
