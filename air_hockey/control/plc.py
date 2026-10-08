"""S7-1500 TO_Kinematics DB1/DB18 protocol and non-blocking game link.

Only the PLCLink worker owns the transport. Coordinate mapping is per-link; feedback
is in the same game-coordinate zone as targets, independent of display scaling.
"""

from __future__ import annotations

import math
import queue
import struct
import threading
import time
from dataclasses import dataclass

from air_hockey import core_config

try:
    import snap7
    from snap7.type import Areas
except ImportError:
    snap7 = None
    Areas = None

HAS_SNAP7 = snap7 is not None

DB_MOTION = 1
DB_COMM = 18
MOTION_CMD_BYTE = 0
HOMING_BYTE = 1
MOTION_STATUS_BYTE = 1
MOTION_BLOCK_START = 20
MOTION_BLOCK_SIZE = 37
TRIGGER_BYTE = 56
TRIGGER_LINEAR_BIT = 0x01
TRIGGER_OTHER_BITS = 0xFE
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
PHYSICAL_X_MIN, PHYSICAL_X_MAX = -185.0, 195.0
PHYSICAL_Y_MIN, PHYSICAL_Y_MAX = -120.0, 190.0
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
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    px = PHYSICAL_X_MIN + (x - left) * (PHYSICAL_X_MAX - PHYSICAL_X_MIN) / (right - left)
    py = PHYSICAL_Y_MIN + (y - top) * (PHYSICAL_Y_MAX - PHYSICAL_Y_MIN) / (bottom - top)
    return (max(PHYSICAL_X_MIN, min(PHYSICAL_X_MAX, px)),
            max(PHYSICAL_Y_MIN, min(PHYSICAL_Y_MAX, py)))


def _physical_to_game(zone, x, y):
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    return (left + (x - PHYSICAL_X_MIN) * (right - left) / (PHYSICAL_X_MAX - PHYSICAL_X_MIN),
            top + (y - PHYSICAL_Y_MIN) * (bottom - top) / (PHYSICAL_Y_MAX - PHYSICAL_Y_MIN))


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
        self._comm_byte0 = None
        self._trigger_byte = None
        self._multi_read_failed = False
        self.round_trips = 0

    @property
    def connected(self):
        return self._connected

    def connect(self, quiet=False):
        if not HAS_SNAP7:
            return False
        try:
            self._client = snap7.client.Client()
            self._client.connect(self.ip, self.rack, self.slot)
            self._connected = True
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, False)
            self.round_trips += 1
            self._trigger_byte = self._client.db_read(DB_MOTION, TRIGGER_BYTE, 1)[0]
            self.clear_trigger()
            self.reset_tracking()
            self.last_heartbeat_time = 0.0
            self.heartbeat_state = False
            self._comm_byte0 = None
            return True
        except Exception as exc:
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
        self._comm_byte0 = None
        self._trigger_byte = None
        self.reset_tracking()

    def reset_tracking(self):
        self._last_phys_x = None
        self._last_phys_y = None

    def _write_bool(self, db, offset, bit, value):
        self.round_trips += 2
        data = self._client.db_read(db, offset, 1)
        byte = (data[0] | (1 << bit)) if value else (data[0] & ~(1 << bit))
        self._client.db_write(db, offset, bytes([byte]))

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
        # The circular-motion bit DB1.56.1 is not ours; retain the last read value.
        self.round_trips += 1
        self._client.db_write(DB_MOTION, TRIGGER_BYTE,
                              bytes([(self._trigger_byte or 0) & TRIGGER_OTHER_BITS]))

    def enable_axes(self):
        if not self.connected:
            return False
        try:
            self.round_trips += 2
            value = self._client.db_read(DB_MOTION, MOTION_CMD_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, MOTION_CMD_BYTE, bytes([value | 0xC0]))
            return True
        except Exception:
            self._connected = False
            return False

    def disable_axes(self):
        if not self.connected:
            return False
        try:
            self.round_trips += 2
            value = self._client.db_read(DB_MOTION, MOTION_CMD_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, MOTION_CMD_BYTE, bytes([value & ~0xC0]))
            return True
        except Exception:
            self._connected = False
            return False

    def home_axes(self):
        if not self.connected:
            return False
        try:
            self.round_trips += 4
            value = self._client.db_read(DB_MOTION, HOMING_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, HOMING_BYTE, bytes([value | 0x03]))
            time.sleep(0.05)
            current = self._client.db_read(DB_MOTION, HOMING_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, HOMING_BYTE, bytes([current & 0xFC]))
            return True
        except Exception:
            self._connected = False
            return False

    def axes_reset(self):
        if not self.connected:
            return False
        try:
            self.round_trips += 8
            value = self._client.db_read(DB_MOTION, MOTION_CMD_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, MOTION_CMD_BYTE, bytes([value | 0x30]))
            group = self._client.db_read(DB_MOTION, GROUP_CTRL_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, GROUP_CTRL_BYTE,
                                  bytes([group | (1 << GROUP_RESET_BIT)]))
            time.sleep(0.05)
            value = self._client.db_read(DB_MOTION, MOTION_CMD_BYTE, 1)[0]
            group = self._client.db_read(DB_MOTION, GROUP_CTRL_BYTE, 1)[0]
            self._client.db_write(DB_MOTION, MOTION_CMD_BYTE, bytes([value & ~0x30]))
            self._client.db_write(DB_MOTION, GROUP_CTRL_BYTE,
                                  bytes([group & ~(1 << GROUP_RESET_BIT)]))
            return True
        except Exception:
            self._connected = False
            return False

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    def send_linear_move(self, game_x, game_y):
        """Write the measured 37-byte [X,0,Y,0], mode, DB1.56.0 block."""
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
            self.last_send_dist_mm = distance
            buffer_mode = 0 if distance > BUFFER_MODE_IMMEDIATE_MM else 5
        else:
            buffer_mode = 5
        try:
            data = bytearray(MOTION_BLOCK_SIZE)
            # The machine's TO_Kinematics uses components 1=X, 3=Y, NOT [X,Y,0,0].
            struct.pack_into(">ddddi", data, 0, phys_x, 0.0, phys_y, 0.0, buffer_mode)
            data[36] = ((self._trigger_byte or 0) & TRIGGER_OTHER_BITS) | TRIGGER_LINEAR_BIT
            self.round_trips += 1
            self._client.db_write(DB_MOTION, MOTION_BLOCK_START, data)
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
            self._comm_byte0 = db18[COMM_BYTE]
            self._trigger_byte = db1[TRIGGER_BYTE]
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
            if self._comm_byte0 is None:
                self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, self.heartbeat_state)
            else:
                self.round_trips += 1
                byte = self._comm_byte0
                byte = (byte | 1) if self.heartbeat_state else (byte & ~1)
                self._client.db_write(DB_COMM, COMM_BYTE, bytes([byte]))
            self.last_heartbeat_time = now
        except Exception:
            self._connected = False


class PLCLink:
    """Single-owner worker; start/stop and target publication do no network I/O."""

    def __init__(self, ip, rack=0, slot=1, period=DEFAULT_PERIOD,
                 client_factory=None, zone=None):
        self.ip = ip
        self.zone = _validated_zone(DEFAULT_ZONE if zone is None else zone)
        self.period = max(0.005, float(period))
        if not math.isfinite(self.period):
            raise ValueError("PLC period must be finite")
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

    @property
    def ready(self):
        """Motion may be sent only while both armed and current status are safe."""
        return self.armed

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    def set_target(self, game_x, game_y):
        x, y = _finite_pair(game_x, game_y)
        with self._lock:
            if not self._stop_event.is_set():
                self._target = x, y
                self._target_at = time.monotonic()

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
                self._messages.put("未安装 snap7：球槌走本地模拟，请先 pip install python-snap7")
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
        except Exception:
            client.disconnect()
            self._invalidate_feedback()
            return
        if self._stop_event.wait(TRIGGER_LOW_TIME):
            return
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
            client.send_linear_move(*target)
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
                except Exception:
                    pass
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
