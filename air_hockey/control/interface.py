"""PLCInterface：阻塞式 S7 传输 + 运动指令的坐标换算、死区与终点下发。

只由 PLCLink 的工作线程调用。
"""

from __future__ import annotations

import ctypes
import struct
import time

from . import protocol
from .limits import (
    DEFAULT_LIMITS, DEFAULT_ZONE, ClampStats,
    game_to_physical, game_to_physical_raw, physical_to_game, validated_zone,
)
from .protocol import (
    AXIS_ENABLE_BITS, AXIS_HOME_BITS, AXIS_NAMES, AXIS_RESET_BITS, COMM_AXES_READY_BIT,
    COMM_BYTE, COMM_HEARTBEAT_BIT, COMM_KIN_ERROR_OFFSET, COMM_KIN_REACTION_OFFSET,
    COMM_LINK_LOST_BIT, COMM_MC_ERRID_OFFSET, COMM_MC_ERR_BIT, COMM_MC_ERR_OFFSET,
    COMM_MC_REMAINING_OFFSET, COMM_PLC_HEARTBEAT_BIT, COMM_READ_SIZE,
    COMM_READ_START, COMM_SEQ_ACK_OFFSET, COMM_SEQ_OFFSET,
    COMM_X_ERRID_OFFSET, COMM_X_ERR_BIT,
    COMM_Y_ERRID_OFFSET, COMM_Y_ERR_BIT, COMM_Y_ERR_OFFSET, DB_COMM, DB_MOTION,
    CYLINDER_BYTE, CYLINDER_START_BIT,
    GROUP_COMM_LOST_LATCH_BIT, GROUP_CTRL_BYTE, GROUP_RESET_BIT, GROUP_STOP_BIT,
    HOMING_BYTE, MOTION_BLOCK_START, MOTION_BUSY_BYTE, MOTION_CMD_BYTE,
    MOTION_STATUS_BYTE, STATUS_READ_SIZE, STATUS_READ_START, TRIGGER_BYTE,
    TRIGGER_LINEAR_BIT, X_POS_OFFSET, Y_POS_OFFSET,
)

DEAD_ZONE_MM = 3.0

# 只下发 AI 的终点，不切中间点。
# PLC 是 X/Y 各一个 MC_MoveAbsolute，新序号到来时同一实例重新触发 = 中止当前定位、改去新终点。
# 轴忙时新终点离上次下发的终点不够远就不发，避免 AI 目标的小抖动让轴反复中止重启。
# ponytail: 10mm 是估计值；实机若反应迟钝就调小，若又出现一顿一顿就调大。
ENDPOINT_RESEND_MM = 10.0


class PLCInterface:
    """Blocking S7 transport, exclusively operated by PLCLink's worker."""

    def __init__(self, ip="192.168.0.64", rack=0, slot=1, zone=None, limits=None):
        self.ip, self.rack, self.slot = ip, rack, slot
        # 轴的实际位置（游戏坐标）和 busy，由 PLCLink 每周期从反馈更新，_prepare_move 的死区判据要用
        self.actual_game = None
        self.actual_busy = False
        self.zone = validated_zone(DEFAULT_ZONE if zone is None else zone)
        # 命令包络（PLCLink 会用它自己的那份覆盖，和 zone 一样）
        self.limits = DEFAULT_LIMITS if limits is None else limits
        self.clamp_stats = ClampStats()
        self._client = None
        self._connected = False
        self.last_heartbeat_time = 0.0
        self.heartbeat_state = False
        self.last_phys_x = None
        self.last_phys_y = None
        # 序号握手：每成功下发一条 +1。种子取当前时间，避免重启后恰好与 PLC 里记的"上次序号"相同、
        # 第一条指令被当成旧指令丢掉。
        self.sequence = int(time.time()) % 1000000 + 1
        self.last_send_state = "unknown"
        self.last_send_dist_mm = 0.0
        self.last_error = ""
        self._multi_read_failed = False

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
        self.last_phys_x = None
        self.last_phys_y = None
        return True

    def _write_bool(self, db, offset, bit, value):
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
                return self._read_multi_vars(specs)
            except Exception as exc:
                self._multi_read_failed = True
                print(f"[PLC] 多变量读取不可用，回退逐块读取: {exc}")
        return [self._client.db_read(db, start, size) for db, start, size in specs]

    def _read_multi_vars(self, specs):
        """一次请求读多个区域：直接调底层 Cli_ReadMultiVars，把两次往返（约 9ms）压成一次。"""
        library = getattr(self._client, "_library", None)
        pointer = getattr(self._client, "_pointer", None)
        if library is None or pointer is None:
            raise RuntimeError("snap7 native handle unavailable")
        from snap7.types import S7AreaDB, S7DataItem, S7WLByte

        count = len(specs)
        items = (S7DataItem * count)()
        buffers = []
        for index, (db, start, size) in enumerate(specs):
            buffer = (ctypes.c_ubyte * size)()
            buffers.append(buffer)
            items[index].Area = S7AreaDB
            items[index].WordLen = S7WLByte
            items[index].DBNumber = db
            items[index].Start = start
            items[index].Amount = size
            items[index].pData = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        result = library.Cli_ReadMultiVars(pointer, items, count)
        if result != 0:
            raise RuntimeError(self._client.error_text(result))
        for index, buffer in enumerate(buffers):
            if items[index].Result != 0:
                raise RuntimeError(f"item {index} failed: {items[index].Result}")
        return [bytearray(bytes(buffer)) for buffer in buffers]

    def clear_trigger(self):
        self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, False)

    def _write_command_bit(self, byte, bit, value=None):
        """写 DB1 命令位；value=None 时发一个 50ms 脉冲（写 1 再写 0）。失败返回 False 并标记断线。"""
        if not self.connected or bit is None:
            return False
        try:
            if value is None:
                self._write_bool(DB_MOTION, byte, bit, True)
                time.sleep(0.05)
                self._write_bool(DB_MOTION, byte, bit, False)
            else:
                self._write_bool(DB_MOTION, byte, bit, value)
            return True
        except Exception:
            self._connected = False
            return False

    def set_axis_enable(self, axis, enabled):
        """单轴使能 / 撤销（axis 为 "x" / "y"）；两个轴都使能才允许下发运动（见 PLCLink._motion_safe）。"""
        return self._write_command_bit(MOTION_CMD_BYTE, AXIS_ENABLE_BITS.get(axis), bool(enabled))

    def reset_axis(self, axis):
        """单轴复位（清该轴的报警）。"""
        return self._write_command_bit(MOTION_CMD_BYTE, AXIS_RESET_BITS.get(axis))

    def home_axis(self, axis):
        """单轴回零（脉冲 DB1.1.0 / 1.1）。

        当前 TIA 里 MC_Home 的 Mode 没接线 = 0：轴不会动，PLC 直接把当前位置设为原点。
        轴不在参考位置时按下，坐标系随之错位、命令包络失效，只剩硬件限位开关兜底。
        """
        return self._write_command_bit(HOMING_BYTE, AXIS_HOME_BITS.get(axis))

    def enable_axes(self):
        return all(self.set_axis_enable(axis, True) for axis in AXIS_NAMES)

    def disable_axes(self):
        # 用列表而不是生成器：两个轴都要写，不能在第一个失败时短路
        return all([self.set_axis_enable(axis, False) for axis in AXIS_NAMES])

    def home_axes(self):
        return all(self.home_axis(axis) for axis in AXIS_NAMES)

    def axes_reset(self):
        """两个单轴复位 + 组复位。"""
        ok = all([self.reset_axis(axis) for axis in AXIS_NAMES])
        return self._write_command_bit(GROUP_CTRL_BYTE, GROUP_RESET_BIT) and ok

    def set_cylinder(self, on):
        """DB1.118.0 气缸启动：开关量，状态由界面持有（PLC 无回读）。"""
        self._write_bool(DB_MOTION, CYLINDER_BYTE, CYLINDER_START_BIT, bool(on))
        return True

    def game_to_physical(self, x, y):
        return game_to_physical(self.zone, self.limits, x, y, self.clamp_stats)

    def game_to_physical_raw(self, x, y):
        return game_to_physical_raw(self.zone, x, y)

    def physical_to_game(self, x, y):
        return physical_to_game(self.zone, x, y)

    def _prepare_move(self, game_x, game_y):
        """坐标换算 + 死区：只下发 AI 终点。

        返回 None（不发）或 (物理x, 物理y, 步距)。步距在**第一次**下发时是 None。
        """
        try:
            phys_x, phys_y = self.game_to_physical(game_x, game_y)
        except (ValueError, OverflowError):
            self.last_send_state = "invalid"
            return None
        # 执行中：新终点离上次下发的终点不到 ENDPOINT_RESEND_MM 就不再排队。
        if (self.actual_busy and self.last_phys_x is not None
                and max(abs(phys_x - self.last_phys_x),
                        abs(phys_y - self.last_phys_y)) < ENDPOINT_RESEND_MM):
            self.last_send_state = "pending"
            return None
        # 起点：优先用上次真正发出去的位置，还没有就用轴的实际位置。
        # 只用局部变量：self.last_phys_* 只在 _commit_move 里更新，写失败时跟踪才能保持清空。
        origin_x, origin_y = self.last_phys_x, self.last_phys_y
        actual = self.actual_game
        if origin_x is None and actual is not None:
            try:
                origin_x, origin_y = self.game_to_physical_raw(actual[0], actual[1])
            except (ValueError, OverflowError):
                origin_x = origin_y = None
        if origin_x is None:
            return phys_x, phys_y, None
        distance = max(abs(phys_x - origin_x), abs(phys_y - origin_y))
        # 同点：轴也已到位就跳过；空闲但未到位时重发，执行中的同点已在上面跳过。
        if distance < DEAD_ZONE_MM:
            actual_gap = None
            if actual is not None:
                try:
                    ax, ay = self.game_to_physical_raw(actual[0], actual[1])
                    actual_gap = max(abs(phys_x - ax), abs(phys_y - ay))
                except (ValueError, OverflowError):
                    actual_gap = None
            if actual_gap is None or actual_gap < DEAD_ZONE_MM:
                self.last_send_state = "deadzone"
                return None
            self.last_send_state = "deadzone-resend"
        return phys_x, phys_y, distance

    def _write_payload(self, phys_x, phys_y):
        # ponytail: 工程档案没导出 Position 的数组下标，Y 同时写 [2]、[3]；TIA 里确认后去掉多余那个。
        self._client.db_write(DB_MOTION, MOTION_BLOCK_START,
                              struct.pack(">ffff", phys_x, phys_y, phys_y, 0.0))

    def _commit_move(self, phys_x, phys_y, distance):
        self.last_phys_x, self.last_phys_y = phys_x, phys_y
        if distance is not None:
            self.last_send_dist_mm = distance
        self.last_send_state = "sent"

    def send_linear_move(self, game_x, game_y):
        """序号握手：先写目标块，再写递增的序号，PLC 比较序号后执行。

        顺序不能反，PLC 看到新序号时目标必须已经稳定；序号只在整条写成功后才推进。
        """
        if not self.connected:
            self.last_send_state = "disconnected"
            return False
        prepared = self._prepare_move(game_x, game_y)
        if prepared is None:
            return self.last_send_state != "invalid"
        phys_x, phys_y, distance = prepared
        try:
            self._write_payload(phys_x, phys_y)
            candidate = (self.sequence + 1) & 0x7FFFFFFF
            self._client.db_write(DB_MOTION, COMM_SEQ_OFFSET, struct.pack(">i", candidate))
            self.sequence = candidate
            self._commit_move(phys_x, phys_y, distance)
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

            def word(offset):
                # 按无符号读，16#8001 这类码才和 TIA 显示一致
                return struct.unpack_from(">H", db18, offset)[0]

            return {
                "x_en": bool(status & 0x04), "y_en": bool(status & 0x08),
                "x_pos": struct.unpack_from(">f", db1, X_POS_OFFSET)[0],
                "y_pos": struct.unpack_from(">f", db1, Y_POS_OFFSET)[0],
                "x_err": bool(comm & (1 << COMM_X_ERR_BIT)),
                "y_err": bool(db18[COMM_Y_ERR_OFFSET] & (1 << COMM_Y_ERR_BIT)),
                "x_errid": word(COMM_X_ERRID_OFFSET),
                "y_errid": word(COMM_Y_ERRID_OFFSET),
                # DB18.32 以后（运动指令 Error/ErrorId、ErrorDetail、RemainingDistance）1211C 程序还没写，恒为 0
                "mc_error": bool(db18[COMM_MC_ERR_OFFSET] & (1 << COMM_MC_ERR_BIT)),
                "mc_errid": word(COMM_MC_ERRID_OFFSET),
                "busy": bool(db1[MOTION_BUSY_BYTE] & 1),
                "plc_heartbeat": bool(comm & (1 << COMM_PLC_HEARTBEAT_BIT)),
                "comm_lost": bool(comm & (1 << COMM_LINK_LOST_BIT)),
                "axes_ready": bool(comm & (1 << COMM_AXES_READY_BIT)),
                "group_stop": bool(group & (1 << GROUP_STOP_BIT)),
                "comm_lost_latch": bool(group & (1 << GROUP_COMM_LOST_LATCH_BIT)),
                "kinematics_error": word(COMM_KIN_ERROR_OFFSET),
                "kin_reaction": struct.unpack_from(">i", db18, COMM_KIN_REACTION_OFFSET)[0],
                # PLC 已处理到的序号（DB1.94），和本地序号一比就是"落后几条"
                "seq_ack": struct.unpack_from(">i", db1, COMM_SEQ_ACK_OFFSET)[0],
                "mc_remaining": struct.unpack_from(">d", db18, COMM_MC_REMAINING_OFFSET)[0],
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
