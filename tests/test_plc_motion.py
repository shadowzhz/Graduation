"""S7 运动学运动目标与断故障互锁的全内存传输测试；不触达实机。"""

import struct
import time
import ctypes
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from air_hockey import core_config as core
from air_hockey.control import protocol
from air_hockey.control.interface import PLCInterface
from air_hockey.control.limits import (
    COORD_EPSILON, DEFAULT_LIMITS, DEFAULT_ZONE, TRAVEL_X_MAX, TRAVEL_X_MIN, TRAVEL_Y_MAX, TRAVEL_Y_MIN,
    MotionLimits,
)
from air_hockey.control.link import ECHO_TIMEOUT, TARGET_MAX_AGE, PLCLink


@contextmanager
def memory_binding():
    with patch.multiple(protocol,
                        Areas=SimpleNamespace(DB=SimpleNamespace(value=0x84)),
                        WordLen=SimpleNamespace(Bit=SimpleNamespace(value=0x01)),
                        HAS_SNAP7=True,
                        snap7=SimpleNamespace(client=SimpleNamespace(Client=MemoryS7)),
                        create=True):
        yield


class MemoryS7:
    def __init__(self):
        self.db = {1: bytearray(200), 18: bytearray(64)}
        self.db[1][1] = 0x0C  # enabled；home Done 脉冲已经结束
        self.db[1][56] = 0x02  # 其他调用者的圆弧启动位
        self.db[18][0] = 0x08  # 轴已就绪
        self.moves = []
        self.connected = False
        self._library = self
        self._pointer = ctypes.c_void_p(1)
        self.operations = []
        self.pending_payload = None
        self.before_bit = None
        self.fail_bit = None
        self.fail_payload = False
        self.fail_transition = False
        self.fail_sequence = False
        self.echo = True

    def Cli_WriteArea(self, pointer, area, db, start, amount, wordlen, buffer):
        assert pointer is (getattr(self, "_s7_client", None) or self._pointer)
        assert area == protocol.Areas.DB.value
        assert amount == 1 and wordlen == protocol.WordLen.Bit.value
        value = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8))[0]
        return self.write_area(protocol.Areas.DB, db, start, bytearray([value]), protocol.WordLen.Bit)

    def write_area(self, area, db, start, data, word_len):
        assert area is protocol.Areas.DB and word_len is protocol.WordLen.Bit
        assert len(data) == 1
        offset, bit = divmod(start, 8)
        value = bool(data[0])
        self.operations.append(("bit", db, offset, bit, value))
        if self.before_bit is not None:
            self.before_bit(db, offset, bit, value)
        if self.fail_bit == (db, offset, bit, value):
            return 0x00200000
        mask = 1 << bit
        self.db[db][offset] = ((self.db[db][offset] | mask) if value
                               else (self.db[db][offset] & ~mask))
        if (db, offset, bit) == (18, 0, 0) and self.echo:
            self.db[18][0] = (self.db[18][0] & ~2) | (int(value) << 1)
        return 0

    def error_text(self, result):
        return "BIT write failed: %s" % result

    def connect(self, ip, rack, slot):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def db_read(self, db, start, size):
        return bytearray(self.db[db][start:start + size])

    def db_write(self, db, start, data):
        data = bytes(data)
        if (db, start, len(data)) == (1, protocol.COMM_SEQ_OFFSET, 4):
            # ★ 序号握手：序号写下来就代表"一条新指令"。
            #   真实 PLC 也是这个语义：先拿到已稳定的目标块，看到新序号就执行。
            self.operations.append(("sequence", db, start, data))
            if self.fail_sequence:
                raise RuntimeError("sequence write failed")
            self.db[db][start:start + len(data)] = data
            # 模拟 PLC 把已处理的序号回写到 DB1.64（诊断用"落后几条"）
            self.db[1][protocol.COMM_SEQ_ACK_OFFSET:protocol.COMM_SEQ_ACK_OFFSET + 4] = data
            assert self.pending_payload is not None
            self.moves.append(self.pending_payload)
            self.pending_payload = None
            return
        if (db, start, len(data)) == (1, protocol.COMM_TRANSITION_OFFSET, 8):
            # ★ 精磨距离（TransitionParameter[1]，DB1.DBD150）：随步距自适应写回。
            #   它不触发运动，只是给 PLC 一个"混合起点"的值，所以单独记一类操作。
            self.operations.append(("transition", db, start, data))
            if self.fail_transition:
                raise RuntimeError("transition write failed")
            self.db[db][start:start + len(data)] = data
            return
        assert (db, start, len(data)) == (1, 20, 36)
        self.operations.append(("payload", db, start, data))
        if self.fail_payload:
            raise RuntimeError("payload write failed")
        self.db[db][start:start + len(data)] = data
        self.pending_payload = data

    def read_multi_vars(self, items):
        return 0, [self.db[item["db_number"]][item["start"]:item["start"] + item["size"]]
                   for item in items]


class ModernMemoryS7(MemoryS7):
    def __init__(self):
        super().__init__()
        self._lib, self._s7_client = self._library, self._pointer
        del self._library, self._pointer


class PureMemoryS7(MemoryS7):
    def __init__(self):
        super().__init__()
        del self._library, self._pointer
        self.protocol = SimpleNamespace(
            build_write_request=self.build_write_request,
            check_write_response=self.check_write_response)

    def build_write_request(self, area, db, start, word_len, data):
        assert area is protocol.Areas.DB and word_len is protocol.WordLen.Bit
        request = bytearray(29)
        struct.pack_into(">BBHHHH", request, 0, 0x32, 1, 0, 1, 14, 5)
        request[10:16] = bytes.fromhex("0501120a1001")
        struct.pack_into(">HHB", request, 16, 1, db, area.value)
        request[21:24] = start.to_bytes(3, "big")
        request[25] = 3
        struct.pack_into(">H", request, 26, 8)
        request[28] = data[0]
        return bytes(request)

    def _send_receive(self, request):
        if struct.unpack_from(">H", request, 26)[0] != 1:
            return {"result": 7}
        db = struct.unpack_from(">H", request, 18)[0]
        start = int.from_bytes(request[21:24], "big")
        result = self.write_area(protocol.Areas.DB, db, start, request[28:], protocol.WordLen.Bit)
        return {"result": result}

    def check_write_response(self, response):
        if response["result"]:
            raise RuntimeError(self.error_text(response["result"]))


class MemoryPLC(PLCInterface):
    def __init__(self, ip="memory", rack=0, slot=1):
        super().__init__(ip, rack, slot)
        self.transport = MemoryS7()

    def connect(self, quiet=False):
        self.transport.connect(self.ip, self.rack, self.slot)
        self._client = self.transport
        self._connected = True
        try:
            self._write_bool(18, 0, 0, False)
            self.clear_trigger()
            self.reset_tracking()
            self.last_heartbeat_time = 0.0
            self.heartbeat_state = False
            return True
        except Exception:
            self.disconnect()
            return False


def until(predicate, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@memory_binding()
def test_measured_kinematics_component_order_and_circular_bit_survive():
    plc = MemoryPLC()
    try:
        assert plc.connect()
        plc.zone = (core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_CENTER_Y)
        assert plc.send_linear_move(core.RINK_RIGHT, core.RINK_CENTER_Y)
        move = plc.transport.moves[-1]
        # 用符号常量而不是写死数值：这个断言的本意是验证**分量顺序**
        # （X, 0, Y, 0 —— 实机 TO_Kinematics 用的是分量 1 和 3），
        # 而不是验证行程的具体毫米数。写死数值会在改行程时误报。
        # 上限是命令包络（limits.command_*），比机械行程内缩 margin_mm。
        assert struct.unpack_from(">ddddi", move) == (
            DEFAULT_LIMITS.command_x_max, 0.0, DEFAULT_LIMITS.command_y_max, 0.0, 5)
        assert len(move) == 36
        assert plc.transport.operations[-3:] == [
            ("payload", 1, 20, move),
            # 终点模式：舍入间距写 -1 = 由 TO 取最大可能值
            ("transition", 1, protocol.COMM_TRANSITION_OFFSET, struct.pack(">d", -1.0)),
            ("sequence", 1, protocol.COMM_SEQ_OFFSET, struct.pack(">i", plc.sequence))]
        assert plc.transport.db[1][56] == 0x02  # 圆弧启动位不被碰
        assert plc.transport.db[1][0] == 0  # 不能覆盖使能与复位控制位
    finally:
        plc.disconnect()


@memory_binding()
def test_reset_tracking_reissues_same_target_without_reporting_failure():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        link.clear_target()
        link.drain_messages()

        assert link.reset_tracking()
        assert until(lambda: client._last_phys_x is None)
        assert not link.drain_messages()

        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 2)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_atomic_trigger_preserves_deadzone_clipping_and_buffer_modes():
    client = MemoryPLC()
    try:
        assert client.connect()
        # ★ buffer_mode 只发 5（混合）：TO_Kinematics 只认 1/2/5，1 会停稳，
        #   轴版的 0（立即中止）在运动机构版里是**非法值**，作业会被拒绝。
        for x, expected_mode in ((0.0, 5), (2.0, None), (8.0, 5), (25.0, 5)):
            game = client.physical_to_game(x, 0.0)
            before = len(client.transport.moves)
            client.clear_trigger()
            assert client.send_linear_move(*game)
            if expected_mode is None:
                assert client.last_send_state == "deadzone"
                assert len(client.transport.moves) == before
            else:
                assert len(client.transport.moves) == before + 1
                assert struct.unpack_from(">i", client.transport.moves[-1], 32)[0] == expected_mode
        client.clear_trigger()
        before = len(client.transport.moves)
        before_operations = len(client.transport.operations)
        previous_target = (client._last_phys_x, client._last_phys_y)
        assert not client.send_linear_move(1e9, 1e9)
        assert client.last_send_state == "invalid"
        assert len(client.transport.moves) == before
        assert not any(operation[0] == "payload"
                       for operation in client.transport.operations[before_operations:])
        assert (client._last_phys_x, client._last_phys_y) == previous_target
        for invalid in (float("nan"), float("inf"), -float("inf")):
            assert not client.send_linear_move(invalid, 0)
            assert client.last_send_state == "invalid"
            assert len(client.transport.moves) == before
    finally:
        client.disconnect()


@memory_binding()
def test_plc_game_zone_edges_and_epsilon_tolerance():
    client = MemoryPLC()
    try:
        assert client.connect()
        left, right, top, bottom = DEFAULT_ZONE
        # 场地边缘映射到 **COMMAND_*（命令限位）**，不是 TRAVEL_*（机械行程/限位开关）。
        # 两者的差值就是 margin_mm —— 命令不主动去碰限位开关。
        for game_point, physical_point in (
            ((left, top), (DEFAULT_LIMITS.command_x_min, DEFAULT_LIMITS.command_y_min)),
            ((right, bottom), (DEFAULT_LIMITS.command_x_max, DEFAULT_LIMITS.command_y_max)),
        ):
            client.reset_tracking()
            client.clear_trigger()
            assert client.send_linear_move(*game_point)
            move = client.transport.moves[-1]
            assert struct.unpack_from(">dddd", move) == (physical_point[0], 0.0, physical_point[1], 0.0)

        client.reset_tracking()
        client.clear_trigger()
        assert client.send_linear_move(left - COORD_EPSILON / 2,
                                       top - COORD_EPSILON / 2)
        move = client.transport.moves[-1]
        assert struct.unpack_from(">dddd", move) == (
            DEFAULT_LIMITS.command_x_min, 0.0, DEFAULT_LIMITS.command_y_min, 0.0,
        )

        before = len(client.transport.moves)
        client.clear_trigger()
        assert not client.send_linear_move(left - COORD_EPSILON * 2, top)
        assert client.last_send_state == "invalid"
        assert len(client.transport.moves) == before
    finally:
        client.disconnect()


def test_plc_link_rejects_out_of_zone_targets_without_replacing_target():
    client = MemoryPLC()
    link = PLCLink("memory", client_factory=lambda *args: client)
    left, _right, top, _bottom = link.zone
    assert link.set_target(left, top)
    previous_target = link._target
    assert not link.set_target(1e9, 1e9)
    assert link._target == previous_target
    assert not link.set_target(float("nan"), 0)
    assert not client.transport.moves


@memory_binding()
def test_real_motion_requires_manual_arm_and_revokes_on_801_alarm():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        link.set_target(300, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert client.transport.moves == []
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        # DB1.1.4/1.5 是 MC_HOME.Done 瞬态脉冲：均为零时仍允许轴就绪运动。
        assert client.transport.db[1][1] & 0x30 == 0

        link.drain_messages()
        client.transport.db[18][56:58] = (801).to_bytes(2, "big", signed=True)
        assert until(lambda: not link.armed and link.feedback.kinematics_error == 801)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count > cycles)
        assert len(link.drain_messages()) == 1
        before = len(client.transport.moves)
        link.set_target(400, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == before
        assert link.drain_messages() == []
        client.transport.db[18][56:58] = b"\x00\x00"
        assert until(lambda: link.feedback.kinematics_error == 0)
        link.set_target(450, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == before  # 报警清除不能自动复发运动
        assert link.enable_axes()
        assert until(lambda: link.armed)
        # 报警中断的旧目标没有完成；人工重新授权后，同一目标也必须能重发。
        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) > before)
    finally:
        assert link.stop(timeout=2.0)
    assert not link.connected
    assert not client.transport.db[1][0] & 0xC0


@memory_binding()
def test_reconnection_discards_previous_motion_and_requires_new_manual_enable():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        client._connected = False  # 假传输模拟网络丢失，无物理网络活动
        assert until(lambda: not link.armed and not link.feedback.valid)
        link.set_target(400, 180)
        assert until(lambda: link.connected and link.feedback.valid and link.feedback.plc_echo_ok, 4.0)
        link.set_target(450, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == 1
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(450, 180)
        assert until(lambda: len(client.transport.moves) == 2)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_home_and_reset_pulses_preserve_plc_changes_to_neighbor_bits():
    client = MemoryPLC()
    client.connect()

    def plc_updates_neighbors(db, offset, bit, value):
        masks = {(18, 0): 0x14, (1, 56): 0x82, (1, 0): 0x08,
                 (1, 1): 0x30, (1, 140): 0x14}
        client.transport.db[db][offset] |= masks[(db, offset)]

    client.transport.before_bit = plc_updates_neighbors
    try:
        client.keep_alive()
        assert client.transport.db[18][0] & 0x14 == 0x14
        client.clear_trigger()
        assert client.send_linear_move(300, 180)
        assert client.transport.db[1][56] & 0x82 == 0x82
        assert client.enable_axes()
        assert client.disable_axes()
        assert client.transport.db[1][0] & 0x08
        assert client.home_axes()
        assert client.transport.db[1][1] & 0x3C == 0x3C
        assert client.transport.db[1][1] & 0x03 == 0
        assert client.axes_reset()
        assert client.transport.db[1][0] & 0x08
        assert client.transport.db[1][0] & 0x30 == 0
        assert client.transport.db[1][140] & 0x14 == 0x14
        assert client.transport.db[1][140] & 0x08 == 0
    finally:
        client.disconnect()


@memory_binding()
def test_home_and_reset_revoke_motion_until_rearmed():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        for command, target in ((link.home_axes, 450), (link.axes_reset, 200)):
            assert command()
            assert not link.armed
            before = len(client.transport.moves)
            link.set_target(target, 180)
            cycles = link.cycle_count
            assert until(lambda: link.cycle_count >= cycles + 3)
            assert len(client.transport.moves) == before
            assert link.enable_axes() and until(lambda: link.armed)
            link.set_target(target, 180)
            assert until(lambda: len(client.transport.moves) > before)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_queued_enable_before_home_or_reset_does_not_rearm_after_command():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        for command in (link.home_axes, link.axes_reset):
            assert link.enable_axes()
            assert command()
            link.set_target(400, 200)
            cycles = link.cycle_count
            assert until(lambda: link.cycle_count >= cycles + 3)
            assert not link.armed
            assert client.transport.moves == []
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(400, 200)
        assert until(lambda: len(client.transport.moves) == 1)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_every_safety_interlock_revokes_and_never_auto_rearms():
    cases = ((18, 0, 0x10, True, "x_err"),
             (18, 4, 0x01, True, "y_err"),
             (18, 0, 0x04, True, "comm_lost"),
             (1, 140, 0x10, True, "comm_lost_latch"),
             (1, 140, 0x04, True, "group_stop"),
             (18, 0, 0x08, False, "axes_ready"),
             (1, 1, 0x04, False, "x_en"),
             (1, 1, 0x08, False, "y_en"))
    for db, offset, mask, fault_value, field in cases:
        client = MemoryPLC()
        link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
        link.start()
        try:
            assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
            assert link.enable_axes() and until(lambda: link.armed)
            link.set_target(300, 180)
            assert until(lambda: len(client.transport.moves) == 1)
            original = client.transport.db[db][offset] & mask
            if fault_value:
                client.transport.db[db][offset] |= mask
            else:
                client.transport.db[db][offset] &= ~mask
            assert until(lambda: getattr(link.feedback, field) == fault_value
                         and not link.armed)
            before = len(client.transport.moves)
            link.set_target(450, 180)
            cycles = link.cycle_count
            assert until(lambda: link.cycle_count >= cycles + 3)
            assert len(client.transport.moves) == before
            client.transport.db[db][offset] = (
                client.transport.db[db][offset] & ~mask) | original
            assert until(lambda: getattr(link.feedback, field) != fault_value)
            link.set_target(450, 180)
            cycles = link.cycle_count
            assert until(lambda: link.cycle_count >= cycles + 3)
            assert not link.armed and len(client.transport.moves) == before
            assert link.enable_axes() and until(lambda: link.armed)
            link.set_target(450, 180)
            assert until(lambda: len(client.transport.moves) == before + 1)
        finally:
            assert link.stop(timeout=2.0)


@memory_binding()
def test_heartbeat_freeze_revokes_motion_and_requires_manual_rearm():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        client.transport.echo = False
        assert until(lambda: not link.feedback.plc_echo_ok and not link.armed,
                     ECHO_TIMEOUT + 2.0)
        link.set_target(450, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert client.transport.moves == []
        client.transport.echo = True
        assert until(lambda: link.feedback.plc_echo_ok)
        assert not link.armed
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_stale_target_does_not_emit_motion_or_revoke_arm():
    client = MemoryPLC()
    link = PLCLink("memory", client_factory=lambda *args: client)
    try:
        assert client.connect()
        link._echo_seen = True
        link._echo_change_at = time.perf_counter()
        link._last_plc_heartbeat = False
        link._armed = True
        link.set_target(450, 180)
        with link._lock:
            link._target_at = time.perf_counter() - TARGET_MAX_AGE - 0.01
        link._cycle()
        assert client.transport.moves == []
        assert link.armed
    finally:
        client.disconnect()


@memory_binding()
def test_payload_or_sequence_failure_does_not_publish_or_cache_success():
    for failure in ("payload", "sequence"):
        client = MemoryPLC()
        try:
            assert client.connect()
            assert client.send_linear_move(300, 180)
            previous = (client._last_phys_x, client._last_phys_y, client.last_buffer_mode, client.sequence)
            before = len(client.transport.moves)
            start = len(client.transport.operations)
            client.transport.fail_payload = failure == "payload"
            client.transport.fail_sequence = failure == "sequence"
            assert not client.send_linear_move(450, 180)
            assert not client.connected and client.last_send_state == "disconnected"
            assert len(client.transport.moves) == before
            # 序号没写成功就不能推进，否则下一条会跳号
            assert (client._last_phys_x, client._last_phys_y,
                    client.last_buffer_mode, client.sequence) == previous
            operations = client.transport.operations[start:]
            # 每次下发按顺序写：目标块 →（精磨距离）→ 序号。
            #   终点模式的精磨距离固定 -1，第一次下发已写过，这次不再重写。
            #   · payload 失败在第 1 步 → 只有 1 次操作
            #   · 序号失败 → 目标块已写出去，但没有多出一次运动
            assert len(operations) == (1 if failure == "payload" else 2)
        finally:
            client.disconnect()


@memory_binding()
def test_native_bit_failure_fails_connection_and_pulses_closed():
    for bit in ((18, 0, 0, False), (1, 56, 0, False)):
        client = MemoryPLC()
        try:
            client.transport.fail_bit = bit
            assert not client.connect()
            assert not client.connected
            assert client.transport.moves == []
        finally:
            client.disconnect()
    for command, bit in (("enable_axes", (1, 0, 7, True)),
                         ("disable_axes", (1, 0, 7, False)),
                         ("home_axes", (1, 1, 1, False)),
                         ("axes_reset", (1, 140, 3, False))):
        client = MemoryPLC()
        try:
            assert client.connect()
            client.transport.fail_bit = bit
            assert not getattr(client, command)()
            assert not client.connected
        finally:
            client.disconnect()


@memory_binding()
def test_production_connection_probes_native_bits_before_accepting_motion():
    factory = protocol.snap7.client.Client
    try:
        for failure in (None, (18, 0, 0, False), (1, 56, 0, False)):
            transport = MemoryS7()
            transport.fail_bit = failure
            protocol.snap7.client.Client = lambda: transport
            client = PLCInterface()
            try:
                assert client.connect(quiet=True) == (failure is None)
                assert client.connected == (failure is None)
                assert transport.moves == []
                if failure is None:
                    assert transport.operations == [
                        ("bit", 18, 0, 0, False), ("bit", 1, 56, 0, False)]
                else:
                    assert not client.send_linear_move(300, 180)
            finally:
                client.disconnect()
    finally:
        protocol.snap7.client.Client = factory


@memory_binding()
def test_connection_and_motion_preserve_other_bits_with_modern_clients():
    for factory in (ModernMemoryS7, PureMemoryS7):
        transport = factory()
        transport.db[1][0] = 0x0D
        transport.db[18][0] = 0x18
        with patch.object(protocol.snap7.client, "Client", return_value=transport):
            client = PLCInterface("memory")
            try:
                assert client.connect(quiet=True)
                assert transport.db[18][0] == 0x18
                assert client.enable_axes()
                assert transport.db[1][0] == 0xCD
                assert client.send_linear_move(300, 180)
                assert transport.db[1][56] == 0x02
                assert client.disable_axes()
                assert transport.db[1][0] == 0x0D
                assert struct.unpack_from(">d", transport.moves[0])[0] == client.game_to_physical(300, 180)[0]
            finally:
                client.disconnect()
        transport = factory()
        transport.fail_bit = (18, 0, 0, False)
        with patch.object(protocol.snap7.client, "Client", return_value=transport):
            client = PLCInterface("memory")
            try:
                assert not client.connect(quiet=True)
                assert not client.connected
                assert not client.send_linear_move(300, 180)
                assert transport.moves == []
            finally:
                client.disconnect()


@memory_binding()
def test_native_motion_error_invalidates_worker_feedback_and_arm():
    for failure in ("payload", "sequence", "heartbeat"):
        client = MemoryPLC()
        link = PLCLink("memory", client_factory=lambda *args: client)
        try:
            assert client.connect()
            link._echo_seen = True
            link._echo_change_at = time.perf_counter()
            link._last_plc_heartbeat = False
            link._armed = True
            link.set_target(450, 180)
            if failure == "payload":
                client.transport.fail_payload = True
            elif failure == "sequence":
                client.transport.fail_sequence = True
            else:
                client.transport.fail_bit = (18, 0, 0, True)
            link._cycle()
            assert not client.connected and not link.armed
            assert not link.feedback.valid and link._target is None
            assert client.transport.moves == []
            assert client._last_phys_x is None and client.last_buffer_mode is None
        finally:
            client.disconnect()


@memory_binding()
def test_each_send_advances_sequence_once():
    """序号握手：写目标块 + 递增序号，PLC 看到序号变了才执行。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        first = struct.unpack_from(">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0]
        assert first == client.sequence > 0

        assert link.set_target(340, 200)
        assert until(lambda: len(client.transport.moves) == 2)
        assert struct.unpack_from(
            ">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0] == first + 1
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_sequence_does_not_advance_on_deadzone_skip():
    """目标没变 **且轴已到位** 时才跳过 —— 死区是压抖动的，不是压重发的。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        # ★ 起始时把"轴的实际位置"放在别处 —— 第一条指令才会真的发出去。
        #   （若一开始就放在目标上，新逻辑会正确地判定"已经到位、不用发"。）
        #   必须在 enable 之前写好，否则和工作线程的周期抢时序（测试会抖）。
        px, py = client.game_to_physical(250, 150)
        struct.pack_into(">d", client.transport.db[1], 2, px)
        struct.pack_into(">d", client.transport.db[1], 10, py)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) >= 1)
        # ★ 轴走到目标处：写进去之后要**等反馈确认**，否则下一句可能在
        #   工作线程读到新位置之前就判定了（会偶发失败）。
        px, py = client.game_to_physical(300, 180)
        struct.pack_into(">d", client.transport.db[1], 2, px)
        struct.pack_into(">d", client.transport.db[1], 10, py)
        assert until(lambda: abs(link.feedback.x - 300.0) < 1.0
                     and abs(link.feedback.y - 180.0) < 1.0)
        sent = struct.unpack_from(">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0]
        # 到这里轴已到位、目标也稳定下来，再取基准计数并让 worker 空转几个周期
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 4)
        moves = len(client.transport.moves)
        # 目标只挪 0.4mm（< DEAD_ZONE_MM）→ 才该被死区挡掉
        assert link.set_target(300.4, 180.0)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 4)
        assert len(client.transport.moves) == moves
        assert struct.unpack_from(
            ">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0] == sent
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_target_is_resent_when_axis_never_arrived():
    """PLC 空闲但实际位置未到目标时仍要重发，不能只凭上次目标跳过。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        moves = len(client.transport.moves)
        # PLC 空闲，轴的实际位置离目标很远（模拟"根本没走到"），而目标不再变化
        far_x, far_y = client.game_to_physical(300, 320)
        struct.pack_into(">d", client.transport.db[1], 2, far_x)
        struct.pack_into(">d", client.transport.db[1], 10, far_y)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 4)
        # 空闲但未到位时必须继续下发，不能因为"目标没变"就一条都不发
        assert len(client.transport.moves) > moves
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_busy_target_is_not_requeued_but_changed_and_idle_targets_are_sent():
    client = MemoryPLC()
    client.transport.db[1][protocol.MOTION_BUSY_BYTE] |= 1
    link = PLCLink("memory", period=0.02, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(350, 250)
        assert until(lambda: bool(client.transport.moves))
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 4)
        assert len(client.transport.moves) == 1

        assert link.set_target(350.1, 250.1)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == 1

        assert link.set_target(450, 220)
        assert until(lambda: len(client.transport.moves) >= 2)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == 2
        assert link.armed and link.feedback.busy

        client.transport.db[1][protocol.MOTION_BUSY_BYTE] &= ~1
        assert link.set_target(450, 220)
        assert until(lambda: len(client.transport.moves) > 2)
        assert not link.feedback.busy and link.armed
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_single_axis_enable_only_touches_that_axis_and_blocks_motion():
    """★ 单轴使能 / 撤销必须安全：

    · 只写**对应那个轴**的命令位（DB1.0 bit6=X / bit7=Y），不能顺手写另一个
    · 单轴使能后**绝不能**开始下发运动 —— _motion_safe 要求 x_en 和 y_en 同时为真
      （现场用它把报 801 的那个轴单独撤下来，另一个继续可用）
    """
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.set_axis_enable("x", True)
        assert until(lambda: (client.transport.db[1][0] >> 6) & 1 == 1)
        # ★ 只动了 X 的命令位，Y 的没被碰
        assert (client.transport.db[1][0] >> 7) & 1 == 0, "单轴使能写到了另一个轴"

        # 模拟 PLC 回报：只有 X 已使能（DB1.1 bit2=X / bit3=Y）
        client.transport.db[1][1] = (client.transport.db[1][1] | 0x04) & ~0x08
        assert until(lambda: link.feedback.x_en and not link.feedback.y_en)

        # ★ 单轴使能下发目标 → 不能有任何运动、不能 arm
        assert link.set_target(300, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 5)
        assert client.transport.moves == [], "单轴使能居然下发了运动！"
        assert not link.armed

        # 再把 Y 也使能 → 才允许运动
        assert link.set_axis_enable("y", True)
        assert until(lambda: (client.transport.db[1][0] >> 7) & 1 == 1)
        client.transport.db[1][1] |= 0x08
        assert until(lambda: link.armed)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_single_axis_disable_clears_only_that_axis_and_disarms():
    """撤下单个轴：只清自己的位，并且立刻停止运动授权。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_axis_enable("y", False)
        assert until(lambda: (client.transport.db[1][0] >> 7) & 1 == 0)
        assert (client.transport.db[1][0] >> 6) & 1 == 1, "撤 Y 不该动 X"
        assert until(lambda: not link.armed)
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_diagnostics_runs_without_client_local_and_reports_transition():
    """★ 回归：diagnostics() 里把 client 当局部变量用过 → NameError。

    后果比看上去严重：它是在**每个 [stats] 行**上崩，现场表现成
    "实时处理失败：name 'client' is not defined"，看起来像整条视觉管线坏了。
    """
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        # 离线、还没下发过：不能崩，且要如实说"未发"
        text = link.diagnostics()
        assert "精磨 未发" in text, text

        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) >= 1)
        text = link.diagnostics()
        # 下发过之后必须报出舍入间距：终点模式固定 -1（TO 最大）
        assert "精磨 TO最大" in text and "关（" not in text, text
        assert client.last_transition_mm == -1.0
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_transition_write_failure_keeps_motion_connected():
    client = MemoryPLC()
    try:
        assert client.connect()
        client.actual_game = (300, 140)
        client.transport.fail_transition = True
        assert client.send_linear_move(350, 220)
        assert client.send_linear_move(400, 250)
        assert struct.unpack_from(">i", client.transport.db[1], protocol.COMM_SEQ_ACK_OFFSET)[0] == client.sequence
        assert client.connected and client.transport.connected
        assert not client.transition_enabled
        assert client.last_error
        assert client.last_send_state == "sent"
        assert len(client.transport.moves) == 2
        assert sum(operation[0] == "transition" for operation in client.transport.operations) == 1
    finally:
        client.disconnect()


@memory_binding()
def test_position_inside_physical_limits_can_arm_outside_command_margin():
    limits = MotionLimits(margin_mm=40)
    link = None
    try:
        client = MemoryPLC()
        struct.pack_into(">dd", client.transport.db[1], 2, 0, -100)
        link = PLCLink("memory", period=0.03,
                       client_factory=lambda *args: client, limits=limits)
        link.start()
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert not link.armed and not client.transport.moves
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: client.last_send_state == "sent")
        px, _, py, _, _ = struct.unpack(">ddddi", client.transport.moves[-1])
        assert limits.command_x_min <= px <= limits.command_x_max
        assert limits.command_y_min <= py <= limits.command_y_max
        assert link.armed
    finally:
        if link is not None:
            assert link.stop(timeout=2.0)


@memory_binding()
def test_position_past_travel_does_not_trip_software_limit():
    """软件不再设软限位：行程越界交给硬件限位开关（触发后 TIA 报轴故障再撤销）。"""
    for x, y in ((TRAVEL_X_MIN - 0.1, 0),
                 (TRAVEL_X_MAX + 0.1, 0),
                 (0, TRAVEL_Y_MIN - 0.1),
                 (0, TRAVEL_Y_MAX + 0.1)):
        client = MemoryPLC()
        link = PLCLink("memory", period=0.03,
                       client_factory=lambda *args: client)
        link.start()
        try:
            assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
            assert link.enable_axes() and until(lambda: link.armed)
            struct.pack_into(">dd", client.transport.db[1], 2, x, y)
            cycles = link.cycle_count
            assert until(lambda: link.cycle_count >= cycles + 3)
            assert link.armed
            # 从越界位置往回走：指令不比当前位置更往外
            assert link.set_target(300, 180)
            assert until(lambda: client.last_send_state == "sent")
            px, _, py, _, _ = struct.unpack(">ddddi", client.transport.moves[-1])
            assert min(DEFAULT_LIMITS.command_x_min, x) <= px <= max(DEFAULT_LIMITS.command_x_max, x)
            assert min(DEFAULT_LIMITS.command_y_min, y) <= py <= max(DEFAULT_LIMITS.command_y_max, y)
        finally:
            assert link.stop(timeout=2.0)


@memory_binding()
def test_worker_cycle_feeds_latency_report_and_motion_snapshot():
    """真实工作线程每个周期都要喂延迟统计；记录用的快照要带实际位置和最近下发点。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03,
                   client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        link.latency_report()                       # 清掉授权前的窗口
        assert link.set_target(300, 300)
        assert until(lambda: client.last_send_state == "sent")
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        text, _hints = link.latency_report()
        assert "S7 读 -" not in text and "轴落后指令" in text, text
        # 内存 PLC 的轴不会动：指令发出去了、实际位置没跟上，必须统计成"落后"
        assert "轴落后指令 0% 周期" not in text, text
        snapshot = link.motion_snapshot()
        assert snapshot is not None and snapshot["armed"]
        assert snapshot["sent"] is not None and snapshot["sent"]["age"] >= 0.0
        assert snapshot["age"] >= 0.0
    finally:
        assert link.stop(timeout=2.0)
