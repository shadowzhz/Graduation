"""S7 运动学运动目标与断故障互锁的全内存传输测试；不触达实机。"""

import struct
import time
import ctypes
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from air_hockey import core_config as core
from air_hockey.control import plc as protocol
from air_hockey.control.plc import PLCInterface, PLCLink


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
        if (db, offset, bit, value) == (1, 56, 0, True):
            assert self.pending_payload is not None
            self.moves.append(self.pending_payload)
            self.pending_payload = None
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
            # ★ 序号握手：序号写下来就代表"一条新指令"，等价于脉冲模式里抬触发位。
            #   真实 PLC 也是这个语义：先拿到已稳定的目标块，看到新序号就执行。
            self.operations.append(("sequence", db, start, data))
            if self.fail_payload:
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
        # 上限是 COMMAND_*（命令限位），比 TIA 软限位内缩 MOTION_MARGIN_MM。
        assert struct.unpack_from(">ddddi", move) == (
            protocol.COMMAND_X_MAX, 0.0, protocol.COMMAND_Y_MAX, 0.0, 5)
        assert len(move) == 36
        assert plc.transport.operations[-2:] == [
            ("payload", 1, 20, move), ("bit", 1, 56, 0, True)]
        assert plc.transport.db[1][56] == 0x03
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
        # ★ buffer_mode 只允许 TO_Kinematics 认的 1/2/5：
        #     1 = 附加运动（先走完当前、停下、再走新的）
        #     5 = 滤波处理（达到混合距离时混合，较高速度）
        #   ⚠️ 轴版的 0（立即中止）在运动机构版里是**非法值**，作业会被拒绝。
        for x, expected_mode in ((0.0, 5), (2.0, None), (8.0, 5), (25.0, 1)):
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
        assert not client.transport.db[1][56] & 1
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
        left, right, top, bottom = protocol.DEFAULT_ZONE
        # 场地边缘映射到 **COMMAND_*（命令限位）**，不是 PHYSICAL_*（TIA 软限位）。
        # 两者的差值就是 MOTION_MARGIN_MM —— 代码的目标必须比软限位保守，
        # 否则命令正好落在软限位上，稍有超调就触发轴故障。
        for game_point, physical_point in (
            ((left, top), (protocol.COMMAND_X_MIN, protocol.COMMAND_Y_MIN)),
            ((right, bottom), (protocol.COMMAND_X_MAX, protocol.COMMAND_Y_MAX)),
        ):
            client.reset_tracking()
            client.clear_trigger()
            assert client.send_linear_move(*game_point)
            move = client.transport.moves[-1]
            assert struct.unpack_from(">dddd", move) == (physical_point[0], 0.0, physical_point[1], 0.0)

        client.reset_tracking()
        client.clear_trigger()
        assert client.send_linear_move(left - protocol.COORD_EPSILON / 2,
                                       top - protocol.COORD_EPSILON / 2)
        move = client.transport.moves[-1]
        assert struct.unpack_from(">dddd", move) == (
            protocol.COMMAND_X_MIN, 0.0, protocol.COMMAND_Y_MIN, 0.0,
        )

        before = len(client.transport.moves)
        client.clear_trigger()
        assert not client.send_linear_move(left - protocol.COORD_EPSILON * 2, top)
        assert client.last_send_state == "invalid"
        assert len(client.transport.moves) == before
        assert not client.transport.db[1][56] & 1
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
    assert not client.transport.db[1][56] & 1


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
                     protocol.ECHO_TIMEOUT + 2.0)
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
        link._echo_change_at = time.monotonic()
        link._last_plc_heartbeat = False
        link._armed = True
        link.set_target(450, 180)
        with link._lock:
            link._target_at = time.monotonic() - protocol.TARGET_MAX_AGE - 0.01
        link._cycle()
        assert client.transport.moves == []
        assert link.armed
    finally:
        client.disconnect()


@memory_binding()
def test_payload_or_native_trigger_failure_does_not_publish_or_cache_success():
    for failure in ("payload", "trigger"):
        client = MemoryPLC()
        try:
            assert client.connect()
            assert client.send_linear_move(300, 180)
            previous = (client._last_phys_x, client._last_phys_y, client.last_buffer_mode)
            client.clear_trigger()
            before = len(client.transport.moves)
            start = len(client.transport.operations)
            client.transport.fail_payload = failure == "payload"
            if failure == "trigger":
                client.transport.fail_bit = (1, 56, 0, True)
            assert not client.send_linear_move(450, 180)
            assert not client.connected and client.last_send_state == "disconnected"
            assert len(client.transport.moves) == before
            assert (client._last_phys_x, client._last_phys_y,
                    client.last_buffer_mode) == previous
            operations = client.transport.operations[start:]
            # 每次下发会按顺序写三个东西：目标块 → 精磨距离 → 触发位。
            #   · payload 失败在第 1 步 → 只有 1 次操作
            #   · trigger 失败在第 3 步 → 前 2 次都已经写出去
            # 关键是**没有多出一次运动**（上面断言 moves 不变），精磨距离只是
            # 一个"混合起点"的数值、幂等，写出去不影响安全。
            assert len(operations) == (1 if failure == "payload" else 3)
            assert client.transport.db[1][56] & 1 == 0
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
                assert transport.db[1][56] == 0x03
                client.clear_trigger()
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
    for failure in ("payload", "trigger", "clear", "heartbeat"):
        client = MemoryPLC()
        link = PLCLink("memory", client_factory=lambda *args: client)
        try:
            assert client.connect()
            link._echo_seen = True
            link._echo_change_at = time.monotonic()
            link._last_plc_heartbeat = False
            link._armed = True
            link.set_target(450, 180)
            if failure == "payload":
                client.transport.fail_payload = True
            else:
                client.transport.fail_bit = {
                    "trigger": (1, 56, 0, True),
                    "clear": (1, 56, 0, False),
                    "heartbeat": (18, 0, 0, True),
                }[failure]
            link._cycle()
            assert not client.connected and not link.armed
            assert not link.feedback.valid and link._target is None
            assert client.transport.moves == []
            assert client._last_phys_x is None and client.last_buffer_mode is None
        finally:
            client.disconnect()


@contextmanager
def _clocked_plc(period=0.01, pulse=0.01):
    clock = [2.0]
    edges = []

    class ClockEvent:
        callback = None
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, seconds):
            clock[0] += max(0.0, seconds)
            callback, self.callback = self.callback, None
            if callback is not None:
                callback()
            return self.stopped

    class TimedS7(MemoryS7):
        def write_area(self, *args):
            clock[0] += 0.002
            result = super().write_area(*args)
            _area, db, bit_offset, data, _word = args
            if (db, bit_offset) == (1, 56 * 8):
                edges.append((clock[0], bool(data[0]), bytes(self.db[1][20:56])))
            return result

        def read_multi_vars(self, items):
            clock[0] += 0.002
            return super().read_multi_vars(items)

        def db_write(self, *args):
            clock[0] += 0.002
            return super().db_write(*args)

    with memory_binding(), patch.object(protocol.time, "monotonic", side_effect=lambda: clock[0]):
        client = MemoryPLC()
        client.transport = TimedS7()
        assert client.connect()
        link = PLCLink("memory", period=period, trigger_low_time=pulse,
                       client_factory=lambda *args: client)
        link._stop_event = ClockEvent()
        link._echo_seen = True
        link._echo_change_at = clock[0]
        link._last_plc_heartbeat = False
        link._armed = True
        try:
            yield link, client, clock, edges
        finally:
            client.disconnect()


def test_native_trigger_pulses_survive_asynchronous_plc_scans():
    with _clocked_plc() as (link, client, clock, edges):
        points = [(200 + step * 10, 250) for step in range(20)]
        expected = [client.game_to_physical(*point) for point in points]
        for point in points:
            assert link.set_target(*point)
            start = clock[0]
            link._cycle()
            clock[0] += max(0.0, link.period - (clock[0] - start))
        link.clear_target()
        link._cycle()

        for phase in (0.0, 0.001, 0.003, 0.009):
            captured = []
            index = 0
            current = previous = False
            payload = None
            tick = 2.0 + phase
            while tick <= edges[-1][0] + 0.010:
                while index < len(edges) and edges[index][0] <= tick:
                    _stamp, current, payload = edges[index]
                    index += 1
                if current and not previous:
                    x, _unused1, y, _unused2 = struct.unpack_from(">dddd", payload)
                    captured.append((x, y))
                previous = current
                tick += 0.010
            assert captured == expected, (phase, len(captured), len(expected))


def test_target_changed_during_trigger_wait_sends_latest_position():
    with _clocked_plc() as (link, client, _clock, _edges):
        assert link.set_target(200, 250)
        link._stop_event.callback = lambda: link.set_target(400, 200)
        link._cycle()
        expected_x, expected_y = client.game_to_physical(400, 200)
        assert struct.unpack_from(">dddd", client.transport.moves[0]) == (expected_x, 0.0, expected_y, 0.0)


def test_target_cleared_during_trigger_wait_never_generates_motion():
    with _clocked_plc() as (link, client, _clock, _edges):
        assert link.set_target(200, 250)
        link._stop_event.callback = link.clear_target
        link._cycle()
        assert client.transport.moves == []
        assert client.transport.pending_payload is None
        assert client.transport.db[1][56] & 1 == 0


def test_stop_during_high_pulse_hold_prevents_the_next_command():
    with _clocked_plc() as (link, client, _clock, _edges):
        assert link.set_target(200, 250)
        link._cycle()
        before = list(client.transport.moves)
        assert link.set_target(400, 200)
        link._stop_event.callback = link.stop
        link._cycle()
        assert not link.armed and link._target is None
        assert client.transport.moves == before


@memory_binding()
def test_sequence_trigger_advances_sequence_instead_of_pulsing():
    """序号握手模式：写目标块 + 递增序号，不写触发位。

    这是替代脉冲触发的方案 —— 脉冲要求高低电平各跨过一个 PLC 扫描周期
    （实测 2~3ms、偶发 5ms），把周期锁在约 28ms（35Hz）；序号握手没有这个限制。
    """
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, sequence_trigger=True,
                   client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        first = struct.unpack_from(">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0]
        assert first == link._sequence > 0
        # ★ 序号模式绝不能再抬触发位 —— 否则 PLC 会被脉冲和序号双重触发
        assert client.transport.db[1][protocol.TRIGGER_BYTE] & 1 == 0

        assert link.set_target(340, 200)
        assert until(lambda: len(client.transport.moves) == 2)
        assert struct.unpack_from(
            ">i", client.transport.db[1], protocol.COMM_SEQ_OFFSET)[0] == first + 1
        assert client.transport.db[1][protocol.TRIGGER_BYTE] & 1 == 0
    finally:
        assert link.stop(timeout=2.0)


@memory_binding()
def test_sequence_does_not_advance_on_deadzone_skip():
    """目标没变 **且轴已到位** 时才跳过 —— 死区是压抖动的，不是压重发的。"""
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, sequence_trigger=True,
                   client_factory=lambda *args: client)
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
    """★ 回归"假死"：目标没变但轴没走到时，必须持续重发。

    现场症状：target(98,340)、轴停在(98,219)，下发 0.0Hz、死区跳过 312，
    机械臂就此卡住不动 —— 因为死区只比"上次发送的目标"，不比轴的实际位置。
    """
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, sequence_trigger=True,
                   client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        assert link.enable_axes() and until(lambda: link.armed)
        assert link.set_target(300, 180)
        assert until(lambda: len(client.transport.moves) == 1)
        moves = len(client.transport.moves)
        # 轴的实际位置离目标很远（模拟"根本没走到"），而目标不再变化
        far_x, far_y = client.game_to_physical(300, 320)
        struct.pack_into(">d", client.transport.db[1], 2, far_x)
        struct.pack_into(">d", client.transport.db[1], 10, far_y)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 4)
        # ★ 必须继续下发 —— 不能因为"目标没变"就一条都不发
        assert len(client.transport.moves) > moves
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
    link = PLCLink("memory", period=0.03, target_slew_mm=20.0,
                   blend_threshold_mm=20.0, client_factory=lambda *args: client)
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
        # 下发过之后必须报出实际的精磨距离（步距 20mm 以内 → 0.75 倍，且 < 步距）
        assert "精磨" in text and "关（" not in text, text
        value = client.last_transition_mm
        assert value is not None and 0.0 < value < 20.0, value
        # 精磨距离必须严格小于步距 —— 等于或大于就会把整段圆掉、到不了目标
        assert value < client.last_send_dist_mm + 1e-9, (value, client.last_send_dist_mm)
    finally:
        assert link.stop(timeout=2.0)
