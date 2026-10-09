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
        assert struct.unpack_from(">ddddi", move) == (195.0, 0.0, 190.0, 0.0, 5)
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
        for x, expected_mode in ((0.0, 5), (2.0, None), (8.0, 5), (25.0, 0)):
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
        assert client.send_linear_move(1e9, 1e9)
        assert struct.unpack_from(">dddd", client.transport.moves[-1]) == (
            protocol.PHYSICAL_X_MAX, 0.0, protocol.PHYSICAL_Y_MAX, 0.0)
        before = len(client.transport.moves)
        assert not client.send_linear_move(float("nan"), 0)
        assert client.last_send_state == "invalid"
        assert len(client.transport.moves) == before
    finally:
        client.disconnect()


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

        client.transport.db[18][56:58] = (801).to_bytes(2, "big", signed=True)
        assert until(lambda: not link.armed and link.feedback.kinematics_error == 801)
        before = len(client.transport.moves)
        link.set_target(400, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == before
        client.transport.db[18][56:58] = b"\x00\x00"
        assert until(lambda: link.feedback.kinematics_error == 0)
        link.set_target(450, 180)
        cycles = link.cycle_count
        assert until(lambda: link.cycle_count >= cycles + 3)
        assert len(client.transport.moves) == before  # 报警清除不能自动复发运动
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(450, 180)
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
            assert len(operations) == (1 if failure == "payload" else 2)
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
