"""S7 运动学运动目标与断故障互锁的全内存传输测试；不触达实机。"""

import struct
import time

from air_hockey import core_config as core
from air_hockey.control.plc import PLCInterface, PLCLink


class MemoryS7:
    def __init__(self):
        self.db = {1: bytearray(200), 18: bytearray(64)}
        self.db[1][1] = 0x0C  # enabled；home Done 脉冲已经结束
        self.db[1][56] = 0x02  # 其他调用者的圆弧启动位
        self.db[18][0] = 0x08  # 轴已就绪
        self.moves = []
        self.connected = False

    def connect(self, ip, rack, slot):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def db_read(self, db, start, size):
        return bytearray(self.db[db][start:start + size])

    def db_write(self, db, start, data):
        data = bytes(data)
        self.db[db][start:start + len(data)] = data
        if (db, start) == (18, 0):
            self.db[18][0] = (self.db[18][0] & ~2) | ((self.db[18][0] & 1) << 1)
        if (db, start) == (1, 20):
            self.moves.append(data)

    def read_multi_vars(self, items):
        return 0, [self.db[item["db_number"]][item["start"]:item["start"] + item["size"]]
                   for item in items]


class MemoryPLC(PLCInterface):
    def __init__(self, ip="memory", rack=0, slot=1):
        super().__init__(ip, rack, slot)
        self.transport = MemoryS7()

    def connect(self, quiet=False):
        self.transport.connect(self.ip, self.rack, self.slot)
        self._client = self.transport
        self._connected = True
        self._trigger_byte = self.transport.db_read(1, 56, 1)[0]
        self.clear_trigger()
        self.reset_tracking()
        self._comm_byte0 = None
        return True


def until(predicate, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_measured_kinematics_component_order_and_circular_bit_survive():
    plc = MemoryPLC()
    plc.connect()
    plc.zone = (core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_CENTER_Y)
    assert plc.send_linear_move(core.RINK_RIGHT, core.RINK_CENTER_Y)
    move = plc.transport.moves[-1]
    assert struct.unpack_from(">ddddi", move) == (195.0, 0.0, 190.0, 0.0, 5)
    assert move[36] == 0x03
    assert len(move) == 37
    assert plc.transport.db[1][56] == 0x03
    assert plc.transport.db[1][0] == 0  # 不能覆盖使能与复位控制位
    plc.disconnect()


def test_real_motion_requires_manual_arm_and_revokes_on_801_alarm():
    client = MemoryPLC()
    link = PLCLink("memory", period=0.03, client_factory=lambda *args: client)
    link.start()
    try:
        assert until(lambda: link.feedback.valid and link.feedback.plc_echo_ok)
        link.set_target(300, 180)
        time.sleep(0.12)
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
        time.sleep(0.12)
        assert len(client.transport.moves) == before
        client.transport.db[18][56:58] = b"\x00\x00"
        assert until(lambda: link.feedback.kinematics_error == 0)
        link.set_target(450, 180)
        time.sleep(0.12)
        assert len(client.transport.moves) == before  # 报警清除不能自动复发运动
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(450, 180)
        assert until(lambda: len(client.transport.moves) > before)
    finally:
        assert link.stop(timeout=2.0)
    assert not link.connected
    assert not client.transport.db[1][0] & 0xC0


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
        time.sleep(0.12)
        assert len(client.transport.moves) == 1
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(450, 180)
        assert until(lambda: len(client.transport.moves) == 2)
    finally:
        assert link.stop(timeout=2.0)


def test_home_and_reset_pulses_preserve_plc_changes_to_neighbor_bits():
    client = MemoryPLC()
    client.connect()
    write = client.transport.db_write
    changed = set()

    def plc_updates_status_during_pulse(db, start, data):
        write(db, start, data)
        if (db, start) == (1, 1) and data[0] & 3 and "home" not in changed:
            client.transport.db[1][1] |= 0x30  # PLC 更新 MC_HOME.Done 脉冲
            changed.add("home")
        if (db, start) == (1, 0) and data[0] & 0x30 and "axis" not in changed:
            client.transport.db[1][0] |= 0x40  # HMI 同时更改使能位
            changed.add("axis")
        if (db, start) == (1, 140) and data[0] & 0x08 and "group" not in changed:
            client.transport.db[1][140] |= 0x04  # PLC 同时置停止所有运动位
            changed.add("group")

    client.transport.db_write = plc_updates_status_during_pulse
    try:
        assert client.home_axes()
        assert client.transport.db[1][1] & 0x3C == 0x3C
        assert client.transport.db[1][1] & 0x03 == 0
        assert client.axes_reset()
        assert client.transport.db[1][0] & 0x40
        assert client.transport.db[1][0] & 0x30 == 0
        assert client.transport.db[1][140] & 0x04
        assert client.transport.db[1][140] & 0x08 == 0
    finally:
        client.disconnect()


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
            time.sleep(0.15)
            assert len(client.transport.moves) == before
            assert link.enable_axes() and until(lambda: link.armed)
            link.set_target(target, 180)
            assert until(lambda: len(client.transport.moves) > before)
    finally:
        assert link.stop(timeout=2.0)


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
            time.sleep(0.2)
            assert not link.armed
            assert client.transport.moves == []
        assert link.enable_axes()
        assert until(lambda: link.armed)
        link.set_target(400, 200)
        assert until(lambda: len(client.transport.moves) == 1)
    finally:
        assert link.stop(timeout=2.0)
