"""AI -> ControlCommand -> PLC 控制适配层测试。

覆盖：
- AIDecision -> ControlCommand（方向、速度、坐标范围检查、输出限幅）
- ControlCommand -> PlcWriteRequest -> PLCInterface 字段映射与时间戳
- 非阻塞输出 worker（覆盖式最新请求、断连/失败处理、后台写入）
"""

import math
import time

from air_hockey.ai import AIDecision
from air_hockey import core_config as core
from air_hockey.control import PlcControlAdapter, PlcOutputWorker, PlcWriteRequest
from air_hockey.planning import ControlCommand
from air_hockey.prediction import PredictionState, TrajectoryPredictor
from game_state import CurlingState

# 必须与 冰壶仿真/plc_interface.py 的 write_game_state 形参完全一致
PLC_ARGUMENTS = {
    "ai_target_x",
    "ai_target_y",
    "ai_x",
    "ai_y",
    "stone_x",
    "stone_y",
    "stone_vx",
    "stone_vy",
    "player_score",
    "ai_score",
}


class FakePlc:
    """记录写入调用的假 PLC。"""

    def __init__(self, connected=True, result=True, raises=False):
        self._connected = connected
        self._result = result
        self._raises = raises
        self.writes = []

    @property
    def connected(self):
        return self._connected

    def write_game_state(self, **kwargs):
        if self._raises:
            raise RuntimeError("boom")
        self.writes.append(kwargs)
        return self._result


def _decision(target_x, target_y):
    return AIDecision(target_x=target_x, target_y=target_y, stalled_stone_phase="")


def _prediction(duration=1.0, endpoint=(300.0, 300.0)):
    return PredictionState(
        trajectory=[(300.0, 300.0), tuple(endpoint)],
        endpoint=tuple(endpoint),
        duration=duration,
        source_state=CurlingState(x=300.0, y=300.0),
    )


def _state(x=300.0, y=300.0, vx=0.0, vy=0.0):
    return CurlingState(x=x, y=y, vx=vx, vy=vy, timestamp=1.0, confidence=0.9, radius=14.0)


def test_write_request_as_kwargs_matches_plc_signature():
    request = PlcWriteRequest(
        ai_target_x=1.0, ai_target_y=2.0, ai_x=3.0, ai_y=4.0,
        stone_x=5.0, stone_y=6.0, stone_vx=7.0, stone_vy=8.0,
        player_score=1, ai_score=2, timestamp=9.0,
    )
    payload = request.as_kwargs()
    assert set(payload.keys()) == PLC_ARGUMENTS
    # 时间戳只用于本地上报，不进 PLC DB
    assert "timestamp" not in payload
    assert request.to_dict()["timestamp"] == 9.0


def test_build_command_direction_and_speed():
    adapter = PlcControlAdapter()
    command = adapter.build_command(_decision(400.0, 300.0), _state(), _prediction(1.0))

    assert isinstance(command, ControlCommand)
    assert abs(command.direction[0] - 1.0) < 1e-9
    assert abs(command.direction[1]) < 1e-9
    assert abs(math.hypot(*command.direction) - 1.0) < 1e-9
    assert abs(command.speed - 100.0) < 1e-9  # 距离 100 / 预测时长 1.0


def test_build_command_clamps_target_into_rink():
    adapter = PlcControlAdapter()
    low = adapter.build_command(_decision(-500.0, -500.0), _state(), _prediction())
    assert low.target_position == (core.RINK_LEFT + core.MALLET_RADIUS, core.RINK_TOP + core.MALLET_RADIUS)

    high = adapter.build_command(_decision(10_000.0, 10_000.0), _state(), _prediction())
    assert high.target_position == (core.RINK_RIGHT - core.MALLET_RADIUS, core.RINK_BOTTOM - core.MALLET_RADIUS)

    inside = adapter.build_command(_decision(320.0, 340.0), _state(), _prediction())
    assert inside.target_position == (320.0, 340.0)


def test_build_command_rejects_non_finite_target():
    adapter = PlcControlAdapter()
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            adapter.build_command(_decision(bad, 300.0), _state(), _prediction())
            assert False, "非有限目标应抛 ValueError"
        except ValueError:
            pass


def test_build_command_limits_single_step_jump():
    adapter = PlcControlAdapter(max_target_jump=10.0)
    first = adapter.build_command(_decision(300.0, 300.0), _state(), _prediction())
    assert first.target_position == (300.0, 300.0)

    second = adapter.build_command(_decision(400.0, 300.0), _state(), _prediction())
    assert abs(second.target_position[0] - 310.0) < 1e-9
    assert abs(second.target_position[1] - 300.0) < 1e-9

    adapter.reset()
    after_reset = adapter.build_command(_decision(400.0, 300.0), _state(), _prediction())
    assert after_reset.target_position == (400.0, 300.0)


def test_to_write_request_maps_context_and_timestamp():
    adapter = PlcControlAdapter()
    curling = _state(x=310.0, y=520.0, vx=110.0, vy=-70.0)
    command = adapter.build_command(_decision(350.0, 200.0), curling, _prediction())
    request = adapter.to_write_request(
        command,
        ai_position=(300.0, 600.0),
        curling_state=curling,
        timestamp=12.5,
        player_score=3,
        ai_score=4,
    )

    assert request.ai_target_x == command.target_position[0]
    assert request.ai_target_y == command.target_position[1]
    assert (request.ai_x, request.ai_y) == (300.0, 600.0)
    assert (request.stone_x, request.stone_y) == (310.0, 520.0)
    assert (request.stone_vx, request.stone_vy) == (110.0, -70.0)
    assert (request.player_score, request.ai_score) == (3, 4)
    assert request.timestamp == 12.5


def test_to_write_request_clamps_out_of_range_command():
    adapter = PlcControlAdapter()
    command = ControlCommand(target_position=(9999.0, -9999.0), direction=(1.0, 0.0), speed=10.0)
    request = adapter.to_write_request(command, ai_position=(300.0, 600.0), curling_state=_state(), timestamp=0.0)
    assert request.ai_target_x == core.RINK_RIGHT - core.MALLET_RADIUS
    assert request.ai_target_y == core.RINK_TOP + core.MALLET_RADIUS


def test_adapter_parameter_validation():
    for kwargs in ({"max_speed": 0.0}, {"speed_gain": 0.0}, {"min_travel_time": 0.0}, {"max_target_jump": 0.0}):
        try:
            PlcControlAdapter(**kwargs)
            assert False, f"{kwargs} 应该抛 ValueError"
        except ValueError:
            pass


def test_output_worker_keeps_only_latest_request():
    plc = FakePlc()
    worker = PlcOutputWorker(plc)
    first = PlcWriteRequest(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, timestamp=1.0)
    second = PlcWriteRequest(2, 2, 0, 0, 0, 0, 0, 0, 0, 0, timestamp=2.0)

    worker.submit(first)
    worker.submit(second)
    assert worker.write_pending() is True
    assert len(plc.writes) == 1
    assert plc.writes[0]["ai_target_x"] == 2.0
    assert worker.write_count == 1
    assert worker.last_request is second
    # 无新请求时不再写入
    assert worker.write_pending() is False
    assert len(plc.writes) == 1


def test_output_worker_skips_when_disconnected():
    plc = FakePlc(connected=False)
    worker = PlcOutputWorker(plc)
    worker.submit(PlcWriteRequest(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, timestamp=1.0))
    assert worker.write_pending() is False
    assert plc.writes == []
    assert worker.write_count == 0


def test_output_worker_counts_failures():
    failing = FakePlc(result=False)
    worker = PlcOutputWorker(failing)
    worker.submit(PlcWriteRequest(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, timestamp=1.0))
    assert worker.write_pending() is False
    assert worker.error_count == 1

    raising = FakePlc(raises=True)
    worker2 = PlcOutputWorker(raising)
    worker2.submit(PlcWriteRequest(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, timestamp=1.0))
    assert worker2.write_pending() is False
    assert worker2.error_count == 1


def test_output_worker_interval_validation():
    try:
        PlcOutputWorker(FakePlc(), interval=0.0)
        assert False, "interval=0 应抛 ValueError"
    except ValueError:
        pass


def test_output_worker_background_thread_writes():
    plc = FakePlc()
    worker = PlcOutputWorker(plc, interval=0.005)
    worker.start()
    assert worker.running
    worker.submit(PlcWriteRequest(1, 2, 3, 4, 5, 6, 7, 8, 0, 0, timestamp=1.0))

    deadline = time.monotonic() + 0.5
    while not plc.writes and time.monotonic() < deadline:
        time.sleep(0.005)

    worker.stop()
    assert not worker.running
    assert plc.writes and plc.writes[0]["ai_target_x"] == 1.0


def test_end_to_end_decision_to_plc_payload():
    curling = CurlingState(x=300.0, y=500.0, vx=120.0, vy=-80.0, timestamp=2.0)
    prediction = TrajectoryPredictor().predict(curling)
    adapter = PlcControlAdapter()
    decision = AIDecision(target_x=350.0, target_y=100.0, stalled_stone_phase="idle")

    command = adapter.build_command(decision, curling, prediction, timestamp=2.0)
    request = adapter.to_write_request(command, ai_position=(300.0, 600.0), curling_state=curling, timestamp=2.0)

    plc = FakePlc()
    worker = PlcOutputWorker(plc)
    worker.submit(request)
    assert worker.write_pending() is True

    written = plc.writes[0]
    assert set(written.keys()) == PLC_ARGUMENTS
    assert core.RINK_LEFT + core.MALLET_RADIUS <= written["ai_target_x"] <= core.RINK_RIGHT - core.MALLET_RADIUS
    assert core.RINK_TOP + core.MALLET_RADIUS <= written["ai_target_y"] <= core.RINK_BOTTOM - core.MALLET_RADIUS
    assert written["stone_vx"] == 120.0 and written["stone_vy"] == -80.0
