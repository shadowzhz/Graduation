"""AI 目标的 PLC 安全边界和事件唤醒、限频输出测试。"""

import math
import threading
import time

from air_hockey.ai import AIDecision
from air_hockey import core_config as core
from air_hockey.control import PlcControlAdapter, PlcOutputWorker, PlcWriteRequest
from game_state import CurlingState


class FakePlc:
    """记录实际写入时刻，使用条件变量等待后台线程完成。"""

    def __init__(self, connected=True, result=True, raises=False):
        self._connected = connected
        self._result = result
        self._raises = raises
        self.writes = []
        self.write_times = []
        self._written = threading.Condition()

    @property
    def connected(self):
        return self._connected

    def write_game_state(self, **kwargs):
        if self._raises:
            raise RuntimeError("boom")
        with self._written:
            self.writes.append(kwargs)
            self.write_times.append(time.perf_counter())
            self._written.notify_all()
        return self._result

    def wait_for_writes(self, count, timeout):
        with self._written:
            return self._written.wait_for(lambda: len(self.writes) >= count, timeout)


def _decision(target_x, target_y):
    return AIDecision(target_x=target_x, target_y=target_y, stalled_stone_phase="")


def _state(x=300.0, y=300.0, vx=0.0, vy=0.0):
    return CurlingState(x=x, y=y, vx=vx, vy=vy, timestamp=1.0, confidence=0.9, radius=14.0)


def _request(x):
    return PlcWriteRequest(x, 300.0, 300.0, 600.0, 300.0, 500.0, 0.0, 0.0, 0, 0, time.perf_counter())


def _expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("invalid control limit or target should be rejected")


def test_build_request_clamps_target_into_rink():
    adapter = PlcControlAdapter()
    def target(x, y):
        request = adapter.build_request(_decision(x, y), ai_position=(300, 600), curling_state=_state(), timestamp=2.0)
        return request.ai_target_x, request.ai_target_y

    assert target(-500.0, -500.0) == (core.RINK_LEFT + core.MALLET_RADIUS, core.RINK_TOP + core.MALLET_RADIUS)
    assert target(10_000.0, 10_000.0) == (core.RINK_RIGHT - core.MALLET_RADIUS, core.RINK_BOTTOM - core.MALLET_RADIUS)
    assert target(320.0, 340.0) == (320.0, 340.0)


def test_custom_bounds_cannot_expand_physical_plc_limits():
    adapter = PlcControlAdapter(bounds=(-1000, 10000, -1000, 10000), margin=0)
    request = adapter.build_request(_decision(-500, 10000), ai_position=(300, 600), curling_state=_state(), timestamp=2)
    assert (request.ai_target_x, request.ai_target_y) == (
        core.RINK_LEFT + core.MALLET_RADIUS, core.RINK_BOTTOM - core.MALLET_RADIUS
    )
    adapter = PlcControlAdapter(bounds=(250, 350, 250, 350), margin=10)
    request = adapter.build_request(_decision(100, 1000), ai_position=(300, 600), curling_state=_state(), timestamp=2)
    assert (request.ai_target_x, request.ai_target_y) == (260, 340)


def test_build_request_rejects_non_finite_target_without_advancing_jump_limit():
    adapter = PlcControlAdapter(max_target_jump=10.0)
    adapter.build_request(_decision(300, 300), ai_position=(300, 600), curling_state=_state(), timestamp=1)
    for bad in (float("nan"), float("inf"), float("-inf")):
        for x, y in ((bad, 300), (300, bad)):
            _expect_value_error(lambda: adapter.build_request(
                _decision(x, y), ai_position=(300, 600), curling_state=_state(), timestamp=2
            ))
    next_request = adapter.build_request(_decision(400, 300), ai_position=(300, 600), curling_state=_state(), timestamp=3)
    assert (next_request.ai_target_x, next_request.ai_target_y) == (310, 300)


def test_build_request_limits_single_step_jump_and_reset():
    adapter = PlcControlAdapter(max_target_jump=10.0)
    def target(x, y):
        request = adapter.build_request(_decision(x, y), ai_position=(300, 600), curling_state=_state(), timestamp=2)
        return request.ai_target_x, request.ai_target_y

    assert target(300, 300) == (300, 300)
    expected_step = 10 / math.sqrt(2)
    assert math.dist(target(400, 400), (300 + expected_step, 300 + expected_step)) < 1e-9
    assert math.dist(target(400, 400), (300 + 2 * expected_step, 300 + 2 * expected_step)) < 1e-9
    adapter.reset()
    assert target(400, 400) == (400, 400)


def test_build_request_maps_game_state_and_local_timestamp():
    adapter = PlcControlAdapter()
    curling = _state(x=310.0, y=520.0, vx=110.0, vy=-70.0)
    request = adapter.build_request(
        _decision(350.0, 200.0),
        ai_position=(300.0, 600.0),
        curling_state=curling,
        timestamp=12.5,
        player_score=3,
        ai_score=4,
    )

    assert (request.ai_target_x, request.ai_target_y) == (350, 200)
    assert (request.ai_x, request.ai_y) == (300.0, 600.0)
    assert (request.stone_x, request.stone_y) == (310.0, 520.0)
    assert (request.stone_vx, request.stone_vy) == (110.0, -70.0)
    assert (request.player_score, request.ai_score) == (3, 4)
    assert request.timestamp == 12.5
    assert request.as_kwargs() == {
        "ai_target_x": 350.0, "ai_target_y": 200.0, "ai_x": 300.0, "ai_y": 600.0,
        "stone_x": 310.0, "stone_y": 520.0, "stone_vx": 110.0, "stone_vy": -70.0,
        "player_score": 3, "ai_score": 4,
    }
    assert request.to_dict()["timestamp"] == 12.5


def test_adapter_parameter_validation():
    for bad in (0, -1, float("inf"), float("nan")):
        _expect_value_error(lambda: PlcControlAdapter(max_target_jump=bad))
    for bad in (float("nan"), float("inf"), -1):
        _expect_value_error(lambda: PlcControlAdapter(margin=bad))
    for bad_bounds in (
        (0, float("nan"), 0, 500),
        (0, 500, float("inf"), 500),
        (400, 300, 0, 500),
        (0, 500, 400, 300),
        (0, 500, 0),
        (1000, 2000, 1000, 2000),
    ):
        _expect_value_error(lambda: PlcControlAdapter(bounds=bad_bounds))


def test_output_worker_keeps_only_latest_request():
    plc = FakePlc()
    worker = PlcOutputWorker(plc)
    first = _request(1)
    second = _request(2)

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
    write_ms, frame_to_write_ms = worker.timing_samples()
    assert len(write_ms) == len(frame_to_write_ms) == 1
    assert write_ms[0] >= 0 and frame_to_write_ms[0] >= 0


def test_output_worker_skips_when_disconnected():
    plc = FakePlc(connected=False)
    worker = PlcOutputWorker(plc)
    worker.submit(_request(1))
    assert worker.write_pending() is False
    assert plc.writes == []
    assert worker.write_count == 0
    assert worker.timing_samples() == ([], [])


def test_output_worker_counts_failures():
    failing = FakePlc(result=False)
    worker = PlcOutputWorker(failing)
    worker.submit(_request(1))
    assert worker.write_pending() is False
    assert worker.error_count == 1
    assert len(worker.timing_samples()[0]) == 1
    assert worker.timing_samples()[1] == []

    raising = FakePlc(raises=True)
    worker2 = PlcOutputWorker(raising)
    worker2.submit(_request(1))
    assert worker2.write_pending() is False
    assert worker2.error_count == 1
    assert len(worker2.timing_samples()[0]) == 1


def test_output_worker_interval_validation():
    for interval in (0.0, -1.0, math.nan, math.inf):
        _expect_value_error(lambda: PlcOutputWorker(FakePlc(), interval=interval))


def test_output_worker_first_write_wakes_without_waiting_for_interval():
    plc = FakePlc()
    worker = PlcOutputWorker(plc, interval=1.0)
    worker.start()
    try:
        assert worker.running
        submitted_at = time.perf_counter()
        worker.submit(_request(1))
        assert plc.wait_for_writes(1, 0.4), "first request should not wait for the periodic timer"
        assert plc.write_times[0] - submitted_at < 0.4
    finally:
        worker.stop()
    assert not worker.running
    assert worker.write_count == 1
    assert plc.writes[0]["ai_target_x"] == 1


def test_output_worker_limits_rate_and_coalesces_requests():
    plc = FakePlc()
    interval = 0.18
    worker = PlcOutputWorker(plc, interval=interval)
    worker.start()
    try:
        worker.submit(_request(1))
        assert plc.wait_for_writes(1, 0.6)
        worker.submit(_request(2))
        worker.submit(_request(3))
        assert not plc.wait_for_writes(2, 0.06), "new submissions cannot bypass the rate limit"
        assert plc.wait_for_writes(2, 0.8)
        assert [write["ai_target_x"] for write in plc.writes] == [1, 3]
        assert plc.write_times[1] - plc.write_times[0] >= interval - 0.01
        assert not plc.wait_for_writes(3, interval + 0.05), "idle worker must not repeat the last request"
    finally:
        worker.stop()
    assert worker.write_count == 2


def test_output_worker_stop_interrupts_cooldown_without_writing_pending_request():
    plc = FakePlc()
    worker = PlcOutputWorker(plc, interval=10.0)
    worker.start()
    try:
        worker.submit(_request(1))
        assert plc.wait_for_writes(1, 0.6)
        worker.submit(_request(2))
        stopped_at = time.perf_counter()
        worker.stop()
        assert time.perf_counter() - stopped_at < 0.4
        assert not worker.running
        assert len(plc.writes) == 1
    finally:
        worker.stop()


def test_output_worker_restart_does_not_replay_stale_pending_target():
    plc = FakePlc()
    worker = PlcOutputWorker(plc, interval=0.15)
    worker.start()
    try:
        worker.submit(_request(1))
        assert plc.wait_for_writes(1, 0.6)
        worker.submit(_request(2))
        worker.stop()
        assert not worker.running
        worker.start()
        assert not plc.wait_for_writes(2, 0.25), "restart must not replay the request discarded on stop"
        worker.submit(_request(3))
        assert plc.wait_for_writes(2, 0.6)
    finally:
        worker.stop()
    assert [write["ai_target_x"] for write in plc.writes] == [1, 3]


def test_output_worker_stop_during_write_keeps_thread_visible_until_join():
    class BlockingPlc(FakePlc):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def write_game_state(self, **kwargs):
            self.entered.set()
            assert self.release.wait(2), "PLC write must be unblocked by test teardown"
            return super().write_game_state(**kwargs)

    plc = BlockingPlc()
    worker = PlcOutputWorker(plc)
    worker.start()
    try:
        worker.submit(_request(1))
        assert plc.entered.wait(0.6)
        worker.stop(timeout=0.01)
        assert worker.running
        worker.start()  # must not clear the stop event or spawn another writer while the old one runs
        assert worker.running
        plc.release.set()
        worker.stop(timeout=1)
        assert not worker.running
        assert len(plc.writes) == 1

        worker.submit(_request(3))
        worker.start()
        assert plc.wait_for_writes(2, 0.6)
    finally:
        plc.release.set()
        worker.stop(timeout=1)
    assert not worker.running
    assert [write["ai_target_x"] for write in plc.writes] == [1, 3]
