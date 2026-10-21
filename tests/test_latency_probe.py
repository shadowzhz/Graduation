"""LatencyProbe：用已知延迟的轴模型验证分段统计读数。"""

from air_hockey.control.latency import LatencyProbe

PERIOD = 1.0 / 30.0


def _run(axis_delay, cycles=150, step_mm=12.0, ack_cycles=1, start_delay=0.0):
    """指令点每周期沿 x 前进 step_mm；轴位置 = axis_delay 秒前的指令点（纯时延模型）。"""
    probe = LatencyProbe(PERIOD)
    commands = []           # (t, x)
    sequence = 100
    for k in range(cycles):
        now = k * PERIOD
        moving_from = start_delay
        # 状态读取：轴在 axis_delay 前的指令位置；起步前保持 0
        lagged = [x for t, x in commands if t <= now - axis_delay and now >= moving_from]
        actual_x = lagged[-1] if lagged else 0.0
        ack = sequence - ack_cycles + 1 if k >= ack_cycles else None
        probe.on_status(now, 2.0, (actual_x, 0.0), armed=True, busy=True,
                        seq_ack=ack, remaining=5.0, ob1_ms=12.0)
        x = step_mm * (k + 1)
        sequence += 1
        probe.on_send(now + 0.001, 3.0, 0.010, x, 0.0, seq=sequence)
        commands.append((now + 0.001, x))
    return probe.report()


def _value(text, label):
    """从汇总文本里取 '<label> p50 123ms' 的数值。"""
    tail = text.split(label, 1)[1]
    number = tail.split("p50", 1)[1].strip().split("ms", 1)[0].split("mm", 1)[0]
    return float(number)


def test_axis_lag_is_measured_and_blamed_on_plc_side():
    text, hints = _run(axis_delay=0.3)
    lag = _value(text, "时间滞后")
    assert 270.0 <= lag <= 370.0, text
    assert any("瓶颈在 PLC/TO 执行" in hint for hint in hints), hints


def test_axis_that_keeps_up_points_to_tia_motion_settings():
    text, hints = _run(axis_delay=0.0)
    assert "轴落后指令 0% 周期" in text, text
    assert any("瓶颈是 TIA 运动机构" in hint for hint in hints), hints
    assert not any("PLC/TO" in hint for hint in hints), hints


def test_sequence_ack_latency_counts_polling_cycles():
    text, hints = _run(axis_delay=0.0, ack_cycles=1)
    assert 30.0 <= _value(text, "序号确认") <= 40.0, text
    text, hints = _run(axis_delay=0.0, ack_cycles=4)
    assert _value(text, "序号确认") > 2.5 * PERIOD * 1000.0, text
    assert any("PLC 取新序号慢" in hint for hint in hints), hints


def test_start_delay_from_standstill_is_reported():
    text, hints = _run(axis_delay=0.0, start_delay=0.25)
    start = _value(text, "起步等待")
    assert 200.0 <= start <= 300.0, text
    assert any("轴起步等待长" in hint for hint in hints), hints


def test_report_resets_window_and_handles_no_data():
    probe = LatencyProbe(PERIOD)
    text, hints = probe.report()
    assert "轴未授权" in text and hints == []
    _run(axis_delay=0.3)
    text, _ = probe.report()
    assert "时间滞后" not in text
