"""实际运动目标的 AI 半场边界和异常输入保护。"""

import math

from air_hockey import core_config as core
from air_hockey.ai import AIDecision
from air_hockey.control import PlcControlAdapter


def decision(x, y):
    return AIDecision(target_x=x, target_y=y, stalled_stone_phase="idle")


def test_plc_target_stays_within_ai_half_and_mallet_clearance():
    adapter = PlcControlAdapter()
    far_left = adapter.build_target(decision(-1e6, -1e6), timestamp=1)
    far_right = adapter.build_target(decision(1e6, 1e6), timestamp=2)
    assert (far_left.target_x, far_left.target_y) == (
        core.RINK_LEFT + core.MALLET_RADIUS,
        core.RINK_TOP + core.MALLET_RADIUS,
    )
    assert (far_right.target_x, far_right.target_y) == (
        core.RINK_RIGHT - core.MALLET_RADIUS,
        core.RINK_CENTER_Y - core.MALLET_RADIUS,
    )
    assert adapter.build_target(decision(300, 180), timestamp=3).target_x == 300


def test_custom_limits_can_only_reduce_physical_reach():
    adapter = PlcControlAdapter(bounds=(-1e6, 1e6, -1e6, 1e6), margin=0)
    target = adapter.build_target(decision(1e6, 1e6), timestamp=1)
    assert target.target_y == core.RINK_CENTER_Y - core.MALLET_RADIUS
    smaller = PlcControlAdapter(bounds=(250, 350, 100, 250), margin=10)
    target = smaller.build_target(decision(-100, 500), timestamp=2)
    assert (target.target_x, target.target_y) == (260, 240)


def test_nonfinite_target_cannot_advance_jump_state():
    adapter = PlcControlAdapter(max_target_jump=10)
    adapter.build_target(decision(300, 200), timestamp=0)
    for invalid in (math.nan, math.inf, -math.inf):
        try:
            adapter.build_target(decision(invalid, 200), timestamp=1)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite PLC target accepted")
    target = adapter.build_target(decision(400, 200), timestamp=2)
    assert (target.target_x, target.target_y) == (310, 200)
    adapter.reset()
    assert adapter.build_target(decision(400, 200), timestamp=3).target_x == 400
