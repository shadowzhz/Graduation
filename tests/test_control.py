"""实际运动目标的 AI 半场边界和异常输入保护。"""

import math

from air_hockey import core_config as core
from air_hockey.control import PlcControlAdapter


def test_plc_target_stays_within_ai_half_and_mallet_clearance():
    adapter = PlcControlAdapter()
    far_left = adapter.build_target(-1e6, -1e6, 1)
    far_right = adapter.build_target(1e6, 1e6, 2)
    assert (far_left.target_x, far_left.target_y) == (
        core.RINK_LEFT + core.MALLET_RADIUS,
        core.RINK_TOP + core.MALLET_RADIUS,
    )
    assert (far_right.target_x, far_right.target_y) == (
        core.RINK_RIGHT - core.MALLET_RADIUS,
        core.RINK_CENTER_Y - core.MALLET_RADIUS,
    )
    assert adapter.build_target(300, 180, 3).target_x == 300


def test_nonfinite_target_is_rejected():
    adapter = PlcControlAdapter()
    for invalid in (math.nan, math.inf, -math.inf):
        try:
            adapter.build_target(invalid, 200, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite PLC target accepted")
