from air_hockey.game_state import StoneState, TrackingState


def test_sim_can_build_stone_state_without_tracking_info():
    # 仿真侧自己构造冰壶状态，tracking 字段用默认值
    stone = StoneState(x=1.0, y=2.0, vx=3.0, vy=4.0)
    assert stone.tracking_state == TrackingState.UNKNOWN
    assert (stone.x, stone.y, stone.vx, stone.vy) == (1.0, 2.0, 3.0, 4.0)
