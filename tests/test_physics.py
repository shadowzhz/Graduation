import air_hockey_config as layout
from air_hockey_physics import (
    StoneMotion,
    clamp,
    goal_scorer,
    stone_inside_goal_mouth,
    reflected_coordinate,
)


def test_clamp():
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0


def test_reflected_coordinate_folds_back():
    assert reflected_coordinate(-5.0, 0.0, 100.0) == 5.0
    assert reflected_coordinate(105.0, 0.0, 100.0) == 95.0
    assert reflected_coordinate(50.0, 0.0, 100.0) == 50.0


def test_wall_bounce_reflects_velocity():
    stone = StoneMotion(x=layout.RINK_LEFT + layout.STONE_RADIUS - 5.0, y=layout.RINK_CENTER_Y)
    stone.set_immediate_velocity(-300.0, 0.0)
    stone.x += stone.vx * 0.01
    assert stone.resolve_walls()
    assert stone.x == layout.RINK_LEFT + layout.STONE_RADIUS
    assert abs(stone.vx - 300.0 * layout.WALL_RESTITUTION) < 1e-9


def test_friction_eventually_stops_stone():
    stone = StoneMotion()
    stone.set_immediate_velocity(100.0, 0.0)
    for _ in range(600):
        stone.advance_velocity(1 / 60)
    assert stone.vx == 0.0 and stone.vy == 0.0


def test_goal_detection():
    # 球门口内整体越线才算得分
    top_stone = StoneMotion(x=layout.RINK_CENTER_X, y=layout.RINK_TOP - layout.STONE_RADIUS - 1.0)
    assert goal_scorer(top_stone) == "player"
    bottom_stone = StoneMotion(x=layout.RINK_CENTER_X, y=layout.RINK_BOTTOM + layout.STONE_RADIUS + 1.0)
    assert goal_scorer(bottom_stone) == "ai"
    side_stone = StoneMotion(x=layout.RINK_LEFT + 10.0, y=layout.RINK_TOP - layout.STONE_RADIUS - 1.0)
    assert goal_scorer(side_stone) is None
    assert stone_inside_goal_mouth(layout.RINK_CENTER_X)
    assert not stone_inside_goal_mouth(layout.RINK_LEFT + 10.0)
