from game_state import GameState, PuckState, StoneState, TrackingState
from vision.types import Track, TrackState


def make_track(state=TrackState.ACTIVE):
    return Track(
        track_id=1, center_x=10.0, center_y=20.0, radius=14.0,
        vx=1.5, vy=-2.5, last_timestamp=0.5, state=state,
    )


def test_from_tracker_maps_fields():
    stone = StoneState.from_tracker(make_track())
    assert stone.x == 10.0 and stone.y == 20.0
    assert stone.vx == 1.5 and stone.vy == -2.5
    assert stone.tracking_state == TrackingState.ACTIVE
    assert stone.timestamp == 0.5


def test_from_tracker_tolerates_unknown_state():
    track = make_track()
    track.state = "whatever"  # 兼容不规范的 tracker 输出
    assert StoneState.from_tracker(track).tracking_state == TrackingState.UNKNOWN


def test_lost_track_maps_to_lost():
    stone = StoneState.from_tracker(make_track(TrackState.LOST))
    assert stone.tracking_state == TrackingState.LOST


def test_puck_state_from_stone():
    puck = PuckState.from_stone(StoneState.from_tracker(make_track()))
    assert (puck.x, puck.y, puck.vx, puck.vy) == (10.0, 20.0, 1.5, -2.5)


def test_from_vision_fills_neutral_ai_fields():
    state = GameState.from_vision(StoneState.from_tracker(make_track()))
    assert state.current_server == "none"
    assert state.difficulty is None
    assert state.puck.x == 10.0
    assert state.stone.tracking_state == TrackingState.ACTIVE
