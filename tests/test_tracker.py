from air_hockey.vision.tracker import StoneTracker
from air_hockey.vision.types import Detection


def make_detection(x, y, t, radius=25.0):
    return Detection(
        center_x=x, center_y=y, radius=radius, area=1963.0,
        timestamp=t, circularity=0.9,
    )


def test_new_detection_creates_track():
    tracks = StoneTracker().update(make_detection(100, 200, 0.0))
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state.value == "active"


def test_near_detection_keeps_same_id():
    tracker = StoneTracker()
    tracker.update(make_detection(100, 200, 0.0))
    track = tracker.update(make_detection(110, 200, 0.033))[0]
    assert track.track_id == 1
    assert track.vx > 0


def test_far_detection_gets_new_id():
    tracker = StoneTracker()
    tracker.update(make_detection(100, 200, 0.0))
    track = tracker.update(make_detection(500, 500, 0.033))[0]
    assert track.track_id == 2


def test_missed_frames_go_lost_then_drop():
    tracker = StoneTracker(max_missed_frames=2)
    tracker.update(make_detection(100, 200, 0.0))
    assert tracker.update(None)[0].state.value == "lost"
    assert tracker.update(None)[0].state.value == "lost"
    assert tracker.update(None) == []
    assert tracker.track is None


def test_velocity_is_smoothed():
    tracker = StoneTracker(velocity_alpha=0.5)
    tracker.update(make_detection(0, 0, 0.0))
    tracker.update(make_detection(10, 0, 1.0))  # 瞬时速度 10
    assert tracker.track.vx == 5.0
    tracker.update(make_detection(30, 0, 2.0))  # 瞬时速度 20，平滑后取一半
    assert tracker.track.vx == 12.5


def test_scheduled_predictions_preserve_state_and_missing_detection_budget():
    for budget in (0, 1):
        tracker = StoneTracker(max_missed_frames=budget, velocity_alpha=1.0)
        tracker.update(make_detection(100, 200, 0.0))
        tracker.update(make_detection(110, 200, 1.0))
        for timestamp in (1.1, 1.2):
            predicted = tracker.predict(timestamp)[0]
            assert predicted.state.value == "active"
            assert predicted.missed_frames == 0
            assert abs(predicted.center_x - (110 + 10 * (timestamp - 1))) < 1e-9
        assert tracker.track.center_x == 110
        assert tracker.track.last_timestamp == 1.0

        tracker.update(None)
        if budget:
            predicted = tracker.predict(1.3)[0]
            assert predicted.state.value == "lost"
            assert predicted.missed_frames == 1
            assert tracker.update(None) == []
        assert tracker.predict(1.4) == []
        recovered = tracker.update(make_detection(110, 200, 1.5))[0]
        assert recovered.state.value == "active"
        assert recovered.missed_frames == 0
