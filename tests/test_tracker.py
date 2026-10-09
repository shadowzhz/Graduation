from air_hockey.vision.tracker import StoneTracker
from air_hockey.vision.types import Detection


def make_detection(x, y, t, radius=25.0):
    return Detection(
        center_x=x, center_y=y, radius=radius, area=1963.0,
        timestamp=t, circularity=0.9,
    )


def test_new_detection_creates_track():
    tracker = StoneTracker()
    tracks = tracker.update(make_detection(100, 200, 0.0))
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state.value == "tentative"
    track = tracker.update(make_detection(100, 200, 0.033))[0]
    assert track.track_id == 1
    assert track.state.value == "active"


def test_near_detection_keeps_same_id():
    tracker = StoneTracker()
    tracker.update(make_detection(100, 200, 0.0))
    track = tracker.update(make_detection(110, 200, 0.033))[0]
    assert track.track_id == 1
    assert track.vx > 0


def test_far_detection_gets_new_id():
    tracker = StoneTracker()
    tracker.update(make_detection(100, 200, 0.0))
    tracker.update(make_detection(100, 200, 0.033))
    track = tracker.update(make_detection(500, 500, 0.066))[0]
    assert track.track_id == 2
    assert track.state.value == "tentative"


def test_track_requires_two_real_detections_to_confirm():
    tracker = StoneTracker()
    first = tracker.update(make_detection(100, 200, 0.0))[0]
    assert first.state.value == "tentative" and first.hits == 1

    for timestamp in (0.033, 0.066):
        predicted = tracker.predict(timestamp)[0]
        assert predicted.state.value == "tentative" and predicted.hits == 1
    assert tracker.track.state.value == "tentative" and tracker.track.hits == 1

    confirmed = tracker.update(make_detection(100, 200, 0.099))[0]
    assert confirmed.track_id == 1
    assert confirmed.state.value == "active" and confirmed.hits == 2


def test_far_detection_creates_new_tentative_identity():
    tracker = StoneTracker()
    tracker.update(make_detection(100, 200, 0.0))
    tracker.update(make_detection(100, 200, 0.033))

    candidate = tracker.update(make_detection(500, 500, 0.066))[0]
    assert candidate.track_id == 2
    assert candidate.state.value == "tentative"
    assert tracker.update(make_detection(500, 500, 0.099))[0].state.value == "active"


def test_lost_identity_requires_confirmation_after_recreation():
    tracker = StoneTracker(max_missed_frames=1)
    tracker.update(make_detection(100, 200, 0.0))
    tracker.update(make_detection(100, 200, 0.033))
    tracker.update(None)
    assert tracker.update(None) == []

    candidate = tracker.update(make_detection(100, 200, 0.2))[0]
    assert candidate.track_id == 2
    assert candidate.state.value == "tentative"
    assert tracker.update(make_detection(100, 200, 0.233))[0].state.value == "active"


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
        assert recovered.track_id == 2
        assert recovered.state.value == "tentative"
        assert recovered.missed_frames == 0
        assert tracker.update(make_detection(110, 200, 1.6))[0].state.value == "active"
