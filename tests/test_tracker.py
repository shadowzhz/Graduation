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


def test_single_far_false_detection_does_not_take_over_live_track():
    tracker = StoneTracker(min_confirmed_hits=1, replacement_hits=2)
    for i in range(3):
        tracker.update(make_detection(100 + i, 200, i * 0.005))
    # 一帧远处误检：旧轨迹保留（按漏检外推），不换 ID
    track = tracker.update(make_detection(900, 600, 0.015))[0]
    assert track.track_id == 1 and track.state.value == "lost"
    # 真冰壶下一帧回来：仍是原轨迹
    track = tracker.update(make_detection(104, 200, 0.020))[0]
    assert track.track_id == 1 and track.state.value == "active"


def test_two_scattered_far_detections_do_not_take_over():
    tracker = StoneTracker(min_confirmed_hits=1, replacement_hits=2)
    tracker.update(make_detection(100, 200, 0.0))
    tracker.update(make_detection(900, 600, 0.005))
    track = tracker.update(make_detection(500, 100, 0.010))[0]
    assert track.track_id == 1


def test_consistent_far_detections_take_over_after_confirmation():
    tracker = StoneTracker(min_confirmed_hits=1, replacement_hits=2)
    tracker.update(make_detection(100, 200, 0.0))
    assert tracker.update(make_detection(900, 600, 0.005))[0].track_id == 1
    track = tracker.update(make_detection(905, 600, 0.010))[0]
    assert track.track_id == 2 and track.state.value == "active"


def test_without_live_track_first_detection_is_immediate():
    tracker = StoneTracker(min_confirmed_hits=1, replacement_hits=2)
    assert tracker.update(make_detection(900, 600, 0.0))[0].state.value == "active"
