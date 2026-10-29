"""测试 StoneDetector 候选检测与轨迹候选关联功能。

验证：
1. detect_candidates 真实单次检测返回按原 score/area 稳定降序排列的 CurlingState 候选。
2. select_candidate 优先匹配原轨迹（predicted_position / previous_position / previous_radius）。
3. 尺寸做软评分不硬丢弃，避免仅按距预测最近而误选噪声。
4. 关联不成立时回退形状最佳候选（candidates[0]），支持新球与快速换向。
5. 空图返回空列表 / None。
6. 默认 detect(frame) 保持原有选最高分行为。
7. VisionRuntime 两帧合成图端到端关联旧球回归验证。
"""

import cv2
import numpy as np

from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.vision.detector import StoneDetector
from air_hockey.game_state import CurlingState


def _create_multi_circle_frame(circles, shape=(720, 1280, 3), timestamp=1.0):
    """生成合成红圆图像：circles 为 [(center_x, center_y, radius), ...] 列表。"""
    image = np.zeros(shape, dtype=np.uint8)
    for cx, cy, r in circles:
        cv2.circle(image, (int(cx), int(cy)), int(r), (0, 0, 255), -1)
    return Frame(image=image, timestamp=timestamp)


def test_old_trajectory_priority():
    """多候选场景：优先匹配符合原轨迹预测的球，即便远端候选得分/面积稍高。"""
    detector = StoneDetector(min_area=50.0, min_radius=5.0)

    # 圆1: 原轨迹上的球（半径20）
    # 圆2: 远端干扰球（半径35，面积与形状得分更高）
    frame = _create_multi_circle_frame([(200, 200, 20), (700, 500, 35)])
    candidates = detector.detect_candidates(frame)

    assert len(candidates) == 2
    # 远端大球因面积更大，默认形状得分在首位
    assert candidates[0].x > 600.0

    # 给定原轨迹预测位置 (205, 200)，上一帧位置 (195, 200)，半径 20.0
    selected = StoneDetector.select_candidate(
        candidates,
        predicted_position=(205.0, 200.0),
        previous_position=(195.0, 200.0),
        previous_radius=20.0,
        max_distance=80.0,
    )

    assert selected is not None
    assert abs(selected.x - 200.0) < 3.0
    assert abs(selected.y - 200.0) < 3.0
    assert abs(selected.radius - 20.0) < 3.0


def test_avoid_closest_noise_favor_real_stone():
    """避免仅选离预测位置最近的候选：真实球置信度与尺寸软评分优先于极小噪声候选。"""
    detector = StoneDetector(min_area=20.0, min_radius=3.0)

    # 两圆物理间距清晰，避免轮廓黏连：真实球在 (235, 200)，小噪点在 (190, 200)
    frame = _create_multi_circle_frame([(235, 200, 20), (190, 200, 4)])
    candidates = detector.detect_candidates(frame)
    assert len(candidates) == 2

    # 预测在 (200, 200)，原半径 20.0：噪点距预测 10px，真实球距预测 35px
    selected = StoneDetector.select_candidate(
        candidates,
        predicted_position=(200.0, 200.0),
        previous_position=(190.0, 200.0),
        previous_radius=20.0,
        max_distance=80.0,
    )

    assert selected is not None
    # 应当选中置信度更高、尺寸匹配的真实球 (235, 200)，而非距预测更近但尺寸形状差的噪点
    assert abs(selected.x - 235.0) < 3.0
    assert abs(selected.radius - 20.0) < 3.0


def test_unmatched_fallback_to_best_candidate():
    """关联不成立时直接回退形状得分最高候选，无需二次确认。"""
    detector = StoneDetector(min_area=50.0, min_radius=5.0)

    # 图像中只有远端新球 (600, 400, 25) 和次优球 (800, 400, 15)
    frame = _create_multi_circle_frame([(600, 400, 25), (800, 400, 15)])
    candidates = detector.detect_candidates(frame)
    assert len(candidates) == 2

    # 历史轨迹在 (100, 100)，超出 max_distance
    selected = StoneDetector.select_candidate(
        candidates,
        predicted_position=(100.0, 100.0),
        previous_position=(90.0, 100.0),
        previous_radius=25.0,
        max_distance=80.0,
    )

    # 关联不成立，应回退至 candidates[0]
    assert selected is not None
    assert selected == candidates[0]
    assert abs(selected.x - 600.0) < 3.0


def test_empty_image():
    """纯黑空图像应安全返回空列表与 None。"""
    detector = StoneDetector()
    frame = Frame(image=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp=2.0)

    candidates = detector.detect_candidates(frame)
    assert candidates == []

    detect_result = detector.detect(frame)
    assert detect_result is None

    assert StoneDetector.select_candidate([]) is None


def test_default_detect_unchanged():
    """默认 detect(frame) 保持选取原管线最高分候选的行为。"""
    detector = StoneDetector(min_area=50.0, min_radius=5.0)

    frame = _create_multi_circle_frame([(300, 300, 20), (500, 300, 30)])
    candidates = detector.detect_candidates(frame)
    best = detector.detect(frame)

    assert isinstance(best, CurlingState)
    assert len(candidates) >= 2
    # detect 结果与 candidates 排序首位完全一致
    assert best.x == candidates[0].x
    assert best.y == candidates[0].y
    assert best.confidence == candidates[0].confidence
    assert best.radius == candidates[0].radius


def test_soft_radius_scoring_not_hard_discarded():
    """候选尺寸即使与 previous_radius 存在差异也只软评分，不硬丢弃。"""
    detector = StoneDetector(min_area=50.0, min_radius=5.0)

    # 单球在 (300, 300)，半径 18
    frame = _create_multi_circle_frame([(300, 300, 18)])
    candidates = detector.detect_candidates(frame)
    assert len(candidates) == 1

    # 历史半径为 35.0（明显差异），但在位置允许范围内
    selected = StoneDetector.select_candidate(
        candidates,
        predicted_position=(302.0, 300.0),
        previous_radius=35.0,
        max_distance=80.0,
    )

    # 绝不能因为尺寸不一致硬丢弃
    assert selected is not None
    assert abs(selected.x - 300.0) < 3.0


def test_vision_runtime_two_frames_multi_target_association():
    """验证 VisionRuntime 处理两帧合成图时，第二帧即便出现更大远端干扰球，仍稳定关联旧球轨迹。"""
    runtime = VisionRuntime(table_calibration_file=None, disable_undistort=True)

    # Frame 1: 原球在 (500, 300)
    img1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img1, (500, 300), 25, (0, 0, 255), -1)
    res1 = runtime.process_frame(Frame(image=img1, timestamp=1.0))
    assert res1.curling_state is not None
    assert runtime.tracker.track is not None
    track_id_1 = runtime.tracker.track.track_id
    assert abs(runtime.tracker.track.center_x - 500.0) < 3.0

    # Frame 2: 原球运动至 (510, 305)；同时远端出现面积更大的干扰球 (800, 500, 半径 35)
    img2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img2, (510, 305), 25, (0, 0, 255), -1)
    cv2.circle(img2, (800, 500), 35, (0, 0, 255), -1)
    res2 = runtime.process_frame(Frame(image=img2, timestamp=1.03))

    assert res2.curling_state is not None
    assert runtime.tracker.track is not None
    # 轨迹 ID 保持连续，且追踪位置落在旧球 (510, 305)，未被远端干扰球劫持
    assert runtime.tracker.track.track_id == track_id_1
    assert abs(runtime.tracker.track.center_x - 510.0) < 3.0
    assert abs(runtime.tracker.track.center_y - 305.0) < 3.0
