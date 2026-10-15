"""实时视觉运行核心。

统一单向数据流：
相机帧 -> Detector(CurlingState) -> Tracker -> KalmanFilter(状态估计) -> CurlingState -> Predictor(PredictionState) -> AI 决策。
每处理一帧返回明确的 VisionResult，完全与 GUI 和显示层解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import TYPE_CHECKING, Optional

from .. import core_config as core
from ..ai import AIDecision, AirHockeyAI
from ..camera.types import Frame
from game_state import CurlingState, GameState, TrackingState
from ..estimation import KalmanFilter
from ..prediction import PredictionState, TrajectoryPredictor
from ..vision import StoneDetector, VisionPipeline
from ..vision.tracker import StoneTracker, TrackState
from ..vision.types import Detection, ROI, Track

if TYPE_CHECKING:
    from .rally import RallyDecision

DETECTION_INTERVAL = 1
MAX_DETECTION_GAP = 0.08
AI_HOME_Y = core.RINK_TOP + (core.RINK_CENTER_Y - core.RINK_TOP) * 0.28


@dataclass
class TrajectoryDebug:
    """轨迹调试信息：当前坐标、速度方向、预测终点。"""

    position: tuple[float, float]
    direction: tuple[float, float]
    predicted_endpoint: tuple[float, float]


def _track_usable(track) -> bool:
    """轨道是否可用于下游决策（**含漏检宽限期内**）。

    ★ 为什么不能用 `track.state == TrackState.ACTIVE`：
    追踪器在**漏检第一帧**就把 state 置成 LOST（见 tracker._handle_missing_detection），
    而"真正丢弃"（`_track = None`）要等 missed_frames 超过 max_missed_frames。
    也就是说 `state` 不是"还有没有球"的判据，`track is not None` 才是。

    现场后果：用 state 判断会让下游**每两帧翻一次** —— rally 在
    DEFENDING/ATTACKING 与 RECOVERING(READY) 之间反复横跳，机械臂刚起步
    就被叫回 (300,140)，**永远接不到球**。

    宽限期内的位置是 Kalman 按模型外推的（管线里 detection is None 走 predict），
    不是新鲜观测，但对"球还在往哪走"的判断依然有效。
    """
    return track is not None and track.state != TrackState.TENTATIVE

def _compute_dynamic_roi(
    track: Optional[Track],
    pred_pos: tuple[float, float],
    img_w: int,
    img_h: int,
    base_box_size: int = 140,
) -> Optional[ROI]:
    """根据当前轨迹速度、时间差与反向包络自适应扩展动态 ROI。"""
    if track is None or img_w <= 0 or img_h <= 0:
        return None

    pred_raw_x, pred_raw_y = pred_pos
    if not (math.isfinite(pred_raw_x) and math.isfinite(pred_raw_y)):
        return None

    # 原框：基于当前预测位置的基础 140 框，作为尺寸与范围的下限保护
    orig_rx = max(0, min(int(pred_raw_x - base_box_size / 2), img_w - 1))
    orig_ry = max(0, min(int(pred_raw_y - base_box_size / 2), img_h - 1))
    orig_rw = max(1, min(base_box_size, img_w - orig_rx))
    orig_rh = max(1, min(base_box_size, img_h - orig_ry))
    orig_rx_max = orig_rx + orig_rw
    orig_ry_max = orig_ry + orig_rh

    radius = float(getattr(track, "radius", 0.0) or 0.0)
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 25.0

    obs_x = float(getattr(track, "center_x", pred_raw_x))
    obs_y = float(getattr(track, "center_y", pred_raw_y))
    if not (math.isfinite(obs_x) and math.isfinite(obs_y)):
        obs_x, obs_y = pred_raw_x, pred_raw_y

    # 上一观测点与预测位置都包含，加球半径余量，且不缩小到比原框更小
    min_x = min(orig_rx, obs_x - radius, pred_raw_x - radius)
    max_x = max(orig_rx_max, obs_x + radius, pred_raw_x + radius)
    min_y = min(orig_ry, obs_y - radius, pred_raw_y - radius)
    max_y = max(orig_ry_max, obs_y + radius, pred_raw_y + radius)

    rx = max(0, min(int(math.floor(min_x)), img_w - 1))
    ry = max(0, min(int(math.floor(min_y)), img_h - 1))
    rx_end = max(rx + 1, min(int(math.ceil(max_x)), img_w))
    ry_end = max(ry + 1, min(int(math.ceil(max_y)), img_h))

    rw = rx_end - rx
    rh = ry_end - ry
    return ROI(x=rx, y=ry, width=rw, height=rh)


@dataclass
class VisionResult:
    """单帧视觉与 AI 处理的完整结果快照。

    curling_state 是唯一权威状态（Kalman 输出升级为带 tracking_state 的 StoneState）；
    stone_state 作为只读别名保留，不再单独保存第二份等价状态。
    """

    frame: Frame
    detection: Optional[Detection] = None
    track: Optional[Track] = None
    track_confirmed: bool = False
    curling_state: Optional[CurlingState] = None
    prediction: Optional[PredictionState] = None
    trajectory: Optional[list[tuple[float, float]]] = None
    trajectory_debug: Optional[TrajectoryDebug] = None
    ai_target: Optional[tuple[float, float]] = None
    fps: float = 0.0
    ai_decision: Optional[AIDecision] = None
    rally: Optional[RallyDecision] = None

    @property
    def stone_state(self) -> Optional[CurlingState]:
        """兼容别名：与 curling_state 指向同一对象。"""
        return self.curling_state


class VisionRuntime:
    """实时视觉处理运行时。

    负责：相机帧 -> Detector -> Tracker -> KalmanFilter -> CurlingState -> Predictor -> AI 决策。
    每处理一帧返回明确的 VisionResult 对象，不负责 GUI 显示或绘制。
    """

    def __init__(
        self,
        table_roi: tuple[int, int, int, int] = (4, 10, 1216, 710),
        lower: tuple[int, int, int] = (170, 100, 80),
        upper: tuple[int, int, int] = (10, 255, 255),
        calibration_file: str = "calibration/camera_calibration.npz",
        table_calibration_file: str = "calibration/table_homography.npz",
        disable_undistort: bool = False,
        detection_interval: int = DETECTION_INTERVAL,
        detector: Optional[StoneDetector] = None,
        tracker: Optional[StoneTracker] = None,
        kalman: Optional[KalmanFilter] = None,
        predictor: Optional[TrajectoryPredictor] = None,
        ai: Optional[AirHockeyAI] = None,
        mallet_speed: Optional[float] = None,
        vision_pipeline: Optional[VisionPipeline] = None,
        profile: bool = False,
    ) -> None:
        self.table_roi = tuple(table_roi)
        self.detection_interval = int(detection_interval)
        self.profile = profile

        self.detector = detector or StoneDetector(
            roi=self.table_roi,
            lower=tuple(lower),
            upper=tuple(upper),
            min_area=500.0,
            min_radius=25.0,
            # ponytail: rink-line overlaps reach 0.578; lower values risk matching red markings.
            min_circularity=0.55,
        )
        self.tracker = tracker or StoneTracker(
            # ★ 宽限期：漏检这么多帧之内仍然算"有球"（位置由 Kalman 外推）。
            # 原来 ×4（≈24ms@165fps）太短 —— 球被球槌挡住、或被撞飞的瞬间
            # 很容易连丢好几帧，轨道一断 rally 就切回 READY、机械臂被叫回去，
            # 表现就是"接不到球"。×8 ≈ 48ms 更实际。
            max_missed_frames=self.detection_interval * 8, min_confirmed_hits=1,
        )
        self.kalman = kalman or KalmanFilter()
        self.predictor = predictor or TrajectoryPredictor()
        self._reuse_prediction_for_ai = ai is None
        # mallet_speed 给了才会做"提前量"（球槌响应延迟补偿）；None 时行为与原来一致。
        self.mallet_speed = float(mallet_speed) if mallet_speed else None
        self.ai = (AirHockeyAI(predictor=self.predictor, mallet_speed=self.mallet_speed)
                   if ai is None else ai)
        # Real hardware follows each observation without simulation reaction or aim noise.
        self._ai_difficulty = replace(
            core.DIFFICULTIES["普通"], aim_error=0.0, prediction=1.0,
            reaction_delay=0.0, attack_line=core.RINK_CENTER_Y,
        )

        if vision_pipeline is not None:
            self.vision_pipeline = vision_pipeline
            self.camera_geometry = vision_pipeline.geometry
        else:
            self.vision_pipeline = VisionPipeline(
                calibration_file=calibration_file,
                enabled=not disable_undistort,
                table_roi=self.table_roi,
                rink_bounds=(core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM),
                table_calibration_file=table_calibration_file,
            )
            self.camera_geometry = self.vision_pipeline.geometry

        self.last_timestamp = None
        self._last_detection_timestamp = None
        self._kalman_track_id = None
        self.frame_index = 0
        self.display_fps = 0.0
        self.detect_ms = 0.0
        self.frame_ms = 0.0
        self.last_detect_ms: Optional[float] = None
        self.last_predict_ms: Optional[float] = None
        self.last_ai_ms: Optional[float] = None

        self.ai_home_y = AI_HOME_Y
        self.target = [core.RINK_CENTER_X, self.ai_home_y]
        self.reaction_timer = 0.0
        self.stalled_phase = "idle"
        self.ai_current_pos = [core.RINK_CENTER_X, self.ai_home_y]
        self._ai_feedback_position = None

    def set_ai_feedback(self, feedback) -> None:
        """只接受新鲜的轴位置；反馈丢失时不把旧位置当作实机当前位置。"""
        self._ai_feedback_position = None
        if feedback is None or not feedback.valid:
            return
        age = time.monotonic() - feedback.stamp
        if 0.0 <= age <= 0.25 and all(math.isfinite(value) for value in (feedback.x, feedback.y)):
            self._ai_feedback_position = (feedback.x, feedback.y)

    def _detect(self, frame, dynamic_roi=None):
        # 自定义检测器继续遵循原 detect 契约。
        if not isinstance(self.detector, StoneDetector) or type(self.detector).detect is not StoneDetector.detect:
            return self.detector.detect(frame, dynamic_roi=dynamic_roi)
        candidates = self.detector.detect_candidates(frame, dynamic_roi=dynamic_roi)
        track = self.tracker.track
        if not _track_usable(track):
            return self.detector.select_candidate(candidates)
        return self.detector.select_candidate(
            candidates,
            predicted_position=self.tracker.predict_position(frame.timestamp),
            previous_position=(track.center_x, track.center_y),
            previous_radius=track.radius,
            max_distance=self.tracker.max_distance,
        )

    def process_frame(self, frame: Frame) -> VisionResult:
        """处理单帧：检测/追踪预测 -> 坐标转换 -> 状态估计 -> 轨迹预测 -> AI 决策。"""
        t0 = time.perf_counter()
        self.last_detect_ms = None
        self.last_predict_ms = None
        self.last_ai_ms = None
        self.frame_index += 1

        processed_frame = self.vision_pipeline.process(frame)

        now = processed_frame.timestamp
        raw_dt = 0.0 if self.last_timestamp is None else (now - self.last_timestamp)
        self.last_timestamp = now
        dt = min(max(raw_dt, 0.001), 0.1)

        if raw_dt > 0:
            instant = 1.0 / raw_dt
            self.display_fps = instant if self.display_fps == 0.0 else self.display_fps * 0.9 + instant * 0.1

        detection = None
        t_detect_start = time.perf_counter()

        if (self._last_detection_timestamp is None
                or self.frame_index % self.detection_interval == 0
                or now - self._last_detection_timestamp >= MAX_DETECTION_GAP):
            dynamic_roi = None
            box_size = 140

            if _track_usable(self.tracker.track):
                pred_pos = self.tracker.predict_position(processed_frame.timestamp)
                img_h, img_w = processed_frame.image.shape[:2]
                dynamic_roi = _compute_dynamic_roi(
                    self.tracker.track,
                    pred_pos,
                    img_w,
                    img_h,
                    base_box_size=box_size,
                )

            detection = self._detect(processed_frame, dynamic_roi=dynamic_roi)
            if detection is None and dynamic_roi is not None:
                detection = self._detect(processed_frame)
            self._last_detection_timestamp = now
            detected_ms = (time.perf_counter() - t_detect_start) * 1000
            self.detect_ms = self.detect_ms * 0.9 + detected_ms * 0.1
            if self.profile:
                self.last_detect_ms = detected_ms
            tracks = self.tracker.update(detection)
        else:
            tracks = self.tracker.predict(processed_frame.timestamp)

        track = tracks[0] if tracks else None

        stone = None
        curling = None
        prediction_state = None
        table_trajectory = None
        trajectory_debug = None
        ai_target = None
        decision = None

        if self._ai_feedback_position is not None:
            self.ai_current_pos[:] = self._ai_feedback_position

        if track is not None:
            if self._kalman_track_id is not None and self._kalman_track_id != track.track_id:
                self.kalman.reset()
                self.stalled_phase = "idle"
                self.reaction_timer = 0.0
            self._kalman_track_id = track.track_id
            # 统一单向坐标流: raw pixel -> undistorted pixel -> rink/table coordinate
            undist_x, undist_y = self.camera_geometry.raw_to_undistorted(track.center_x, track.center_y)
            table_x, table_y = self.camera_geometry.undistorted_to_table(undist_x, undist_y)

            # 状态估计层：KalmanFilter 平滑位置/速度，输出统一的 CurlingState
            # 有检测则用观测校正，检测间隔帧则按模型外推
            if detection is not None:
                estimated = self.kalman.update(
                    table_x,
                    table_y,
                    track.last_timestamp,
                    confidence=track.confidence,
                    radius=track.radius,
                )
            else:
                estimated = self.kalman.predict(
                    processed_frame.timestamp,
                    confidence=track.confidence,
                    radius=track.radius,
                )

            # 唯一权威状态：StoneState 只是给 CurlingState 补上追踪状态，
            # 预测与 AI 共用同一个对象，避免同帧重复构造两份等价状态。
            tracking_state = TrackingState.LOST if track.state == TrackState.LOST else TrackingState.ACTIVE
            curling = estimated.to_stone_state(tracking_state)
            stone = curling

            # 预测数据层：业务层只与 PredictionState 交互，不再直接传递裸点列表
            predict_start = time.perf_counter() if self.profile else 0.0
            prediction_state = self.predictor.predict(curling)
            if self.profile:
                self.last_predict_ms = (time.perf_counter() - predict_start) * 1000
            table_trajectory = prediction_state.trajectory
            trajectory_debug = TrajectoryDebug(
                position=curling.position,
                direction=curling.direction,
                predicted_endpoint=prediction_state.endpoint,
            )

            state = GameState(
                ai_x=self.ai_current_pos[0],
                ai_y=self.ai_current_pos[1],
                ai_home_y=self.ai_home_y,
                target_x=self.target[0],
                target_y=self.target[1],
                stone=stone,
                awaiting_serve=False,
                current_server="player",
                serve_phase="idle",
                stalled_stone_phase=self.stalled_phase,
                reaction_timer=self.reaction_timer,
                difficulty=self._ai_difficulty,
            )

            ai_start = time.perf_counter() if self.profile else 0.0
            if self._reuse_prediction_for_ai:
                decision = self.ai.update(state, dt, prediction=prediction_state)
            else:
                decision = self.ai.update(state, dt)
            if self.profile:
                self.last_ai_ms = (time.perf_counter() - ai_start) * 1000
            self.target[0], self.target[1] = decision.target_x, decision.target_y
            self.reaction_timer = decision.reaction_timer
            self.stalled_phase = decision.stalled_stone_phase

            ai_target = (decision.target_x, decision.target_y)
        else:
            # 轨迹丢失时清空估计器，避免下一个冰壶被旧状态污染
            self.kalman.reset()
            self._kalman_track_id = None
            self.stalled_phase = "idle"
            self.reaction_timer = 0.0
            self.target[:] = (core.RINK_CENTER_X, self.ai_home_y)
            ai_target = tuple(self.target)

        if self._ai_feedback_position is None:
            smooth_alpha = min(1.0, dt * 12.0)
            self.ai_current_pos[0] += (self.target[0] - self.ai_current_pos[0]) * smooth_alpha
            self.ai_current_pos[1] += (self.target[1] - self.ai_current_pos[1]) * smooth_alpha

        self.frame_ms = self.frame_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

        return VisionResult(
            frame=processed_frame,
            detection=detection,
            track=track,
            track_confirmed=_track_usable(track),
            curling_state=curling,
            prediction=prediction_state,
            trajectory=table_trajectory,
            trajectory_debug=trajectory_debug,
            ai_target=ai_target,
            fps=self.display_fps,
            ai_decision=decision,
        )
