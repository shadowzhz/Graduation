"""实时视觉运行核心。

统一单向数据流：
相机帧 -> Detector(CurlingState) -> Tracker -> KalmanFilter(状态估计) -> CurlingState -> Predictor(PredictionState) -> AI 决策。
每处理一帧返回明确的 VisionResult，完全与 GUI 和显示层解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

from .. import core_config as core
from ..ai import AirHockeyAI
from ..camera.types import Frame
from game_state import CurlingState, GameState, TrackingState
from ..estimation import KalmanFilter
from ..prediction import PredictionState, TrajectoryPredictor
from ..vision import StoneDetector, VisionPipeline
from ..vision.tracker import StoneTracker, TrackState
from ..vision.types import Detection, ROI, Track

DETECTION_INTERVAL = 3
AI_HOME_Y = core.RINK_TOP + (core.RINK_CENTER_Y - core.RINK_TOP) * 0.28


@dataclass
class TrajectoryDebug:
    """轨迹调试信息：当前坐标、速度方向、预测终点。"""

    position: tuple[float, float]
    direction: tuple[float, float]
    predicted_endpoint: tuple[float, float]


@dataclass
class VisionResult:
    """单帧视觉与 AI 处理的完整结果快照。

    curling_state 是唯一权威状态（Kalman 输出升级为带 tracking_state 的 StoneState）；
    stone_state 作为只读别名保留，不再单独保存第二份等价状态。
    """

    frame: Frame
    detection: Optional[Detection] = None
    track: Optional[Track] = None
    curling_state: Optional[CurlingState] = None
    prediction: Optional[PredictionState] = None
    trajectory: Optional[list[tuple[float, float]]] = None
    trajectory_debug: Optional[TrajectoryDebug] = None
    ai_target: Optional[tuple[float, float]] = None
    fps: float = 0.0

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
        table_roi: tuple[int, int, int, int] = (350, 0, 580, 650),
        lower: tuple[int, int, int] = (170, 100, 80),
        upper: tuple[int, int, int] = (179, 255, 255),
        calibration_file: str = "calibration/camera_calibration.npz",
        table_calibration_file: str = "calibration/table_homography.npz",
        disable_undistort: bool = False,
        detection_interval: int = DETECTION_INTERVAL,
        detector: Optional[StoneDetector] = None,
        tracker: Optional[StoneTracker] = None,
        kalman: Optional[KalmanFilter] = None,
        predictor: Optional[TrajectoryPredictor] = None,
        ai: Optional[AirHockeyAI] = None,
        vision_pipeline: Optional[VisionPipeline] = None,
    ) -> None:
        self.table_roi = tuple(table_roi)
        self.detection_interval = int(detection_interval)

        self.detector = detector or StoneDetector(
            roi=self.table_roi,
            lower=tuple(lower),
            upper=tuple(upper),
            min_area=500.0,
            min_radius=25.0,
            min_circularity=0.65,
        )
        self.tracker = tracker or StoneTracker(max_missed_frames=self.detection_interval * 4)
        self.kalman = kalman or KalmanFilter()
        self.predictor = predictor or TrajectoryPredictor()
        self.ai = ai or AirHockeyAI(predictor=self.predictor)

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
        self.frame_index = 0
        self.display_fps = 0.0
        self.detect_ms = 0.0
        self.frame_ms = 0.0

        self.ai_home_y = AI_HOME_Y
        self.target = [core.RINK_CENTER_X, self.ai_home_y]
        self.reaction_timer = 0.0
        self.stalled_phase = "idle"
        self.ai_current_pos = [core.RINK_CENTER_X, self.ai_home_y]

    def process_frame(self, frame: Frame) -> VisionResult:
        """处理单帧：检测/追踪预测 -> 坐标转换 -> 状态估计 -> 轨迹预测 -> AI 决策。"""
        t0 = time.perf_counter()
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

        if self.frame_index == 1 or self.frame_index % self.detection_interval == 0:
            dynamic_roi = None
            box_size = 140

            if self.tracker.track is not None and self.tracker.track.state == TrackState.ACTIVE:
                pred_raw_x, pred_raw_y = self.tracker.predict_position(processed_frame.timestamp)
                img_h, img_w = processed_frame.image.shape[:2]
                rx = max(0, min(int(pred_raw_x - box_size / 2), img_w - 1))
                ry = max(0, min(int(pred_raw_y - box_size / 2), img_h - 1))
                rw = max(1, min(box_size, img_w - rx))
                rh = max(1, min(box_size, img_h - ry))
                dynamic_roi = ROI(x=rx, y=ry, width=rw, height=rh)

            detection = self.detector.detect(processed_frame, dynamic_roi=dynamic_roi)
            self.detect_ms = self.detect_ms * 0.9 + (time.perf_counter() - t_detect_start) * 1000 * 0.1
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

        if track is not None:
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
                    track.last_timestamp,
                    confidence=track.confidence,
                    radius=track.radius,
                )

            # 唯一权威状态：StoneState 只是给 CurlingState 补上追踪状态，
            # 预测与 AI 共用同一个对象，避免同帧重复构造两份等价状态。
            tracking_state = TrackingState.LOST if track.state == TrackState.LOST else TrackingState.ACTIVE
            curling = estimated.to_stone_state(tracking_state)
            stone = curling

            # 预测数据层：业务层只与 PredictionState 交互，不再直接传递裸点列表
            prediction_state = self.predictor.predict(curling)
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
                difficulty=core.DIFFICULTIES["普通"],
            )

            decision = self.ai.update(state, dt)
            self.target[0], self.target[1] = decision.target_x, decision.target_y
            self.reaction_timer = decision.reaction_timer
            self.stalled_phase = decision.stalled_stone_phase

            smooth_alpha = min(1.0, dt * 12.0)
            self.ai_current_pos[0] += (self.target[0] - self.ai_current_pos[0]) * smooth_alpha
            self.ai_current_pos[1] += (self.target[1] - self.ai_current_pos[1]) * smooth_alpha

            ai_target = (decision.target_x, decision.target_y)
        else:
            # 轨迹丢失时清空估计器，避免下一个冰壶被旧状态污染
            self.kalman.reset()

        self.frame_ms = self.frame_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

        return VisionResult(
            frame=processed_frame,
            detection=detection,
            track=track,
            curling_state=curling,
            prediction=prediction_state,
            trajectory=table_trajectory,
            trajectory_debug=trajectory_debug,
            ai_target=ai_target,
            fps=self.display_fps,
        )
