"""视觉和游戏之间共享的状态定义。"""

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Any


class TrackingState(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass
class CurlingState:
    """统一冰壶状态：视觉、预测、控制共享的数据模型。

    核心是 position(x, y) 与 velocity(vx, vy)，附带 timestamp 与 confidence。
    radius 只在视觉检测/追踪路径上有意义，控制侧可以忽略。
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    timestamp: float = 0.0
    confidence: float = 0.0
    radius: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        """当前位置 (x, y)。"""
        return (self.x, self.y)

    @property
    def velocity(self) -> tuple[float, float]:
        """当前速度 (vx, vy)。"""
        return (self.vx, self.vy)

    @property
    def speed(self) -> float:
        """速度大小。"""
        return hypot(self.vx, self.vy)

    @property
    def direction(self) -> tuple[float, float]:
        """单位速度方向，静止时返回 (0.0, 0.0)。"""
        speed = self.speed
        if speed <= 1e-9:
            return (0.0, 0.0)
        return (self.vx / speed, self.vy / speed)

    # 视觉层沿用 center_x/center_y 的像素坐标命名，这里作为 x/y 的只读别名。
    @property
    def center_x(self) -> float:
        return self.x

    @property
    def center_y(self) -> float:
        return self.y

    @classmethod
    def from_detection(cls, detection) -> "CurlingState":
        """由检测结果（center_x/center_y/radius/score）构建统一状态。"""
        return cls(
            x=float(detection.center_x),
            y=float(detection.center_y),
            vx=float(getattr(detection, "vx", 0.0)),
            vy=float(getattr(detection, "vy", 0.0)),
            timestamp=float(getattr(detection, "timestamp", 0.0)),
            confidence=float(getattr(detection, "confidence", getattr(detection, "score", 0.0))),
            radius=float(getattr(detection, "radius", 0.0)),
        )

    @classmethod
    def from_track(cls, track) -> "CurlingState":
        """由 Tracker 轨迹构建统一状态。"""
        return cls(
            x=float(track.center_x),
            y=float(track.center_y),
            vx=float(track.vx),
            vy=float(track.vy),
            timestamp=float(track.last_timestamp),
            confidence=float(getattr(track, "confidence", 0.0)),
            radius=float(track.radius),
        )

    @classmethod
    def from_any(cls, source: Any, y: float | None = None, vx: float | None = None, vy: float | None = None) -> "CurlingState":
        """把裸坐标、CurlingState 或任意带 x/y/vx/vy 的运动实体统一成 CurlingState。

        这是预测器唯一的状态归一化入口，避免各处重复实现坐标/速度转换。
        """
        if y is not None:
            return cls(x=float(source), y=float(y), vx=float(vx or 0.0), vy=float(vy or 0.0))
        if isinstance(source, cls):
            return source
        return cls(
            x=float(source.x),
            y=float(source.y),
            vx=float(getattr(source, "vx", 0.0)),
            vy=float(getattr(source, "vy", 0.0)),
            timestamp=float(getattr(source, "timestamp", 0.0)),
            confidence=float(getattr(source, "confidence", 0.0)),
            radius=float(getattr(source, "radius", 0.0)),
        )

    def to_stone_state(self, tracking_state: TrackingState = TrackingState.UNKNOWN) -> "StoneState":
        """补齐追踪字段，升级为 StoneState。"""
        return StoneState(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            timestamp=self.timestamp,
            confidence=self.confidence,
            radius=self.radius,
            tracking_state=tracking_state,
        )


@dataclass
class StoneState(CurlingState):
    """冰壶状态：在 CurlingState 之上补充追踪状态。

    视觉追踪出来的会带 tracking_state；
    仿真侧自己构造时可以不填，默认 unknown。
    """

    tracking_state: TrackingState = TrackingState.UNKNOWN

    @classmethod
    def from_tracker(cls, track) -> "StoneState":
        # state 可能是枚举也可能是裸值，统一转成 TrackingState
        try:
            tracking_state = TrackingState(str(getattr(track.state, "value", track.state)))
        except ValueError:
            tracking_state = TrackingState.UNKNOWN
        state = CurlingState.from_track(track)
        return state.to_stone_state(tracking_state)


@dataclass
class GameState:
    """AI 决策需要的游戏快照，stone 是冰壶的运动状态。"""

    ai_x: float
    ai_y: float
    ai_home_y: float
    target_x: float
    target_y: float
    stone: StoneState
    awaiting_serve: bool
    current_server: str
    serve_phase: str
    stalled_stone_phase: str
    reaction_timer: float
    difficulty: Any

    @classmethod
    def from_vision(cls, stone) -> "GameState":
        """只有视觉结果、还没接 AI 时的快照，AI 字段给中性初值。"""
        return cls(
            ai_x=0.0,
            ai_y=0.0,
            ai_home_y=0.0,
            target_x=0.0,
            target_y=0.0,
            stone=stone,
            awaiting_serve=False,
            current_server="none",
            serve_phase="idle",
            stalled_stone_phase="idle",
            reaction_timer=0.0,
            difficulty=None,
        )
