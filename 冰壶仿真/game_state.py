"""Vision 与 Game 之间共享的只读状态契约。

本模块不依赖 Camera、Detector 或 StoneTracker。视觉层只需把公开的追踪字段
转为 :class:`StoneState`，游戏和 AI 只消费这里定义的状态对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class TrackingState(str, Enum):
    """统一的视觉跟踪生命周期，不暴露视觉实现自己的枚举类型。"""

    ACTIVE = "active"
    LOST = "lost"
    UNKNOWN = "unknown"


class TrackOutput(Protocol):
    """Tracker 输出适配所需的最小公共字段。"""

    center_x: float
    center_y: float
    vx: float
    vy: float
    radius: float
    last_timestamp: float
    state: Any


@dataclass(frozen=True)
class StoneState:
    """视觉层追踪到的冰壶状态，坐标及速度单位由调用方保持一致。"""

    x: float
    y: float
    vx: float
    vy: float
    tracking_state: TrackingState
    radius: float = 0.0
    timestamp: float = 0.0

    @classmethod
    def from_tracker(cls, track: TrackOutput) -> "StoneState":
        """将任意兼容 Tracker 的公共输出转为统一状态。"""
        raw_state = getattr(track.state, "value", track.state)
        try:
            tracking_state = TrackingState(str(raw_state))
        except ValueError:
            tracking_state = TrackingState.UNKNOWN
        return cls(
            x=float(track.center_x),
            y=float(track.center_y),
            vx=float(track.vx),
            vy=float(track.vy),
            tracking_state=tracking_state,
            radius=float(track.radius),
            timestamp=float(track.last_timestamp),
        )


@dataclass(frozen=True)
class PuckState:
    """游戏和 AI 消费的冰球/冰壶运动状态。"""

    x: float
    y: float
    vx: float
    vy: float

    @classmethod
    def from_stone(cls, stone: StoneState) -> "PuckState":
        return cls(x=stone.x, y=stone.y, vx=stone.vx, vy=stone.vy)


@dataclass(frozen=True)
class GameState:
    """AI 决策所需的统一游戏快照。

    ``stone`` 保留视觉追踪状态；``puck`` 是游戏坐标系中的运动实体。离线视觉
    管线通过 ``PuckState.from_stone`` 桥接两者，纯物理仿真则可直接构造 ``puck``。
    标量属性保留给既有 AI 策略使用，避免把策略与任何视觉实现耦合。
    """

    ai_x: float
    ai_y: float
    ai_home_y: float
    target_x: float
    target_y: float
    puck: PuckState
    awaiting_serve: bool
    current_server: str
    serve_phase: str
    stalled_puck_phase: str
    reaction_timer: float
    difficulty: Any
    stone: StoneState | None = None

    @classmethod
    def from_vision(cls, stone: StoneState) -> "GameState":
        """由视觉追踪结果创建尚未接入 AI 的游戏快照。

        AI 相关字段保留中性初值，使实时视觉管线也能生产和检查完整的
        ``GameState``，而不需要伪造摄像头、检测器或控制器。
        """
        return cls(
            ai_x=0.0,
            ai_y=0.0,
            ai_home_y=0.0,
            target_x=0.0,
            target_y=0.0,
            puck=PuckState.from_stone(stone),
            awaiting_serve=False,
            current_server="none",
            serve_phase="idle",
            stalled_puck_phase="idle",
            reaction_timer=0.0,
            difficulty=None,
            stone=stone,
        )

    @property
    def puck_x(self) -> float:
        return self.puck.x

    @property
    def puck_y(self) -> float:
        return self.puck.y

    @property
    def puck_velocity_x(self) -> float:
        return self.puck.vx

    @property
    def puck_velocity_y(self) -> float:
        return self.puck.vy


__all__ = ["GameState", "PuckState", "StoneState", "TrackingState"]
