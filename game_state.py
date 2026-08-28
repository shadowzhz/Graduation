"""视觉和游戏之间共享的状态定义。"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TrackingState(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass
class StoneState:
    """冰壶状态：位置和速度。

    视觉追踪出来的会带 tracking_state、radius、timestamp；
    仿真侧自己构造时这些字段可以不填。
    """

    x: float
    y: float
    vx: float
    vy: float
    tracking_state: TrackingState = TrackingState.UNKNOWN
    radius: float = 0.0
    timestamp: float = 0.0

    @classmethod
    def from_tracker(cls, track) -> "StoneState":
        # state 可能是枚举也可能是裸值，统一转成 TrackingState
        try:
            tracking_state = TrackingState(str(getattr(track.state, "value", track.state)))
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
