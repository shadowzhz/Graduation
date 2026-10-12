"""实时对打：直接使用 AI 防守/击球目标，状态只用于显示。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .. import core_config as core
from ..control.plc import TARGET_MAX_AGE
from .vision_runtime import AI_HOME_Y

if TYPE_CHECKING:
    from .vision_runtime import VisionResult

READY_POSITION = (core.RINK_CENTER_X, AI_HOME_Y)
ARRIVAL_TOLERANCE = core.STONE_RADIUS


class RallyState(Enum):
    WAITING = "WAITING"
    DEFENDING = "DEFENDING"
    ATTACKING = "ATTACKING"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class RallyDecision:
    state: RallyState
    target: tuple[float, float] | None
    reason: str
    direction: str
    mallet_position: tuple[float, float]
    mallet_source: str
    transitions: tuple[str, ...] = ()


class RallyController:
    def __init__(self):
        self.state = RallyState.WAITING

    def update(self, result: VisionResult, *, now: float, feedback=None,
               mallet_position=READY_POSITION) -> RallyDecision:
        actual = (feedback is not None and feedback.valid
                  and 0 <= now - feedback.stamp <= TARGET_MAX_AGE
                  and all(math.isfinite(v) for v in (feedback.x, feedback.y)))
        position = (feedback.x, feedback.y) if actual else tuple(mallet_position)
        source = "PLC" if actual else "software"
        previous = self.state
        self.state = (RallyState.WAITING if math.dist(position, READY_POSITION) <= ARRIVAL_TOLERANCE
                      else RallyState.RECOVERING)
        target = READY_POSITION
        reason = "no puck; READY"
        direction = "invalid"
        stone = result.curling_state
        fresh = (math.isfinite(now) and math.isfinite(result.frame.timestamp)
                 and 0 <= now - result.frame.timestamp <= core.MAX_FRAME_GAP)
        if not fresh:
            target = None
            reason = "stale camera frame; target cleared"
        elif stone is not None and (not all(math.isfinite(v) for v in
                                           (stone.x, stone.y, stone.vx, stone.vy, stone.timestamp))
                                    or not 0 <= now - stone.timestamp <= core.MAX_FRAME_GAP
                                    or not core.RINK_LEFT <= stone.x <= core.RINK_RIGHT
                                    or not core.RINK_TOP <= stone.y <= core.RINK_BOTTOM):
            target = None
            reason = "invalid/stale puck state; target cleared"
        elif result.track_confirmed and stone is not None:
            direction = ("still" if stone.speed <= core.STONE_STOP_SPEED else
                         "incoming" if stone.vy < 0 else "outgoing" if stone.vy > 0 else "lateral")
            if stone.y <= core.RINK_CENTER_Y or stone.vy <= 0:
                self.state = RallyState.ATTACKING if stone.y <= core.RINK_CENTER_Y else RallyState.DEFENDING
                decision = result.ai_decision
                if decision is None or not all(math.isfinite(v) for v in (decision.target_x, decision.target_y)):
                    target = None
                    reason = "invalid/missing AI target; target cleared"
                else:
                    target = (decision.target_x, decision.target_y)
                    reason = ("machine-half puck: strike/retrieve" if self.state == RallyState.ATTACKING
                              else "human-half puck: defend/prepare")
            else:
                reason = "puck on human side moving away; READY"
        transitions = (() if self.state == previous else
                       (f"{previous.value} -> {self.state.value}: {reason}",))
        return RallyDecision(self.state, target, reason, direction, position, source, transitions)
