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


# 主防御的"深入上限"（球台单位，1 单位 ≈ 0.928mm）。
#
# ⚠️ 默认是 None = **不限制**。
#
# 2026-10-13 现场选的是「全部追球」，但最初我把默认值设成 90，结果 Y 被卡在
# AI_HOME_Y + 90 = 229.52，而 ai.py 自己允许到 RINK_CENTER_Y - MALLET_RADIUS = 349
# —— **少了 119.5 单位 ≈ 111mm 的前向行程**，表现为"左右能到极限、Y 到不了"。
# 所以默认改成不限制，需要保守时再用 --defense-forward-limit 显式收窄。
#
# 物理对照：球台 y∈[46,380] 对应 Y 轴 -125~185（现场实测），防守线 AI_HOME_Y=139.52。
#     0    → 钉死在防守线（Y 完全不动）
#    50    → 往前约 41mm
#    90    → 往前约 75mm
#    210   → 约等于不限制（= ai.py 自身的上限 349）
DEFAULT_DEFENSE_FORWARD_LIMIT = None


class RallyController:
    def __init__(self, defense_forward_limit: float | None = DEFAULT_DEFENSE_FORWARD_LIMIT):
        """defense_forward_limit: 主防御的"深入上限"，单位是球台坐标（1 单位 ≈ 0.928mm）。

        传 None = 不限制（等于原来的主动追球行为）。
        见模块常量 DEFAULT_DEFENSE_FORWARD_LIMIT 的物理对照表。
        """
        self.state = RallyState.WAITING
        self.defense_forward_limit = defense_forward_limit

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
            # 用上面已经带 STONE_STOP_SPEED 死区的 direction 判断，
            # 不要用 stone.vy 的原始符号。
            #
            # stone.vy 是卡尔曼滤波的估计值，静止冰壶的 vy 会在 0 附近抖正负。
            # 用符号判断会让这一支每帧翻转：vy 略正走 else（状态留在 RECOVERING、
            # target 保持 READY），vy 略负走 if（状态 DEFENDING、target 换成 AI 目标）。
            # 实机 165FPS 下实测目标在 (300,140)/(338,140) 之间交替 207/102 次，
            # DEFENDING<->RECOVERING 各 119/118 次；机械臂因此被反复中止、几乎不动。
            #
            # direction 为 "still" 时（速度低于 STONE_STOP_SPEED）按"未知速度，
            # 先准备防守"处理，这与 test_human_half_unknown_velocity_* 的约定一致。
            machine_half = stone.y <= core.RINK_CENTER_Y
            puck_stopped = direction == "still"
            if machine_half or direction != "outgoing":
                self.state = RallyState.ATTACKING if machine_half else RallyState.DEFENDING
                decision = result.ai_decision
                if decision is None or not all(math.isfinite(v) for v in (decision.target_x, decision.target_y)):
                    target = None
                    reason = "invalid/missing AI target; target cleared"
                else:
                    target_y = decision.target_y
                    if machine_half and puck_stopped:
                        # 死球（冰壶在自己半场停住）必须上去回收，否则回合停住 —— 不设上限。
                        reason = "machine-half puck stopped: retrieve"
                    elif machine_half:
                        # 主防御：冰壶还在运动时，限制机械臂离开防守线往前的幅度。
                        # 原来这一支会一路跟到冰壶所在的深度（实机日志里 target y
                        # 到过 239/262/289），离开防线之后正前方就空了。
                        if self.defense_forward_limit is not None:
                            target_y = min(target_y, AI_HOME_Y + self.defense_forward_limit)
                        reason = "machine-half puck moving: limited pursuit"
                    else:
                        reason = "human-half puck: defend/prepare"
                    target = (decision.target_x, target_y)
            else:
                reason = "puck on human side moving away; READY"
        transitions = (() if self.state == previous else
                       (f"{previous.value} -> {self.state.value}: {reason}",))
        return RallyDecision(self.state, target, reason, direction, position, source, transitions)
