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
    held_target: bool = False


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

# ★ 遮挡保持：球槌压到球上时，球被球槌挡住 → 检测消失 → 轨道丢弃 →
# track_confirmed=False → rally 切 RECOVERING、目标回 READY → 球槌退开 →
# 球又可见 → 再去 → 再次遮挡 …… 现场日志里连续循环好几轮。
#
# 判据故意放宽：**只要刚追击过、且球槌还在那个目标附近，就当是它自己挡的**。
# 现场实测把半径/窗口收紧时会"只保持一帧就放弃"，所以取宽值；
# 真正兜底的是窗口上限（超时一定会回 READY），不会永远抱着一个陈旧目标。
OCCLUSION_HOLD_SECONDS = 1.0       # 刚丢多久之内算"可能只是被挡"
OCCLUSION_KEEP_RADIUS = 150.0      # 球槌离最后目标多近算"就是它挡的"（球台单位，约 139mm）

class RallyController:
    def __init__(self, defense_forward_limit: float | None = DEFAULT_DEFENSE_FORWARD_LIMIT,
                 reachable_bounds: tuple | None = None):
        """defense_forward_limit: 主防御的"深入上限"，单位是球台坐标（1 单位 ≈ 0.928mm）。

        传 None = 不限制（等于原来的主动追球行为）。
        见模块常量 DEFAULT_DEFENSE_FORWARD_LIMIT 的物理对照表。

        reachable_bounds: (left, right, top, bottom) —— 球槌**真能到**的范围（游戏坐标），
        来自 `plc.reachable_bounds`。用来判断"这颗停住的球还回不回收得动"。
        """
        self.state = RallyState.WAITING
        self.defense_forward_limit = defense_forward_limit
        self.reachable_bounds = reachable_bounds
        # ★ 遮挡保持用的记忆（见 update 里的说明）
        self._last_target = None
        self._last_seen_at = 0.0
        self._last_active_state = RallyState.WAITING

    def can_retrieve(self, stone) -> bool:
        """命令范围内是否存在朝真人方向接触的位置；不要求整段回撤间距。"""
        if self.reachable_bounds is None:
            return True
        try:
            left, right, top, bottom = self.reachable_bounds
            closest_x = min(max(stone.x, left), right)
            closest_y = min(stone.y, bottom)
            return (stone.y > top and math.hypot(stone.x - closest_x, stone.y - closest_y)
                    < core.STONE_RADIUS + core.MALLET_RADIUS)
        except (TypeError, ValueError, IndexError):
            return True

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
        held_target = False
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
                    unreachable = machine_half and puck_stopped and not self.can_retrieve(stone)
                    if unreachable:
                        # 不能盲目直穿身后的球回 READY，那会从前侧推向自家球门。
                        self.state = RallyState.RECOVERING
                        reason = (f"puck unreachable (stopped at y={stone.y:.0f}, "
                                  f"no goal-side contact): target cleared; move puck manually")
                        self._last_target = None
                    elif machine_half and puck_stopped:
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
                    if unreachable:
                        target = None
                    else:
                        target = (decision.target_x, target_y)
                        # ★ 记下这次的有效目标 —— 供"球槌遮挡"时保持用
                        self._last_target = target
                        self._last_active_state = self.state
            else:
                reason = "puck on human side moving away; READY"
                # ★ 主动回 READY 时清掉旧目标 —— 否则后面一次遮挡会把一个
                #   陈旧的追击点"复活"，机械臂又冲回去。
                self._last_target = None
            # ★ 只要这一帧有确认的轨道（不管走哪一支），就刷新时间戳。
            #   原来只在"机器半场追击"那一支里刷新 —— 如果中间夹了一帧
            #   "球飞走"（target=READY），时间戳就不动，保持窗口会悄悄过期，
            #   表现为"遮挡保持只成功一帧、下一帧就放弃"（现场日志里正是如此）。
            self._last_seen_at = now
        else:
            # ★★★ 遮挡保持：球槌压到球上时，球会被球槌挡住 —— 检测消失、
            # 轨道在宽限期（max_missed_frames）之后被丢弃、track_confirmed 变 False。
            # 而"球槌压到球上"恰恰是它该做的事，于是出现
            #     逼近球 → ATTACKING->RECOVERING(no puck) → 目标回 READY → 退开
            #     → 球又可见 → 再去 → 再次遮挡 → 循环
            # 现场日志里连续好几轮（mallet 239→175→143 退开 → 246→…→143 再来）。
            #
            # 判据：**球槌就停在最后那个目标附近**（说明是它自己挡的），
            # 且刚丢不久 → 保持原目标和原状态，让它把这一下做完。
            if (self._last_target is not None
                    and 0 <= now - self._last_seen_at <= OCCLUSION_HOLD_SECONDS
                    and math.dist(position, self._last_target) <= OCCLUSION_KEEP_RADIUS):
                target = self._last_target
                self.state = self._last_active_state
                held_target = True
                reason = "puck hidden by mallet: hold target"
            elif self._last_target is not None:
                # ★ 没保持成功时把原因写进 reason —— 现场一次就能看出是
                #   "超时"还是"球槌离太远"。
                #   age 取整到秒：否则每帧都变，日志会被这行刷屏，
                #   把真正的故障信息（比如"轴故障位=X"）淹掉。
                age = now - self._last_seen_at
                gap = math.dist(position, self._last_target)
                reason = f"no puck; READY (hold rejected: age={age:.0f}s gap={gap:.0f})"
        transitions = (() if self.state == previous else
                       (f"{previous.value} -> {self.state.value}: {reason}",))
        return RallyDecision(self.state, target, reason, direction, position, source, transitions,
                             held_target=held_target)
