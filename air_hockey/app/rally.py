"""实时回合控制：把 AI 目标转成下发目标，处理放手、遮挡保持和回收不了的死球。

目标坐标直接取 AI 的连续决策，不按状态改写；RallyState（WAITING / DEFENDING /
ATTACKING / RECOVERING）主要用于显示与诊断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .. import core_config as core
from ..control.link import TARGET_MAX_AGE

if TYPE_CHECKING:
    from .vision_runtime import VisionResult

READY_POSITION = (core.RINK_CENTER_X, core.AI_HOME_Y)
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


# 遮挡保持：球槌压到球上时球被挡住，检测消失、轨道在宽限期后被丢弃；这时若回 READY，
# 球槌一退开球又出现，就会"逼近 → 遮挡 → 回位"循环。所以刚追过、且球槌还在最后目标附近时，
# 保持原目标让它把这一下做完。判据故意放宽（收紧时实测只保持一帧就放弃），超时一定回 READY。
OCCLUSION_HOLD_SECONDS = 1.0       # 刚丢多久之内算"可能只是被挡"
OCCLUSION_KEEP_RADIUS = 150.0      # 球槌离最后目标多近算"就是它挡的"（球台单位，约 136mm）

# 迟滞与防抖参数：消除状态机在分界线和速度零点附近的高频横跳
# （2026-10-29 场地 Y 方向改为各向同性单位，下面纯 Y 的量按 ×1.449 换算，毫米含义不变。）
MIDLINE_HYSTERESIS = 17.0          # 中线迟滞带（球台单位，约 16mm），进攻态需退至中线 +17 才解除
OUTGOING_VEL_THRESHOLD = 29.0      # 人类半场离开球门的速度阈值，低于此值不提前放弃防守
STATE_DWELL_SECONDS = 0.08         # 最小状态驻留时间（80ms），防止单帧图像噪点引起状态来回抽搐
# 放手迟滞：往外飞的球预测路线越过中线就放手；放手后要回落到中线以内这么多才重新进攻。
# 打软的球预测随速度估计抖动，没有迟滞会在放手/进攻之间来回切。
LET_GO_HYSTERESIS = 58.0
# 只有"刚被我们打过"的球才放手：球槌与冰壶中心距离进过接触距离 + TOUCH_MARGIN 才算碰过
# （PLC 位置约晚 20~30ms，留余量）。碰过之后 TOUCH_MEMORY 秒内往外飞才放手；
# 没碰过的球（例如从机器端角落弹出来的）照样进攻。play2 里三次真实击打最近距离 14~34。
TOUCH_MARGIN = 30.0
TOUCH_MEMORY = 1.0

class RallyController:
    def __init__(self, reachable_bounds: tuple | None = None):
        """reachable_bounds: (left, right, top, bottom) —— 球槌**真能到**的范围（游戏坐标），
        来自 `plc.reachable_bounds`。用来判断"这颗停住的球还回不回收得动"。
        """
        self.state = RallyState.WAITING
        self.reachable_bounds = reachable_bounds
        # 遮挡保持用的记忆（见 OCCLUSION_* 常量）
        self._last_target = None
        self._last_seen_at = 0.0
        self._last_active_state = RallyState.WAITING
        self._state_changed_at = 0.0
        self._letting_go = False
        self._touched_at = -math.inf

    def can_retrieve(self, stone) -> bool:
        """命令范围内是否存在朝真人方向接触的位置；不要求整段回撤间距。"""
        if self.reachable_bounds is None:
            return True
        left, right, top, bottom = self.reachable_bounds
        closest_x = min(max(stone.x, left), right)
        closest_y = min(stone.y, bottom)
        return (stone.y > top and math.hypot(stone.x - closest_x, stone.y - closest_y)
                < core.STONE_RADIUS + core.MALLET_RADIUS)

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
            # 速度方向判断引入死区：在真人半场只有明确朝对方门滑行（vy > OUTGOING_VEL_THRESHOLD）才算 outgoing；
            # 速度在零点附近波动或横向滑动时维持防守准备，避免水平横跳。
            if stone.speed <= core.STONE_STOP_SPEED:
                direction = "still"
            elif stone.vy > OUTGOING_VEL_THRESHOLD:
                direction = "outgoing"
            elif stone.vy < -core.STONE_STOP_SPEED:
                direction = "incoming"
            else:
                direction = "lateral"

            # 中线空间迟滞：若当前正在进攻，冰壶需退回到中线以远（> 中线 + MIDLINE_HYSTERESIS）才解除进攻；
            # 尚未进攻时，冰壶一旦踏入机器半场（<= 中线）立即触发进攻。
            if previous == RallyState.ATTACKING:
                machine_half = stone.y <= (core.RINK_CENTER_Y + MIDLINE_HYSTERESIS)
            else:
                machine_half = stone.y <= core.RINK_CENTER_Y

            puck_stopped = direction == "still"
            if math.dist(position, (stone.x, stone.y)) <= core.STONE_RADIUS + core.MALLET_RADIUS + TOUCH_MARGIN:
                self._touched_at = now
            # 冰壶往对方飞时，只有**刚被我们打过**、且它自己能飞出机器半场才放手回 READY（不追第二次）；
            # 打软了、按物理会停在机器半场的慢球，以及没碰过、从角落弹出来的球照样进攻。
            prediction = result.prediction
            # 判据是"预测路线会不会越过中线"，不是停点：快球会在对方半场撞墙弹回、停点落回我方半场，
            # 那也是先进了人那半场，回来时按来球防守。放手后回落到中线以内 LET_GO_HYSTERESIS 才重新进攻。
            line = core.RINK_CENTER_Y - (LET_GO_HYSTERESIS if self._letting_go else 0.0)
            leaving = direction == "outgoing" and (
                not machine_half
                or (now - self._touched_at <= TOUCH_MEMORY
                    and (prediction is None or max(point[1] for point in prediction.trajectory) > line)))
            self._letting_go = leaving
            if not leaving:
                # 状态防抖：在 ATTACKING 与 DEFENDING 之间切换时受最小驻留时间保护
                desired_state = RallyState.ATTACKING if machine_half else RallyState.DEFENDING
                if (previous in (RallyState.ATTACKING, RallyState.DEFENDING)
                        and desired_state != previous
                        and now - self._state_changed_at < STATE_DWELL_SECONDS):
                    self.state = previous
                else:
                    self.state = desired_state
                decision = result.ai_decision
                if decision is None or not all(math.isfinite(v) for v in (decision.target_x, decision.target_y)):
                    target = None
                    reason = "invalid/missing AI target; target cleared"
                elif machine_half and puck_stopped and not self.can_retrieve(stone):
                    # 不能盲目直穿身后的球回 READY，那会从前侧推向自家球门。
                    self.state = RallyState.RECOVERING
                    target = None
                    reason = (f"puck unreachable (stopped at y={stone.y:.0f}, "
                              f"no goal-side contact): target cleared; move puck manually")
                    self._last_target = None
                else:
                    if machine_half and puck_stopped:
                        # 死球（冰壶在自己半场停住）必须上去回收，否则回合停住
                        reason = "machine-half puck stopped: retrieve"
                    elif machine_half:
                        reason = "machine-half puck moving: pursue"
                    else:
                        reason = "human-half puck: defend/prepare"
                    target = (decision.target_x, decision.target_y)
                    self._last_target = target
                    self._last_active_state = self.state
            else:
                reason = "puck moving away; READY"
                # 主动回 READY 时清掉旧目标，否则之后一次遮挡会把陈旧的追击点复活
                self._last_target = None
            # 只要这一帧有确认的轨道（不管走哪一支）就刷新，否则夹一帧"球飞走"保持窗口就会悄悄过期
            self._last_seen_at = now
        else:
            # 遮挡保持（见 OCCLUSION_* 常量）
            if (self._last_target is not None
                    and 0 <= now - self._last_seen_at <= OCCLUSION_HOLD_SECONDS
                    and math.dist(position, self._last_target) <= OCCLUSION_KEEP_RADIUS):
                target = self._last_target
                self.state = self._last_active_state
                held_target = True
                reason = "puck hidden by mallet: hold target"
            elif self._last_target is not None:
                # 没保持成功时写明是超时还是球槌离太远；age 取整到秒，免得每帧都变、把日志刷屏
                age = now - self._last_seen_at
                gap = math.dist(position, self._last_target)
                reason = f"no puck; READY (hold rejected: age={age:.0f}s gap={gap:.0f})"
        if self.state != previous:
            self._state_changed_at = now
            transitions = (f"{previous.value} -> {self.state.value}: {reason}",)
        else:
            transitions = ()
        return RallyDecision(self.state, target, reason, direction, position, source, transitions,
                             held_target=held_target)
