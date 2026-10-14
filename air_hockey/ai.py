"""电脑球槌的目标决策。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from . import core_config as core
from .physics import clamp
from game_state import GameState
from .prediction import PredictionState, TrajectoryPredictor

AI_SERVE_SETUP_GAP = 6.0

# 近门保护：冰壶离自家球门太近（y <= back_line）时按兵不动，避免把它推进自家门。
#
# 2026-10-13 现场要求"全部追球"，所以关掉。想恢复防守式保护把它改回 True。
GUARD_OWN_GOAL_NEAR_LINE = False


@dataclass
class AIDecision:
    target_x: float
    target_y: float
    stalled_stone_phase: str
    reaction_timer: float = 0.0


class AirHockeyAI:
    """只根据状态快照给目标，不直接改游戏实体。"""

    def __init__(self, predictor=None, mallet_speed=None, recovery_bounds=None) -> None:
        self.predictor = predictor or TrajectoryPredictor()
        self._random = random.Random()
        # 球槌速度估计（球台单位/秒）。给了才做"提前量"：目标瞄冰壶再走 lead_time
        # 之后的位置，而不是它现在的位置。不给（None）= 保持原行为，仿真不受影响。
        self.mallet_speed = float(mallet_speed) if mallet_speed else None
        # ★ "死球回收"模式的粘性开关。见 choose_target 里静止球那一支：
        #   门控必须是粘性的，否则绕到球后面时（离球约 75 单位 > 门限）会被
        #   踢回"固定瞄球位置"，球槌再被拉回球边，来回震荡永远推不到球。
        self._retrieving = False
        # 回收阶段：GameState.stalled_stone_phase 在实机 live 下没有任何人推进它，
        # 永远是 "idle"，导致停滞球规划器永远进不了 "striking" —— 球槌绕到球后面
        # 又退回侧移，无限循环。所以由 AI 自己算，且只前进不后退。
        self._retrieve_step = 0
        # 可覆盖范围（left, right, top, bottom，游戏坐标）：由 main.py 从 PLC 的
        # 实际行程灌进来（plc.reachable_bounds）。None = 不额外限制。
        self.recovery_bounds = recovery_bounds

    def update(self, state: GameState, dt: float, *, prediction: PredictionState | None = None) -> AIDecision:
        """按难度反应延迟刷新目标。"""
        if state.awaiting_serve and state.current_server == "ai":
            return replace(self.choose_target(state, prediction=prediction), reaction_timer=state.reaction_timer)
        reaction_timer = state.reaction_timer - dt
        if reaction_timer <= 0:
            reaction_timer += state.difficulty.reaction_delay
            return replace(self.choose_target(state, prediction=prediction), reaction_timer=reaction_timer)
        return AIDecision(state.target_x, state.target_y, state.stalled_stone_phase, reaction_timer)

    def choose_target(self, state: GameState, *, prediction: PredictionState | None = None) -> AIDecision:
        if state.awaiting_serve:
            return self._choose_serve_target(state)

        difficulty = state.difficulty
        error = self._random.uniform(-difficulty.aim_error, difficulty.aim_error)
        safe_left = core.RINK_LEFT + core.MALLET_RADIUS
        safe_right = core.RINK_RIGHT - core.MALLET_RADIUS
        # 允许退到防守线后方：下限放到"球槌边缘正好贴到后墙"。
        # 原来下限是 state.ai_home_y，物理上从防守线(-33.2mm)往后到 -120mm
        # 那 87mm 的轴行程一直用不到。2026-10-13 现场要求"全部追"，改为放开。
        safe_top = core.RINK_TOP + core.MALLET_RADIUS
        safe_bottom = core.RINK_CENTER_Y - core.MALLET_RADIUS
        # ★★★ 机器真实可达范围（recovery_bounds，来自 PLC 的命令包络）必须在**这里**
        #     就收窄 safe_*，一次覆盖**所有**决策分支。
        #
        # 现场症状（2026-10 实机日志）：`目标被夹取 Y 31%` / `Y 43%` —— 三到四成的
        # 目标落在轴够不到的地方，PLC 在下发时默默夹到包络边缘。后果是机械臂长期
        # **顶在包络边缘**，任何过冲都会顶进「包络→软限位」那段余量，越界检测于是
        # 开火、撤销使能，一局就跑不下去。
        #
        # ⚠️ 原来这段只在下面"回收停滞球"那一支里做（_choose_stalled_stone_target
        #    之前），所以**防守/追击/拦截全部没受约束** —— 那才是夹取率高的原因。
        #    bounds 只该收窄、不该放宽，所以用 max/min 而不是直接赋值。
        if self.recovery_bounds is not None:
            left, right, top, bottom = self.recovery_bounds
            safe_left, safe_right = max(safe_left, left), min(safe_right, right)
            safe_top, safe_bottom = max(safe_top, top), min(safe_bottom, bottom)
        stone_speed = math.hypot(state.stone.vx, state.stone.vy)
        # ★ 球一旦动起来（或离开中线区）就退出"死球回收"模式。
        if stone_speed > core.STONE_STOP_SPEED:
            self._retrieving = False
            self._retrieve_step = 0
        stone_near_center = state.stone.y <= core.RINK_CENTER_Y + core.STONE_RADIUS + 2.0
        stone_behind_ai = state.stone.y < state.ai_y - core.STONE_RADIUS * 0.35
        if stone_behind_ai or (state.stalled_stone_phase != "idle" and stone_speed <= core.STONE_STOP_SPEED):
            # （safe_* 已经在上面按 recovery_bounds 收窄过了，这里不用再收一次）
            return self._choose_stalled_stone_target(state, safe_left, safe_right, safe_top, safe_bottom)
        stalled_stone_phase = "idle"
        stone_threatening_goal = state.stone.vy < -25.0

        stone_in_attack_zone = state.stone.y <= difficulty.attack_line
        if stone_speed <= core.STONE_STOP_SPEED and stone_near_center:
            # ★★★ 关键：目标"冰壶圆心"物理上做不到 —— 球槌和冰壶都是实体，
            # 不可能占同一个圆心。轴开到旁边（约 5 单位）就被死区停住，
            # 于是"顶着球不动"：日志里同一颗球在 (96,238) 附近被回收了 7 轮
            # （track 225→231）都没挪窝，每轮都是 到位 → 遮挡保持 1s 超时 →
            # 退回 READY → 再去 → 再到位 ……
            #
            # 修法分两步，不能一刀切：
            #   · 球槌还没到球边上 → 保持原来的"固定瞄球位置"（防抖动，有测试守着）
            #   · 球槌已经在球边上、球还是不动 → 说明这一支推不动它，
            #     改用停滞球规划器：侧移绕过 → 绕到球后面 → 对齐 → 从后往前推。
            #
            # 还要确认"推得动"：击球点是 stone.y + MALLET_RADIUS（球后面一个
            # 球槌半径），它必须还在球槌的行程里（≤ safe_bottom）。否则像停在
            # 中线上的球，推的位置在 349 之外、轴根本到不了，切规划器只是徒劳。
            #
            # ★ 门控必须是粘性的（self._retrieving）：绕到球后面时球槌离球约
            #   75 单位，已经超出 reach，若只看距离就会被踢回"固定瞄球位置"，
            #   球槌又被拉回球边 → sidestepping/positioning/idle 三态震荡，
            #   永远到不了击球点（模拟 14 步实测）。所以一旦进入回收就一直用
            #   规划器，直到球真的动起来（下面统一清标志）。
            reach = (core.STONE_RADIUS + core.MALLET_RADIUS) * 1.6
            can_push = state.stone.y + core.MALLET_RADIUS <= safe_bottom
            if not can_push:
                self._retrieving = False
                self._retrieve_step = 0
            elif self._retrieving or math.hypot(state.ai_x - state.stone.x,
                                                state.ai_y - state.stone.y) <= reach:
                self._retrieving = True
                # ★★ 阶段由谁决定（合并两个分支时定的规则）：
                #   · 状态里已经有非 idle 的阶段（仿真、以及将来 live 真的去推进它）
                #     → **以状态为准**，尊重上游的状态机
                #   · 状态是 "idle"（实机 live 的情况：vision_runtime 没人推进这个
                #     字段）→ 由 AI 自己算，否则规划器永远进不了 "striking"，
                #     球槌绕到球后面又退回侧移，无限循环
                phase = self._retrieve_phase(state)
                if state.stalled_stone_phase != "idle":
                    phase = state.stalled_stone_phase
                return self._choose_stalled_stone_target(
                    replace(state, stalled_stone_phase=phase),
                    safe_left, safe_right, safe_top, safe_bottom)
            # ⚠️ 这里**故意瞄球心** —— 击球本来就是"朝球撞过去"，压上去是预期的。
            #   （曾经想在这里改成"停在接触圆上"以防压住球，结果 10 个测试挂了：
            #    击球、首次进攻、回收判定全都依赖"朝球心去"。防护必须做在
            #    **只有"停在球边待命"那一种情形**上，不能一刀切。）
            return AIDecision(
                clamp(state.stone.x + error, safe_left, safe_right),
                clamp(state.stone.y, safe_top, safe_bottom),
                stalled_stone_phase,
            )

        # 冰壶在自己前方且进了攻击区才主动追。
        # 这里原来直接瞄冰壶当前位置 → 球动了轴才追，永远慢一拍；
        # 现在过一遍 _lead_target 加上球槌响应延迟的提前量。
        if stone_in_attack_zone and not stone_behind_ai:
            lead_x, lead_y = self._lead_target(state, state.stone.x, state.stone.y)
            return AIDecision(
                clamp(lead_x + error, safe_left, safe_right),
                clamp(lead_y, safe_top, safe_bottom),
                stalled_stone_phase,
            )

        if stone_threatening_goal:
            # 冲门的球：预判它会从防守线哪个 X 穿过（这一支本来就有提前量）。
            predicted_x = self._predict_stone_x(state, state.ai_home_y, prediction)
            blended_x = core.RINK_CENTER_X * (1.0 - difficulty.prediction) + predicted_x * difficulty.prediction
            target_x = clamp(blended_x + error, safe_left, safe_right)
        elif difficulty.attack_line >= core.RINK_CENTER_Y:
            # 实时首帧速度尚未知，先按当前位置对齐防守；仿真难度不变。
            lead_x, _lead_y = self._lead_target(state, state.stone.x, state.stone.y)
            target_x = clamp(lead_x + error, safe_left, safe_right)
        else:
            target_x = clamp(core.RINK_CENTER_X + error * 0.35, safe_left, safe_right)
        return AIDecision(target_x, state.ai_home_y, stalled_stone_phase)

    def advance_serve_phase(self, state: GameState, dt: float) -> str:
        """AI 开球时，就位了才切到击球阶段。"""
        if not (state.awaiting_serve and state.current_server == "ai" and state.serve_phase == "positioning"):
            return state.serve_phase
        setup_y = core.RINK_CENTER_Y - core.MALLET_RADIUS - core.STONE_RADIUS - AI_SERVE_SETUP_GAP
        arrival_tolerance = max(2.0, state.difficulty.ai_speed * dt)
        if math.hypot(state.ai_x - core.RINK_CENTER_X, state.ai_y - setup_y) <= arrival_tolerance:
            return "striking"
        return state.serve_phase

    @staticmethod
    def _choose_serve_target(state: GameState) -> AIDecision:
        if state.current_server == "ai":
            if state.serve_phase == "positioning":
                target_y = core.RINK_CENTER_Y - core.MALLET_RADIUS - core.STONE_RADIUS - AI_SERVE_SETUP_GAP
            else:
                target_y = core.RINK_CENTER_Y
        else:
            target_y = state.ai_home_y
        return AIDecision(core.RINK_CENTER_X, target_y, state.stalled_stone_phase)

    def _choose_stalled_stone_target(self, state, safe_left, safe_right, safe_top, safe_bottom) -> AIDecision:
        """先侧移、后退、对齐，避免穿过身后的冰壶造成乌龙球。"""
        minimum_distance = core.STONE_RADIUS + core.MALLET_RADIUS
        clearance = minimum_distance + 8.0
        tolerance = max(3.0, minimum_distance * 0.1)
        # 侧移途中沿用已经选择的方向，不因球心在球槌左右抖动而反复换边。
        side_x = state.target_x if state.stalled_stone_phase == "sidestepping" else state.ai_x
        side = 1.0 if side_x >= state.stone.x else -1.0
        if not safe_left <= state.stone.x + side * clearance <= safe_right:
            side = -side
        staging_x = clamp(state.stone.x + side * clearance, safe_left, safe_right)
        back_line = core.RINK_TOP + core.MALLET_RADIUS + core.GOAL_POST_RADIUS + 8.0
        staging_y = clamp(max(back_line, state.stone.y - clearance), safe_top, safe_bottom)
        striking_x = clamp(state.stone.x, safe_left, safe_right)
        striking = state.stalled_stone_phase == "striking" and state.stone.y >= state.ai_y
        if not striking and state.ai_y > staging_y + tolerance:
            if abs(state.ai_x - state.stone.x) < clearance - tolerance:
                return AIDecision(staging_x, state.ai_y, "sidestepping")
            return AIDecision(staging_x, staging_y, "positioning")
        # ponytail: below the post-clearance line, hold off rather than push toward our goal.
        # 2026-10-13 现场要求"全部追球"，这条近门保护由 GUARD_OWN_GOAL_NEAR_LINE 控制，
        # 默认已关闭。想恢复防守式保护把它改回 True。
        # 实机若连球心后方都退不到，不能对齐后从前侧把球推向自家门。
        if ((self.recovery_bounds is not None and state.stone.y <= safe_top)
                or (GUARD_OWN_GOAL_NEAR_LINE and state.stone.y <= back_line)):
            return AIDecision(staging_x, staging_y, "positioning")
        if not striking and abs(state.ai_x - striking_x) > tolerance:
            return AIDecision(striking_x, staging_y, "aligning")
        return AIDecision(
            striking_x,
            clamp(state.stone.y + core.MALLET_RADIUS, safe_top, safe_bottom),
            "striking",
        )

    # 提前量的上限（秒）。再长没有意义 —— 球早就变向或被撞了。
    MAX_LEAD_TIME = 0.6

    RETRIEVE_PHASES = ("sidestepping", "positioning", "aligning", "striking")

    def _retrieve_phase(self, state) -> str:
        """按球槌当前所在位置判断回收走到哪一步，且**只前进不后退**。

        停滞球规划器靠 state.stalled_stone_phase == "striking" 才肯给出
        "从后往前推"的目标；实机下那个字段永远是 "idle"，所以这里自己算。

        只前进很关键：开始推球时球槌会越过 staging 线，若按位置重算就会退回
        "sidestepping"，于是又绕一圈 —— 模拟里就是这样无限循环的。
        """
        clearance = core.STONE_RADIUS + core.MALLET_RADIUS + 8.0
        back_line = core.RINK_TOP + core.MALLET_RADIUS + core.GOAL_POST_RADIUS + 8.0
        staging_y = max(back_line, state.stone.y - clearance)
        tolerance = max(3.0, (core.STONE_RADIUS + core.MALLET_RADIUS) * 0.1)
        behind = state.ai_y <= staging_y + tolerance
        aligned = abs(state.ai_x - state.stone.x) <= tolerance
        if behind and aligned:
            natural = "striking"
        elif behind:
            natural = "aligning"
        elif abs(state.ai_x - state.stone.x) < clearance - tolerance:
            natural = "sidestepping"
        else:
            natural = "positioning"
        current = self.RETRIEVE_PHASES[self._retrieve_step]
        if self.RETRIEVE_PHASES.index(natural) > self._retrieve_step:
            current = natural
            self._retrieve_step = self.RETRIEVE_PHASES.index(natural)
        return current

    def _lead_target(self, state: GameState, lead_x: float, lead_y: float):
        """把目标从"冰壶现在的位置"提前到"球槌赶到时冰壶的位置"。

        现场反馈：轴总是慢一拍 —— 球离开一个位置之后轴才追到那里。
        原因是原来的目标就是冰壶的当前位置，完全没有提前量。

        做法是解拦截时间 t（球槌以速度 s 直线赶去拦截匀速运动的冰壶）：
            |d + v·t| = s·t     →     (v·v − s²)t² + 2(d·v)t + d·d = 0
        取最小的正根。

        ⚠️ 不能用"反复迭代距离/速度"的写法：当冰壶比球槌快时那个迭代会发散，
        目标会被一路推到场地角落（实测 mallet_speed=300、球速 447 时目标变成
        (533, 77) 那个远角，比直接瞄当前位置还差）。这里用闭式解，
        无正根（即追不上）时直接退回瞄当前位置。

        没有球槌速度模型时原样返回 —— 仿真与不传该参数时行为不变。
        """
        if self.mallet_speed is None or self.mallet_speed <= 0.0:
            return lead_x, lead_y
        stone = state.stone
        if math.hypot(stone.vx, stone.vy) <= core.STONE_STOP_SPEED:
            return lead_x, lead_y          # 基本静止，提前量没有意义

        dx = stone.x - state.ai_x
        dy = stone.y - state.ai_y
        s = self.mallet_speed
        a = stone.vx * stone.vx + stone.vy * stone.vy - s * s
        b = 2.0 * (dx * stone.vx + dy * stone.vy)
        c = dx * dx + dy * dy

        t = None
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                t = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                for candidate in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    if candidate > 0.0 and (t is None or candidate < t):
                        t = candidate
        if t is None or not math.isfinite(t) or t <= 0.0:
            return lead_x, lead_y          # 追不上：不外推，直接瞄现在的位置
        t = min(t, self.MAX_LEAD_TIME)
        return stone.x + stone.vx * t, stone.y + stone.vy * t

    def _predict_stone_x(
        self, state: GameState, target_y: float, prediction: PredictionState | None = None
    ) -> float:
        """从共享 TrajectoryPredictor 的 PredictionState 中获取冰壶到达 target_y 横线附近时的横坐标。"""
        trajectory = (prediction if prediction is not None else self.predictor.predict(state.stone)).trajectory
        if not trajectory:
            return state.stone.x
        if len(trajectory) == 1:
            return trajectory[0][0]

        for (x0, y0), (x1, y1) in zip(trajectory[:-1], trajectory[1:]):
            if (y0 <= target_y <= y1) or (y1 <= target_y <= y0):
                dy = y1 - y0
                if abs(dy) > 1e-9:
                    t = (target_y - y0) / dy
                    return x0 + t * (x1 - x0)
                return (x0 + x1) * 0.5

        closest_point = min(trajectory, key=lambda pt: abs(pt[1] - target_y))
        return closest_point[0]
