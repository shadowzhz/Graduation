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
        stone_speed = math.hypot(state.stone.vx, state.stone.vy)
        stone_near_center = state.stone.y <= core.RINK_CENTER_Y + core.STONE_RADIUS + 2.0
        stone_behind_ai = state.stone.y < state.ai_y - core.STONE_RADIUS * 0.35
        if stone_behind_ai or (state.stalled_stone_phase != "idle" and stone_speed <= core.STONE_STOP_SPEED):
            if self.recovery_bounds is not None:
                left, right, top, bottom = self.recovery_bounds
                safe_left, safe_right = max(safe_left, left), min(safe_right, right)
                safe_top, safe_bottom = max(safe_top, top), min(safe_bottom, bottom)
            return self._choose_stalled_stone_target(state, safe_left, safe_right, safe_top, safe_bottom)
        stalled_stone_phase = "idle"
        stone_threatening_goal = state.stone.vy < -25.0

        stone_in_attack_zone = state.stone.y <= difficulty.attack_line
        if stone_speed <= core.STONE_STOP_SPEED and stone_near_center:
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
