"""电脑球槌的目标决策。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from . import core_config as core
from .physics import StoneMotion, clamp
from .game_state import GameState
from .prediction import PredictionState, TrajectoryPredictor

AI_SERVE_SETUP_GAP = 6.0

# 冲门快球"守门线"模式的迟滞。守门线（Y=防守线）和追球（Y≈冰壶）两种目标在 Y 上
# 相差约 160 单位（≈210mm），判据若只是单个阈值，速度估计的噪声就会让 Y 目标在两者
# 之间来回跳（仿真：±15 速度噪声下 1.2s 内跳 37 次），实机表现为前后急刹、反复折返。
# 进入要求明确冲门，退出要求明确减速或明确偏离球门，中间保持上一次的判断。
# （2026-10-29 场地 Y 方向改为各向同性单位，原 -60 / -20 按 ×1.449 换算，毫米含义不变。）
FAST_SHOT_ENTER_VY = -87.0
FAST_SHOT_EXIT_VY = -29.0
# 非冲门来球的判据（同上换算，原 -25）
INCOMING_VY = -36.0
FAST_SHOT_ENTER_MARGIN = core.STONE_RADIUS
FAST_SHOT_EXIT_MARGIN = core.STONE_RADIUS * 3.0
# 迎击：终点模式下球槌开到目标就停住，球只是撞上一个静止的球槌弹回去。
# 冰壶再过 STRIKE_WINDOW 秒就碰到球槌时，把目标沿来球方向往回推 STRIKE_THROUGH，
# 球槌在接触瞬间还在朝球冲，把球打回去。窗口之前照常等在原位，不提前离开防守位置。
STRIKE_WINDOW = 0.15
STRIKE_THROUGH = 60.0


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
        # 球槌速度估计（球台单位/秒）；给了才做"提前量"，None 时直接瞄冰壶当前位置
        self.mallet_speed = float(mallet_speed) if mallet_speed else None
        # 死球回收的粘性开关：绕到球后面时已离球超出门限，不粘住会被拉回球边来回震荡
        self._retrieving = False
        # 回收走到第几步（只前进不后退）。实机没人推进 GameState.stalled_stone_phase，由 AI 自己算
        self._retrieve_step = 0
        # 可覆盖范围（left, right, top, bottom，游戏坐标），来自 plc.reachable_bounds；None = 不额外限制
        self.recovery_bounds = recovery_bounds
        # 冲门快球"守门线"模式（带迟滞，见 FAST_SHOT_* 常量）
        self._goal_guard = False

    def reset(self) -> None:
        """清掉跨帧的粘性状态（换了一颗球 / 轨迹丢失时由调用方调用）。

        死球回收和冲门迟滞都是"针对这颗球"的记忆；不清的话，上一颗球回收到 striking
        之后，新放的一颗静止球会直接从 striking 开始，跳过侧移/绕后/对齐斜着去推。
        """
        self._retrieving = False
        self._retrieve_step = 0
        self._goal_guard = False

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
        # 下限放到球槌边缘贴后墙，防守线后方的行程也能用
        safe_top = core.RINK_TOP + core.MALLET_RADIUS
        safe_bottom = core.RINK_CENTER_Y - core.MALLET_RADIUS
        # 机器真实可达范围在这里统一收窄、覆盖所有分支：否则 AI 会瞄轴够不到的点，
        # 下发时被夹到包络边缘，机械臂长期顶在边上。只收窄不放宽。
        if self.recovery_bounds is not None:
            left, right, top, bottom = self.recovery_bounds
            safe_left, safe_right = max(safe_left, left), min(safe_right, right)
            safe_top, safe_bottom = max(safe_top, top), min(safe_bottom, bottom)
        stone_speed = math.hypot(state.stone.vx, state.stone.vy)
        # 球一旦动起来就退出死球回收
        if stone_speed > core.STONE_STOP_SPEED:
            self._retrieving = False
            self._retrieve_step = 0
        stone_near_center = state.stone.y <= core.RINK_CENTER_Y + core.STONE_RADIUS + 2.0
        # 冲门判断与下面的防守分支共用一次预测（实时链路会直接传进来）
        if prediction is None and state.stone.vy < FAST_SHOT_EXIT_VY:
            prediction = self.predictor.predict(state.stone)
        stone_behind_ai = state.stone.y < state.ai_y - core.STONE_RADIUS * 0.35
        if stone_behind_ai or (state.stalled_stone_phase != "idle" and stone_speed <= core.STONE_STOP_SPEED):
            self._goal_guard = False
            return self._choose_stalled_stone_target(state, safe_left, safe_right, safe_top, safe_bottom)
        stalled_stone_phase = "idle"

        stone_in_attack_zone = state.stone.y <= difficulty.attack_line
        if stone_speed <= core.STONE_STOP_SPEED and stone_near_center:
            self._goal_guard = False
            # 球心本身到不了（两个实体不能重叠，轴停在旁边被死区卡住就顶着球不动）。
            # 球槌还没到球边：照旧瞄球心；已在球边而球不动：改用停滞球规划器绕到后面推。
            # 击球点 stone.y + MALLET_RADIUS 必须在行程内才推得动；进入回收后保持粘性直到球动起来。
            reach = (core.STONE_RADIUS + core.MALLET_RADIUS) * 1.6
            can_push = state.stone.y + core.MALLET_RADIUS <= safe_bottom
            if not can_push:
                self._retrieving = False
                self._retrieve_step = 0
            elif self._retrieving or math.hypot(state.ai_x - state.stone.x,
                                                state.ai_y - state.stone.y) <= reach:
                self._retrieving = True
                # 状态里已有非 idle 阶段（仿真）就以状态为准，否则（实机）由 AI 自己推进
                phase = self._retrieve_phase(state)
                if state.stalled_stone_phase != "idle":
                    phase = state.stalled_stone_phase
                return self._choose_stalled_stone_target(
                    replace(state, stalled_stone_phase=phase),
                    safe_left, safe_right, safe_top, safe_bottom)
            # 这里故意瞄球心：击球就是朝球撞过去
            return AIDecision(
                clamp(state.stone.x + error, safe_left, safe_right),
                clamp(state.stone.y, safe_top, safe_bottom),
                stalled_stone_phase,
            )

        # 冲门快球拦截优先：冰壶冲向我方球门（vy 足够负、在球槌前方），
        # 预测轨迹真的能滑到防守线，且穿过防守线的位置落在球门有效防御区内：
        # 不前扑迎球（迎球会缩短反应时间导致漏球），守在防守线（Y = ai_home_y）
        # 提前横移到预判截获点堵截。进入/退出带迟滞，见 FAST_SHOT_* 常量。
        guard = self._goal_guard_decision(state, prediction, error,
                                          safe_left, safe_right, safe_top, safe_bottom)
        if guard is not None:
            return guard

        # 冰壶在自己前方且进了攻击区才主动追；_lead_target 补上球槌响应延迟的提前量
        if stone_in_attack_zone and not stone_behind_ai:
            lead_x, lead_y = self._lead_target(state, state.stone.x, state.stone.y)
            strike_x, strike_y = self._strike_through(state, lead_x + error, lead_y)
            return AIDecision(
                clamp(strike_x, safe_left, safe_right),
                clamp(strike_y, safe_top, safe_bottom),
                stalled_stone_phase,
            )

        if state.stone.vy < INCOMING_VY:
            # 冲门快球之外的来球（含先撞墙再进门的）：预判它从防守线哪个 X 穿过。
            # 仿真难度的 prediction 系数靠这一支区分强弱，不能并进冲门快球那一支。
            predicted_x = self._predict_stone_x(state, state.ai_home_y, prediction)
            target_x = clamp(self._blend_with_center(predicted_x, difficulty) + error, safe_left, safe_right)
        elif difficulty.attack_line >= core.RINK_CENTER_Y:
            # 实时首帧速度尚未知，先按当前位置对齐防守；仿真难度不变。
            lead_x, _lead_y = self._lead_target(state, state.stone.x, state.stone.y)
            target_x = clamp(lead_x + error, safe_left, safe_right)
        else:
            target_x = clamp(core.RINK_CENTER_X + error * 0.35, safe_left, safe_right)
        strike_x, strike_y = self._strike_through(state, target_x, state.ai_home_y)
        return AIDecision(clamp(strike_x, safe_left, safe_right), clamp(strike_y, safe_top, safe_bottom),
                          stalled_stone_phase)

    def _goal_guard_decision(self, state, prediction, error, safe_left, safe_right, safe_top, safe_bottom):
        """冲门球：守在防守线上、横移到穿线点堵截；不是冲门球返回 None。"""
        guard_y = state.ai_home_y
        vy_limit = FAST_SHOT_EXIT_VY if self._goal_guard else FAST_SHOT_ENTER_VY
        margin = FAST_SHOT_EXIT_MARGIN if self._goal_guard else FAST_SHOT_ENTER_MARGIN
        self._goal_guard = False
        if state.stone.vy >= vy_limit:
            return None
        # 滑不到这条线就停下的慢球不算冲门 —— 这种球应该上前击打
        if not any(point_y <= guard_y for _x, point_y in prediction.trajectory):
            return None
        predicted_x = self._predict_stone_x(state, guard_y, prediction)
        if not core.GOAL_LEFT - margin <= predicted_x <= core.GOAL_RIGHT + margin:
            return None
        self._goal_guard = True
        blended_x = self._blend_with_center(predicted_x, state.difficulty)
        strike_x, strike_y = self._strike_through(state, blended_x + error, guard_y)
        return AIDecision(clamp(strike_x, safe_left, safe_right), clamp(strike_y, safe_top, safe_bottom), "idle")

    @staticmethod
    def _blend_with_center(predicted_x, difficulty):
        """按难度的 prediction 系数，在场地中线和预测穿线点之间取值。"""
        return core.RINK_CENTER_X * (1.0 - difficulty.prediction) + predicted_x * difficulty.prediction

    @staticmethod
    def _strike_through(state: GameState, target_x: float, target_y: float):
        """冰壶快要碰到等在 target 的球槌时，把目标沿来球方向往回推，让球槌迎着球打出去。"""
        vx, vy = state.stone.vx, state.stone.vy
        speed = math.hypot(vx, vy)
        if speed <= core.STONE_STOP_SPEED:
            return target_x, target_y
        dx, dy = target_x - state.stone.x, target_y - state.stone.y
        contact = core.STONE_RADIUS + core.MALLET_RADIUS
        gap = math.hypot(dx, dy) - contact
        # 球槌得已经**等在**那个位置附近：还在赶路时它本来就在动，往回推只会把它带离拦截点
        waiting = math.hypot(state.ai_x - target_x, state.ai_y - target_y) <= STRIKE_THROUGH
        # 等的位置要在冰壶**现在这条直线**上（中间不撞墙）才迎击；要先撞墙的，按现在方向冲出去会打空
        on_path = abs(dx * vy - dy * vx) / speed <= contact
        if not waiting or dx * vx + dy * vy <= 0.0 or not on_path or gap / speed > STRIKE_WINDOW:
            return target_x, target_y
        return target_x - vx / speed * STRIKE_THROUGH, target_y - vy / speed * STRIKE_THROUGH

    def advance_serve_phase(self, state: GameState, dt: float) -> str:
        """AI 开球时，就位了才切到击球阶段。"""
        if not (state.awaiting_serve and state.current_server == "ai" and state.serve_phase == "positioning"):
            return state.serve_phase
        arrival_tolerance = max(2.0, state.difficulty.ai_speed * dt)
        if math.hypot(state.ai_x - core.RINK_CENTER_X, state.ai_y - self._serve_setup_y()) <= arrival_tolerance:
            return "striking"
        return state.serve_phase

    @staticmethod
    def _serve_setup_y() -> float:
        """AI 开球时球槌的站位：中线后方隔一个冰壶。"""
        return core.RINK_CENTER_Y - core.MALLET_RADIUS - core.STONE_RADIUS - AI_SERVE_SETUP_GAP

    def _choose_serve_target(self, state: GameState) -> AIDecision:
        if state.current_server == "ai":
            if state.serve_phase == "positioning":
                target_y = self._serve_setup_y()
            else:
                target_y = core.RINK_CENTER_Y
        else:
            target_y = state.ai_home_y
        return AIDecision(core.RINK_CENTER_X, target_y, state.stalled_stone_phase)

    def _choose_stalled_stone_target(self, state, safe_left, safe_right, safe_top, safe_bottom) -> AIDecision:
        """先侧移、后退、对齐，避免穿过身后的冰壶造成乌龙球。"""
        clearance, tolerance, staging_y = self._retrieve_geometry(state.stone.y)
        staging_y = clamp(staging_y, safe_top, safe_bottom)
        # 侧移途中沿用已经选择的方向，不因球心在球槌左右抖动而反复换边。
        side_x = state.target_x if state.stalled_stone_phase == "sidestepping" else state.ai_x
        side = 1.0 if side_x >= state.stone.x else -1.0
        if not safe_left <= state.stone.x + side * clearance <= safe_right:
            side = -side
        staging_x = clamp(state.stone.x + side * clearance, safe_left, safe_right)
        striking_x = clamp(state.stone.x, safe_left, safe_right)
        striking = state.stalled_stone_phase == "striking" and state.stone.y >= state.ai_y
        if not striking and state.ai_y > staging_y + tolerance:
            if abs(state.ai_x - state.stone.x) < clearance - tolerance:
                return AIDecision(staging_x, state.ai_y, "sidestepping")
            return AIDecision(staging_x, staging_y, "positioning")
        # 实机若连球心后方都退不到，不能对齐后从前侧把球推向自家门。
        if self.recovery_bounds is not None and state.stone.y <= safe_top:
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

    @staticmethod
    def _retrieve_geometry(stone_y):
        """回收用的 (侧移间距, 到位容差, 未夹取的绕后站位 y)。"""
        contact = core.STONE_RADIUS + core.MALLET_RADIUS
        clearance = contact + 8.0
        back_line = core.RINK_TOP + core.MALLET_RADIUS + core.GOAL_POST_RADIUS + 8.0
        return clearance, max(3.0, contact * 0.1), max(back_line, stone_y - clearance)

    def _retrieve_phase(self, state) -> str:
        """按球槌当前位置判断回收走到哪一步，只前进不后退。

        推球时球槌会越过绕后站位线，若按位置重算就会退回侧移、又绕一圈。
        """
        clearance, tolerance, staging_y = self._retrieve_geometry(state.stone.y)
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
        """把目标从"冰壶现在的位置"提前到"球槌来得及赶到的、冰壶将要经过的位置"。

        按真实物理（摩擦、撞墙反弹）逐微步推演冰壶，取第一个满足
            |冰壶(t) − 球槌| ≤ mallet_speed · t
        的点；MAX_LEAD_TIME 内都赶不上时，瞄冰壶在 MAX_LEAD_TIME 时的位置（仍在它前方的路径上）。
        追不上时不能退回瞄冰壶当前位置（会把球槌从拦截点叫走、一路追在球后面），
        也不能反复迭代"距离/速度"（冰壶比球槌快时会发散）。没有球槌速度模型时原样返回。
        """
        if self.mallet_speed is None or self.mallet_speed <= 0.0:
            return lead_x, lead_y
        stone = state.stone
        if math.hypot(stone.vx, stone.vy) <= core.STONE_STOP_SPEED:
            return lead_x, lead_y          # 基本静止，提前量没有意义

        motion = StoneMotion(stone.x, stone.y)
        motion.set_immediate_velocity(stone.vx, stone.vy)
        substep = core.PREDICTION_SUBSTEP
        elapsed = 0.0
        while elapsed < self.MAX_LEAD_TIME - 1e-9:
            step_dt = min(substep, self.MAX_LEAD_TIME - elapsed)
            motion.step(step_dt)
            elapsed += step_dt
            if math.hypot(motion.x - state.ai_x, motion.y - state.ai_y) <= self.mallet_speed * elapsed:
                break
            if abs(motion.vx) < 1e-9 and abs(motion.vy) < 1e-9 and not motion.response_active:
                break
        return motion.x, motion.y

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
