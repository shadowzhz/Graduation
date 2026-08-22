"""空气冰球电脑球槌的目标决策。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import air_hockey_config as layout
from air_hockey_physics import clamp, reflected_coordinate


AI_SERVE_SETUP_GAP = 6.0


@dataclass(frozen=True)
class AIGameState:
    """AI 在一次决策中需要的只读游戏状态快照。"""

    ai_x: float
    ai_y: float
    ai_home_y: float
    target_x: float
    target_y: float
    puck_x: float
    puck_y: float
    puck_velocity_x: float
    puck_velocity_y: float
    awaiting_serve: bool
    current_server: str
    serve_phase: str
    stalled_puck_phase: str
    reaction_timer: float
    difficulty: layout.Difficulty


@dataclass(frozen=True)
class AIDecision:
    """AI 返回给游戏控制器的目标与内部阶段变更。"""

    target_x: float
    target_y: float
    stalled_puck_phase: str
    reaction_timer: float = 0.0


class AirHockeyAI:
    """根据状态快照选择 AI 球槌目标；不修改游戏实体。"""

    def __init__(self) -> None:
        self._random = random.Random()

    def update(self, state: AIGameState, dt: float) -> AIDecision:
        """按既有反应延迟刷新 AI 目标，并返回下一次反应计时。"""
        if state.awaiting_serve and state.current_server == "ai":
            return replace(self.choose_target(state), reaction_timer=state.reaction_timer)
        reaction_timer = state.reaction_timer - dt
        if reaction_timer <= 0:
            reaction_timer += state.difficulty.reaction_delay
            return replace(self.choose_target(state), reaction_timer=reaction_timer)
        return AIDecision(state.target_x, state.target_y, state.stalled_puck_phase, reaction_timer)

    def choose_target(self, state: AIGameState) -> AIDecision:
        """返回当前状态下的球槌目标和静止球阶段。"""
        if state.awaiting_serve:
            return self._choose_serve_target(state)

        difficulty = state.difficulty
        error = self._random.uniform(-difficulty.aim_error, difficulty.aim_error)
        safe_left = layout.RINK_LEFT + layout.MALLET_RADIUS
        safe_right = layout.RINK_RIGHT - layout.MALLET_RADIUS
        puck_speed = math.hypot(state.puck_velocity_x, state.puck_velocity_y)
        puck_near_center = state.puck_y <= layout.RINK_CENTER_Y + layout.PUCK_RADIUS + 2 * layout.UI_SCALE

        if state.stalled_puck_phase != "idle":
            if puck_speed <= layout.PUCK_STOP_SPEED:
                return self._choose_stalled_puck_target(state, safe_left, safe_right)
            stalled_puck_phase = "idle"
        else:
            stalled_puck_phase = "idle"

        # 冰球已经跑到 AI 球槌的“身后”时，绝不能继续追球。
        # 此时追击会从冰球下方再次击球，把冰球重新推向 AI 自己的球门。
        puck_behind_ai = state.puck_y < state.ai_y - layout.PUCK_RADIUS * 0.35
        puck_threatening_goal = state.puck_velocity_y < -25 * layout.UI_SCALE
        if puck_behind_ai:
            # 冰球在 AI 身后停住时，不能只回防到中路，否则双方都会等球而死锁。
            if puck_speed <= layout.PUCK_STOP_SPEED:
                return self._choose_stalled_puck_target(
                    state,
                    safe_left,
                    safe_right,
                    stalled_puck_phase="positioning",
                )
            if puck_threatening_goal:
                # 防守时只做横向封堵，不主动向冰球移动。
                predicted_x = self._predict_puck_x(state, state.ai_home_y)
                target_x = clamp(predicted_x + error * 0.5, safe_left, safe_right)
            else:
                target_x = clamp(layout.RINK_CENTER_X + error * 0.25, safe_left, safe_right)
            return AIDecision(target_x, state.ai_home_y, stalled_puck_phase)

        puck_in_attack_zone = state.puck_y <= difficulty.attack_line
        if puck_speed <= layout.PUCK_STOP_SPEED and puck_near_center:
            return AIDecision(
                clamp(state.puck_x + error, safe_left, safe_right),
                clamp(state.puck_y, layout.RINK_TOP + layout.MALLET_RADIUS, layout.RINK_CENTER_Y - layout.MALLET_RADIUS),
                stalled_puck_phase,
            )

        # 只有冰球位于 AI 前方并进入攻击区时才主动追击。
        if puck_in_attack_zone and not puck_behind_ai:
            return AIDecision(
                clamp(state.puck_x + error, safe_left, safe_right),
                clamp(state.puck_y, state.ai_home_y, layout.RINK_CENTER_Y - layout.MALLET_RADIUS),
                stalled_puck_phase,
            )

        if puck_threatening_goal:
            predicted_x = self._predict_puck_x(state, state.ai_home_y)
            blended_x = layout.RINK_CENTER_X * (1.0 - difficulty.prediction) + predicted_x * difficulty.prediction
            target_x = clamp(blended_x + error, safe_left, safe_right)
        else:
            target_x = clamp(layout.RINK_CENTER_X + error * 0.35, safe_left, safe_right)
        return AIDecision(target_x, state.ai_home_y, stalled_puck_phase)

    def advance_serve_phase(self, state: AIGameState, dt: float) -> str:
        """根据实际移动后的位置决定 AI 开球是否由就位转为击球。"""
        if not (state.awaiting_serve and state.current_server == "ai" and state.serve_phase == "positioning"):
            return state.serve_phase
        setup_y = layout.RINK_CENTER_Y - layout.MALLET_RADIUS - layout.PUCK_RADIUS - AI_SERVE_SETUP_GAP * layout.UI_SCALE
        arrival_tolerance = max(2.0 * layout.UI_SCALE, state.difficulty.ai_speed * dt)
        if math.hypot(state.ai_x - layout.RINK_CENTER_X, state.ai_y - setup_y) <= arrival_tolerance:
            return "striking"
        return state.serve_phase

    @staticmethod
    def _choose_serve_target(state: AIGameState) -> AIDecision:
        if state.current_server == "ai":
            if state.serve_phase == "positioning":
                target_y = layout.RINK_CENTER_Y - layout.MALLET_RADIUS - layout.PUCK_RADIUS - AI_SERVE_SETUP_GAP * layout.UI_SCALE
            else:
                target_y = layout.RINK_CENTER_Y
        else:
            target_y = state.ai_home_y
        return AIDecision(layout.RINK_CENTER_X, target_y, state.stalled_puck_phase)

    @staticmethod
    def _choose_stalled_puck_target(
        state: AIGameState,
        safe_left: float,
        safe_right: float,
        stalled_puck_phase: str | None = None,
    ) -> AIDecision:
        """从球门侧绕到静止冰球旁边，再把它朝玩家半场击出。"""
        phase = state.stalled_puck_phase if stalled_puck_phase is None else stalled_puck_phase
        minimum_distance = layout.PUCK_RADIUS + layout.MALLET_RADIUS
        side = 1.0 if state.puck_x <= layout.RINK_CENTER_X else -1.0
        staging_x = clamp(state.puck_x + side * (minimum_distance + 8.0 * layout.UI_SCALE), safe_left, safe_right)
        staging_y = layout.RINK_TOP + layout.MALLET_RADIUS
        if phase == "positioning":
            if math.hypot(state.ai_x - staging_x, state.ai_y - staging_y) <= max(3.0 * layout.UI_SCALE, minimum_distance * 0.1):
                phase = "striking"
            return AIDecision(staging_x, staging_y, phase)
        return AIDecision(
            clamp(state.puck_x - side * minimum_distance, safe_left, safe_right),
            clamp(state.puck_y + layout.MALLET_RADIUS, layout.RINK_TOP + layout.MALLET_RADIUS, layout.RINK_CENTER_Y - layout.MALLET_RADIUS),
            phase,
        )

    @staticmethod
    def _predict_puck_x(state: AIGameState, target_y: float) -> float:
        """按现有减速和边墙反射公式预测冰球在目标横线的横坐标。"""
        speed = math.hypot(state.puck_velocity_x, state.puck_velocity_y)
        if speed <= layout.COLLISION_EPSILON or state.puck_velocity_y >= -layout.COLLISION_EPSILON:
            return reflected_coordinate(state.puck_x, layout.RINK_LEFT + layout.PUCK_RADIUS, layout.RINK_RIGHT - layout.PUCK_RADIUS)
        direction_x = state.puck_velocity_x / speed
        direction_y = state.puck_velocity_y / speed
        distance_to_target = (target_y - state.puck_y) / direction_y
        if distance_to_target <= 0:
            return reflected_coordinate(state.puck_x, layout.RINK_LEFT + layout.PUCK_RADIUS, layout.RINK_RIGHT - layout.PUCK_RADIUS)
        if layout.PUCK_FRICTION_DECELERATION <= layout.COLLISION_EPSILON:
            travel_distance = distance_to_target
        else:
            stopping_distance = speed * speed / (2.0 * layout.PUCK_FRICTION_DECELERATION)
            travel_distance = min(distance_to_target, stopping_distance)
        projected_x = state.puck_x + direction_x * travel_distance
        return reflected_coordinate(projected_x, layout.RINK_LEFT + layout.PUCK_RADIUS, layout.RINK_RIGHT - layout.PUCK_RADIUS)
