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


@dataclass
class AIDecision:
    target_x: float
    target_y: float
    stalled_stone_phase: str
    reaction_timer: float = 0.0


class AirHockeyAI:
    """只根据状态快照给目标，不直接改游戏实体。"""

    def __init__(self, predictor=None) -> None:
        self.predictor = predictor or TrajectoryPredictor()
        self._random = random.Random()

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
        stone_speed = math.hypot(state.stone.vx, state.stone.vy)
        stone_near_center = state.stone.y <= core.RINK_CENTER_Y + core.STONE_RADIUS + 2.0
        stone_behind_ai = state.stone.y < state.ai_y - core.STONE_RADIUS * 0.35
        if stone_behind_ai or (state.stalled_stone_phase != "idle" and stone_speed <= core.STONE_STOP_SPEED):
            return self._choose_stalled_stone_target(state, safe_left, safe_right)
        stalled_stone_phase = "idle"
        stone_threatening_goal = state.stone.vy < -25.0

        stone_in_attack_zone = state.stone.y <= difficulty.attack_line
        if stone_speed <= core.STONE_STOP_SPEED and stone_near_center:
            return AIDecision(
                clamp(state.stone.x + error, safe_left, safe_right),
                clamp(state.stone.y, core.RINK_TOP + core.MALLET_RADIUS, core.RINK_CENTER_Y - core.MALLET_RADIUS),
                stalled_stone_phase,
            )

        # 冰壶在自己前方且进了攻击区才主动追
        if stone_in_attack_zone and not stone_behind_ai:
            return AIDecision(
                clamp(state.stone.x + error, safe_left, safe_right),
                clamp(state.stone.y, state.ai_home_y, core.RINK_CENTER_Y - core.MALLET_RADIUS),
                stalled_stone_phase,
            )

        if stone_threatening_goal:
            predicted_x = self._predict_stone_x(state, state.ai_home_y, prediction)
            blended_x = core.RINK_CENTER_X * (1.0 - difficulty.prediction) + predicted_x * difficulty.prediction
            target_x = clamp(blended_x + error, safe_left, safe_right)
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

    @staticmethod
    def _choose_stalled_stone_target(state, safe_left, safe_right) -> AIDecision:
        """先侧移、后退、对齐，避免穿过身后的冰壶造成乌龙球。"""
        minimum_distance = core.STONE_RADIUS + core.MALLET_RADIUS
        clearance = minimum_distance + 8.0
        tolerance = max(3.0, minimum_distance * 0.1)
        side = 1.0 if state.ai_x >= state.stone.x else -1.0
        if not safe_left <= state.stone.x + side * clearance <= safe_right:
            side = -side
        staging_x = clamp(state.stone.x + side * clearance, safe_left, safe_right)
        back_line = core.RINK_TOP + core.MALLET_RADIUS + core.GOAL_POST_RADIUS + 8.0
        staging_y = max(back_line, state.stone.y - clearance)
        striking = state.stalled_stone_phase == "striking" and state.stone.y >= state.ai_y
        if not striking and state.ai_y > staging_y + tolerance:
            if abs(state.ai_x - state.stone.x) < clearance - tolerance:
                return AIDecision(staging_x, state.ai_y, "sidestepping")
            return AIDecision(state.ai_x, staging_y, "positioning")
        # ponytail: below the post-clearance line, hold off rather than push toward our goal.
        if state.stone.y <= back_line:
            return AIDecision(staging_x, staging_y, "positioning")
        if not striking and abs(state.ai_x - state.stone.x) > tolerance:
            return AIDecision(clamp(state.stone.x, safe_left, safe_right), staging_y, "aligning")
        return AIDecision(
            clamp(state.stone.x, safe_left, safe_right),
            clamp(state.stone.y + core.MALLET_RADIUS, core.RINK_TOP + core.MALLET_RADIUS, core.RINK_CENTER_Y - core.MALLET_RADIUS),
            "striking",
        )

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
