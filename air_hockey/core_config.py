"""空气冰壶共享核心算法与物理配置。

包含所有与 GUI/显示解耦的物理场地几何、尺寸、动力学参数、预测器参数及难度等级。
所有参数均为固定基准值，不受仿真窗口或屏幕缩放影响。
"""

from __future__ import annotations

from dataclasses import dataclass

# 场地基准尺寸与边界
BASE_CANVAS_WIDTH = 600.0
BASE_CANVAS_HEIGHT = 760.0

RINK_LEFT = 36.0
RINK_RIGHT = BASE_CANVAS_WIDTH - 36.0  # 564.0
RINK_TOP = 46.0
RINK_BOTTOM = BASE_CANVAS_HEIGHT - 46.0  # 714.0
RINK_CENTER_X = BASE_CANVAS_WIDTH / 2    # 300.0
RINK_CENTER_Y = BASE_CANVAS_HEIGHT / 2   # 380.0

# 球门几何参数
GOAL_HALF_WIDTH = 92.0
GOAL_LEFT = RINK_CENTER_X - GOAL_HALF_WIDTH   # 208.0
GOAL_RIGHT = RINK_CENTER_X + GOAL_HALF_WIDTH  # 392.0
GOAL_DEPTH = 28.0

# 球与球槌尺寸
STONE_RADIUS = 14.0
MALLET_RADIUS = 31.0
GOAL_POST_RADIUS = 6.0
GOAL_POSTS = (
    (GOAL_LEFT, RINK_TOP),
    (GOAL_RIGHT, RINK_TOP),
    (GOAL_LEFT, RINK_BOTTOM),
    (GOAL_RIGHT, RINK_BOTTOM),
)

# 物理动力学参数
PLAYER_MAX_SPEED = 12_000.0
MAX_STONE_SPEED = 1_100.0
STONE_FRICTION_DECELERATION = 72.0
STONE_STOP_SPEED = 8.0
MIN_WALL_BOUNCE_SPEED = 96.0
WALL_RESTITUTION = 0.97
GOAL_POST_RESTITUTION = 0.90
MALLET_RESTITUTION = 0.88
PLAYER_IMPACT_SPEED_SCALE = 0.05
AI_IMPACT_SPEED_SCALE = 0.72
STONE_RESPONSE_ACCELERATION = 9_000.0
MAX_FRAME_TIME = 0.04
MAX_FRAME_GAP = 0.25
MAX_PHYSICS_STEP = 1 / 360
FRAME_INTERVAL_MS = 2

# 轨迹预测参数
PREDICTION_POINT_INTERVAL = 0.06
PREDICTION_SUBSTEP = 1 / 240
PREDICTION_REFRESH_INTERVAL = 0.08
PREDICTION_DISPLAY_SMOOTHING = 0.34
PREDICTION_MAX_BENDS = 1
PREDICTION_MAX_SIMULATION_STEPS = 1024
PREDICTION_COLOR = "#5bacca"
PREDICTION_POINT_COUNT = 18
COLLISION_EPSILON = 1e-9


@dataclass
class Difficulty:
    ai_speed: float
    reaction_delay: float
    aim_error: float
    prediction: float
    attack_line: float


DIFFICULTY_BASES = {
    "简单": (345.0, 0.18, 55.0, 0.45, -145.0),
    "普通": (500.0, 0.085, 24.0, 0.78, -95.0),
    "困难": (680.0, 0.035, 7.0, 1.0, -55.0),
}


def build_difficulties(scale: float = 1.0) -> dict[str, Difficulty]:
    return {
        name: Difficulty(
            ai_speed * scale,
            reaction_delay,
            aim_error * scale,
            prediction,
            RINK_CENTER_Y + attack_offset * scale,
        )
        for name, (ai_speed, reaction_delay, aim_error, prediction, attack_offset)
        in DIFFICULTY_BASES.items()
    }


DIFFICULTIES = build_difficulties(1.0)
