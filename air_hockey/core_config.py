"""空气冰壶共享核心算法与物理配置。

包含所有与 GUI/显示解耦的物理场地几何、尺寸、动力学参数、预测器参数及难度等级。
所有参数均为固定基准值，不受仿真窗口或屏幕缩放影响。
"""

from __future__ import annotations

from dataclasses import dataclass

# 场地基准尺寸与边界
# 游戏坐标必须各向同性（X、Y 每单位代表同样的毫米数）：物理里的半径、hypot 速度、
# 摩擦、接触距离全按圆和各向同性计算。真实台面（墙内）480mm x 880mm，
# X 取 528 单位宽 -> 1 单位 = 480/528 ≈ 0.909mm，Y 按同一比例 = 968 单位。
# （2026-10-29 前是 528 x 668，Y 方向每单位比 X 大 1.45 倍，Y 方向的物理全部失真。）
TABLE_WIDTH_MM = 480.0
TABLE_LENGTH_MM = 880.0
BASE_CANVAS_WIDTH = 600.0

RINK_LEFT = 36.0
RINK_RIGHT = BASE_CANVAS_WIDTH - 36.0  # 564.0
UNITS_PER_MM = (RINK_RIGHT - RINK_LEFT) / TABLE_WIDTH_MM  # 1.1
RINK_TOP = 46.0
RINK_BOTTOM = RINK_TOP + TABLE_LENGTH_MM * UNITS_PER_MM  # 1014.0
BASE_CANVAS_HEIGHT = RINK_BOTTOM + 46.0  # 1060.0
RINK_CENTER_X = BASE_CANVAS_WIDTH / 2    # 300.0
RINK_CENTER_Y = (RINK_TOP + RINK_BOTTOM) / 2   # 530.0
# 机器端防守线（READY 位置的 y）：球门线到中线的 28%
AI_HOME_Y = RINK_TOP + (RINK_CENTER_Y - RINK_TOP) * 0.28   # 181.52

# 球门几何参数
GOAL_HALF_WIDTH = 92.0
GOAL_LEFT = RINK_CENTER_X - GOAL_HALF_WIDTH   # 208.0
GOAL_RIGHT = RINK_CENTER_X + GOAL_HALF_WIDTH  # 392.0
GOAL_DEPTH = 28.0

# 球与球槌尺寸
# 2026-10-21 现场冰壶：相机里半径约 31px，按球台 480mm 宽折算约 26 单位（原 14 是早期游戏里的值）
STONE_RADIUS = 26.0
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
# 2026-10-29 用 play2 记录按毫米重新拟合（只有 3 段干净的自由滑行，都沿长边方向）：
# 225 / 318 / 561 mm/s²（561 那段转角 15°，可能蹭了墙）；HANDOFF 的"实测约 156"换成毫米约 205。
# 取约 210mm/s² ≈ 230 单位/s²。原值 72（仿真时代）让预测滑行距离长出 2~3 倍。
# 数据量少，上实机后用 --record 采更多直线滑行段复核。
STONE_FRICTION_DECELERATION = 230.0
STONE_STOP_SPEED = 8.0
MIN_WALL_BOUNCE_SPEED = 96.0
# 2026-10-21 play2 现场 4 次边墙反弹实测：法向保留 0.54~0.78、沿墙保留 0.59~0.88；
# 按这两个值拟合反弹后轨迹，误差从 82 降到 21（原 0.97 且沿墙不损失）。
WALL_RESTITUTION = 0.65
WALL_TANGENT_RETENTION = 0.6
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
PREDICTION_MAX_SIMULATION_STEPS = 4000
PREDICTION_COLOR = "#5bacca"
COLLISION_EPSILON = 1e-9


@dataclass
class Difficulty:
    ai_speed: float
    reaction_delay: float
    aim_error: float
    prediction: float
    attack_line: float


# 最后一列（进攻线相对中线的偏移）是纯 Y 距离：场地 Y 方向由 668 拉长到 968 单位后
# 按 ×1.449 换算，保持它在半场里的比例不变。
DIFFICULTY_BASES = {
    "简单": (345.0, 0.18, 55.0, 0.45, -210.0),
    "普通": (500.0, 0.085, 24.0, 0.78, -138.0),
    "困难": (680.0, 0.035, 7.0, 1.0, -80.0),
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
