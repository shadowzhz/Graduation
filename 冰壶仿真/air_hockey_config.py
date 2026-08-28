"""空气冰球的场地尺寸和难度配置。"""

import threading
import tkinter as tk
from dataclasses import dataclass


BASE_CANVAS_WIDTH = 600.0
BASE_CANVAS_HEIGHT = 760.0
UI_SCALE = 1.0

CANVAS_WIDTH = BASE_CANVAS_WIDTH
CANVAS_HEIGHT = BASE_CANVAS_HEIGHT

RINK_LEFT = 36.0
RINK_RIGHT = CANVAS_WIDTH - 36.0
RINK_TOP = 46.0
RINK_BOTTOM = CANVAS_HEIGHT - 46.0
RINK_CENTER_X = CANVAS_WIDTH / 2
RINK_CENTER_Y = CANVAS_HEIGHT / 2

GOAL_HALF_WIDTH = 92.0
GOAL_LEFT = RINK_CENTER_X - GOAL_HALF_WIDTH
GOAL_RIGHT = RINK_CENTER_X + GOAL_HALF_WIDTH
GOAL_DEPTH = 28.0

PUCK_RADIUS = 14.0
MALLET_RADIUS = 31.0
GOAL_POST_RADIUS = 6.0
GOAL_POSTS = (
    (GOAL_LEFT, RINK_TOP),
    (GOAL_RIGHT, RINK_TOP),
    (GOAL_LEFT, RINK_BOTTOM),
    (GOAL_RIGHT, RINK_BOTTOM),
)

PLAYER_MAX_SPEED = 12_000.0
MAX_PUCK_SPEED = 1_100.0
PUCK_FRICTION_DECELERATION = 72.0
PUCK_STOP_SPEED = 8.0
MIN_WALL_BOUNCE_SPEED = 96.0
WALL_RESTITUTION = 0.97
GOAL_POST_RESTITUTION = 0.90
MALLET_RESTITUTION = 0.88
PLAYER_IMPACT_SPEED_SCALE = 0.05
AI_IMPACT_SPEED_SCALE = 0.72
PUCK_RESPONSE_ACCELERATION = 9_000.0
MAX_FRAME_TIME = 0.04
MAX_FRAME_GAP = 0.25
MAX_PHYSICS_STEP = 1 / 360
FRAME_INTERVAL_MS = 2

PREDICTION_POINT_INTERVAL = 0.06
PREDICTION_SUBSTEP = 1 / 240
PREDICTION_REFRESH_INTERVAL = 0.08
PREDICTION_DISPLAY_SMOOTHING = 0.34
PREDICTION_MAX_BENDS = 1
PREDICTION_MAX_SIMULATION_STEPS = 1024
PREDICTION_SECOND_SEGMENT_SCALE = 1 / 3
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


def build_difficulties(scale):
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


DIFFICULTIES = build_difficulties(UI_SCALE)
LAYOUT_CONFIG_LOCK = threading.RLock()
LAYOUT_THREAD_ID = None
LAYOUT_CONFIGURED = False
LAYOUT_ROOT = None


def configure_responsive_layout(root):
    """按屏幕大小缩放场地，只能在 Tk 主线程调一次。"""
    global LAYOUT_THREAD_ID, LAYOUT_CONFIGURED, LAYOUT_ROOT
    thread_id = threading.get_ident()
    with LAYOUT_CONFIG_LOCK:
        if LAYOUT_THREAD_ID is None:
            LAYOUT_THREAD_ID = thread_id
        elif LAYOUT_THREAD_ID != thread_id:
            raise RuntimeError("configure_responsive_layout 必须在 Tk 主线程调用")
        if LAYOUT_CONFIGURED:
            if LAYOUT_ROOT is root:
                return
            raise RuntimeError("布局已经绑定到另一个窗口，不能在运行中重新配置")
        _apply_responsive_layout(root)
        LAYOUT_CONFIGURED = True
        LAYOUT_ROOT = root


def _apply_responsive_layout(root):
    global UI_SCALE
    global CANVAS_WIDTH, CANVAS_HEIGHT
    global RINK_LEFT, RINK_RIGHT, RINK_TOP, RINK_BOTTOM
    global RINK_CENTER_X, RINK_CENTER_Y, GOAL_HALF_WIDTH, GOAL_LEFT, GOAL_RIGHT
    global GOAL_DEPTH, PUCK_RADIUS, MALLET_RADIUS, GOAL_POST_RADIUS, GOAL_POSTS
    global PLAYER_MAX_SPEED, MAX_PUCK_SPEED, PUCK_FRICTION_DECELERATION
    global PUCK_STOP_SPEED, MIN_WALL_BOUNCE_SPEED, PUCK_RESPONSE_ACCELERATION
    global DIFFICULTIES
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    available_height = max(360.0, screen_height - 220.0)
    available_width = max(360.0, screen_width - 80.0)
    UI_SCALE = min(1.0, available_width / BASE_CANVAS_WIDTH, available_height / BASE_CANVAS_HEIGHT)
    CANVAS_WIDTH = round(BASE_CANVAS_WIDTH * UI_SCALE)
    CANVAS_HEIGHT = round(BASE_CANVAS_HEIGHT * UI_SCALE)
    RINK_LEFT = 36.0 * UI_SCALE
    RINK_RIGHT = CANVAS_WIDTH - 36.0 * UI_SCALE
    RINK_TOP = 46.0 * UI_SCALE
    RINK_BOTTOM = CANVAS_HEIGHT - 46.0 * UI_SCALE
    RINK_CENTER_X = CANVAS_WIDTH / 2
    RINK_CENTER_Y = CANVAS_HEIGHT / 2
    GOAL_HALF_WIDTH = 92.0 * UI_SCALE
    GOAL_LEFT = RINK_CENTER_X - GOAL_HALF_WIDTH
    GOAL_RIGHT = RINK_CENTER_X + GOAL_HALF_WIDTH
    GOAL_DEPTH = 28.0 * UI_SCALE
    PUCK_RADIUS = 14.0 * UI_SCALE
    MALLET_RADIUS = 31.0 * UI_SCALE
    GOAL_POST_RADIUS = 6.0 * UI_SCALE
    GOAL_POSTS = ((GOAL_LEFT, RINK_TOP), (GOAL_RIGHT, RINK_TOP), (GOAL_LEFT, RINK_BOTTOM), (GOAL_RIGHT, RINK_BOTTOM))
    PLAYER_MAX_SPEED = 12_000.0 * UI_SCALE
    MAX_PUCK_SPEED = 1_100.0 * UI_SCALE
    PUCK_FRICTION_DECELERATION = 72.0 * UI_SCALE
    PUCK_STOP_SPEED = 8.0 * UI_SCALE
    MIN_WALL_BOUNCE_SPEED = 96.0 * UI_SCALE
    PUCK_RESPONSE_ACCELERATION = 9_000.0 * UI_SCALE
    DIFFICULTIES = build_difficulties(UI_SCALE)


def sync_layout_globals(namespace):
    """把缩放后的常量同步给直接引用常量的界面模块。"""
    for name, value in globals().items():
        if name.isupper() or name in {"DIFFICULTIES", "Difficulty"}:
            namespace[name] = value


def center_window(root):
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
