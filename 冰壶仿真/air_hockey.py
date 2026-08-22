"""鼠标对战电脑的虚拟空气冰球。

运行方式：
    python air_hockey.py

只使用 Python 标准库 tkinter，不需要安装第三方依赖。
玩家按住鼠标左键控制下方蓝色球槌，电脑控制上方红色球槌。
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field, replace
from tkinter import ttk


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
# 仅用于冰球贴住外角时的脱困反弹；普通墙面慢速碰撞不使用该下限。
MIN_WALL_BOUNCE_SPEED = 96.0
WALL_RESTITUTION = 0.97
GOAL_POST_RESTITUTION = 0.90
MALLET_RESTITUTION = 0.88
# 球槌保持高速跟随鼠标；碰撞时单独缩放为“有效击打速度”，避免轻碰也满速。
PLAYER_IMPACT_SPEED_SCALE = 0.05
AI_IMPACT_SPEED_SCALE = 0.72
# 撞击只设置目标速度，冰球以恒定加速度逐步达到，不在单帧内跳变。
PUCK_RESPONSE_ACCELERATION = 9_000.0
MAX_FRAME_TIME = 0.04
MAX_FRAME_GAP = 0.25
MAX_PHYSICS_STEP = 1 / 360
# Tk 的 after 参数是“本次回调结束后再等待”的时间。2 ms 给绘制和
# Tk 事件处理留下余量，目标周期约 4-5 ms，适合 200 Hz 以上采样。
FRAME_INTERVAL_MS = 2

PREDICTION_POINT_INTERVAL = 0.06
PREDICTION_SUBSTEP = 1 / 240
PREDICTION_REFRESH_INTERVAL = 0.08
PREDICTION_DISPLAY_SMOOTHING = 0.34
PREDICTION_MAX_BENDS = 1
PREDICTION_MAX_SIMULATION_STEPS = 1024
PREDICTION_SECOND_SEGMENT_SCALE = 1 / 3
PREDICTION_COLORS = (
    "#1583ae",
    "#228bb4",
    "#3094ba",
    "#3e9cc0",
    "#4da4c5",
    "#5bacca",
    "#69b1d0",
    "#76b9d4",
    "#83c0d8",
    "#8fc7dc",
    "#9acde0",
    "#a5d3e3",
    "#afd8e6",
    "#b8dde9",
    "#c1e1ec",
    "#cae6ef",
    "#d2eaf2",
    "#d9edf4",
)
PREDICTION_POINT_COUNT = len(PREDICTION_COLORS)
COLLISION_EPSILON = 1e-9


@dataclass(frozen=True)
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


def build_difficulties(scale: float) -> dict[str, Difficulty]:
    """从未缩放的基准值生成当前画布对应的难度参数。"""
    return {
        name: Difficulty(
            ai_speed * scale,
            reaction_delay,
            aim_error * scale,
            prediction,
            RINK_CENTER_Y + attack_offset * scale,
        )
        for name, (
            ai_speed,
            reaction_delay,
            aim_error,
            prediction,
            attack_offset,
        ) in DIFFICULTY_BASES.items()
    }


DIFFICULTIES = build_difficulties(UI_SCALE)
LAYOUT_CONFIG_LOCK = threading.RLock()
LAYOUT_THREAD_ID: int | None = None
LAYOUT_CONFIGURED = False
LAYOUT_ROOT: tk.Tk | None = None


def configure_responsive_layout(root: tk.Tk) -> None:
    """在创建游戏前配置布局，并限制布局状态只由 Tk 线程修改。"""
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


def _apply_responsive_layout(root: tk.Tk) -> None:
    """根据屏幕尺寸统一缩放画布、棋盘几何和速度参数。

    物理计算仍使用画布坐标，但所有距离和速度按同一比例缩放，
    因此缩小窗口不会改变仿真的运动比例。
    """
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
    # 标题区、底部提示和窗口边框占用的高度不属于游戏画布。
    available_height = max(360.0, screen_height - 220.0)
    available_width = max(360.0, screen_width - 80.0)
    UI_SCALE = min(
        1.0,
        available_width / BASE_CANVAS_WIDTH,
        available_height / BASE_CANVAS_HEIGHT,
    )

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
    GOAL_POSTS = (
        (GOAL_LEFT, RINK_TOP),
        (GOAL_RIGHT, RINK_TOP),
        (GOAL_LEFT, RINK_BOTTOM),
        (GOAL_RIGHT, RINK_BOTTOM),
    )

    PLAYER_MAX_SPEED = 12_000.0 * UI_SCALE
    MAX_PUCK_SPEED = 1_100.0 * UI_SCALE
    PUCK_FRICTION_DECELERATION = 72.0 * UI_SCALE
    PUCK_STOP_SPEED = 8.0 * UI_SCALE
    MIN_WALL_BOUNCE_SPEED = 96.0 * UI_SCALE
    PUCK_RESPONSE_ACCELERATION = 9_000.0 * UI_SCALE

    DIFFICULTIES = build_difficulties(UI_SCALE)


def center_window(root: tk.Tk) -> None:
    """让缩放后的窗口在当前屏幕内居中显示。"""
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def reflected_coordinate(value: float, low: float, high: float) -> float:
    """把越界坐标按两侧边界的镜面反射折回有效区间。"""
    span = high - low
    if span <= 0:
        return low
    folded = (value - low) % (2 * span)
    if folded > span:
        folded = 2 * span - folded
    return low + folded


def puck_inside_goal_mouth(x: float) -> bool:
    return GOAL_LEFT + PUCK_RADIUS < x < GOAL_RIGHT - PUCK_RADIUS


def circle_post_contact(
    circle_x: float,
    circle_y: float,
    post_x: float,
    post_y: float,
    minimum_distance: float,
) -> tuple[float, float, float] | None:
    """返回圆与球门柱的接触距离和法向量，避免重合时除零。"""
    dx = circle_x - post_x
    dy = circle_y - post_y
    distance_sq = dx * dx + dy * dy
    if distance_sq >= minimum_distance * minimum_distance:
        return None

    if distance_sq > COLLISION_EPSILON:
        distance = math.sqrt(distance_sq)
        return distance, dx / distance, dy / distance

    # 圆心与球门柱完全重合时，使用从球门柱指向台面内部的法向量。
    nx = RINK_CENTER_X - post_x
    ny = RINK_CENTER_Y - post_y
    length = math.hypot(nx, ny)
    if length <= COLLISION_EPSILON:
        return 0.0, 0.0, 1.0
    return 0.0, nx / length, ny / length


@dataclass
class PuckMotion:
    """真实冰球和预测轨迹共用的运动状态。"""

    x: float = field(default_factory=lambda: RINK_CENTER_X)
    y: float = field(default_factory=lambda: RINK_CENTER_Y)
    vx: float = 0.0
    vy: float = 0.0
    target_vx: float = 0.0
    target_vy: float = 0.0
    response_active: bool = False

    @staticmethod
    def _limited_velocity(vx: float, vy: float) -> tuple[float, float]:
        speed = math.hypot(vx, vy)
        if speed <= MAX_PUCK_SPEED:
            return vx, vy
        scale = MAX_PUCK_SPEED / speed
        return vx * scale, vy * scale

    def collision_velocity(self) -> tuple[float, float]:
        if self.response_active and math.hypot(self.target_vx, self.target_vy) > 1e-9:
            return self.target_vx, self.target_vy
        return self.vx, self.vy

    def set_target_velocity(self, target_vx: float, target_vy: float) -> None:
        self.vx, self.vy = self._limited_velocity(self.vx, self.vy)
        self.target_vx, self.target_vy = self._limited_velocity(target_vx, target_vy)

        current_speed = math.hypot(self.vx, self.vy)
        target_speed = math.hypot(self.target_vx, self.target_vy)
        if current_speed > 1e-9 and target_speed > 1e-9:
            self.vx = self.target_vx / target_speed * current_speed
            self.vy = self.target_vy / target_speed * current_speed

        self.response_active = math.hypot(
            self.target_vx - self.vx,
            self.target_vy - self.vy,
        ) > 1e-9

    def set_immediate_velocity(self, vx: float, vy: float) -> None:
        """提交碰撞后的即时速度，避免边界处重复追踪旧的目标速度。"""
        self.vx, self.vy = self._limited_velocity(vx, vy)
        self.target_vx = self.vx
        self.target_vy = self.vy
        self.response_active = False

    def set_wall_reflection(
        self,
        vx: float,
        vy: float,
        target_vx: float,
        target_vy: float,
    ) -> None:
        """反射实际速度，同时反射响应目标，避免撞墙时瞬间跳到目标速度。"""
        self.vx, self.vy = self._limited_velocity(vx, vy)
        if self.response_active:
            self.target_vx, self.target_vy = self._limited_velocity(
                target_vx,
                target_vy,
            )
            self.response_active = math.hypot(
                self.target_vx - self.vx,
                self.target_vy - self.vy,
            ) > COLLISION_EPSILON
        else:
            self.target_vx = self.vx
            self.target_vy = self.vy

    def advance_velocity(self, dt: float) -> None:
        if self.response_active and dt > 0:
            delta_x = self.target_vx - self.vx
            delta_y = self.target_vy - self.vy
            delta_speed = math.hypot(delta_x, delta_y)
            velocity_step = PUCK_RESPONSE_ACCELERATION * dt
            if delta_speed <= velocity_step + 1e-9:
                self.vx = self.target_vx
                self.vy = self.target_vy
                self.response_active = False
            else:
                scale = velocity_step / delta_speed
                self.vx += delta_x * scale
                self.vy += delta_y * scale

        speed = math.hypot(self.vx, self.vy)
        new_speed = max(0.0, speed - PUCK_FRICTION_DECELERATION * dt)
        if speed <= 1e-9 or new_speed <= 1e-9:
            self.vx = self.vy = 0.0
        else:
            scale = new_speed / speed
            self.vx *= scale
            self.vy *= scale
        self.vx, self.vy = self._limited_velocity(self.vx, self.vy)

    def resolve_walls(self) -> bool:
        left_limit = RINK_LEFT + PUCK_RADIUS
        right_limit = RINK_RIGHT - PUCK_RADIUS
        top_limit = RINK_TOP + PUCK_RADIUS
        bottom_limit = RINK_BOTTOM - PUCK_RADIUS
        vx, vy = self.vx, self.vy
        target_vx, target_vy = self.target_vx, self.target_vy
        reflected_vx, reflected_vy = vx, vy
        reflected_target_vx, reflected_target_vy = target_vx, target_vy
        bounced = False

        if self.x < left_limit - COLLISION_EPSILON:
            self.x = left_limit
            if vx < -COLLISION_EPSILON:
                reflected_vx = abs(vx) * WALL_RESTITUTION
                if target_vx < -COLLISION_EPSILON:
                    reflected_target_vx = abs(target_vx) * WALL_RESTITUTION
                bounced = True
        elif self.x > right_limit + COLLISION_EPSILON:
            self.x = right_limit
            if vx > COLLISION_EPSILON:
                reflected_vx = -abs(vx) * WALL_RESTITUTION
                if target_vx > COLLISION_EPSILON:
                    reflected_target_vx = -abs(target_vx) * WALL_RESTITUTION
                bounced = True
        elif self.x <= left_limit + COLLISION_EPSILON and vx < -COLLISION_EPSILON:
            self.x = left_limit
            reflected_vx = abs(vx) * WALL_RESTITUTION
            if target_vx < -COLLISION_EPSILON:
                reflected_target_vx = abs(target_vx) * WALL_RESTITUTION
            bounced = True
        elif self.x >= right_limit - COLLISION_EPSILON and vx > COLLISION_EPSILON:
            self.x = right_limit
            reflected_vx = -abs(vx) * WALL_RESTITUTION
            if target_vx > COLLISION_EPSILON:
                reflected_target_vx = -abs(target_vx) * WALL_RESTITUTION
            bounced = True

        if not puck_inside_goal_mouth(self.x):
            if self.y < top_limit - COLLISION_EPSILON:
                self.y = top_limit
                if vy < -COLLISION_EPSILON:
                    reflected_vy = abs(vy) * WALL_RESTITUTION
                    if target_vy < -COLLISION_EPSILON:
                        reflected_target_vy = abs(target_vy) * WALL_RESTITUTION
                    bounced = True
            elif self.y > bottom_limit + COLLISION_EPSILON:
                self.y = bottom_limit
                if vy > COLLISION_EPSILON:
                    reflected_vy = -abs(vy) * WALL_RESTITUTION
                    if target_vy > COLLISION_EPSILON:
                        reflected_target_vy = -abs(target_vy) * WALL_RESTITUTION
                    bounced = True
            elif self.y <= top_limit + COLLISION_EPSILON and vy < -COLLISION_EPSILON:
                self.y = top_limit
                reflected_vy = abs(vy) * WALL_RESTITUTION
                if target_vy < -COLLISION_EPSILON:
                    reflected_target_vy = abs(target_vy) * WALL_RESTITUTION
                bounced = True
            elif self.y >= bottom_limit - COLLISION_EPSILON and vy > COLLISION_EPSILON:
                self.y = bottom_limit
                reflected_vy = -abs(vy) * WALL_RESTITUTION
                if target_vy > COLLISION_EPSILON:
                    reflected_target_vy = -abs(target_vy) * WALL_RESTITUTION
                bounced = True

        at_left_or_right = (
            self.x <= left_limit + COLLISION_EPSILON
            or self.x >= right_limit - COLLISION_EPSILON
        )
        at_top_or_bottom = (
            self.y <= top_limit + COLLISION_EPSILON
            or self.y >= bottom_limit - COLLISION_EPSILON
        )
        current_speed = math.hypot(vx, vy)
        if at_left_or_right and at_top_or_bottom and current_speed <= COLLISION_EPSILON:
            # 零速冰球没有可供镜面反射的入射方向；用两面墙的内法线给
            # 一个连续的小速度，避免它永远贴在外角，且不直接改坐标。
            direction_x = 1.0 if self.x <= left_limit + COLLISION_EPSILON else -1.0
            direction_y = 1.0 if self.y <= top_limit + COLLISION_EPSILON else -1.0
            component_speed = MIN_WALL_BOUNCE_SPEED / math.sqrt(2.0)
            reflected_vx = direction_x * component_speed
            reflected_vy = direction_y * component_speed
            bounced = True
        elif bounced and current_speed > COLLISION_EPSILON and at_left_or_right and at_top_or_bottom:
            reflected_speed = math.hypot(reflected_vx, reflected_vy)
            if reflected_speed < MIN_WALL_BOUNCE_SPEED:
                scale = MIN_WALL_BOUNCE_SPEED / reflected_speed
                reflected_vx *= scale
                reflected_vy *= scale

        if bounced:
            # 两个方向在同一个物理步碰壁时同时反射，角落自然形成镜面反弹。
            if current_speed <= COLLISION_EPSILON:
                self.set_immediate_velocity(reflected_vx, reflected_vy)
            else:
                self.set_wall_reflection(
                    reflected_vx,
                    reflected_vy,
                    reflected_target_vx,
                    reflected_target_vy,
                )
        return bounced

    def resolve_goal_posts(self) -> bool:
        bounced = False
        minimum_distance = PUCK_RADIUS + GOAL_POST_RADIUS
        for post_x, post_y in GOAL_POSTS:
            contact = circle_post_contact(
                self.x,
                self.y,
                post_x,
                post_y,
                minimum_distance,
            )
            if contact is None:
                continue

            distance, nx, ny = contact
            self.x = post_x + nx * minimum_distance
            self.y = post_y + ny * minimum_distance
            vx, vy = self.collision_velocity()
            normal_speed = vx * nx + vy * ny
            if normal_speed < 0:
                impulse = (1.0 + GOAL_POST_RESTITUTION) * normal_speed
                self.set_immediate_velocity(vx - impulse * nx, vy - impulse * ny)
                bounced = True
        return bounced


def goal_scorer(puck: PuckMotion) -> str | None:
    if not puck_inside_goal_mouth(puck.x):
        return None
    if puck.y + PUCK_RADIUS < RINK_TOP:
        return "player"
    if puck.y - PUCK_RADIUS > RINK_BOTTOM:
        return "ai"
    return None


class AirHockeyGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.random = random.Random()
        self.closed = False
        self.game_loop_id: str | None = None

        self.player_score = self.ai_score = 0
        self.running = False
        self.started = False
        initial_message = self._serve_instruction("player")
        self.mouse_control_active = False

        self.ai_home_y = RINK_TOP + (RINK_CENTER_Y - RINK_TOP) * 0.28

        self.score_var = tk.StringVar(value="0 : 0")
        self.status_text = initial_message
        self.status_header_var = tk.StringVar(value=initial_message)
        self.last_rendered_status = initial_message
        self.difficulty = DIFFICULTIES["普通"]
        self.prediction_cache: list[tuple[float, float]] = []
        self.prediction_display_points: list[tuple[float, float]] = []
        self.prediction_cache_age = PREDICTION_REFRESH_INTERVAL
        self.prediction_cache_origin = (RINK_CENTER_X, RINK_CENTER_Y)
        self.prediction_cache_velocity = (0.0, 0.0, 0.0, 0.0)
        self.prediction_cache_response_active = False
        self.prediction_line_visible = False
        self.prediction_render_counter = 0
        self.difficulty_var = tk.StringVar(value="普通")
        self.start_button_var = tk.StringVar(value="开始游戏")

        self._configure_window()
        self._build_ui()
        self._bind_controls()
        self._reset_round(initial_message, "player")
        self._render()

        self.last_frame_time = time.perf_counter()
        self._schedule_game_loop()

    def _schedule_game_loop(self) -> None:
        if self.closed:
            return
        try:
            self.game_loop_id = self.root.after(FRAME_INTERVAL_MS, self._game_loop)
        except tk.TclError:
            self.game_loop_id = None
            self.closed = True

    def _configure_window(self) -> None:
        self.root.title("虚拟空气冰球")
        self.root.configure(bg="#0b2239")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Difficulty.TRadiobutton",
            background="#102f4c",
            foreground="#eaf7ff",
            font=("Microsoft YaHei UI", 10),
            padding=(8, 5),
        )
        style.map(
            "Difficulty.TRadiobutton",
            background=[("active", "#174469")],
            foreground=[("active", "#ffffff")],
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#0b2239", height=122)
        header.pack(fill="x", padx=16, pady=(12, 7))
        header.pack_propagate(False)

        title_row = tk.Frame(header, bg="#0b2239")
        title_row.pack(fill="x")

        title_box = tk.Frame(title_row, bg="#0b2239")
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="虚拟空气冰球",
            bg="#0b2239",
            fg="#ffffff",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="红方电脑在上 · 蓝方玩家在下",
            bg="#0b2239",
            fg="#9fc6df",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            title_box,
            textvariable=self.status_header_var,
            bg="#0b2239",
            fg="#f5cf70",
            font=("Microsoft YaHei UI", 9, "bold"),
            anchor="w",
            wraplength=max(240, int(CANVAS_WIDTH * 0.58)),
        ).pack(anchor="w", pady=(2, 0))

        score_box = tk.Frame(title_row, bg="#0b2239")
        score_box.pack(side="right")
        tk.Label(
            score_box,
            text="玩家          电脑",
            bg="#0b2239",
            fg="#9fc6df",
            font=("Microsoft YaHei UI", 9),
        ).pack()
        self.score_label = tk.Label(
            score_box,
            textvariable=self.score_var,
            bg="#0b2239",
            fg="#ffffff",
            font=("Consolas", 24, "bold"),
            anchor="center",
        )
        self.score_label.pack()

        controls = tk.Frame(header, bg="#0b2239")
        controls.pack(fill="x", pady=(8, 0))

        difficulty_box = tk.Frame(controls, bg="#102f4c", padx=5, pady=3)
        difficulty_box.pack(side="left")
        tk.Label(
            difficulty_box,
            text="难度",
            bg="#102f4c",
            fg="#9fc6df",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="left", padx=(4, 2))
        for name in DIFFICULTIES:
            ttk.Radiobutton(
                difficulty_box,
                text=name,
                value=name,
                variable=self.difficulty_var,
                command=self._difficulty_changed,
                style="Difficulty.TRadiobutton",
                takefocus=False,
            ).pack(side="left")

        button_box = tk.Frame(controls, bg="#0b2239")
        button_box.pack(side="right")
        tk.Button(
            button_box,
            textvariable=self.start_button_var,
            command=self.toggle_pause,
            width=10,
            relief="flat",
            bd=0,
            bg="#18a7d6",
            activebackground="#36b9e3",
            fg="#ffffff",
            activeforeground="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
            cursor="hand2",
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            button_box,
            text="重新开局",
            command=self.reset_match,
            width=9,
            relief="flat",
            bd=0,
            bg="#274c69",
            activebackground="#356685",
            fg="#ffffff",
            activeforeground="#ffffff",
            font=("Microsoft YaHei UI", 10),
            cursor="hand2",
        ).pack(side="left")

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#dff4fc",
            highlightthickness=0,
            cursor="arrow",
        )
        self.canvas.pack(padx=20, pady=(0, 7))
        self._draw_rink()

        tk.Label(
            self.root,
            text="按住左键拖动：控制球槌    预测线：预测轨迹    空格：开始/暂停    R：重新开局",
            bg="#0b2239",
            fg="#86adc5",
            font=("Microsoft YaHei UI", 9),
            wraplength=CANVAS_WIDTH,
        ).pack(pady=(0, 12))

    def _draw_rink(self) -> None:
        c = self.canvas
        scale = UI_SCALE
        line_width_4 = max(1, round(4 * scale))
        line_width_3 = max(1, round(3 * scale))
        line_width_2 = max(1, round(2 * scale))

        # 球门和球网
        goal_ranges = (
            (RINK_TOP - GOAL_DEPTH, RINK_TOP),
            (RINK_BOTTOM, RINK_BOTTOM + GOAL_DEPTH),
        )
        for y1, y2 in goal_ranges:
            c.create_rectangle(
                GOAL_LEFT,
                y1,
                GOAL_RIGHT,
                y2,
                fill="#c7e8f4",
                outline="#e84b5f",
                width=line_width_4,
            )
        for y1, y2 in goal_ranges:
            for x in range(
                int(GOAL_LEFT + 14 * scale),
                int(GOAL_RIGHT),
                max(1, round(16 * scale)),
            ):
                c.create_line(x, y1, x, y2, fill="#98c6d7")

        c.create_rectangle(
            RINK_LEFT,
            RINK_TOP,
            RINK_RIGHT,
            RINK_BOTTOM,
            fill="#edfaff",
            outline="#167ca8",
            width=max(1, round(5 * scale)),
        )

        # 中线和开球圈
        c.create_line(
            RINK_LEFT + 3 * scale,
            RINK_CENTER_Y,
            RINK_RIGHT - 3 * scale,
            RINK_CENTER_Y,
            fill="#e45b69",
            width=line_width_4,
        )
        c.create_oval(
            RINK_CENTER_X - 82 * scale,
            RINK_CENTER_Y - 82 * scale,
            RINK_CENTER_X + 82 * scale,
            RINK_CENTER_Y + 82 * scale,
            outline="#55aacf",
            width=line_width_3,
        )
        c.create_oval(
            RINK_CENTER_X - 7 * scale,
            RINK_CENTER_Y - 7 * scale,
            RINK_CENTER_X + 7 * scale,
            RINK_CENTER_Y + 7 * scale,
            fill="#e45b69",
            outline="",
        )

        # 上下半场装饰线和争球点
        for y in (RINK_TOP + 205 * scale, RINK_BOTTOM - 205 * scale):
            c.create_line(
                RINK_LEFT + 3 * scale,
                y,
                RINK_RIGHT - 3 * scale,
                y,
                fill="#7cc5df",
                width=line_width_2,
            )
            for x in (RINK_CENTER_X - 145 * scale, RINK_CENTER_X + 145 * scale):
                c.create_oval(
                    x - 29 * scale,
                    y - 29 * scale,
                    x + 29 * scale,
                    y + 29 * scale,
                    outline="#7cc5df",
                    width=line_width_2,
                )
                c.create_oval(
                    x - 5 * scale,
                    y - 5 * scale,
                    x + 5 * scale,
                    y + 5 * scale,
                    fill="#e45b69",
                    outline="",
                )

        for y, text in (
            (RINK_TOP + 25 * scale, "电脑半场"),
            (RINK_BOTTOM - 25 * scale, "玩家半场"),
        ):
            c.create_text(
                RINK_CENTER_X,
                y,
                text=text,
                fill="#5b9fbd",
                font=("Microsoft YaHei UI", max(8, round(11 * scale)), "bold"),
            )

        # 预测轨迹使用一条连续折线；拆成很多短线段会在低速时显示成点状残线。
        self.prediction_line_item = c.create_line(
            0,
            0,
            0,
            0,
            fill=PREDICTION_COLORS[5],
            width=max(2, round(3.2 * scale)),
            capstyle=tk.BUTT,
            joinstyle=tk.ROUND,
            smooth=False,
            dash="",
            state="hidden",
        )

        # 动态物体只绘制主体，不使用阴影。
        self.player_item = c.create_oval(
            0, 0, 0, 0, fill="#118fc5", outline="#075e83", width=4
        )
        self.ai_item = c.create_oval(
            0, 0, 0, 0, fill="#eb4f5d", outline="#9f2633", width=4
        )
        self.puck_item = c.create_oval(
            0, 0, 0, 0, fill="#172b3b", outline="#07121b", width=3
        )
        self.player_glint = c.create_oval(0, 0, 0, 0, fill="#76d1ee", outline="")
        self.ai_glint = c.create_oval(0, 0, 0, 0, fill="#ff9ca5", outline="")

        self.status_item = c.create_text(
            RINK_CENTER_X,
            RINK_CENTER_Y,
            text=self.status_text,
            fill="#173e55",
            font=("Microsoft YaHei UI", max(8, round(14 * scale)), "bold"),
            state="hidden",
        )

    def _bind_controls(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._mouse_pressed)
        self.canvas.bind("<B1-Motion>", self._mouse_dragged)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_released)
        self.root.bind("<space>", self._space_pressed)
        self.root.bind("<Key-r>", lambda _event: self.reset_match())
        self.root.bind("<Key-R>", lambda _event: self.reset_match())
        self.root.bind("1", lambda _event: self._set_difficulty("简单"))
        self.root.bind("2", lambda _event: self._set_difficulty("普通"))
        self.root.bind("3", lambda _event: self._set_difficulty("困难"))

    def _update_mouse_target(self, event: tk.Event) -> None:
        self.mouse_target_x = float(event.x)
        self.mouse_target_y = float(event.y)

    def _mouse_pressed(self, event: tk.Event) -> None:
        self._update_mouse_target(event)
        if not self.running:
            return
        self.mouse_control_active = self.running
        if self.mouse_control_active:
            self.canvas.configure(cursor="none")

    def _mouse_dragged(self, event: tk.Event) -> None:
        if self.mouse_control_active:
            self._update_mouse_target(event)

    def _stop_player_control(self) -> None:
        self.mouse_control_active = False
        self.player_vx = self.player_vy = 0.0
        self.canvas.configure(cursor="arrow")

    def _mouse_released(self, _event: tk.Event) -> None:
        self._stop_player_control()

    def _space_pressed(self, event: tk.Event) -> str | None:
        # 按钮和难度单选框有自己的空格键行为，避免再触发一次全局暂停。
        if event.widget.winfo_class() in {
            "Button",
            "TButton",
            "Radiobutton",
            "TRadiobutton",
        }:
            return None
        self.toggle_pause()
        return "break"

    def read_player_target(self) -> tuple[float, float]:
        """返回玩家目标位置。

        当前版本读取鼠标。后续接 S7-1200/1500 时，只需在这里把 PLC
        坐标转换为 Canvas 像素坐标；游戏物理和 UI 无需跟着改动。
        """
        return self.mouse_target_x, self.mouse_target_y

    def publish_game_state(self) -> None:
        """未来向 PLC 输出球、球槌和比分状态的集中位置；虚拟版无需处理。"""

    def toggle_pause(self) -> None:
        self.running = not self.running
        self.started = True
        self.last_frame_time = time.perf_counter()
        self.prediction_cache_age = PREDICTION_REFRESH_INTERVAL
        if self.running:
            self.start_button_var.set("暂停")
        else:
            self._stop_player_control()
            self.start_button_var.set("继续")
            self.status_text = "已暂停"

    def reset_match(self) -> None:
        self.player_score = self.ai_score = 0
        self.running = False
        self.started = False
        self._stop_player_control()
        self.start_button_var.set("开始游戏")
        self._reset_round(self._serve_instruction("player"), "player")
        self._update_score_text()
        self._render()

    def _difficulty_changed(self) -> None:
        self.ai_reaction_timer = 0.0
        self.difficulty = DIFFICULTIES[self.difficulty_var.get()]
        if self.started and not self.running:
            self.status_text = f"当前难度：{self.difficulty_var.get()}"

    def _set_difficulty(self, name: str) -> None:
        self.difficulty_var.set(name)
        self._difficulty_changed()

    @staticmethod
    def _serve_instruction(server: str) -> str:
        return "玩家开球：按住左键，用球槌击球" if server == "player" else "电脑开球"

    def _reset_round(self, message: str, server: str) -> None:
        self.mouse_control_active = False
        self.player_vx = self.player_vy = 0.0
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="arrow")

        self.current_server = server
        self.player_x = RINK_CENTER_X
        self.player_y = RINK_BOTTOM - (RINK_BOTTOM - RINK_CENTER_Y) * 0.47
        self.player_vx = self.player_vy = 0.0
        self.mouse_target_x = self.player_x
        self.mouse_target_y = self.player_y

        self.ai_x = RINK_CENTER_X
        self.ai_y = self.ai_home_y
        self.ai_vx = self.ai_vy = 0.0
        self.ai_target_x = RINK_CENTER_X
        self.ai_target_y = self.ai_home_y
        self.ai_reaction_timer = 0.0

        self.puck = PuckMotion(x=RINK_CENTER_X, y=RINK_CENTER_Y)
        self.awaiting_serve = True
        self.round_message = message
        self.status_text = message
        self.prediction_cache = []
        self.prediction_display_points = []
        self.prediction_cache_age = PREDICTION_REFRESH_INTERVAL
        self.prediction_cache_origin = (RINK_CENTER_X, RINK_CENTER_Y)
        self.prediction_cache_velocity = (0.0, 0.0, 0.0, 0.0)
        self.prediction_cache_response_active = False

    def _game_loop(self) -> None:
        self.game_loop_id = None
        if self.closed:
            return

        now = time.perf_counter()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        if elapsed > MAX_FRAME_GAP:
            # 窗口最小化或系统休眠后不追赶这段时间，避免恢复时出现跳跃。
            dt = 0.0
            self.ai_reaction_timer = 0.0
            self.prediction_cache_age = PREDICTION_REFRESH_INTERVAL
        else:
            dt = min(elapsed, MAX_FRAME_TIME)
        self.prediction_cache_age += dt

        if self.running:
            step_count = max(1, math.ceil(dt / MAX_PHYSICS_STEP))
            step_time = dt / step_count
            for _ in range(step_count):
                self._move_player(step_time)
                self._move_ai(step_time)
                if self._move_puck(step_time):
                    break

            self.publish_game_state()

        if self.closed:
            return
        self._render()
        self._schedule_game_loop()

    def _move_player(self, dt: float) -> None:
        if not self.mouse_control_active:
            self.player_vx = 0.0
            self.player_vy = 0.0
            (
                self.player_x,
                self.player_y,
                self.player_vx,
                self.player_vy,
            ) = self._resolve_mallet_goal_posts(
                self.player_x,
                self.player_y,
                self.player_vx,
                self.player_vy,
            )
            (
                self.player_x,
                self.player_y,
                self.player_vx,
                self.player_vy,
            ) = self._keep_mallet_clear_of_outer_corners(
                self.player_x,
                self.player_y,
                self.player_vx,
                self.player_vy,
            )
            return

        target_x, target_y = self.read_player_target()
        target_x = clamp(
            target_x,
            RINK_LEFT + MALLET_RADIUS,
            RINK_RIGHT - MALLET_RADIUS,
        )
        player_min_y = RINK_CENTER_Y + MALLET_RADIUS
        if self.awaiting_serve and self.current_server == "ai":
            player_min_y = RINK_CENTER_Y + MALLET_RADIUS + PUCK_RADIUS + 8 * UI_SCALE
        target_y = clamp(
            target_y,
            player_min_y,
            RINK_BOTTOM - MALLET_RADIUS,
        )
        (
            self.player_x,
            self.player_y,
            self.player_vx,
            self.player_vy,
        ) = self._move_towards(
            self.player_x,
            self.player_y,
            target_x,
            target_y,
            PLAYER_MAX_SPEED,
            dt,
        )
        (
            self.player_x,
            self.player_y,
            self.player_vx,
            self.player_vy,
        ) = self._resolve_mallet_goal_posts(
            self.player_x,
            self.player_y,
            self.player_vx,
            self.player_vy,
        )
        (
            self.player_x,
            self.player_y,
            self.player_vx,
            self.player_vy,
        ) = self._keep_mallet_clear_of_outer_corners(
            self.player_x,
            self.player_y,
            self.player_vx,
            self.player_vy,
        )

    def _move_ai(self, dt: float) -> None:
        difficulty = self.difficulty
        self.ai_reaction_timer -= dt
        if self.ai_reaction_timer <= 0:
            self.ai_reaction_timer += difficulty.reaction_delay
            self._choose_ai_target(difficulty)

        ai_min_y = RINK_TOP + MALLET_RADIUS
        ai_max_y = RINK_CENTER_Y - MALLET_RADIUS
        if self.awaiting_serve and self.current_server == "player":
            ai_max_y = RINK_CENTER_Y - MALLET_RADIUS - PUCK_RADIUS - 8 * UI_SCALE
            self.ai_y = min(self.ai_y, ai_max_y)
        self.ai_target_y = clamp(self.ai_target_y, ai_min_y, ai_max_y)

        (
            self.ai_x,
            self.ai_y,
            self.ai_vx,
            self.ai_vy,
        ) = self._move_towards(
            self.ai_x,
            self.ai_y,
            self.ai_target_x,
            self.ai_target_y,
            difficulty.ai_speed,
            dt,
        )
        (
            self.ai_x,
            self.ai_y,
            self.ai_vx,
            self.ai_vy,
        ) = self._resolve_mallet_goal_posts(
            self.ai_x,
            self.ai_y,
            self.ai_vx,
            self.ai_vy,
        )
        (
            self.ai_x,
            self.ai_y,
            self.ai_vx,
            self.ai_vy,
        ) = self._keep_mallet_clear_of_outer_corners(
            self.ai_x,
            self.ai_y,
            self.ai_vx,
            self.ai_vy,
        )

    @staticmethod
    def _resolve_mallet_goal_posts(
        mallet_x: float,
        mallet_y: float,
        mallet_vx: float,
        mallet_vy: float,
    ) -> tuple[float, float, float, float]:
        """把球槌从球门柱外侧推出，并反射朝向球门柱的速度。"""
        minimum_distance = MALLET_RADIUS + GOAL_POST_RADIUS
        for post_x, post_y in GOAL_POSTS:
            contact = circle_post_contact(
                mallet_x,
                mallet_y,
                post_x,
                post_y,
                minimum_distance,
            )
            if contact is None:
                continue

            distance, nx, ny = contact
            mallet_x = post_x + nx * minimum_distance
            mallet_y = post_y + ny * minimum_distance
            normal_speed = mallet_vx * nx + mallet_vy * ny
            if normal_speed < 0:
                impulse = (1.0 + GOAL_POST_RESTITUTION) * normal_speed
                mallet_vx -= impulse * nx
                mallet_vy -= impulse * ny

        return mallet_x, mallet_y, mallet_vx, mallet_vy

    @staticmethod
    def _keep_mallet_clear_of_outer_corners(
        mallet_x: float,
        mallet_y: float,
        mallet_vx: float,
        mallet_vy: float,
    ) -> tuple[float, float, float, float]:
        """避免球槌占据冰球停在外角时无法被击打的几何区域。"""
        puck_x_values = (RINK_LEFT + PUCK_RADIUS, RINK_RIGHT - PUCK_RADIUS)
        puck_y_values = (RINK_TOP + PUCK_RADIUS, RINK_BOTTOM - PUCK_RADIUS)
        minimum_distance = MALLET_RADIUS + PUCK_RADIUS

        for corner_x in puck_x_values:
            for corner_y in puck_y_values:
                dx = mallet_x - corner_x
                dy = mallet_y - corner_y
                distance_sq = dx * dx + dy * dy
                if distance_sq >= minimum_distance * minimum_distance:
                    continue

                if distance_sq > COLLISION_EPSILON:
                    distance = math.sqrt(distance_sq)
                    nx, ny = dx / distance, dy / distance
                else:
                    nx = RINK_CENTER_X - corner_x
                    ny = RINK_CENTER_Y - corner_y
                    length = math.hypot(nx, ny)
                    nx, ny = nx / length, ny / length

                mallet_x = corner_x + nx * minimum_distance
                mallet_y = corner_y + ny * minimum_distance
                mallet_vx = mallet_vy = 0.0

        return mallet_x, mallet_y, mallet_vx, mallet_vy

    def _choose_ai_target(self, difficulty: Difficulty) -> None:
        if self.awaiting_serve:
            self.ai_target_x = RINK_CENTER_X
            if self.current_server == "ai":
                self.ai_target_y = RINK_CENTER_Y - MALLET_RADIUS - 5 * UI_SCALE
            else:
                self.ai_target_y = self.ai_home_y
            return

        puck = self.puck
        error = self.random.uniform(-difficulty.aim_error, difficulty.aim_error)
        safe_left = RINK_LEFT + MALLET_RADIUS
        safe_right = RINK_RIGHT - MALLET_RADIUS
        puck_speed = math.hypot(*puck.collision_velocity())
        puck_near_center = puck.y <= RINK_CENTER_Y + PUCK_RADIUS + 2 * UI_SCALE

        puck_in_attack_zone = puck.y <= difficulty.attack_line
        puck_moving_toward_ai = puck.vy < -25 * UI_SCALE

        if puck_speed <= PUCK_STOP_SPEED and puck_near_center:
            # 冰球停在中线附近时主动接触，避免双方被半场限制卡在球的两侧。
            self.ai_target_x = clamp(puck.x + error, safe_left, safe_right)
            self.ai_target_y = clamp(
                puck.y,
                RINK_TOP + MALLET_RADIUS,
                RINK_CENTER_Y - MALLET_RADIUS,
            )
            return

        if puck_in_attack_zone:
            # 出击：向球的位置移动；球槌仍受速度和半场边界限制。
            self.ai_target_x = clamp(puck.x + error, safe_left, safe_right)
            self.ai_target_y = clamp(
                puck.y,
                self.ai_home_y,
                RINK_CENTER_Y - MALLET_RADIUS,
            )
            return

        self.ai_target_y = self.ai_home_y
        if puck_moving_toward_ai:
            predicted_x = self._predict_puck_x(self.ai_home_y)
            blended_x = (
                RINK_CENTER_X * (1.0 - difficulty.prediction)
                + predicted_x * difficulty.prediction
            )
            self.ai_target_x = clamp(blended_x + error, safe_left, safe_right)
        else:
            # 球离开电脑半场时回守，避免一直贴着中线追球。
            self.ai_target_x = clamp(RINK_CENTER_X + error * 0.35, safe_left, safe_right)

    def _predict_puck_x(self, target_y: float) -> float:
        puck = self.puck
        velocity_x, velocity_y = puck.collision_velocity()

        speed = math.hypot(velocity_x, velocity_y)
        if speed <= COLLISION_EPSILON or velocity_y >= -COLLISION_EPSILON:
            return RINK_CENTER_X

        direction_x = velocity_x / speed
        direction_y = velocity_y / speed
        distance_to_target = (target_y - self.puck.y) / direction_y
        if distance_to_target <= 0:
            return reflected_coordinate(
                self.puck.x,
                RINK_LEFT + PUCK_RADIUS,
                RINK_RIGHT - PUCK_RADIUS,
            )

        # 摩擦沿运动方向消耗速度；如果冰球提前停下，则预测停止位置。
        if PUCK_FRICTION_DECELERATION <= COLLISION_EPSILON:
            travel_distance = distance_to_target
        else:
            stopping_distance = speed * speed / (2.0 * PUCK_FRICTION_DECELERATION)
            travel_distance = min(distance_to_target, stopping_distance)

        projected_x = self.puck.x + direction_x * travel_distance
        return reflected_coordinate(
            projected_x,
            RINK_LEFT + PUCK_RADIUS,
            RINK_RIGHT - PUCK_RADIUS,
        )

    @staticmethod
    def _move_towards(
        x: float,
        y: float,
        target_x: float,
        target_y: float,
        max_speed: float,
        dt: float,
    ) -> tuple[float, float, float, float]:
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        if distance <= 1e-6 or dt <= 0 or max_speed <= 0:
            return x, y, 0.0, 0.0

        travel = min(distance, max_speed * dt)
        if travel <= 1e-9:
            return x, y, 0.0, 0.0
        new_x = x + dx / distance * travel
        new_y = y + dy / distance * travel
        return new_x, new_y, (new_x - x) / dt, (new_y - y) / dt

    def _move_puck(self, dt: float) -> bool:
        puck = self.puck
        puck.x += puck.vx * dt
        puck.y += puck.vy * dt

        if self._check_goal():
            return True

        puck.resolve_walls()
        puck.resolve_goal_posts()
        self._collide_with_mallet(
            self.player_x,
            self.player_y,
            self.player_vx * PLAYER_IMPACT_SPEED_SCALE,
            self.player_vy * PLAYER_IMPACT_SPEED_SCALE,
        )
        self._collide_with_mallet(
            self.ai_x,
            self.ai_y,
            self.ai_vx * AI_IMPACT_SPEED_SCALE,
            self.ai_vy * AI_IMPACT_SPEED_SCALE,
        )
        # 球槌可能在同一物理步内把冰球重新推向边界，必须再次约束。
        puck.resolve_walls()
        puck.resolve_goal_posts()

        puck.advance_velocity(dt)

        moving_speed = math.hypot(puck.vx, puck.vy)
        if puck.response_active:
            moving_speed = max(moving_speed, math.hypot(puck.target_vx, puck.target_vy))
        if self.awaiting_serve and moving_speed > PUCK_STOP_SPEED:
            self.awaiting_serve = False
        return False

    def _check_goal(self) -> bool:
        scorer = goal_scorer(self.puck)
        if scorer is None:
            return False

        if scorer == "player":
            self.player_score += 1
            scorer_name = "玩家"
        else:
            self.ai_score += 1
            scorer_name = "电脑"

        self._update_score_text()
        next_server = "ai" if self.current_server == "player" else "player"
        self._reset_round(
            f"{scorer_name}得分！{self._serve_instruction(next_server)}",
            next_server,
        )
        return True

    @staticmethod
    def _fallback_mallet_contact_normal(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """中心重合时选择安全推出方向，并避开球门开口的外侧方向。"""
        candidates = (
            (1.0, 0.0, RINK_RIGHT - PUCK_RADIUS - x),
            (-1.0, 0.0, x - RINK_LEFT - PUCK_RADIUS),
        )
        if puck_inside_goal_mouth(x):
            if y <= RINK_CENTER_Y:
                candidates += ((0.0, 1.0, RINK_BOTTOM - PUCK_RADIUS - y),)
            else:
                candidates += ((0.0, -1.0, y - RINK_TOP - PUCK_RADIUS),)
        else:
            candidates += (
                (0.0, 1.0, RINK_BOTTOM - PUCK_RADIUS - y),
                (0.0, -1.0, y - RINK_TOP - PUCK_RADIUS),
            )
        nx, ny, _ = max(candidates, key=lambda candidate: candidate[2])
        return nx, ny

    def _collide_with_mallet(
        self,
        mallet_x: float,
        mallet_y: float,
        mallet_vx: float,
        mallet_vy: float,
    ) -> None:
        puck = self.puck
        dx = puck.x - mallet_x
        dy = puck.y - mallet_y
        minimum_distance = PUCK_RADIUS + MALLET_RADIUS
        distance_sq = dx * dx + dy * dy
        if distance_sq >= minimum_distance * minimum_distance:
            return

        collision_vx, collision_vy = puck.collision_velocity()

        if distance_sq <= COLLISION_EPSILON:
            relative_x = collision_vx - mallet_vx
            relative_y = collision_vy - mallet_vy
            relative_length = math.hypot(relative_x, relative_y)
            if relative_length > COLLISION_EPSILON:
                # 中心完全重合时没有几何法线，选取与相对运动相反的方向，
                # 让球被推出并能正确继承球槌的击打速度。
                nx, ny = -relative_x / relative_length, -relative_y / relative_length
            else:
                nx, ny = self._fallback_mallet_contact_normal(puck.x, puck.y)
            distance = 0.0
        else:
            distance = math.sqrt(distance_sq)
            nx, ny = dx / distance, dy / distance

        overlap = minimum_distance - distance
        puck.x += nx * overlap
        puck.y += ny * overlap

        relative_normal_speed = (
            (collision_vx - mallet_vx) * nx + (collision_vy - mallet_vy) * ny
        )
        # 只按接近方向的相对速度传递力度；静止接触只解除重叠，不增加球速。
        if relative_normal_speed < 0:
            impulse = (1.0 + MALLET_RESTITUTION) * relative_normal_speed
            puck.set_target_velocity(
                collision_vx - impulse * nx,
                collision_vy - impulse * ny,
            )

    def _update_score_text(self) -> None:
        score_text = f"{self.player_score} : {self.ai_score}"
        self.score_var.set(score_text)
        if hasattr(self, "score_label"):
            score_width = max(5, len(score_text))
            score_font_size = max(12, min(24, 24 - max(0, len(score_text) - 7)))
            self.score_label.configure(
                width=score_width,
                font=("Consolas", score_font_size, "bold"),
            )

    def _render(self) -> None:
        # 预测线只服务于视觉反馈，不参与物理和采集；100 Hz 已足够平滑，
        # 冰球和球槌本体仍然在每个显示周期更新。
        self.prediction_render_counter += 1
        if self.prediction_render_counter >= 2:
            self.prediction_render_counter = 0
            self._render_puck_prediction()
        puck = self.puck
        glint_offset = MALLET_RADIUS * 0.27
        glint_radius = MALLET_RADIUS * 0.2
        self._position_circle(self.player_item, self.player_x, self.player_y, MALLET_RADIUS)
        self._position_circle(self.ai_item, self.ai_x, self.ai_y, MALLET_RADIUS)
        self._position_circle(self.puck_item, puck.x, puck.y, PUCK_RADIUS)
        self._position_circle(
            self.player_glint,
            self.player_x - glint_offset,
            self.player_y - glint_offset,
            glint_radius,
        )
        self._position_circle(
            self.ai_glint,
            self.ai_x - glint_offset,
            self.ai_y - glint_offset,
            glint_radius,
        )

        if self.running and self.awaiting_serve:
            self.status_text = self.round_message
        elif self.running:
            self.status_text = ""
        elif not self.started:
            self.status_text = self._serve_instruction("player")
        if self.status_text != self.last_rendered_status:
            self.status_header_var.set(self.status_text)
            self.last_rendered_status = self.status_text

    def _render_puck_prediction(self) -> None:
        puck = self.puck
        current_velocity = (puck.vx, puck.vy, puck.target_vx, puck.target_vy)
        cached_vx, cached_vy, cached_target_vx, cached_target_vy = (
            self.prediction_cache_velocity
        )
        velocity_delta = math.hypot(puck.vx - cached_vx, puck.vy - cached_vy)
        target_delta = math.hypot(
            puck.target_vx - cached_target_vx,
            puck.target_vy - cached_target_vy,
        )
        reference_speed = max(
            math.hypot(cached_vx, cached_vy),
            math.hypot(cached_target_vx, cached_target_vy),
            UI_SCALE,
        )
        cache_state_changed = (
            puck.response_active != self.prediction_cache_response_active
            or velocity_delta > max(10.0 * UI_SCALE, reference_speed * 0.25)
            or target_delta > max(10.0 * UI_SCALE, reference_speed * 0.25)
        )
        if self.prediction_cache_age >= PREDICTION_REFRESH_INTERVAL or cache_state_changed:
            self.prediction_cache = self._calculate_predicted_trajectory()
            self.prediction_cache_origin = (puck.x, puck.y)
            self.prediction_cache_velocity = current_velocity
            self.prediction_cache_response_active = puck.response_active
            self.prediction_cache_age = 0.0

        origin_x, origin_y = self.prediction_cache_origin
        target_points = [
            (point_x - origin_x, point_y - origin_y)
            for point_x, point_y in self.prediction_cache
        ]
        if not target_points:
            self.prediction_display_points = []
        elif (
            cache_state_changed
            or len(self.prediction_display_points) != len(target_points)
        ):
            # 碰撞和反向是明确的物理事件，立即切到新方向；普通刷新走平滑追踪。
            self.prediction_display_points = target_points
        else:
            blend = PREDICTION_DISPLAY_SMOOTHING
            self.prediction_display_points = [
                (
                    display_x + (target_x - display_x) * blend,
                    display_y + (target_y - display_y) * blend,
                )
                for (display_x, display_y), (target_x, target_y) in zip(
                    self.prediction_display_points,
                    target_points,
                )
            ]

        predicted_points = [
            (puck.x + point_x, puck.y + point_y)
            for point_x, point_y in self.prediction_display_points
        ]
        path_points = [(puck.x, puck.y)]
        minimum_render_distance = max(1.5, 2.5 * UI_SCALE)
        for point_x, point_y in predicted_points:
            last_x, last_y = path_points[-1]
            dx = point_x - last_x
            dy = point_y - last_y
            if dx * dx + dy * dy >= minimum_render_distance * minimum_render_distance:
                path_points.append((point_x, point_y))
        segment_count = len(path_points) - 1

        if segment_count == 0:
            if self.prediction_line_visible:
                self.canvas.itemconfigure(self.prediction_line_item, state="hidden")
                self.prediction_line_visible = False
            return

        coordinates = [coordinate for point in path_points for coordinate in point]
        self.canvas.coords(self.prediction_line_item, *coordinates)
        if not self.prediction_line_visible:
            self.canvas.itemconfigure(self.prediction_line_item, state="normal")
            self.prediction_line_visible = True

    def _calculate_predicted_trajectory(self) -> list[tuple[float, float]]:
        """同步模拟加速与摩擦；球槌会移动，因此预测在首次触槌前停止。"""
        motion = replace(self.puck)
        current_speed = math.hypot(motion.vx, motion.vy)
        target_speed = math.hypot(motion.target_vx, motion.target_vy)
        if self.awaiting_serve or (
            current_speed <= PUCK_STOP_SPEED
            and (not motion.response_active or target_speed <= PUCK_STOP_SPEED)
        ):
            return []

        sample_elapsed = 0.0
        bend_count = 0
        first_bend_index: int | None = None
        predicted_points: list[tuple[float, float]] = []
        simulation_steps = 0

        while (
            len(predicted_points) < PREDICTION_POINT_COUNT
            and simulation_steps < PREDICTION_MAX_SIMULATION_STEPS
        ):
            simulation_steps += 1
            motion.x += motion.vx * PREDICTION_SUBSTEP
            motion.y += motion.vy * PREDICTION_SUBSTEP
            if goal_scorer(motion):
                break

            bounced_this_step = motion.resolve_walls()
            bounced_this_step = motion.resolve_goal_posts() or bounced_this_step
            if bounced_this_step:
                bend_count += 1
                if not predicted_points or math.hypot(
                    motion.x - predicted_points[-1][0],
                    motion.y - predicted_points[-1][1],
                ) > 1.0:
                    predicted_points.append((motion.x, motion.y))
                if bend_count == 1:
                    first_bend_index = len(predicted_points) - 1
                sample_elapsed = 0.0
                if (
                    bend_count > PREDICTION_MAX_BENDS
                    or len(predicted_points) >= PREDICTION_POINT_COUNT
                ):
                    break

            collision_distance = PUCK_RADIUS + MALLET_RADIUS
            if any(
                (motion.x - mallet_x) ** 2 + (motion.y - mallet_y) ** 2
                <= collision_distance * collision_distance
                for mallet_x, mallet_y in (
                    (self.player_x, self.player_y),
                    (self.ai_x, self.ai_y),
                )
            ):
                break

            motion.advance_velocity(PREDICTION_SUBSTEP)
            if motion.vx == 0.0 and motion.vy == 0.0 and not motion.response_active:
                break

            sample_elapsed += PREDICTION_SUBSTEP
            if sample_elapsed + 1e-9 >= PREDICTION_POINT_INTERVAL:
                sample_elapsed -= PREDICTION_POINT_INTERVAL
                predicted_points.append((motion.x, motion.y))

        if first_bend_index is not None and len(predicted_points) > first_bend_index + 1:
            bend_x, bend_y = predicted_points[first_bend_index]
            end_x, end_y = predicted_points[-1]
            target_x = bend_x + (end_x - bend_x) * PREDICTION_SECOND_SEGMENT_SCALE
            target_y = bend_y + (end_y - bend_y) * PREDICTION_SECOND_SEGMENT_SCALE
            target_length = math.hypot(target_x - bend_x, target_y - bend_y)

            shortened_points = predicted_points[: first_bend_index + 1]
            for point_x, point_y in predicted_points[first_bend_index + 1 :]:
                if math.hypot(point_x - bend_x, point_y - bend_y) >= target_length:
                    break
                shortened_points.append((point_x, point_y))
            shortened_points.append((target_x, target_y))
            predicted_points = shortened_points

        if predicted_points:
            # 活动状态下固定预测点数量，避免摩擦导致末端线段逐帧减少，
            # 从而让轨迹尾部出现视觉抖动。补齐点重合，不会画出额外线段。
            predicted_points.extend(
                [predicted_points[-1]]
                * (PREDICTION_POINT_COUNT - len(predicted_points))
            )
        return predicted_points

    def _position_circle(self, item: int, x: float, y: float, radius: float) -> None:
        self.canvas.coords(item, x - radius, y - radius, x + radius, y + radius)

    def _close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.game_loop_id is not None:
            try:
                self.root.after_cancel(self.game_loop_id)
            except tk.TclError:
                pass
            finally:
                self.game_loop_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    # 某些本机终端没有继承桌面环境变量，但仍可连接已运行的 X11 会话。
    if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"
        xauthority = "/run/user/1000/gdm/Xauthority"
        if os.path.exists(xauthority) and not os.environ.get("XAUTHORITY"):
            os.environ["XAUTHORITY"] = xauthority
    root = tk.Tk()
    configure_responsive_layout(root)
    AirHockeyGame(root)
    center_window(root)
    root.mainloop()


if __name__ == "__main__":
    main()
