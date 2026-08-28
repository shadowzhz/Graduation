"""鼠标对战电脑的虚拟空气冰球，只用 tkinter。

运行：python air_hockey.py
玩家按住左键控制下方蓝色球槌，电脑控制上方红色球槌。
"""

import math
import os
import sys
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from air_hockey_config import *
from air_hockey_ai import AIDecision, AirHockeyAI
from air_hockey_physics import (
    PuckMotion,
    circle_post_contact,
    clamp,
    goal_scorer,
    puck_inside_goal_mouth,
)
from game_state import GameState, PuckState


class AirHockeyGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.ai_controller = AirHockeyAI()
        self.closed = False
        self.game_loop_id = None
        self.player_score = self.ai_score = 0
        self.running = False
        self.started = False
        initial_message = self._serve_instruction("player")
        self.mouse_button_down = False
        self.mouse_control_active = False
        self.ai_home_y = RINK_TOP + (RINK_CENTER_Y - RINK_TOP) * 0.28
        self.score_var = tk.StringVar(value="0 : 0")
        self.status_text = initial_message
        self.status_header_var = tk.StringVar(value=initial_message)
        self.last_rendered_status = initial_message
        self.difficulty = DIFFICULTIES["普通"]
        self.prediction_cache = []
        self.prediction_display_points = []
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
        style.configure("Difficulty.TRadiobutton", background="#102f4c", foreground="#eaf7ff", font=("Microsoft YaHei UI", 10), padding=(8, 5))
        style.map("Difficulty.TRadiobutton", background=[("active", "#174469")], foreground=[("active", "#ffffff")])

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#0b2239", height=154)
        header.pack(fill="x", padx=16, pady=(12, 7))
        header.pack_propagate(False)
        title_row = tk.Frame(header, bg="#0b2239")
        title_row.pack(fill="x")
        title_box = tk.Frame(title_row, bg="#0b2239")
        title_box.pack(side="left")
        tk.Label(title_box, text="虚拟空气冰球", bg="#0b2239", fg="#ffffff", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        tk.Label(title_box, text="红方电脑在上 · 蓝方玩家在下", bg="#0b2239", fg="#9fc6df", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(2, 0))
        tk.Label(title_box, textvariable=self.status_header_var, bg="#0b2239", fg="#f5cf70", font=("Microsoft YaHei UI", 9, "bold"), anchor="w", wraplength=max(240, int(CANVAS_WIDTH * 0.58))).pack(anchor="w", pady=(2, 0))
        score_box = tk.Frame(title_row, bg="#0b2239")
        score_box.pack(side="right")
        tk.Label(score_box, text="玩家          电脑", bg="#0b2239", fg="#9fc6df", font=("Microsoft YaHei UI", 9)).pack()
        self.score_label = tk.Label(score_box, textvariable=self.score_var, bg="#0b2239", fg="#ffffff", font=("Consolas", 24, "bold"), anchor="center")
        self.score_label.pack()
        controls = tk.Frame(header, bg="#0b2239")
        controls.pack(fill="x", pady=(8, 0))
        difficulty_box = tk.Frame(controls, bg="#102f4c", padx=5, pady=3)
        difficulty_box.pack(side="left")
        tk.Label(difficulty_box, text="难度", bg="#102f4c", fg="#9fc6df", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(4, 2))
        for name in DIFFICULTIES:
            ttk.Radiobutton(difficulty_box, text=name, value=name, variable=self.difficulty_var, command=self._difficulty_changed, style="Difficulty.TRadiobutton", takefocus=False).pack(side="left")
        button_box = tk.Frame(controls, bg="#0b2239")
        button_box.pack(side="right")
        tk.Button(button_box, textvariable=self.start_button_var, command=self.toggle_pause, width=10, relief="flat", bd=0, bg="#18a7d6", activebackground="#36b9e3", fg="#ffffff", activeforeground="#ffffff", font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(button_box, text="重新开局", command=self.reset_match, width=9, relief="flat", bd=0, bg="#274c69", activebackground="#356685", fg="#ffffff", activeforeground="#ffffff", font=("Microsoft YaHei UI", 10), cursor="hand2").pack(side="left")
        self.canvas = tk.Canvas(self.root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#dff4fc", highlightthickness=0, cursor="arrow")
        self.canvas.pack(padx=20, pady=(0, 7))
        self._draw_rink()
        tk.Label(self.root, text="按住左键拖动：控制球槌    预测线：预测轨迹    空格：开始/暂停    R：重新开局", bg="#0b2239", fg="#86adc5", font=("Microsoft YaHei UI", 9), wraplength=CANVAS_WIDTH).pack(pady=(0, 12))

    def _draw_rink(self) -> None:
        c = self.canvas
        scale = UI_SCALE
        line_width_4 = max(1, round(4 * scale))
        line_width_3 = max(1, round(3 * scale))
        line_width_2 = max(1, round(2 * scale))
        goal_ranges = ((RINK_TOP - GOAL_DEPTH, RINK_TOP), (RINK_BOTTOM, RINK_BOTTOM + GOAL_DEPTH))
        for y1, y2 in goal_ranges:
            c.create_rectangle(GOAL_LEFT, y1, GOAL_RIGHT, y2, fill="#c7e8f4", outline="#e84b5f", width=line_width_4)
        for y1, y2 in goal_ranges:
            for x in range(int(GOAL_LEFT + 14 * scale), int(GOAL_RIGHT), max(1, round(16 * scale))):
                c.create_line(x, y1, x, y2, fill="#98c6d7")
        c.create_rectangle(RINK_LEFT, RINK_TOP, RINK_RIGHT, RINK_BOTTOM, fill="#edfaff", outline="#167ca8", width=max(1, round(5 * scale)))
        c.create_line(RINK_LEFT + 3 * scale, RINK_CENTER_Y, RINK_RIGHT - 3 * scale, RINK_CENTER_Y, fill="#e45b69", width=line_width_4)
        c.create_oval(RINK_CENTER_X - 82 * scale, RINK_CENTER_Y - 82 * scale, RINK_CENTER_X + 82 * scale, RINK_CENTER_Y + 82 * scale, outline="#55aacf", width=line_width_3)
        c.create_oval(RINK_CENTER_X - 7 * scale, RINK_CENTER_Y - 7 * scale, RINK_CENTER_X + 7 * scale, RINK_CENTER_Y + 7 * scale, fill="#e45b69", outline="")
        for y in (RINK_TOP + 205 * scale, RINK_BOTTOM - 205 * scale):
            c.create_line(RINK_LEFT + 3 * scale, y, RINK_RIGHT - 3 * scale, y, fill="#7cc5df", width=line_width_2)
            for x in (RINK_CENTER_X - 145 * scale, RINK_CENTER_X + 145 * scale):
                c.create_oval(x - 29 * scale, y - 29 * scale, x + 29 * scale, y + 29 * scale, outline="#7cc5df", width=line_width_2)
                c.create_oval(x - 5 * scale, y - 5 * scale, x + 5 * scale, y + 5 * scale, fill="#e45b69", outline="")
        for y, text in ((RINK_TOP + 25 * scale, "电脑半场"), (RINK_BOTTOM - 25 * scale, "玩家半场")):
            c.create_text(RINK_CENTER_X, y, text=text, fill="#5b9fbd", font=("Microsoft YaHei UI", max(8, round(11 * scale)), "bold"))
        self.prediction_line_item = c.create_line(0, 0, 0, 0, fill=PREDICTION_COLOR, width=max(2, round(3.2 * scale)), capstyle=tk.BUTT, joinstyle=tk.ROUND, state="hidden")
        self.player_item = c.create_oval(0, 0, 0, 0, fill="#118fc5", outline="#075e83", width=4)
        self.ai_item = c.create_oval(0, 0, 0, 0, fill="#eb4f5d", outline="#9f2633", width=4)
        self.puck_item = c.create_oval(0, 0, 0, 0, fill="#172b3b", outline="#07121b", width=3)
        self.player_glint = c.create_oval(0, 0, 0, 0, fill="#76d1ee", outline="")
        self.ai_glint = c.create_oval(0, 0, 0, 0, fill="#ff9ca5", outline="")

    def _bind_controls(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._mouse_pressed)
        self.canvas.bind("<B1-Motion>", self._mouse_dragged)
        self.root.bind("<ButtonRelease-1>", self._mouse_released)
        self.root.bind("<space>", self._space_pressed)
        self.root.bind("<Key-r>", lambda _event: self.reset_match())
        self.root.bind("<Key-R>", lambda _event: self.reset_match())
        self.root.bind("1", lambda _event: self._set_difficulty("简单"))
        self.root.bind("2", lambda _event: self._set_difficulty("普通"))
        self.root.bind("3", lambda _event: self._set_difficulty("困难"))

    def _update_mouse_target(self, event) -> None:
        self.mouse_target_x = float(event.x)
        self.mouse_target_y = float(event.y)

    def _mouse_pressed(self, event) -> None:
        self._update_mouse_target(event)
        self.mouse_button_down = True
        self._sync_player_mouse_control()

    def _mouse_dragged(self, event: tk.Event) -> None:
        if self.mouse_button_down:
            self._update_mouse_target(event)

    def _stop_player_control(self) -> None:
        self.mouse_control_active = False
        self.player_vx = self.player_vy = 0.0
        self.canvas.configure(cursor="arrow")

    def _sync_player_mouse_control(self) -> None:
        self.mouse_control_active = self.running and self.mouse_button_down
        if self.mouse_control_active:
            self.canvas.configure(cursor="none")
        else:
            self._stop_player_control()

    def _mouse_released(self, _event) -> None:
        self.mouse_button_down = False
        self._sync_player_mouse_control()

    def _space_pressed(self, event):
        if event.widget.winfo_class() in {"Button", "TButton", "Radiobutton", "TRadiobutton"}:
            return None
        self.toggle_pause()
        return "break"

    def read_player_target(self):
        return self.mouse_target_x, self.mouse_target_y

    def toggle_pause(self) -> None:
        self.running = not self.running
        self.started = True
        self.last_frame_time = time.perf_counter()
        self.prediction_cache_age = PREDICTION_REFRESH_INTERVAL
        if self.running:
            self._sync_player_mouse_control()
            self.start_button_var.set("暂停")
        else:
            self._stop_player_control()
            self.start_button_var.set("继续")
            self.status_text = "已暂停"

    def reset_match(self) -> None:
        self.player_score = self.ai_score = 0
        self.running = False
        self.started = False
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
    def _serve_instruction(server):
        return "玩家开球：按住左键，用球槌击球" if server == "player" else "电脑开球"

    def _reset_round(self, message, server) -> None:
        previous_player_position = (self.player_x, self.player_y) if self.running and self.mouse_button_down else None
        previous_mouse_target = (self.mouse_target_x, self.mouse_target_y) if self.mouse_button_down else None
        self.mouse_control_active = False
        self.canvas.configure(cursor="arrow")
        self.current_server = server
        if previous_player_position is None:
            self.player_x = RINK_CENTER_X
            self.player_y = RINK_BOTTOM - (RINK_BOTTOM - RINK_CENTER_Y) * 0.47
        else:
            player_min_y = RINK_CENTER_Y + MALLET_RADIUS
            if server == "ai":
                player_min_y += PUCK_RADIUS + 8 * UI_SCALE
            self.player_x = clamp(previous_player_position[0], RINK_LEFT + MALLET_RADIUS, RINK_RIGHT - MALLET_RADIUS)
            self.player_y = clamp(previous_player_position[1], player_min_y, RINK_BOTTOM - MALLET_RADIUS)
        self.player_vx = self.player_vy = 0.0
        if previous_mouse_target is None:
            self.mouse_target_x = self.player_x
            self.mouse_target_y = self.player_y
        else:
            self.mouse_target_x, self.mouse_target_y = previous_mouse_target
        self.ai_x = RINK_CENTER_X
        self.ai_y = self.ai_home_y
        self.ai_vx = self.ai_vy = 0.0
        self.ai_target_x = RINK_CENTER_X
        self.ai_target_y = self.ai_home_y
        self.ai_reaction_timer = 0.0
        self.ai_serve_phase = "positioning" if server == "ai" else "idle"
        self.ai_stalled_puck_phase = "idle"
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
        self._sync_player_mouse_control()

    def _game_loop(self) -> None:
        self.game_loop_id = None
        if self.closed:
            return
        now = time.perf_counter()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        if elapsed > MAX_FRAME_GAP:
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
                player_previous = (self.player_x, self.player_y)
                ai_previous = (self.ai_x, self.ai_y)
                self._move_player(step_time)
                self._move_ai(step_time)
                if self._move_puck(step_time, player_previous, ai_previous):
                    break
        if self.closed:
            return
        self._render()
        self._schedule_game_loop()

    def _move_player(self, dt) -> None:
        if not self.mouse_control_active:
            self.player_vx = self.player_vy = 0.0
            self.player_x, self.player_y, self.player_vx, self.player_vy = self._resolve_mallet_goal_posts(self.player_x, self.player_y, self.player_vx, self.player_vy)
            self.player_x, self.player_y, self.player_vx, self.player_vy = self._keep_mallet_clear_of_outer_corners(self.player_x, self.player_y, self.player_vx, self.player_vy)
            return
        target_x, target_y = self.read_player_target()
        target_x = clamp(target_x, RINK_LEFT + MALLET_RADIUS, RINK_RIGHT - MALLET_RADIUS)
        player_min_y = RINK_CENTER_Y + MALLET_RADIUS
        if self.awaiting_serve and self.current_server == "ai":
            player_min_y = RINK_CENTER_Y + MALLET_RADIUS + PUCK_RADIUS + 8 * UI_SCALE
        target_y = clamp(target_y, player_min_y, RINK_BOTTOM - MALLET_RADIUS)
        self.player_x, self.player_y, self.player_vx, self.player_vy = self._move_towards(self.player_x, self.player_y, target_x, target_y, PLAYER_MAX_SPEED, dt)
        self.player_x, self.player_y, self.player_vx, self.player_vy = self._resolve_mallet_goal_posts(self.player_x, self.player_y, self.player_vx, self.player_vy)
        self.player_x, self.player_y, self.player_vx, self.player_vy = self._keep_mallet_clear_of_outer_corners(self.player_x, self.player_y, self.player_vx, self.player_vy)

    def _move_ai(self, dt) -> None:
        difficulty = self.difficulty
        self._apply_ai_decision(self.ai_controller.update(self._game_state(), dt))
        ai_min_y = RINK_TOP + MALLET_RADIUS
        ai_max_y = RINK_CENTER_Y - MALLET_RADIUS
        if self.awaiting_serve and self.current_server == "player":
            ai_max_y = RINK_CENTER_Y - MALLET_RADIUS - PUCK_RADIUS - 8 * UI_SCALE
            self.ai_y = min(self.ai_y, ai_max_y)
        self.ai_target_y = clamp(self.ai_target_y, ai_min_y, ai_max_y)
        self.ai_x, self.ai_y, self.ai_vx, self.ai_vy = self._move_towards(self.ai_x, self.ai_y, self.ai_target_x, self.ai_target_y, difficulty.ai_speed, dt)
        self.ai_x, self.ai_y, self.ai_vx, self.ai_vy = self._resolve_mallet_goal_posts(self.ai_x, self.ai_y, self.ai_vx, self.ai_vy)
        self.ai_x, self.ai_y, self.ai_vx, self.ai_vy = self._keep_mallet_clear_of_outer_corners(self.ai_x, self.ai_y, self.ai_vx, self.ai_vy)
        self.ai_serve_phase = self.ai_controller.advance_serve_phase(self._game_state(), dt)

    def _game_state(self):
        puck_velocity_x, puck_velocity_y = self.puck.collision_velocity()
        return GameState(
            ai_x=self.ai_x,
            ai_y=self.ai_y,
            ai_home_y=self.ai_home_y,
            target_x=self.ai_target_x,
            target_y=self.ai_target_y,
            puck=PuckState(
                x=self.puck.x,
                y=self.puck.y,
                vx=puck_velocity_x,
                vy=puck_velocity_y,
            ),
            awaiting_serve=self.awaiting_serve,
            current_server=self.current_server,
            serve_phase=self.ai_serve_phase,
            stalled_puck_phase=self.ai_stalled_puck_phase,
            reaction_timer=self.ai_reaction_timer,
            difficulty=self.difficulty,
        )

    def _apply_ai_decision(self, decision) -> None:
        self.ai_target_x = decision.target_x
        self.ai_target_y = decision.target_y
        self.ai_stalled_puck_phase = decision.stalled_puck_phase
        self.ai_reaction_timer = decision.reaction_timer

    @staticmethod
    def _resolve_mallet_goal_posts(mallet_x, mallet_y, mallet_vx, mallet_vy):
        minimum_distance = MALLET_RADIUS + GOAL_POST_RADIUS
        for post_x, post_y in GOAL_POSTS:
            contact = circle_post_contact(mallet_x, mallet_y, post_x, post_y, minimum_distance)
            if contact is None:
                continue
            _distance, nx, ny = contact
            mallet_x = post_x + nx * minimum_distance
            mallet_y = post_y + ny * minimum_distance
            normal_speed = mallet_vx * nx + mallet_vy * ny
            if normal_speed < 0:
                impulse = (1.0 + GOAL_POST_RESTITUTION) * normal_speed
                mallet_vx -= impulse * nx
                mallet_vy -= impulse * ny
        return mallet_x, mallet_y, mallet_vx, mallet_vy

    @staticmethod
    def _keep_mallet_clear_of_outer_corners(mallet_x, mallet_y, mallet_vx, mallet_vy):
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

    @staticmethod
    def _move_towards(x, y, target_x, target_y, max_speed, dt):
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

    def _move_puck(self, dt, player_previous, ai_previous) -> bool:
        puck = self.puck
        puck_previous = (puck.x, puck.y)
        puck.x += puck.vx * dt
        puck.y += puck.vy * dt
        puck.resolve_walls()
        puck.resolve_goal_posts()
        self._swept_collide_with_mallet(puck_previous, player_previous, self.player_x, self.player_y, self.player_vx * PLAYER_IMPACT_SPEED_SCALE, self.player_vy * PLAYER_IMPACT_SPEED_SCALE)
        self._collide_with_mallet(self.player_x, self.player_y, self.player_vx * PLAYER_IMPACT_SPEED_SCALE, self.player_vy * PLAYER_IMPACT_SPEED_SCALE)

        # 防守只改 AI 的移动目标，碰撞照常算
        self._swept_collide_with_mallet(puck_previous, ai_previous, self.ai_x, self.ai_y, self.ai_vx * AI_IMPACT_SPEED_SCALE, self.ai_vy * AI_IMPACT_SPEED_SCALE)
        self._collide_with_mallet(self.ai_x, self.ai_y, self.ai_vx * AI_IMPACT_SPEED_SCALE, self.ai_vy * AI_IMPACT_SPEED_SCALE)

        puck.resolve_walls()
        puck.resolve_goal_posts()
        if self._check_goal():
            return True
        puck.advance_velocity(dt)
        moving_speed = math.hypot(puck.vx, puck.vy)
        if puck.response_active:
            moving_speed = max(moving_speed, math.hypot(puck.target_vx, puck.target_vy))
        if self.awaiting_serve and moving_speed > PUCK_STOP_SPEED:
            self.awaiting_serve = False
        return False

    def _swept_collide_with_mallet(self, puck_previous, mallet_previous, mallet_x, mallet_y, mallet_vx, mallet_vy) -> None:
        # 相对运动检测，防止高速时球槌和冰球互相穿过
        puck = self.puck
        start_x = puck_previous[0] - mallet_previous[0]
        start_y = puck_previous[1] - mallet_previous[1]
        minimum_distance = PUCK_RADIUS + MALLET_RADIUS
        minimum_distance_sq = minimum_distance * minimum_distance
        end_x = puck.x - mallet_x
        end_y = puck.y - mallet_y
        # 只处理帧内穿过但帧末没重叠的情况，接触态交给普通碰撞
        if start_x * start_x + start_y * start_y <= minimum_distance_sq or end_x * end_x + end_y * end_y <= minimum_distance_sq:
            return
        delta_x = (puck.x - puck_previous[0]) - (mallet_x - mallet_previous[0])
        delta_y = (puck.y - puck_previous[1]) - (mallet_y - mallet_previous[1])
        delta_length_sq = delta_x * delta_x + delta_y * delta_y
        if delta_length_sq <= COLLISION_EPSILON:
            return
        impact_time = clamp(-(start_x * delta_x + start_y * delta_y) / delta_length_sq, 0.0, 1.0)
        relative_x = start_x + delta_x * impact_time
        relative_y = start_y + delta_y * impact_time
        distance_sq = relative_x * relative_x + relative_y * relative_y
        if distance_sq >= minimum_distance * minimum_distance:
            return
        distance = math.sqrt(distance_sq)
        if distance <= COLLISION_EPSILON:
            nx, ny = self._fallback_mallet_contact_normal(puck.x, puck.y)
        else:
            nx, ny = relative_x / distance, relative_y / distance
        impact_mallet_x = mallet_previous[0] + (mallet_x - mallet_previous[0]) * impact_time
        impact_mallet_y = mallet_previous[1] + (mallet_y - mallet_previous[1]) * impact_time
        puck.x = impact_mallet_x + nx * (minimum_distance - 1e-6)
        puck.y = impact_mallet_y + ny * (minimum_distance - 1e-6)
        self._collide_with_mallet(impact_mallet_x, impact_mallet_y, mallet_vx, mallet_vy)

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
        self._reset_round(f"{scorer_name}得分！{self._serve_instruction(next_server)}", next_server)
        return True

    @staticmethod
    def _fallback_mallet_contact_normal(x, y):
        candidates = ((1.0, 0.0, RINK_RIGHT - PUCK_RADIUS - x), (-1.0, 0.0, x - RINK_LEFT - PUCK_RADIUS))
        if puck_inside_goal_mouth(x):
            if y <= RINK_CENTER_Y:
                candidates += ((0.0, 1.0, RINK_BOTTOM - PUCK_RADIUS - y),)
            else:
                candidates += ((0.0, -1.0, y - RINK_TOP - PUCK_RADIUS),)
        else:
            candidates += ((0.0, 1.0, RINK_BOTTOM - PUCK_RADIUS - y), (0.0, -1.0, y - RINK_TOP - PUCK_RADIUS))
        nx, ny, _ = max(candidates, key=lambda candidate: candidate[2])
        return nx, ny

    def _collide_with_mallet(self, mallet_x, mallet_y, mallet_vx, mallet_vy) -> None:
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
        relative_normal_speed = (collision_vx - mallet_vx) * nx + (collision_vy - mallet_vy) * ny
        if relative_normal_speed < 0:
            impulse = (1.0 + MALLET_RESTITUTION) * relative_normal_speed
            impact_vx = collision_vx - impulse * nx
            impact_vy = collision_vy - impulse * ny
            if math.hypot(puck.vx, puck.vy) <= PUCK_STOP_SPEED:
                puck.set_immediate_velocity(impact_vx, impact_vy)
            else:
                puck.set_target_velocity(impact_vx, impact_vy)

    def _update_score_text(self) -> None:
        score_text = f"{self.player_score} : {self.ai_score}"
        self.score_var.set(score_text)
        if hasattr(self, "score_label"):
            score_width = max(5, len(score_text))
            score_font_size = max(12, min(24, 24 - max(0, len(score_text) - 7)))
            self.score_label.configure(width=score_width, font=("Consolas", score_font_size, "bold"))

    def _render(self) -> None:
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
        self._position_circle(self.player_glint, self.player_x - glint_offset, self.player_y - glint_offset, glint_radius)
        self._position_circle(self.ai_glint, self.ai_x - glint_offset, self.ai_y - glint_offset, glint_radius)
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
        cached_vx, cached_vy, cached_target_vx, cached_target_vy = self.prediction_cache_velocity
        velocity_delta = math.hypot(puck.vx - cached_vx, puck.vy - cached_vy)
        target_delta = math.hypot(puck.target_vx - cached_target_vx, puck.target_vy - cached_target_vy)
        reference_speed = max(math.hypot(cached_vx, cached_vy), math.hypot(cached_target_vx, cached_target_vy), UI_SCALE)
        cache_state_changed = puck.response_active != self.prediction_cache_response_active or velocity_delta > max(10.0 * UI_SCALE, reference_speed * 0.25) or target_delta > max(10.0 * UI_SCALE, reference_speed * 0.25)
        if self.prediction_cache_age >= PREDICTION_REFRESH_INTERVAL or cache_state_changed:
            self.prediction_cache = self._calculate_predicted_trajectory()
            self.prediction_cache_origin = (puck.x, puck.y)
            self.prediction_cache_velocity = current_velocity
            self.prediction_cache_response_active = puck.response_active
            self.prediction_cache_age = 0.0
        origin_x, origin_y = self.prediction_cache_origin
        target_points = [(point_x - origin_x, point_y - origin_y) for point_x, point_y in self.prediction_cache]
        if not target_points:
            self.prediction_display_points = []
        elif cache_state_changed or len(self.prediction_display_points) != len(target_points):
            self.prediction_display_points = target_points
        else:
            blend = PREDICTION_DISPLAY_SMOOTHING
            self.prediction_display_points = [(display_x + (target_x - display_x) * blend, display_y + (target_y - display_y) * blend) for (display_x, display_y), (target_x, target_y) in zip(self.prediction_display_points, target_points)]
        predicted_points = [(puck.x + point_x, puck.y + point_y) for point_x, point_y in self.prediction_display_points]
        path_points = [(puck.x, puck.y)]
        minimum_render_distance = max(1.5, 2.5 * UI_SCALE)
        for point_x, point_y in predicted_points:
            last_x, last_y = path_points[-1]
            if (point_x - last_x) ** 2 + (point_y - last_y) ** 2 >= minimum_render_distance ** 2:
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

    def _calculate_predicted_trajectory(self):
        motion = replace(self.puck)
        current_speed = math.hypot(motion.vx, motion.vy)
        target_speed = math.hypot(motion.target_vx, motion.target_vy)
        if self.awaiting_serve or (current_speed <= PUCK_STOP_SPEED and (not motion.response_active or target_speed <= PUCK_STOP_SPEED)):
            return []
        sample_elapsed = 0.0
        bend_count = 0
        first_bend_index = None
        predicted_points: list[tuple[float, float]] = []
        simulation_steps = 0
        while len(predicted_points) < PREDICTION_POINT_COUNT and simulation_steps < PREDICTION_MAX_SIMULATION_STEPS:
            simulation_steps += 1
            motion.x += motion.vx * PREDICTION_SUBSTEP
            motion.y += motion.vy * PREDICTION_SUBSTEP
            if goal_scorer(motion):
                break
            bounced_this_step = motion.resolve_walls()
            bounced_this_step = motion.resolve_goal_posts() or bounced_this_step
            if bounced_this_step:
                bend_count += 1
                if not predicted_points or math.hypot(motion.x - predicted_points[-1][0], motion.y - predicted_points[-1][1]) > 1.0:
                    predicted_points.append((motion.x, motion.y))
                if bend_count == 1:
                    first_bend_index = len(predicted_points) - 1
                sample_elapsed = 0.0
                if bend_count > PREDICTION_MAX_BENDS or len(predicted_points) >= PREDICTION_POINT_COUNT:
                    break
            collision_distance = PUCK_RADIUS + MALLET_RADIUS
            if any((motion.x - mallet_x) ** 2 + (motion.y - mallet_y) ** 2 <= collision_distance ** 2 for mallet_x, mallet_y in ((self.player_x, self.player_y), (self.ai_x, self.ai_y))):
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
            predicted_points.extend([predicted_points[-1]] * (PREDICTION_POINT_COUNT - len(predicted_points)))
        return predicted_points

    def _position_circle(self, item, x, y, radius) -> None:
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


def main():
    if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"
        xauthority = "/run/user/1000/gdm/Xauthority"
        if os.path.exists(xauthority) and not os.environ.get("XAUTHORITY"):
            os.environ["XAUTHORITY"] = xauthority
    root = tk.Tk()
    configure_responsive_layout(root)
    sync_layout_globals(globals())
    AirHockeyGame(root)
    center_window(root)
    root.mainloop()


if __name__ == "__main__":
    main()
