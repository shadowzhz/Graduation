"""GUI 预览：Tk 窗口、PLC 操作按钮与预览帧编码。"""

import tkinter as tk
from tkinter import font as tkfont

import cv2

from .renderer import render_preview

DISPLAY_WIDTH = 640         # 窗口图片最大宽度 640
SIDEBAR_WIDTH = 300

# 深色仪表盘：窗口放在明亮的白色球台旁边，深色外框让画面成为焦点、不反光。
BG = "#0f1216"
PANEL = "#161b21"
LINE = "#262e38"
FG = "#e6e9ee"
MUTED = "#8b95a3"
BUTTON = "#222a33"
BUTTON_HOVER = "#2e3844"
PRIMARY = "#1f6f8b"
PRIMARY_HOVER = "#2785a6"
OK = "#3ddc97"
BAD = "#ff6b6b"
# 回合状态 → (中文名, 颜色)；颜色与画面叠加层无关，只区分状态
STATE_STYLE = {
    "WAITING": ("待命", MUTED),
    "DEFENDING": ("防守", "#4cc2ff"),
    "ATTACKING": ("进攻", "#ff9f43"),
    "RECOVERING": ("回位", "#b48cff"),
}


def _family(available, *names):
    """取第一个装了的字体（Windows 现场与 Linux 开发机字体不同）。"""
    return next((name for name in names if name in available), "TkDefaultFont")


def encode_preview_ppm(result, table_roi, camera_geometry, brightness_gain=1.0):
    """将最新 VisionResult 渲染并编码为 Tk 可显示的 PPM 原始像素。"""
    small = render_preview(result, table_roi, camera_geometry, DISPLAY_WIDTH)
    if brightness_gain != 1.0:
        cv2.convertScaleAbs(small, small, alpha=brightness_gain)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return f"P6 {w} {h} 255\n".encode() + rgb.tobytes()


class PreviewBrightness:
    """预览亮度补偿：以第一帧亮度为参考，缓慢跟随环境光，抵消整体明暗闪烁。"""

    def __init__(self):
        self.reference = None

    def gain(self, image):
        blue, green, red = cv2.mean(image[::8, ::8])[:3]
        brightness = 0.114 * blue + 0.587 * green + 0.299 * red
        if self.reference is None:
            self.reference = brightness
        gain = max(0.75, min(1.33, self.reference / max(brightness, 1.0)))
        # ponytail: global preview correction cannot remove local light bands; use flicker-free lighting for those.
        self.reference += 0.005 * (brightness - self.reference)
        return gain


class VisionWindow:
    """Tk 窗口，只在主线程使用；定时器从共享区取 PPM 原始像素和最新结果显示。"""

    def __init__(self, dry_run=False):
        self.closed = False
        self.dry_run = dry_run
        self.root = tk.Tk()
        self.root.title("冰壶对打 · 视觉与控制")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("Q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())

        families = set(tkfont.families(self.root))
        ui = _family(families, "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC")
        mono = _family(families, "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "Courier New")
        self.font = tkfont.Font(family=ui, size=10)
        self.font_small = tkfont.Font(family=ui, size=9)
        self.font_state = tkfont.Font(family=ui, size=22, weight="bold")
        self.font_heading = tkfont.Font(family=ui, size=10, weight="bold")
        self.font_value = tkfont.Font(family=mono, size=11)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(12, 8))
        frame = tk.Frame(body, bg=LINE, padx=1, pady=1)
        frame.pack(side="left", anchor="n")
        self._blank = tk.PhotoImage(width=DISPLAY_WIDTH, height=360)    # 没画面时也占住 640x360
        self.label = tk.Label(frame, bg="black", image=self._blank, text="等待画面", fg=MUTED,
                              font=self.font, compound="center", bd=0)
        self.label.pack()

        side = tk.Frame(body, bg=PANEL, width=SIDEBAR_WIDTH, padx=18, pady=12,
                        highlightthickness=1, highlightbackground=LINE)
        side.pack(side="left", fill="both", expand=True, padx=(12, 0))
        side.pack_propagate(False)
        # 侧栏会被底栏撑宽：换行宽度跟着实际宽度走，长故障信息少占几行
        side.bind("<Configure>", lambda event: [label.configure(wraplength=event.width - 40)
                                                for label in (self._reason, self._plc_message)])
        head = tk.Frame(side, bg=PANEL)
        head.pack(fill="x")
        self._state = tk.Label(head, text="等待画面", font=self.font_state, fg=MUTED, bg=PANEL)
        self._state.pack(side="left")
        self._fps = tk.Label(head, text="", font=self.font_value, fg=MUTED, bg=PANEL)
        self._fps.pack(side="right", anchor="s", pady=(0, 6))
        self._reason = tk.Label(side, text="", font=self.font_small, fg=MUTED, bg=PANEL, anchor="w",
                                justify="left", wraplength=SIDEBAR_WIDTH - 40)
        self._reason.pack(fill="x", pady=(0, 8))
        self._divider(side)

        metrics = tk.Frame(side, bg=PANEL)
        metrics.pack(fill="x", pady=6)
        metrics.columnconfigure(1, weight=1)
        self._values = {}
        for row, name in enumerate(("冰壶", "速度", "目标", "球槌")):
            tk.Label(metrics, text=name, font=self.font, fg=MUTED, bg=PANEL).grid(
                row=row, column=0, sticky="w", pady=2)
            value = tk.Label(metrics, text="—", font=self.font_value, fg=FG, bg=PANEL, anchor="e")
            value.grid(row=row, column=1, sticky="e", pady=2)
            self._values[name] = value
        self._divider(side)

        plc = tk.Frame(side, bg=PANEL)
        plc.pack(fill="x", pady=(8, 0))
        tk.Label(plc, text="PLC", font=self.font_heading, fg=FG, bg=PANEL).pack(side="left")
        self._safe = tk.Label(plc, text="● 安全", font=self.font_small, fg=MUTED, bg=PANEL)
        self._safe.pack(side="right")
        self._armed = tk.Label(plc, text="● 授权", font=self.font_small, fg=MUTED, bg=PANEL)
        self._armed.pack(side="right", padx=(0, 12))
        self._plc_message = tk.Label(side, text="未连接 PLC：只看视觉", font=self.font_small, fg=MUTED,
                                     bg=PANEL, anchor="nw", justify="left", wraplength=SIDEBAR_WIDTH - 40)
        self._plc_message.pack(fill="both", expand=True, pady=(4, 0))

        # 底栏：左边放 PLC 按钮（attach_plc 时才加），右边常驻退出提示
        self._bar = tk.Frame(self.root, bg=BG)
        self._bar.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(self._bar, text="Q / Esc 退出", font=self.font_small, fg=MUTED, bg=BG).pack(side="right")

        self._photo = None
        self._seen_seq = -1
        self._lock = None
        self._shared = None
        self._plc = None
        self._axis_buttons = {}

    @staticmethod
    def _divider(parent):
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x")

    def _button(self, parent, text, command, primary=False):
        base, hover = (PRIMARY, PRIMARY_HOVER) if primary else (BUTTON, BUTTON_HOVER)
        button = tk.Button(parent, text=text, command=command, font=self.font, fg=FG, bg=base,
                           activebackground=hover, activeforeground=FG, relief="flat", bd=0,
                           highlightthickness=0, padx=12, pady=6, cursor="hand2")
        button.bind("<Enter>", lambda _event: button.configure(bg=hover))
        button.bind("<Leave>", lambda _event: button.configure(bg=base))
        button.pack(side="left", padx=(0, 6))
        return button

    def _group(self, parent, title):
        group = tk.Frame(parent, bg=BG)
        group.pack(side="left", padx=(0, 18))
        tk.Label(group, text=title, font=self.font_small, fg=MUTED, bg=BG).pack(side="left", padx=(0, 8))
        return group

    def attach_plc(self, link):
        """轴命令由现场操作员主动触发；连接成功不自动使能。

        ★ 单轴按钮：某个轴报 801 / 撞限位之后可以**只把那个轴**撤下来，
          另一个继续用；也能单独回零 / 单独复位。
        ★ 单轴使能**不会**让程序开始追球 —— 运动需要 X、Y 同时使能
          （PLCLink._motion_safe 要求 x_en and y_en）。
        [!] 回零会让该轴动（去找参考点），是刻意的人工动作。
        """
        self._plc = link
        both = self._group(self._bar, "双轴")
        self._button(both, "使能 · E", link.enable_axes, primary=True)
        self._button(both, "回零 · H", link.home_axes)
        self._button(both, "复位 · C", link.axes_reset)
        for axis in ("x", "y"):
            name = axis.upper()
            group = self._group(self._bar, f"{name} 轴")
            self._axis_buttons[axis] = self._button(
                group, f"使能 · {axis}", lambda a=axis: self._toggle_axis(a))
            self._button(group, f"回零 · {name}", lambda a=axis: self._home_axis(a))
            self._button(group, "复位", lambda a=axis: self._reset_axis(a))
        for key, command in (("e", link.enable_axes), ("h", link.home_axes),
                             ("c", link.axes_reset)):
            self.root.bind(key, lambda _event, callback=command: callback())
        # ★ 大小写区分：小写 = 单轴使能/撤销，大写(Shift) = 单轴回零
        self.root.bind("x", lambda _event: self._toggle_axis("x"))
        self.root.bind("y", lambda _event: self._toggle_axis("y"))
        self.root.bind("X", lambda _event: self._home_axis("x"))
        self.root.bind("Y", lambda _event: self._home_axis("y"))

    def _toggle_axis(self, axis):
        """单轴使能 / 撤销。切换后立刻刷新按钮文字。"""
        link = getattr(self, "_plc", None)
        if link is None:
            return
        sent, wanted = link.toggle_axis_enable(axis)
        if not sent:
            return
        button = self._axis_buttons.get(axis)
        if button is not None:
            button.config(text=f"撤销 · {axis}" if wanted else f"使能 · {axis}")
        if not wanted:
            # 撤下一个轴 → 程序立刻停止发运动（_motion_safe 需要两个都使能）
            print(f"[PLC] {axis.upper()} 轴已撤销使能 —— 在它重新使能前不会下发运动")

    def _home_axis(self, axis):
        link = getattr(self, "_plc", None)
        if link is not None:
            print(f"[PLC] {axis.upper()} 轴回零 —— 该轴会移动，注意手别在行程里")
            link.home_axis(axis)

    def _reset_axis(self, axis):
        link = getattr(self, "_plc", None)
        if link is not None:
            link.reset_axis(axis)

    def start(self, lock, shared):
        self._lock = lock
        self._shared = shared
        self.root.after(33, self._tick)

    def _tick(self):
        if self.closed:
            return
        try:
            with self._lock:
                ppm = self._shared["ppm"]
                seq = self._shared["ppm_seq"]
                result = self._shared["result"]
                plc = self._shared["plc"]
                fatal = self._shared["fatal"]
            if ppm is not None and seq != self._seen_seq:
                self._photo = tk.PhotoImage(data=ppm, format="PPM")
                self.label.configure(image=self._photo, text="")
                self._seen_seq = seq
            self._show(result, plc, fatal)
            if fatal:
                self.closed = True
                self.root.after(2500, self.root.destroy)
                return
            self.root.after(33, self._tick)
        except tk.TclError:
            self.closed = True

    def _show(self, result, plc, fatal):
        """把最新结果写进侧栏。只读快照，不改任何状态。"""
        rally = None if result is None else result.rally
        if fatal:
            name, color, reason = "已停止", BAD, fatal
        elif rally is None:
            name, color, reason = "等待画面", MUTED, ""
        else:
            name, color = STATE_STYLE.get(rally.state.name, (rally.state.name, FG))
            reason = rally.reason
        if self.dry_run:
            reason = f"DRY-RUN 只观察，不发运动 · {reason}"
        self._state.configure(text=name, fg=color)
        self._reason.configure(text=reason)

        stone = None if result is None else result.stone_state
        pair = lambda value, fmt="{:.0f}, {:.0f}": "—" if value is None else fmt.format(*value)
        self._fps.configure(text="" if result is None else f"{result.fps:.0f} fps")
        values = {
            "冰壶": pair(None if stone is None else (stone.x, stone.y)),
            "速度": pair(None if stone is None else (stone.vx, stone.vy), "{:+.0f}, {:+.0f}"),
            "目标": pair(None if rally is None else rally.target),
            "球槌": "—" if rally is None else (
                pair(rally.mallet_position) + ("  PLC" if rally.mallet_source == "PLC" else "  软件")),
        }
        for key, text in values.items():
            self._values[key].configure(text=text)

        if plc is None:
            return
        text, (armed, safe) = plc
        self._armed.configure(fg=OK if armed else MUTED)
        self._safe.configure(fg=OK if safe else BAD)
        # plc_status_text 的格式是 " | PLC armed=.. safe=.. | 说明"，说明里自己也可能带 " | "
        message = text.split(" | ", 2)[-1].removeprefix("PLC ").replace(" | ", " / ")
        self._plc_message.configure(text=message, fg=FG if safe else BAD)

    def _close(self):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def close(self):
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass
