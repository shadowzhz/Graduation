"""GUI 预览：Tk 窗口、PLC 操作按钮与预览帧编码。"""

import tkinter as tk

import cv2

from .renderer import render_preview

DISPLAY_WIDTH = 640         # 窗口图片最大宽度 640


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
    """Tk 窗口，只在主线程使用；定时器从共享区取 PPM 原始像素显示。"""

    def __init__(self):
        self.closed = False
        self.root = tk.Tk()
        self.root.title("冰壶视觉演示：检测 -> 追踪 -> AI")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("q", lambda _event: self._close())
        self.root.bind("Q", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.status = tk.StringVar(value="等待画面")
        status_panel = tk.Frame(self.root, width=DISPLAY_WIDTH, height=64)
        status_panel.pack(fill="x")
        status_panel.pack_propagate(False)
        tk.Label(
            status_panel, textvariable=self.status, anchor="nw", justify="left",
            wraplength=DISPLAY_WIDTH - 12,
        ).pack(fill="both", expand=True)
        self._photo = None
        self._seen_seq = -1
        self._lock = None
        self._shared = None

    def attach_plc(self, link):
        """轴命令由现场操作员主动触发；连接成功不自动使能。

        ★ 单轴按钮：某个轴报 801 / 撞限位之后可以**只把那个轴**撤下来，
          另一个继续用；也能单独回零 / 单独复位。
        ★ 单轴使能**不会**让程序开始追球 —— 运动需要 X、Y 同时使能
          （PLCLink._motion_safe 要求 x_en and y_en）。
        [!] 回零会让该轴动（去找参考点），是刻意的人工动作。
        """
        self._plc = link
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        self._axis_buttons = {}
        for axis, name in (("x", "X"), ("y", "Y")):
            button = tk.Button(
                controls, text=f"{name} 轴使能 ( {name.lower()} )",
                command=lambda a=axis: self._toggle_axis(a))
            button.pack(side="left", expand=True)
            self._axis_buttons[axis] = button
            tk.Button(controls, text=f"{name} 回零 ( {name} )",
                      command=lambda a=axis: self._home_axis(a)).pack(side="left", expand=True)
            tk.Button(controls, text=f"{name} 复位",
                      command=lambda a=axis: self._reset_axis(a)).pack(side="left", expand=True)
        second = tk.Frame(self.root)
        second.pack(fill="x")
        for label, command in (("E 双轴使能", link.enable_axes),
                               ("H 双轴回零", link.home_axes),
                               ("C 全部复位", link.axes_reset)):
            tk.Button(second, text=label, command=command).pack(side="left", expand=True)
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
        button = (self._axis_buttons or {}).get(axis)
        if button is not None:
            name = axis.upper()
            button.config(text=(f"{name} 轴撤销 ( {axis} )" if wanted
                                else f"{name} 轴使能 ( {axis} )"))
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
                status = self._shared["status"]
                fatal = self._shared["fatal"]
            if ppm is not None and seq != self._seen_seq:
                self._photo = tk.PhotoImage(data=ppm, format="PPM")
                self.label.configure(image=self._photo)
                self._seen_seq = seq
            self.status.set(fatal or status)
            if fatal:
                self.closed = True
                self.root.after(2500, self.root.destroy)
                return
            self.root.after(33, self._tick)
        except tk.TclError:
            self.closed = True

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
