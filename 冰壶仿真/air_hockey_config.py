"""空气冰壶仿真 GUI 布局配置。

仅包含与 Tkinter 仿真窗口尺寸、响应式缩放计算、窗口居中相关的界面显示配置。
不包含也不得修改任何共享物理与算法常量。
"""

import threading

BASE_CANVAS_WIDTH = 600.0
BASE_CANVAS_HEIGHT = 760.0
UI_SCALE = 1.0

CANVAS_WIDTH = BASE_CANVAS_WIDTH
CANVAS_HEIGHT = BASE_CANVAS_HEIGHT

LAYOUT_CONFIG_LOCK = threading.RLock()
LAYOUT_THREAD_ID = None
LAYOUT_CONFIGURED = False
LAYOUT_ROOT = None


def configure_responsive_layout(root):
    """按屏幕大小缩放画布，只能在 Tk 主线程调一次。"""
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
    global UI_SCALE, CANVAS_WIDTH, CANVAS_HEIGHT
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    available_height = max(360.0, screen_height - 220.0)
    available_width = max(360.0, screen_width - 80.0)
    UI_SCALE = min(1.0, available_width / BASE_CANVAS_WIDTH, available_height / BASE_CANVAS_HEIGHT)
    CANVAS_WIDTH = round(BASE_CANVAS_WIDTH * UI_SCALE)
    CANVAS_HEIGHT = round(BASE_CANVAS_HEIGHT * UI_SCALE)


def center_window(root):
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
