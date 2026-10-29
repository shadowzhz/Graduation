"""空气冰壶仿真 GUI 布局配置。

仅包含与 Tkinter 仿真窗口尺寸、响应式缩放计算、窗口居中相关的界面显示配置。
不包含也不得修改任何共享物理与算法常量。
"""

from air_hockey.core_config import BASE_CANVAS_HEIGHT, BASE_CANVAS_WIDTH
UI_SCALE = 1.0

CANVAS_WIDTH = BASE_CANVAS_WIDTH
CANVAS_HEIGHT = BASE_CANVAS_HEIGHT


def configure_responsive_layout(root):
    """按屏幕大小缩放画布；在创建 AirHockeyGame 之前调用。"""
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
