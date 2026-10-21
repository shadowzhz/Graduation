"""轴的机械行程、命令包络、TIA 动力学限值，以及游戏场地 <-> 机器坐标映射。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from air_hockey import core_config

# [!] 两套常量，用途不同，不要混：
#   TRAVEL_* = 轴的机械行程（行程两端是硬件限位开关），命令绝不越过
#   TABLE_*  = 球台真实边缘在【轴坐标】里的位置（只用于游戏<->机器坐标映射）
#
# 2026-10-13 现场实测（PLC 绝对位置读数 + 卷尺）：
#   球台（墙内）480mm x 880mm，球槌半径实测约 26.5mm
#   在轴 X=-188 / 219 处球槌边缘离左右墙各约 10mm
#   Y 最前处（185）球槌边缘离中线约 30mm
# 由此反推球台边缘（轴坐标）：
#   左墙 -188 - 26.5 - 10 = -224.5    右墙 219 + 26.5 + 10 = 255.5   （差 480 OK）
#   中线  185 + 26.5 + 30 =  241.5    后墙 241.5 - 880/2   = -198.5   （差 440 OK）
# TABLE_* 只由这些**几何测量**决定，与轴行程无关，所以改行程时不要动它们。
#
# 机械行程（TIA 工艺对象 XY轴 → 几何结构 →「3D 显示中的表示」，现场实测）：
#   X -192 .. 224        Y -129 .. 187
# 球台比轴的行程大：游戏坐标的边角（例如靠近中线）换算后会超出行程，
# 必须夹回行程内，否则会把轴直接开进限位开关、触发轴故障。
#
# 2026-10-20 现场已装硬件限位开关，代码不再维护一层"软限位"（原 PHYSICAL_* =
# X -180..215 / Y -125..185、再内缩 40mm 的命令包络，以及越过软限位就撤销授权的检测）。
# 现在只把命令夹到机械行程内缩 MotionLimits.margin_mm，避免主动去碰限位开关。
TRAVEL_X_MIN, TRAVEL_X_MAX = -192.0, 224.0
TRAVEL_Y_MIN, TRAVEL_Y_MAX = -129.0, 187.0
TABLE_X_MIN, TABLE_X_MAX = -224.5, 255.5
TABLE_Y_MIN, TABLE_Y_MAX = -198.5, 241.5

# 游戏坐标容差：AI 目标落在场地边界外这么一点点仍按边界处理。
COORD_EPSILON = 1e-6
DEFAULT_ZONE = (
    core_config.RINK_LEFT, core_config.RINK_RIGHT,
    core_config.RINK_TOP, core_config.RINK_CENTER_Y,
)

# 命令余量：下发目标离机械行程端点（限位开关）再往内收这么多毫米。
# 2026-10-21 现场要求让球槌尽量走满机械行程，改为 2（原 5）：全速冲到边上更容易碰到限位开关停轴，
# 频繁碰到就用 --motion-margin-mm 加回来。
DEFAULT_MOTION_MARGIN_MM = 2.0


# ============================================================================
# 行程与命令包络
# ============================================================================
#   ① 轴的真实机械行程（现场量的；TIA 工艺对象 XY轴 → 几何结构 →「3D 显示
#      中的表示」。注意运动学里 x=X 轴、z=Y 轴，见 _write_payload 注释）
#          X -192 .. 224        Y -129 .. 187
#      行程两端是硬件限位开关（2026-10-20 起代码不再另设软限位）。
#   ② 代码命令包络 = TRAVEL_* 再内缩 MotionLimits.margin_mm（默认 2mm）
#          X -190 .. 222        Y -127 .. 185
#   速度 / 加减速完全由 TIA「运动机构运动」的设置决定，代码只下发终点。


@dataclass(frozen=True)
class MotionLimits:
    """命令余量：决定命令包络。由 --motion-margin-mm 构造，经 PLCLink 交给 PLCInterface。

    margin_mm：轴碰到限位开关时把它调大即可。这是"少走一点"和"不碰限位开关"
    之间的取舍：调大更安全、可覆盖范围更小。
    """

    margin_mm: float = DEFAULT_MOTION_MARGIN_MM

    def __post_init__(self):
        object.__setattr__(self, "margin_mm", float(self.margin_mm))
        if self.margin_mm < 0.0:
            raise ValueError("motion margin must be >= 0")

    # 命令夹取的上/下限（= TRAVEL_* 再内缩 margin_mm）
    @property
    def command_x_min(self):
        return TRAVEL_X_MIN + self.margin_mm

    @property
    def command_x_max(self):
        return TRAVEL_X_MAX - self.margin_mm

    @property
    def command_y_min(self):
        return TRAVEL_Y_MIN + self.margin_mm

    @property
    def command_y_max(self):
        return TRAVEL_Y_MAX - self.margin_mm

    def clamp_command(self, px, py):
        """把机器坐标夹进命令包络。"""
        return (max(self.command_x_min, min(self.command_x_max, px)),
                max(self.command_y_min, min(self.command_y_max, py)))


DEFAULT_LIMITS = MotionLimits()


class ClampStats:
    """现场诊断：有多少次目标被夹取（= 想去的点在轴能到的范围之外）。

    这个比例高说明"游戏场地比机器能覆盖的大"这件事在频繁发生，
    也说明机械臂经常跑到极限位置。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._counts = {"x": 0, "y": 0, "both": 0, "total": 0}

    def record(self, hit_x, hit_y):
        self._counts["total"] += 1
        if hit_x:
            self._counts["x"] += 1
        if hit_y:
            self._counts["y"] += 1
        if hit_x and hit_y:
            self._counts["both"] += 1

    def snapshot(self):
        return dict(self._counts)


def validated_zone(zone):
    left, right, top, bottom = map(float, zone)
    if not all(map(math.isfinite, (left, right, top, bottom))):
        raise ValueError("PLC zone must have finite bounds")
    if right <= left or bottom <= top:
        raise ValueError("PLC zone must have positive width and height")
    return left, right, top, bottom


def finite_pair(x, y):
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("PLC coordinates must be finite")
    return x, y


def game_to_physical_raw(zone, x, y):
    """不做夹取的换算 —— 只给"轴跑到哪了"这类检测用。

    `game_to_physical` 是给**下发**用的，会把结果夹进命令包络；拿它来判断
    "轴是不是跑出去了"永远得到"没跑出去"，等于没检测。
    """
    x, y = finite_pair(x, y)
    left, right, top, bottom = zone
    px = TABLE_X_MIN + (x - left) * (TABLE_X_MAX - TABLE_X_MIN) / (right - left)
    py = TABLE_Y_MIN + (y - top) * (TABLE_Y_MAX - TABLE_Y_MIN) / (bottom - top)
    return px, py


def game_to_physical(zone, limits, x, y, clamp_stats=None):
    """游戏场地坐标 -> 机器坐标。

    映射用 TABLE_*（球台真实边缘在轴坐标里的位置），
    夹取用 limits.command_*（= TRAVEL_* 机械行程再内缩 limits.margin_mm）。

    球台比轴的行程大，不夹的话靠近中线/侧墙的目标会把轴开进限位开关。
    """
    x, y = validate_game_point(zone, x, y)
    left, right, top, bottom = zone
    px = TABLE_X_MIN + (x - left) * (TABLE_X_MAX - TABLE_X_MIN) / (right - left)
    py = TABLE_Y_MIN + (y - top) * (TABLE_Y_MAX - TABLE_Y_MIN) / (bottom - top)
    cx, cy = limits.clamp_command(px, py)
    if clamp_stats is not None:
        clamp_stats.record(cx != px, cy != py)
    return cx, cy


def validate_game_point(zone, x, y):
    x, y = finite_pair(x, y)
    left, right, top, bottom = zone
    if (x < left - COORD_EPSILON or x > right + COORD_EPSILON
            or y < top - COORD_EPSILON or y > bottom + COORD_EPSILON):
        raise ValueError(f"PLC game coordinates ({x}, {y}) are outside zone {zone}")
    return max(left, min(right, x)), max(top, min(bottom, y))


def physical_to_game(zone, x, y):
    """机器坐标 -> 游戏场地坐标（game_to_physical 的逆，用同一套 TABLE_*）。"""
    x, y = finite_pair(x, y)
    left, right, top, bottom = zone
    return (left + (x - TABLE_X_MIN) * (right - left) / (TABLE_X_MAX - TABLE_X_MIN),
            top + (y - TABLE_Y_MIN) * (bottom - top) / (TABLE_Y_MAX - TABLE_Y_MIN))
