"""S7-1500 TO_Kinematics DB1/DB18 protocol and non-blocking game link.

Only the PLCLink worker owns the transport. Coordinate mapping is per-link; feedback
is in the same game-coordinate zone as targets, independent of display scaling.
"""

from __future__ import annotations

import math
import ctypes
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

from air_hockey import core_config
from air_hockey.control.latency import LatencyProbe

try:
    import snap7
    try:
        from snap7.type import Areas, WordLen
    except ModuleNotFoundError as exc:
        if exc.name != "snap7.type":
            raise
        from snap7.types import Areas, WordLen
    SNAP7_ERROR = ""
except ImportError as exc:
    snap7 = None
    Areas = None
    SNAP7_ERROR = f"PLC 无法加载 python-snap7：{exc}"

HAS_SNAP7 = snap7 is not None

DB_MOTION = 1
DB_COMM = 18
MOTION_CMD_BYTE = 0
HOMING_BYTE = 1
# ★ 单轴操作：DB1 里 X/Y 的位本来就是分开的（PLC 侧早就支持），只是代码一直两个一起写。
#   有了单轴使能，就能在某个轴报 801 / 撞限位之后只把它撤下来，另一个继续用。
#
# [!] 这几个映射是**从原来"两个一起写"的代码推出来的**：
#      enable_axes  → DB1.0 bit6(True) + bit7(True)
#      disable_axes → DB1.0 bit6(False) + bit7(False)
#      home_axes    → DB1.1 bit0(True) + bit1(True)
#      axes_reset   → DB1.0 bit4(True) + bit5(True)
#   顺序都是 X 然后 Y。**上机前请在 FC1 里核对一次**：按单轴按钮时
#   只有对应那个轴动作，就说明映射对；若按 X 动了 Y，把这两个字典对调即可。
AXIS_ENABLE_BITS = {"x": 6, "y": 7}
AXIS_RESET_BITS = {"x": 4, "y": 5}
AXIS_HOME_BITS = {"x": 0, "y": 1}
AXIS_NAMES = ("x", "y")
MOTION_STATUS_BYTE = 1
MOTION_BLOCK_START = 20
MOTION_BLOCK_SIZE = 36
TRIGGER_BYTE = 56
TRIGGER_LINEAR_BIT = 0
MOTION_BUSY_BYTE = 18
GROUP_CTRL_BYTE = 140
GROUP_RESET_BIT = 3
GROUP_STOP_BIT = 2
GROUP_COMM_LOST_LATCH_BIT = 4
STATUS_READ_START = 0
# DB1 读取长度：原来 141（0~140）。序号握手加的 上次序号 在 146，
# 所以要读到 149 才拿得到 —— 150 字节正好覆盖 0~149。
# [!] 不能超过 DB1 实际长度（现场 DB1 最后一个变量是 气缸启动@150.0，共 151 字节）。
STATUS_READ_SIZE = 150
COMM_READ_START = 0
# DB18 随现场变量表增长：58（原始）→ 62 → 66（加 Reaction）→ 92 → 96（RemainingDistance
# 挪到 70、infoData 挪到 86）。从新到旧逐个尝试，读到哪个能用就用哪个 ——
# 现场改 PLC 变量表时不该把整个连接判死（那会表现成"PLC 未连接"，很难查）。
# [!] 读**超过** DB 实际长度会失败，所以列表里必须含实际长度（或其以下的值）。
COMM_READ_SIZES = (96, 92, 66, 62, 58)
COMM_READ_SIZE_LEGACY = COMM_READ_SIZES[-1]
COMM_KIN_ERROR_OFFSET = 56
# ★ 轴自己的故障码 —— PLC 早就导出来了，代码一直没读：
#   DB18.2  Word  X轴ERRORID
#   DB18.6  Word  Y轴ERRORID
# 轴撞软限位 / 跟随误差时，这里就是 TIA 里看到的那个具体代码。
# 按**无符号**读，显示成 16#xxxx（和 TIA 一致）。
COMM_X_ERRID_OFFSET = 2
COMM_Y_ERRID_OFFSET = 6
# 接到 MC_MoveLinearAbsolute 上的 Error / ErrorId
COMM_MC_ERR_OFFSET = 58
COMM_MC_ERR_BIT = 3
COMM_MC_ERRID_OFFSET = 60
# ★ TO_Kinematics.ErrorDetail.Reaction（DInt，有符号）—— 6 字节起。
# Number 告诉你"报了什么"，Reaction 告诉你"TO 因此做了什么"，
# 后者才是判断"这是原因还是结果"的关键。
COMM_KIN_REACTION_OFFSET = 62

# ============================================================================
# 序号握手触发（替代脉冲触发）
# ============================================================================
# 脉冲触发的致命约束：脉宽必须跨过 PLC 扫描周期。实测 OB1 = 2~3ms（偶发 5ms），
# 所以高低电平各要 10ms 才安全 —— 加上 S7 往返，周期被锁在约 28ms（35Hz）。
# 而"平滑"需要 ≥100Hz（否则快球的步距超过混合阈值，每周期都中止重开）。
#
# 序号握手：Python 每下发一条就把序号 +1，PLC 比较"指令序号"和"上次序号"，
# 不同就执行并记下 —— **完全不需要脉宽**。周期只剩 4 次 S7 往返 ≈ 6.6ms → ~150Hz。
#
# 顺序很重要：**先写目标块，再写序号** —— PLC 看到新序号时目标已经稳定。
#
# PLC 侧（OB1 里 MC_MoveLinearAbsolute 之前）：
#     IF "DB1".指令序号 <> "DB1".上次序号 THEN
#         "DB1".上次序号 := "DB1".指令序号;
#         #序号触发 := TRUE;
#     ELSE
#         #序号触发 := FALSE;
#     END_IF;
#     // Execute 输入改成:  #序号触发 OR "DB1".XY轴直线绝对启动
# 这样两种模式都能跑，切换不用改 PLC。
#
# [!] 偏移必须和 TIA 里【实际排布】一致，不是我们想放哪就放哪。
# 现场 DB1 里 TIA 把这两个变量追加到了末尾：
#     指令序号   DInt   142.0
#     上次序号   DInt   146.0
# （气缸启动 150.0 是 DB1 的最后一个变量 → DB1 共 151 字节）
COMM_SEQ_OFFSET = 142         # DInt：Python 写，指令序号
COMM_SEQ_ACK_OFFSET = 146     # DInt：PLC 写，已处理的序号（读回来就是"落后几条"）

# ============================================================================
# 补齐诊断用的 DB18 字段（全部用边界检查读，PLC 没加就显示"未读到"）
# ============================================================================
# [!] 读错偏移比"没加"更糟：边界检查能挡住"没加"（显示"未读到"），
#    但读错偏移会**静默返回垃圾值**。所以每个数字都必须对着 TIA 抄。
#
# 2026-10-13 现场实际布局（TIA 变量表，DB18 共 96 字节）：
#   56.0  XY轴ErrorDetail.Number     UInt     ★ 运动学错误明细（就是那个 801 = 16#0321）
#   58.3  error                      Bool     MC 块的 Error
#   60.0  errorid                    Word     MC 块的 ErrorId
#   62.0  XY轴ErrorDetail.Reaction   DInt     ★ TO 对该错误的反应
#   66.0  XY轴.WarningWord           DWord    ★ 预警字（软限位预警在这里）
#   70.0  XY轴RemainingDistance      LReal    ★ 当前运动还差多远（接的是 MC.RemainingDistance）
#   78.0  mcRemaining                LReal    ← 旧字段，现在没人写它（保留不动，代码不读）
#   86.0  infoData                   LTime    ★ OB1 循环时间（RT_INFO，单位**纳秒**）
#   94.0  ret_Val                    Int      ← RT_INFO 的状态码，不是时间
#
# ★ 注意 DB18 里**没有** CommandAborted 字段了（曾经想加在 70.0，后来被 RemainingDistance 占了）。
#   所以代码不再读它 —— 否则会把 RemainingDistance 的字节当成"中止标志"，永远显示 True。
#   MC 块上的 CommandAborted 输出目前也还是写死的 false，没有数据可用。
COMM_WARNING_WORD_OFFSET = 66
COMM_MC_REMAINING_OFFSET = 70      # XY轴RemainingDistance（不是那个旧的 mcRemaining@78）
COMM_OB1_TIME_OFFSET = 86          # LTime，纳秒
COMM_RTINFO_STATUS_OFFSET = 94     # Int，RT_INFO 状态码
COMM_BYTE = 0
COMM_HEARTBEAT_BIT = 0
COMM_PLC_HEARTBEAT_BIT = 1
COMM_LINK_LOST_BIT = 2
COMM_AXES_READY_BIT = 3
COMM_X_ERR_BIT = 4
COMM_Y_ERR_OFFSET = 4
COMM_Y_ERR_BIT = 0
ECHO_TIMEOUT = 1.0
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
# 现在只把命令夹到机械行程内缩 MOTION_MARGIN_MM，避免主动去碰限位开关。
TRAVEL_X_MIN, TRAVEL_X_MAX = -192.0, 224.0
TRAVEL_Y_MIN, TRAVEL_Y_MAX = -129.0, 187.0
TABLE_X_MIN, TABLE_X_MAX = -224.5, 255.5
TABLE_Y_MIN, TABLE_Y_MAX = -198.5, 241.5

# 命令余量：下发目标离机械行程端点（限位开关）再往内收这么多毫米，
# 正常运动和超调都不主动碰限位开关。运行期可用 --motion-margin-mm 覆盖。
MOTION_MARGIN_MM = 5.0

# 命令夹取的上/下限（= TRAVEL_* 再内缩 MOTION_MARGIN_MM）
COMMAND_X_MIN, COMMAND_X_MAX = TRAVEL_X_MIN + MOTION_MARGIN_MM, TRAVEL_X_MAX - MOTION_MARGIN_MM
COMMAND_Y_MIN, COMMAND_Y_MAX = TRAVEL_Y_MIN + MOTION_MARGIN_MM, TRAVEL_Y_MAX - MOTION_MARGIN_MM


def set_motion_margin(margin_mm: float) -> None:
    """运行期改命令余量（--motion-margin-mm）。

    轴碰到限位开关时把它调大即可，不用改代码重跑。
    这是"少走一点"和"不碰限位开关"之间的取舍：调大更安全、可覆盖范围更小。
    """
    global MOTION_MARGIN_MM, COMMAND_X_MIN, COMMAND_X_MAX, COMMAND_Y_MIN, COMMAND_Y_MAX
    margin_mm = float(margin_mm)
    if margin_mm < 0.0:
        raise ValueError("motion margin must be >= 0")
    MOTION_MARGIN_MM = margin_mm
    COMMAND_X_MIN = TRAVEL_X_MIN + margin_mm
    COMMAND_X_MAX = TRAVEL_X_MAX - margin_mm
    COMMAND_Y_MIN = TRAVEL_Y_MIN + margin_mm
    COMMAND_Y_MAX = TRAVEL_Y_MAX - margin_mm


# ============================================================================
# 安全自检
# ============================================================================
# 一张表说清所有限值 —— 现场调参只看这一段。
#
#   ① 轴的真实机械行程（现场量的；TIA 工艺对象 XY轴 → 几何结构 →「3D 显示
#      中的表示」。注意运动学里 x=X 轴、z=Y 轴，见 _write_payload 注释）
#          X -192 .. 224        Y -129 .. 187
#      行程两端是硬件限位开关（2026-10-20 起代码不再另设软限位）。
#   ② 代码命令包络 = TRAVEL_* 再内缩 MOTION_MARGIN_MM（默认 5mm）
#          X -187 .. 219        Y -124 .. 182
#   ③ 超调距离 = v² / (2a)，v = target_slew_mm × 实际周期（不是 --plc-rate），
#      a = TIA 减速度。目标斜坡本身会提前减速、停在包络内（_braking_step）；
#      冲过包络的只可能是轴跟不上、按自身减速度刹车的那一段，所以用 TIA 减速度估算。
#
# ★★ 铁律：② 到 ① 之间的余量必须 > ③，否则全速冲向边缘时会碰到限位开关。
#
# TIA「工艺对象 XY轴 → 动力学 → 预设值和限值」里的**预设值**（不是右边的最大值）。
# 现场实测（2026-10，运动机构运动）：
#     速度    3750 mm/s      （最大值也是 3750，已经顶到）
#     加速度  35000 mm/s^2   ← 决定起步 / 跟不跟得上，**跟随误差看它**
#     减速度  35000 mm/s^2   ← ★★★ 决定**制动距离**，**撞不撞限位看它**
#     加加速度 300000 mm/s^3
#
# ★★ 两者不是一回事，混了就会算错：
#   · --target-accel-mm-s2（软件限幅器）要**低于加速度**，否则等于没限 → 跟随误差
#   · 制动距离必须用**减速度**算 —— 用加速度算会**低估一半以上**
#     （35000 算成 112mm，15000 才是 261mm），那正是"以为算过了、结果还是撞"
#
# ⚠️ 这三个值必须和 TIA 里一致。改了 TIA 就用 --tia-{accel,decel,speed}-* 同步，
#    否则自检要么白丢速度、要么说安全却会撞（后者才要命）。
AXIS_ACCEL_LIMIT_MM_S2 = 35000.0   # 加速度：给软件限幅器做上限参考
AXIS_DECEL_LIMIT_MM_S2 = 35000.0   # ★ 减速度：制动距离 = v^2 / (2 x 这个)
AXIS_SPEED_LIMIT_MM_S = 3750.0


def set_axis_dynamics(accel_mm_s2=None, decel_mm_s2=None, speed_mm_s=None) -> None:
    """运行期同步 TIA 的动力学预设值，供 --tia-accel-mm-s2 / --tia-decel-mm-s2 /
    --tia-speed-mm-s 使用。

    **改了 TIA 就必须同步这里**，否则安全自检拿旧数字算：
      · 减速度填小了 → 明明安全的 slew 被误报"会撞"，白丢一半速度
      · 减速度填大了 → 明明会撞却说安全 —— 那才是要命的
    """
    global AXIS_ACCEL_LIMIT_MM_S2, AXIS_DECEL_LIMIT_MM_S2, AXIS_SPEED_LIMIT_MM_S
    if accel_mm_s2 is not None:
        AXIS_ACCEL_LIMIT_MM_S2 = float(accel_mm_s2)
    if decel_mm_s2 is not None:
        AXIS_DECEL_LIMIT_MM_S2 = float(decel_mm_s2)
    if speed_mm_s is not None:
        AXIS_SPEED_LIMIT_MM_S = float(speed_mm_s)

# TransitionParameter[1]（精磨距离 = 混合起点离目标多远）在 DB1 里的字节偏移。
# 现场在 DB1 建了 `TransitionParameter : Array[1..5] of LReal` @ 150.0，
# FC1 用指针 P#DB1.DBX150.0 传给 MC_MoveLinearAbsolute 的 TransitionParameter。
COMM_TRANSITION_OFFSET = 150

# 精磨距离 = 步距 × 这个比例。必须 **< 1**：等于 1 时 TO 会把整段都圆掉、
# 到不了目标点。0.75 是"最后四分之一才混合"的意思，实测既连续又走得准。
TRANSITION_RATIO = 0.75


def braking_distance_mm(slew_mm_per_cycle, rate_hz, accel_mm_s2=None):
    """从指令速度刹停需要多长距离（mm）。返回 (制动距离, 指令速度mm/s)。"""
    accel = AXIS_DECEL_LIMIT_MM_S2 if accel_mm_s2 is None else float(accel_mm_s2)
    if accel <= 0.0 or rate_hz <= 0.0:
        return 0.0, 0.0
    speed = float(slew_mm_per_cycle) * float(rate_hz)
    return speed * speed / (2.0 * accel), speed


def _braking_step(dist: float, max_delta: float) -> float:
    """按剩余距离算出不超调的最大离散步长。"""
    if dist <= 0.0 or max_delta <= 0.0:
        return 0.0
    if dist <= max_delta:
        return dist
    d = dist - max_delta / 8.0
    disc = max_delta * max_delta + 8.0 * max_delta * d
    return max(0.0, (math.sqrt(disc) - max_delta) / 2.0)

def safety_warnings(slew_mm_per_cycle, rate_hz, accel_mm_s2=None):
    """按当前参数算出安全警告列表（空列表 = 安全）。

    这是把"余量够不够刹得住"从人脑里搬到程序里 —— 现场踩过的撞死挡，
    根源就是没人会在跑之前先算这一步。
    """
    warnings = []
    brake, speed = braking_distance_mm(slew_mm_per_cycle, rate_hz, accel_mm_s2)
    margin = min(COMMAND_X_MIN - TRAVEL_X_MIN, TRAVEL_X_MAX - COMMAND_X_MAX,
                 COMMAND_Y_MIN - TRAVEL_Y_MIN, TRAVEL_Y_MAX - COMMAND_Y_MAX)
    if margin <= 0.0:
        warnings.append(
            f"命令包络没有余量（MOTION_MARGIN_MM={MOTION_MARGIN_MM:.0f}）—— "
            f"指令会直接落在机械行程端点（限位开关）上")
    if speed > AXIS_SPEED_LIMIT_MM_S:
        warnings.append(
            f"指令速度 {speed:.0f}mm/s 超过 TIA 预设最大速度 "
            f"{AXIS_SPEED_LIMIT_MM_S:.0f}mm/s —— TO 会截断，轴跟不上 → 跟随误差")
    if brake > margin:
        # 要让制动距离塞进余量，slew 必须 ≤ sqrt(2·a·margin) / 周期
        accel = AXIS_DECEL_LIMIT_MM_S2 if accel_mm_s2 is None else float(accel_mm_s2)
        safe_slew = (2.0 * accel * margin) ** 0.5 / max(rate_hz, 1e-6)
        warnings.append(
            f"制动距离 {brake:.0f}mm > 包络到限位开关的余量 {margin:.0f}mm"
            f"（缺 {brake - margin:.0f}mm）—— 全速冲向边缘时可能碰到限位开关。"
            f"三选一：① --target-slew-mm 降到 {safe_slew:.0f} 以下；"
            f"② --motion-margin-mm 加到 {brake + 5:.0f}（会明显缩小可覆盖范围）；"
            f"③ 在 TIA 里把加速度从 {accel:.0f} 提高（现按此值估算）")
    return warnings

# 现场诊断：有多少次目标被夹取（= 想去的点在轴能到的范围之外）。
# 这个比例高说明"游戏场地比机器能覆盖的大"这件事在频繁发生，
# 也说明机械臂经常跑到极限位置。
_CLAMP_COUNTS = {"x": 0, "y": 0, "both": 0, "total": 0}


def reset_clamp_counts():
    _CLAMP_COUNTS.update({"x": 0, "y": 0, "both": 0, "total": 0})


def clamp_counts():
    return dict(_CLAMP_COUNTS)
COORD_EPSILON = 1e-6
DEAD_ZONE_MM = 3.0
# Measured on the machine: 50mm with BufferMode=5 stalled streaming motion.
#
# 现场实测的含义：步距 ≤ 这个值时用 BufferMode=5（混合，平滑）；更大就用 0
# （中止重开，会一顿一顿）。50mm 用混合会"卡住"，所以当年把阈值定在 10mm。
#
# ★ 但"卡住"的原因是 TO 的「作业序列 → 最大舍入间距」被设成了 100%：
#   混合半径可以吃掉整个运动，轴一直在拐弯、走不出直线。
#   把舍入间距调小（例如 30%）之后，混合应当可以在更大的步距下正常工作，
#   那时这个阈值就可以放宽 —— 快球才可能真正平滑。
#
# 所以做成可调（--blend-threshold-mm），配合 PLC 侧舍入间距一起实验。
BUFFER_MODE_IMMEDIATE_MM = 10.0
DEFAULT_PERIOD = 0.03
RECONNECT_INTERVAL = 2.0
TRIGGER_LOW_TIME = 0.010
TARGET_MAX_AGE = 0.25
DEFAULT_ZONE = (
    core_config.RINK_LEFT, core_config.RINK_RIGHT,
    core_config.RINK_TOP, core_config.RINK_CENTER_Y,
)


def _validated_zone(zone):
    left, right, top, bottom = map(float, zone)
    if not all(map(math.isfinite, (left, right, top, bottom))):
        raise ValueError("PLC zone must have finite bounds")
    if right <= left or bottom <= top:
        raise ValueError("PLC zone must have positive width and height")
    return left, right, top, bottom


def _finite_pair(x, y):
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("PLC coordinates must be finite")
    return x, y


def _game_to_physical_raw(zone, x, y):
    """不做夹取的换算 —— 只给"轴跑到哪了"这类检测用。

    `_game_to_physical` 是给**下发**用的，会把结果夹进命令包络；拿它来判断
    "轴是不是跑出去了"永远得到"没跑出去"，等于没检测。
    """
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    px = TABLE_X_MIN + (x - left) * (TABLE_X_MAX - TABLE_X_MIN) / (right - left)
    py = TABLE_Y_MIN + (y - top) * (TABLE_Y_MAX - TABLE_Y_MIN) / (bottom - top)
    return px, py


def _game_to_physical(zone, x, y):
    """游戏场地坐标 -> 机器坐标。

    映射用 TABLE_*（球台真实边缘在轴坐标里的位置），
    夹取用 COMMAND_*（= TRAVEL_* 机械行程再内缩 MOTION_MARGIN_MM）。

    球台比轴的行程大，不夹的话靠近中线/侧墙的目标会把轴开进限位开关。
    """
    x, y = _validate_game_point(zone, x, y)
    left, right, top, bottom = zone
    px = TABLE_X_MIN + (x - left) * (TABLE_X_MAX - TABLE_X_MIN) / (right - left)
    py = TABLE_Y_MIN + (y - top) * (TABLE_Y_MAX - TABLE_Y_MIN) / (bottom - top)
    cx = max(COMMAND_X_MIN, min(COMMAND_X_MAX, px))
    cy = max(COMMAND_Y_MIN, min(COMMAND_Y_MAX, py))
    _CLAMP_COUNTS["total"] += 1
    hit_x, hit_y = cx != px, cy != py
    if hit_x:
        _CLAMP_COUNTS["x"] += 1
    if hit_y:
        _CLAMP_COUNTS["y"] += 1
    if hit_x and hit_y:
        _CLAMP_COUNTS["both"] += 1
    return cx, cy


def _validate_game_point(zone, x, y):
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    if (x < left - COORD_EPSILON or x > right + COORD_EPSILON
            or y < top - COORD_EPSILON or y > bottom + COORD_EPSILON):
        raise ValueError(f"PLC game coordinates ({x}, {y}) are outside zone {zone}")
    return max(left, min(right, x)), max(top, min(bottom, y))


def _physical_to_game(zone, x, y):
    """机器坐标 -> 游戏场地坐标（_game_to_physical 的逆，用同一套 TABLE_*）。"""
    x, y = _finite_pair(x, y)
    left, right, top, bottom = zone
    return (left + (x - TABLE_X_MIN) * (right - left) / (TABLE_X_MAX - TABLE_X_MIN),
            top + (y - TABLE_Y_MIN) * (bottom - top) / (TABLE_Y_MAX - TABLE_Y_MIN))


@dataclass(frozen=True)
class PLCFeedback:
    """Immutable worker snapshot; position in game units, velocity in units/s."""

    stamp: float = 0.0
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    x_en: bool = False
    y_en: bool = False
    x_home: bool = False
    y_home: bool = False
    x_err: bool = False
    y_err: bool = False
    # ★ 轴自己的故障码（DB18.2 / DB18.6）。撞软限位、跟随误差等都会在这里给出
    #   TIA 里看到的那个具体代码；None = 该字段读不到（DB18 还是旧长度）。
    x_errid: Optional[int] = None
    y_errid: Optional[int] = None
    # MC_MoveLinearAbsolute 的 Error / ErrorId（现场接到 DB18.58.3 / DB18.60）
    mc_error: Optional[bool] = None
    mc_errid: Optional[int] = None
    busy: bool = False
    comm_lost: bool = False
    axes_ready: bool = False
    plc_echo_ok: bool = True
    group_stop: bool = False
    comm_lost_latch: bool = False
    kinematics_error: int = 0
    # TO_Kinematics.ErrorDetail.Reaction（DInt）：TO 对该错误的**反应**。
    # Number 说"报了什么"，Reaction 说"因此做了什么" —— 后者决定
    # 这到底是一个原因，还是别的原因（比如使能被撤）导致的结果。
    kin_reaction: Optional[int] = None
    # 序号握手：PLC 已处理到的序号（DB1.64）。和本地下发序号一比就是"落后几条"。
    # None = 没读到（PLC 还没加这个字段）。
    seq_ack: Optional[int] = None
    # ---- 诊断补充（PLC 加了多少就显示多少；None = 未读到）----
    warning_word: Optional[int] = None     # ★ 预警字（软限位预警在这里）
    mc_remaining: Optional[float] = None   # ★ 当前运动还差多远（MC.RemainingDistance）
    ob1_ns: Optional[int] = None           # ★ OB1 循环时间（纳秒，来自 RT_INFO）
    rtinfo_status: Optional[int] = None    # RT_INFO 的状态码（0 = 正常）

    @property
    def ob1_ms(self):
        """OB1 循环时间换算成毫秒（RT_INFO 给的是纳秒）。"""
        return None if self.ob1_ns is None else self.ob1_ns / 1e6

    @property
    def axis_fault(self):
        return self.x_err or self.y_err

    def fault_detail(self):
        """把故障相关的码拼成一行，供状态显示和报警快照使用。

        全部按无符号 + 16#xxxx 显示 —— 和 TIA 里的写法一致，
        这样你拿这个码直接去查 TIA 在线帮助或手册就能对上。
        """
        def _hex(value):
            if value is None:
                return "未读到"
            return f"16#{value:04X}({value})"
        return (f"轴码 X={_hex(self.x_errid)} Y={_hex(self.y_errid)} | "
                f"运动指令error={self.mc_error} errorid={_hex(self.mc_errid)} | "
                f"运动学Number={_hex(self.kinematics_error)} "
                f"Reaction={'未读到' if self.kin_reaction is None else self.kin_reaction}")


class PLCInterface:
    """Blocking S7 transport, exclusively operated by PLCLink's worker."""

    def __init__(self, ip="192.168.0.64", rack=0, slot=1, zone=None,
                 blend_threshold_mm=BUFFER_MODE_IMMEDIATE_MM, target_slew_mm=0.0,
                 target_accel_mm_s2=0.0, period=0.0):
        self.ip, self.rack, self.slot = ip, rack, slot
        # 步距超过它就用 BufferMode=1（附加运动：先走完当前再走新的）而不是 5（混合）。
        # 现场实测 50mm + 混合会"卡住"（TO 舍入间距 100%），所以默认 10mm。
        # 调小 TO 的舍入间距后可以放宽这个值让快球也平滑。
        self.blend_threshold_mm = float(blend_threshold_mm)
        # 目标限速（mm/周期）。0 = 关闭。
        # ★ 建议 = blend_threshold_mm：这样步距永远不超阈值 → 永不中止 → 不顿。
        self.target_slew_mm = float(target_slew_mm)
        # ★★★ 目标加速度限幅（mm/s^2），0 = 关闭；period 是这个限幅换算用的周期。
        # 见 _prepare_move 里的说明 —— TIA 状态位显示 Y 轴报的是"跟随误差"，
        # 光限步长治不了（目标突然反向需要的加速度是轴能力的约 5 倍）。
        self.target_accel_mm_s2 = float(target_accel_mm_s2)
        self.period = float(period)
        # 上一个周期真正发出去的步长（mm），加速度限幅要拿它比。
        self._last_step = None
        # 上一次真正发送的时刻 —— 加速度限幅要用**实测** dt，
        # 不能用 --plc-rate 的名义周期（名义 200Hz、实际常只有 70Hz）。
        self._last_send_at = None
        # ★ 轴的实际位置（游戏坐标），由 PLCLink 每个周期从反馈里更新。
        #   _prepare_move 的死区判据要用它 —— 只看"上次发了什么"会假死，见那里的注释。
        self.actual_game = None
        self.actual_busy = False
        self.zone = _validated_zone(DEFAULT_ZONE if zone is None else zone)
        self._client = None
        self._connected = False
        self.last_heartbeat_time = 0.0
        self.heartbeat_state = False
        self._last_phys_x = None
        self._last_phys_y = None
        self.last_send_state = "unknown"
        self.last_buffer_mode = None
        # 必须初始化：它原本只在 send_linear_move 里、且从**第二次**下发才开始赋值
        # （第一次下发时 _last_phys_x 还是 None，那个分支不成立）。
        # 读它的诊断代码会在第一次下发时抛 AttributeError。
        self.last_send_dist_mm = 0.0
        # 精磨距离（TransitionParameter[1]）随步距自适应写回 DB1.DBD150。
        # transition_enabled 会在写入失败（DB 里没这个变量）时自动关掉，
        # 之后退回由 TO「最大舍入间距」决定混合距离 —— 不会影响运动本身。
        self.transition_enabled = True
        self._last_transition = None
        self.last_transition_mm = None
        self.last_error = ""
        self._multi_read_failed = False
        # DB18 当前用哪种长度读（66 / 62 / 58），见 COMM_READ_SIZES
        self._comm_size = None
        self.round_trips = 0

    @property
    def connected(self):
        return self._connected

    def connect(self, quiet=False):
        self.last_error = ""
        if not HAS_SNAP7:
            self.last_error = SNAP7_ERROR
            if not quiet:
                print(f"[PLC] {SNAP7_ERROR}")
            return False
        try:
            self._client = snap7.client.Client()
            self._client.connect(self.ip, self.rack, self.slot)
            self._connected = True
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, False)
            self.clear_trigger()
            self.reset_tracking()
            self.last_heartbeat_time = 0.0
            self.heartbeat_state = False
            return True
        except Exception as exc:
            self.last_error = str(exc)
            if not quiet:
                print(f"[PLC] 连接失败: {exc}")
            self.disconnect()
            return False

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self.reset_tracking()

    def reset_tracking(self):
        self._last_phys_x = None
        self._last_phys_y = None
        self._last_step = None
        self._last_send_at = None
        return True

    def _write_bool(self, db, offset, bit, value):
        self.round_trips += 1
        try:
            library = getattr(self._client, "_lib", None) or getattr(self._client, "_library", None)
            if library is None:
                request = bytearray(self._client.protocol.build_write_request(
                    Areas.DB, db, offset * 8 + bit, WordLen.Bit, bytes([bool(value)])))
                # ponytail: 3.0 错把 1 位编码成 8 位；升级协议接口时重验封包与应答。
                data_start = 10 + struct.unpack_from(">H", request, 6)[0]
                struct.pack_into(">H", request, data_start + 2, 1)
                response = self._client._send_receive(request)
                self._client.protocol.check_write_response(response)
            else:
                pointer = getattr(self._client, "_s7_client", None) or getattr(self._client, "_pointer", None)
                if pointer is None:
                    raise RuntimeError("Snap7 native client handle is unavailable")
                data = ctypes.c_uint8(bool(value))
                result = library.Cli_WriteArea(
                    pointer, Areas.DB.value, db, offset * 8 + bit,
                    1, WordLen.Bit.value, ctypes.byref(data))
                if result != 0:
                    raise RuntimeError(self._client.error_text(result))
        except Exception:
            self._connected = False
            raise

    def _read_db_blocks(self, specs):
        if Areas is not None and not self._multi_read_failed:
            try:
                items = [{"area": Areas.DB, "db_number": db, "start": start, "size": size}
                         for db, start, size in specs]
                self.round_trips += 1
                result, data = self._client.read_multi_vars(items)
                if result == 0 and data is not None and len(data) == len(specs):
                    blocks = [bytearray(block) for block in data]
                    if all(len(block) == spec[2] for block, spec in zip(blocks, specs)):
                        return blocks
                raise RuntimeError(f"read_multi_vars result={result}")
            except Exception as exc:
                self._multi_read_failed = True
                print(f"[PLC] 多变量读取不可用，回退逐块读取: {exc}")
        self.round_trips += len(specs)
        return [self._client.db_read(db, start, size) for db, start, size in specs]

    def _read_status_blocks(self):
        """读 DB1 + DB18，兼容 DB18 的三种长度（58 / 62 / 66）。

        从新的往旧的试，读到哪个能用就记住哪个 —— 现场加减 DB18 字段时
        不该把整个连接判死（那会表现成"PLC 未连接"，很难查）。
        """
        sizes = list(COMM_READ_SIZES)
        if self._comm_size in sizes:                 # 先用上次成功的长度
            sizes.remove(self._comm_size)
            sizes.insert(0, self._comm_size)
        for size in sizes:
            try:
                db1, db18 = self._read_db_blocks(
                    ((DB_MOTION, STATUS_READ_START, STATUS_READ_SIZE),
                     (DB_COMM, COMM_READ_START, size)))
            except Exception:
                continue
            self._comm_size = size
            return db1, db18
        return None, None

    def clear_trigger(self):
        self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, False)

    def set_axis_enable(self, axis, enabled):
        """单轴使能 / 撤销。axis 为 "x" 或 "y"。

        两个轴必须**都**使能程序才允许下发运动（见 PLCLink._motion_safe），
        所以单轴使能只用于现场单独调试或单独撤下一个报故障的轴。
        """
        if not self.connected:
            return False
        bit = AXIS_ENABLE_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, bool(enabled))
            return True
        except Exception:
            self._connected = False
            return False

    def reset_axis(self, axis):
        """单轴复位（清该轴的报警）。"""
        if not self.connected:
            return False
        bit = AXIS_RESET_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, MOTION_CMD_BYTE, bit, False)
            return True
        except Exception:
            self._connected = False
            return False

    def home_axis(self, axis):
        """单轴回零。[!] 回零**会让该轴动**（去找参考点），是刻意的人工动作。"""
        if not self.connected:
            return False
        bit = AXIS_HOME_BITS.get(axis)
        if bit is None:
            return False
        try:
            self._write_bool(DB_MOTION, HOMING_BYTE, bit, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, HOMING_BYTE, bit, False)
            return True
        except Exception:
            self._connected = False
            return False

    def enable_axes(self):
        return all(self.set_axis_enable(axis, True) for axis in AXIS_NAMES)

    def disable_axes(self):
        # 两个都要写：用 all() 会在第一个失败时短路，漏掉另一个。
        return all([self.set_axis_enable(axis, False) for axis in AXIS_NAMES])

    def home_axes(self):
        return all(self.home_axis(axis) for axis in AXIS_NAMES)

    def axes_reset(self):
        # 组复位 + 两个单轴复位
        ok = all([self.reset_axis(axis) for axis in AXIS_NAMES])
        if not self.connected:
            return False
        try:
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, True)
            time.sleep(0.05)
            self._write_bool(DB_MOTION, GROUP_CTRL_BYTE, GROUP_RESET_BIT, False)
            return ok
        except Exception:
            self._connected = False
            return False

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def game_to_physical_raw(self, x, y):
        return _game_to_physical_raw(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    def _prepare_move(self, game_x, game_y):
        """坐标换算 + 目标限速 + 死区 + BufferMode。

        返回 None（不发）或 (物理x, 物理y, mode, 步距)。步距在**第一次**下发时是 None。

        ★ 目标限速（slew）治的是"顿"：rally 状态机切换时目标会**瞬间跳变**
        （现场实测 p95 步距 192.8mm —— Y 在防守线 140 和追击 286 之间来回跳），
        大步使用 BufferMode=1，完成当前运动并停稳后才执行下一条。
        把跳变削成"每周期最多走 target_slew_mm"，只要 slew ≤ blend_threshold，
        步距就永远不超阈值 → 永远走混合、永不中止 —— 运动从"跳"变成"斜坡"。
        """
        try:
            phys_x, phys_y = self.game_to_physical(game_x, game_y)
        except (ValueError, OverflowError):
            self.last_send_state = "invalid"
            return None
        # 执行中的同点不再排队；在限速前比较，避免挡住仍在推进的斜坡目标。
        if (self.actual_busy and self._last_phys_x is not None
                and max(abs(phys_x - self._last_phys_x),
                        abs(phys_y - self._last_phys_y)) < DEAD_ZONE_MM):
            self.last_send_state = "pending"
            return None
        # ★★★ 起点：优先用"上次真正发出去的位置"；还没有就用轴的**实际位置**打底 ——
        # 这样 arming 后的**第一条**指令也走限速斜坡。
        #
        # 现场（TIA 工艺对象 Y轴 → 状态位和错误位：错误 = 跟随误差）：
        # 按 E 使能时 _enable_on_worker 会 reset_tracking() → _last_phys=None，
        # 于是 arming 后的**第一条**指令**完全不限速**，直接跳到当前目标的
        # 命令包络角点（实测 game(533,349) → phys(200,170)，从轴所在的
        # (-117,-75) 算起约 320mm）。轴以最大加速度扑过去 → 跟随误差 →
        # Y 轴 16#8001 → 整机撤销使能。快照 prev_en=(True,True) prev_busy=True
        # 正是"刚使能、第一条指令正在跑"。前面 X 轴的 801 是同一个原因。
        #
        # [!] 只做**局部变量**，绝不写回 self._last_phys_* —— 那是"下发之前就改
        #    状态"，会让"写失败后必须清空跟踪"的保证失效（有测试守着）。
        #    self._last_phys_* 只在 _commit_move 里更新。
        origin_x, origin_y = self._last_phys_x, self._last_phys_y
        if origin_x is None:
            actual = getattr(self, "actual_game", None)
            if actual is not None:
                try:
                    origin_x, origin_y = self.game_to_physical_raw(actual[0], actual[1])
                except (ValueError, OverflowError):
                    origin_x = origin_y = None
        if origin_x is not None:
            goal_phys_x = phys_x
            goal_phys_y = phys_y
            dx = phys_x - origin_x
            dy = phys_y - origin_y
            span = max(abs(dx), abs(dy))
            now_mono = time.perf_counter()
            dt = (now_mono - self._last_send_at
                  if self._last_send_at is not None else self.period)
            if dt <= 0.0 or not math.isfinite(dt):
                dt = self.period
            max_delta = self.target_accel_mm_s2 * dt * dt if self.target_accel_mm_s2 > 0.0 else 0.0

            # 目标限速与提前减速：步长不超 target_slew_mm 且按剩余距离减速刹停
            v_limit = self.target_slew_mm if self.target_slew_mm > 0.0 else span
            if max_delta > 0.0:
                v_limit = min(v_limit, _braking_step(span, max_delta))

            if span > v_limit:
                ratio = v_limit / span
                phys_x = origin_x + dx * ratio
                phys_y = origin_y + dy * ratio

            dx_step = phys_x - origin_x
            dy_step = phys_y - origin_y
            distance = max(abs(dx_step), abs(dy_step))

            # 加速度限幅：限制相邻周期步长变化量
            if self.target_accel_mm_s2 > 0.0 and self._last_step is not None and max_delta > 0.0:
                ddx = dx_step - self._last_step[0]
                ddy = dy_step - self._last_step[1]
                accel_span = max(abs(ddx), abs(ddy))
                if accel_span > max_delta:
                    ratio = max_delta / accel_span
                    phys_x = origin_x + self._last_step[0] + ddx * ratio
                    phys_y = origin_y + self._last_step[1] + ddy * ratio

            # 包络兜底：加速度限幅可能让中间指令冲过目标，中间指令同样不得越出命令包络。
            # 轴本来就在包络外（行程内）时，允许从起点按斜坡逐步回来、不跳到包络边界，
            # 但绝不比起点更往外 —— 不会被命令往限位开关方向走。
            phys_x = max(min(COMMAND_X_MIN, origin_x), min(max(COMMAND_X_MAX, origin_x), phys_x))
            phys_y = max(min(COMMAND_Y_MIN, origin_y), min(max(COMMAND_Y_MAX, origin_y), phys_y))

            # ★ 注意：distance 已经在上面（加速度限幅之前）算好了，这里不要
            #   用限幅后的步长重算 —— 那正是把它锁死的写法。见上面的注释。
            # 同点到位则跳过；空闲但未到位时重发，执行中的同点已在限速前跳过。
            # 只有当总目标差距（span）本身已进入死区时，才进入死区到位跳过判定；
            # 只要总目标还在远方（span >= DEAD_ZONE_MM），哪怕这一拍步距只有 1~2mm，也必须连续下发！
            if span < DEAD_ZONE_MM:
                actual_gap = None
                actual = getattr(self, "actual_game", None)
                if actual is not None:
                    try:
                        ax, ay = self.game_to_physical_raw(actual[0], actual[1])
                        actual_gap = max(abs(goal_phys_x - ax), abs(goal_phys_y - ay))
                    except (ValueError, OverflowError):
                        actual_gap = None
                if actual_gap is None or actual_gap < DEAD_ZONE_MM:
                    self.last_send_state = "deadzone"
                    return None
                self.last_send_state = "deadzone-resend"
            # 步距 ≤ blend_threshold → 混合（平滑）；更大 → 附加运动（先走完、停下、再走）。
            #
            # ★★★ TO_Kinematics 的 BufferMode 取值和**轴版完全不同**，别照搬轴的 0/1：
            #     1 = 附加运动    当前的运动序列已完成，运动系统将停止，之后才执行新作业  ← 会停
            #     2 = 滤波处理（较低速度）  达到混合距离时与当前运动混合
            #     5 = 滤波处理（较高速度）  达到混合距离时与当前运动混合
            #   而轴版 MC_MoveAbsolute 的 0=立即中止 / 1=混合 —— **运动机构版没有 0**。
            #
            # 现场踩过：这里原来发 0（照抄轴版语义），对 TO_Kinematics 是**非法值**，
            # 作业被拒绝、目标直接丢失。日志里 `BufferMode 混合 N / 中止 1` 的那个
            # "中止 1" 就是它。
            #
            # 现在大跳变时发 1（附加运动）= 明确的"走完当前再走新的"，语义最接近原来
            # 想要的中止重开，而且是**合法值**。
            # 限速开启时，所有斜坡步均由软件规划保证连续平滑，统一保持 BufferMode=5（混合滤波），
            # 彻底杜绝中途停稳造成的卡顿；只有未开启限速（slew<=0）且步距超阈值时才发模式 1。
            buffer_mode = 1 if (self.target_slew_mm <= 0.0 and distance > self.blend_threshold_mm + COORD_EPSILON) else 5
            # 精磨距离和诊断步距必须用**实际发出去的步长**（加速度限幅、包络兜底之后）。
            # 限幅前的 distance 可能比实际步长大好几倍（目标反向时实测 20mm vs 6.5mm），
            # 精磨距离按它算就会超过整段运动，TO 把整段圆掉、到不了目标点。
            distance = max(abs(phys_x - origin_x), abs(phys_y - origin_y))
        else:
            distance = None
            buffer_mode = 5
        return phys_x, phys_y, buffer_mode, distance

    def _write_payload(self, phys_x, phys_y, buffer_mode):
        data = bytearray(MOTION_BLOCK_SIZE)
        # The machine's TO_Kinematics uses components 1=X, 3=Y, NOT [X,Y,0,0].
        struct.pack_into(">ddddi", data, 0, phys_x, 0.0, phys_y, 0.0, buffer_mode)
        self.round_trips += 1
        self._client.db_write(DB_MOTION, MOTION_BLOCK_START, data)

    def _commit_move(self, phys_x, phys_y, buffer_mode, distance):
        # 记住这一周期真正发出去的步长 —— 下一周期的加速度限幅要拿它比。
        if self._last_phys_x is not None:
            self._last_step = (phys_x - self._last_phys_x, phys_y - self._last_phys_y)
        self._last_phys_x, self._last_phys_y = phys_x, phys_y
        self._last_send_at = time.perf_counter()
        self.last_buffer_mode = buffer_mode
        if distance is not None:
            self.last_send_dist_mm = distance
        self.last_send_state = "sent"

    def _write_transition(self, distance):
        """写 TransitionParameter[1]（精磨距离）＝ 混合起点离目标多远。

        ★★★ 为什么必须每周期跟着步距写，不能留一个固定值：
        它的语义是「在距离目标还有这么远时，就开始和下一段作业混合」。
          · 步距 20mm、精磨 15mm → 只最后 15mm 混合 ✓ 连续
          · 步距  5mm、精磨 15mm → 精磨距离比整段还长 → TO 把整段圆掉，
            **到不了目标**。小步距恰恰是"瞄准到点"的精细修正，被圆掉就瞄不准。

        ★★★ 步距接近 0 时**绝不能写 0**（现场踩过）：
        目标没变时步距就是 0.1mm 量级，写成 0 的含义是「精磨距离 = 0」=
        **完全不许混合**，等于把刚修好的混合又关掉。日志里
        `步距中位 0.1mm | 精磨 0.0mm` 就是这个 bug。
        这种周期本来就没什么可混合的，**保持上一次的值**才对。
        """
        if not self.transition_enabled:
            return
        if distance is None or distance <= DEAD_ZONE_MM:
            return          # 步距太小：保持上一次的精磨距离，不要写 0
        value = min(self.blend_threshold_mm, distance * TRANSITION_RATIO)
        value = round(value, 1)
        # 变化不大就不写 —— 省一次往返，也避免 PLC 读到中间态。
        if (self._last_transition is not None
                and abs(value - self._last_transition) < 0.2):
            return
        try:
            self._client.db_write(DB_MOTION, COMM_TRANSITION_OFFSET,
                                  struct.pack(">d", value))
        except Exception as exc:
            self.transition_enabled = False
            self.last_error = (
                f"PLC 写 TransitionParameter[1] 失败（DB1.DBD{COMM_TRANSITION_OFFSET}）"
                f"：{exc} —— 已停用该写入；混合距离改回由 TO「最大舍入间距」决定")
            print(f"[PLC] {self.last_error}")
            return
        self.round_trips += 1
        self._last_transition = value
        self.last_transition_mm = value

    def send_linear_move(self, game_x, game_y):
        """脉冲模式：写 36 字节 [X,0,Y,0,mode]，然后抬 DB1.56.0（P 沿触发）。"""
        if not self.connected:
            self.last_send_state = "disconnected"
            return False
        prepared = self._prepare_move(game_x, game_y)
        if prepared is None:
            return self.last_send_state != "invalid"
        phys_x, phys_y, buffer_mode, distance = prepared
        try:
            self._write_payload(phys_x, phys_y, buffer_mode)
            self._write_transition(distance)
            self._write_bool(DB_MOTION, TRIGGER_BYTE, TRIGGER_LINEAR_BIT, True)
            self._commit_move(phys_x, phys_y, buffer_mode, distance)
            return True
        except Exception:
            self._connected = False
            self.last_send_state = "disconnected"
            return False

    def send_linear_move_sequence(self, game_x, game_y, sequence):
        """序号握手模式：写目标块，再写序号 —— PLC 比较序号后执行，**不需要脉宽**。

        [!] 顺序不能反：**先写目标块，再写序号**。这样 PLC 看到新序号时
        目标已经稳定，和"先写目标再抬触发"是同一个思路。
        """
        if not self.connected:
            self.last_send_state = "disconnected"
            return False
        prepared = self._prepare_move(game_x, game_y)
        if prepared is None:
            return self.last_send_state != "invalid"
        phys_x, phys_y, buffer_mode, distance = prepared
        try:
            self._write_payload(phys_x, phys_y, buffer_mode)
            # ★ 精磨距离必须在触发/序号之前写好 —— PLC 看到新序号时两个值都要是新的。
            self._write_transition(distance)
            self.round_trips += 1
            self._client.db_write(DB_MOTION, COMM_SEQ_OFFSET,
                                  struct.pack(">i", sequence & 0x7FFFFFFF))
            self._commit_move(phys_x, phys_y, buffer_mode, distance)
            return True
        except Exception:
            self._connected = False
            self.last_send_state = "disconnected"
            return False

    def read_status_and_position(self):
        if not self.connected:
            return None
        try:
            db1, db18 = self._read_status_blocks()
            if db1 is None:
                self._connected = False
                return None
            status, comm, group = db1[MOTION_STATUS_BYTE], db18[COMM_BYTE], db1[GROUP_CTRL_BYTE]

            def _word(offset):
                """按无符号读一个 Word；DB18 还是旧长度时返回 None。"""
                if offset + 2 > len(db18):
                    return None
                return struct.unpack_from(">H", db18, offset)[0]

            def _bit(offset, bit):
                if offset >= len(db18):
                    return None
                return bool(db18[offset] & (1 << bit))

            def _dword(offset):
                if offset + 4 > len(db18):
                    return None
                return struct.unpack_from(">I", db18, offset)[0]

            def _lreal(offset):
                if offset + 8 > len(db18):
                    return None
                return struct.unpack_from(">d", db18, offset)[0]

            return {
                "x_en": bool(status & 0x04), "y_en": bool(status & 0x08),
                "x_home": bool(status & 0x10), "y_home": bool(status & 0x20),
                "x_pos": struct.unpack_from(">d", db1, 2)[0],
                "y_pos": struct.unpack_from(">d", db1, 10)[0],
                "x_err": bool(comm & (1 << COMM_X_ERR_BIT)),
                "y_err": bool(db18[COMM_Y_ERR_OFFSET] & (1 << COMM_Y_ERR_BIT)),
                # ★ 轴自己的故障码（一直没读）
                "x_errid": _word(COMM_X_ERRID_OFFSET),
                "y_errid": _word(COMM_Y_ERRID_OFFSET),
                # MC_MoveLinearAbsolute 的 Error / ErrorId（现场新接的）
                "mc_error": _bit(COMM_MC_ERR_OFFSET, COMM_MC_ERR_BIT),
                "mc_errid": _word(COMM_MC_ERRID_OFFSET),
                "busy": bool(db1[MOTION_BUSY_BYTE] & 1),
                "plc_heartbeat": bool(comm & (1 << COMM_PLC_HEARTBEAT_BIT)),
                "comm_lost": bool(comm & (1 << COMM_LINK_LOST_BIT)),
                "axes_ready": bool(comm & (1 << COMM_AXES_READY_BIT)),
                "group_stop": bool(group & (1 << GROUP_STOP_BIT)),
                "comm_lost_latch": bool(group & (1 << GROUP_COMM_LOST_LATCH_BIT)),
                # ★ ErrorDetail.Number 在 TO 里是 **UDInt**（DB12.Static.ErrorDetail.Number），
                # 程序用 MOVE 送进 %DB18.DBW56（Word，只保留低 16 位）。
                # 必须按**无符号**读 —— 按有符号读的话，16#8001 这类码会变成 -32767，
                # 和 TIA 里看到的对不上，故障就成了谜。
                "kinematics_error": struct.unpack_from(">H", db18, COMM_KIN_ERROR_OFFSET)[0],
                # ErrorDetail.Reaction（DInt，有符号）—— 读不到就是 None（DB18 短）
                "kin_reaction": (struct.unpack_from(">i", db18, COMM_KIN_REACTION_OFFSET)[0]
                                 if COMM_KIN_REACTION_OFFSET + 4 <= len(db18) else None),
                # PLC 已处理到哪个序号（它写在 DB1.64）。和 Python 的序号一比，
                # 就知道"PLC 落后几条指令" —— 这是判断它跟不跟得上的直接指标。
                "seq_ack": (struct.unpack_from(">i", db1, COMM_SEQ_ACK_OFFSET)[0]
                            if COMM_SEQ_ACK_OFFSET + 4 <= len(db1) else None),
                # ---- 诊断补充字段（PLC 没加就是 None，界面显示"未读到"）----
                # ★ WarningWord：软限位之类的**预警**，能在报故障之前看到征兆
                "warning_word": _dword(COMM_WARNING_WORD_OFFSET),
                # ★ 当前运动还差多远（接的是 MC.RemainingDistance）
                "mc_remaining": _lreal(COMM_MC_REMAINING_OFFSET),
                # ★ OB1 循环时间：infoData 是 LTime（64 位有符号，单位纳秒）
                "ob1_ns": (struct.unpack_from(">q", db18, COMM_OB1_TIME_OFFSET)[0]
                           if COMM_OB1_TIME_OFFSET + 8 <= len(db18) else None),
                "rtinfo_status": (struct.unpack_from(">h", db18, COMM_RTINFO_STATUS_OFFSET)[0]
                                  if COMM_RTINFO_STATUS_OFFSET + 2 <= len(db18) else None),
            }
        except Exception:
            self._connected = False
            return None

    def keep_alive(self):
        if not self.connected:
            return
        now = time.perf_counter()
        if now - self.last_heartbeat_time <= 0.1:
            return
        self.heartbeat_state = not self.heartbeat_state
        try:
            self._write_bool(DB_COMM, COMM_BYTE, COMM_HEARTBEAT_BIT, self.heartbeat_state)
            self.last_heartbeat_time = now
        except Exception:
            self._connected = False


class PLCLink:
    """Single-owner worker; start/stop and target publication do no network I/O."""

    def __init__(self, ip, rack=0, slot=1, period=DEFAULT_PERIOD,
                 client_factory=None, zone=None, trigger_low_time=TRIGGER_LOW_TIME,
                 blend_threshold_mm=BUFFER_MODE_IMMEDIATE_MM, sequence_trigger=False,
                 target_slew_mm=0.0, target_accel_mm_s2=0.0):
        self.ip = ip
        self.zone = _validated_zone(DEFAULT_ZONE if zone is None else zone)
        self.period = max(0.005, float(period))
        if not math.isfinite(self.period):
            raise ValueError("PLC period must be finite")
        # 高、低电平都要跨过 PLC 扫描；仅保证低电平会在高通信频率下漏掉上升沿。
        self.trigger_low_time = max(0.0, float(trigger_low_time))
        if not math.isfinite(self.trigger_low_time):
            raise ValueError("trigger_low_time must be finite")
        self._trigger_high_at = None
        # ★ 序号握手模式：用序号代替脉冲触发，彻底不受脉宽/扫描周期限制。
        #   见 COMM_SEQ_OFFSET 的说明和 PLC 侧需要的改动。
        self.sequence_trigger = bool(sequence_trigger)
        # 序号种子取当前时间：避免"app 重启后序号恰好和 PLC 里记的重合，
        # 第一条指令被当成旧指令丢掉"这种极难查的问题。
        self._sequence = int(time.time()) % 1000000 + 1
        self._sequence_sent = 0
        self._requires_snap7 = client_factory is None
        self.blend_threshold_mm = float(blend_threshold_mm)
        self.target_slew_mm = float(target_slew_mm)
        # ★ 目标加速度限幅（mm/s^2）。见 PLCInterface._prepare_move 的说明：
        #   TIA 里 Y 轴报的是「跟随误差」，纯限步长治不了（急转需要的加速度
        #   是轴能力的约 5 倍）。0 = 关闭。
        self.target_accel_mm_s2 = float(target_accel_mm_s2)
        self._client = (PLCInterface(ip, rack, slot, zone=self.zone,
                                     blend_threshold_mm=self.blend_threshold_mm,
                                     target_slew_mm=self.target_slew_mm,
                                     target_accel_mm_s2=self.target_accel_mm_s2,
                                     period=self.period)
                        if client_factory is None else client_factory(ip, rack, slot))
        if isinstance(self._client, PLCInterface):
            self._client.zone = self.zone
        self._lock = threading.Lock()
        self._target = None
        self._target_at = 0.0
        self._feedback = PLCFeedback()
        self._commands = queue.Queue()
        self._messages = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._disable_on_stop = False
        self._armed = False
        self._arm_pending = False
        self._connect_attempted = False
        self._ever_connected = False
        self._snap7_warned = False
        self._last_retry = 0.0
        self._last_plc_heartbeat = None
        self._echo_change_at = 0.0
        self._echo_seen = False
        self.cycle_count = 0
        # 统计实际频率、同点跳过、模式分布和目标步距；不把它们直接当作 801 的原因。
        self._stats_at = time.perf_counter()
        self._stats_cycles = 0
        self._stats_counters = {"sent": 0, "deadzone": 0, "pending": 0, "other": 0,
                                "mode1": 0, "mode5": 0}
        self._stats_dists = []
        # 现场延迟排查：PC → PLC → 轴 分段统计（见 control/latency.py）
        self.latency = LatencyProbe(self.period)
        self._sent_snapshot = None      # (perf_counter 时刻, 物理 x, 物理 y) 最近一次成功下发

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._target = None
            self._target_at = 0.0
            self._feedback = PLCFeedback()
            self._armed = False
            self._arm_pending = False
            self._trigger_high_at = None
            while not self._commands.empty():
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
            self._last_retry = 0.0
            self._stop_event.clear()
            self._disable_on_stop = False
            self._thread = threading.Thread(target=self._run, name="plc-link", daemon=True)
            self._thread.start()

    def stop(self, disable_axes=True, timeout=1.5):
        """Request worker cleanup; False means timeout, NEVER perform caller-thread I/O.

        A timed-out worker retains ownership and cleans up when its blocking I/O returns.
        A successful return confirms cleanup attempts completed, not that an offline
        PLC physically accepted the disable command; inspect drain_messages().
        """
        with self._lock:
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
            self._stop_event.set()
            self._disable_on_stop |= disable_axes
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        if thread.is_alive():
            return False
        with self._lock:
            if self._thread is thread:
                self._thread = None
        return True

    @property
    def connected(self):
        return self._client.connected

    @property
    def last_send_state(self):
        return self._client.last_send_state

    def diagnostics(self):
        """返回并重置通信频率、BufferMode 分布和相邻目标步距，不代表电机执行确认。"""
        with self._lock:
            now = time.perf_counter()
            elapsed = max(1e-9, now - self._stats_at)
            counters = dict(self._stats_counters)
            dists = self._stats_dists
            cycles = self.cycle_count - self._stats_cycles
            self._stats_at = now
            self._stats_cycles = self.cycle_count
            self._stats_counters = {"sent": 0, "deadzone": 0, "pending": 0, "other": 0,
                                    "mode1": 0, "mode5": 0}
            self._stats_dists = []
            # 序号模式的诊断：PLC 已处理到哪
            seq_ack = self._feedback.seq_ack
            seq_now = self._sequence
            seq_sent = self._sequence_sent

        mode_total = counters["mode1"] + counters["mode5"]
        dists.sort()
        median = dists[len(dists) // 2] if dists else 0.0
        p95 = dists[min(len(dists) - 1, int(len(dists) * 0.95))] if dists else 0.0

        text = (f"周期 {cycles / elapsed:5.1f}Hz | 下发 {counters['sent'] / elapsed:5.1f}Hz | "
                f"死区跳过 {counters['deadzone']} | 执行中同点跳过 {counters['pending']}")
        if mode_total:
            stop_pct = 100.0 * counters["mode1"] / mode_total
            text += (f" | BufferMode 混合 {counters['mode5']} / 附加停稳 {counters['mode1']}"
                     f"（附加停稳 {stop_pct:.0f}%）")
        else:
            text += " | BufferMode 无下发"
        text += f" | 步距中位 {median:.1f}mm p95 {p95:.1f}mm"
        # ★ 精磨距离（TransitionParameter[1]）：混合起点离目标多远，随步距自适应。
        #   如果这里一直显示 "关"，说明写 DB1.DBD150 失败了 —— 那时混合距离
        #   退回由 TO「最大舍入间距」决定，就是"每点停一下"的老毛病。
        # ★ 这里必须用 self._client —— diagnostics() 里没有名为 client 的局部变量。
        #   （踩过：写成 client 会 NameError，而且是在每个 [stats] 行上崩，表现为
        #     "实时处理失败"，看起来像整个视觉管线坏了。）
        client = self._client
        trans = getattr(client, "last_transition_mm", None)
        if getattr(client, "transition_enabled", False):
            text += (f" | 精磨 {trans:.1f}mm" if trans is not None else " | 精磨 未发")
        else:
            text += " | 精磨 关（DB1.DBD150 写失败）"
        if counters["other"]:
            text += f" | 其它状态 {counters['other']}"
        if mode_total and 100.0 * counters["mode1"] / mode_total > 20.0:
            text += "  ← 附加停稳占比偏高（大步目标）；核对 --target-slew-mm 和 --blend-threshold-mm"
        # ★ 夹取比例：目标被"够不到"截住的频率。高 = 经常跑到行程边缘。
        counts = clamp_counts()
        if counts["total"]:
            pct_x = 100.0 * counts["x"] / counts["total"]
            pct_y = 100.0 * counts["y"] / counts["total"]
            text += f" | 目标被夹取 X {pct_x:.0f}% Y {pct_y:.0f}%"
            if pct_x > 30.0 or pct_y > 30.0:
                text += "  ← 频繁要求够不到的位置，机械臂长期顶在命令限位上"
        reset_clamp_counts()
        # ★ 序号模式的直接指标：PLC 落后几条指令。落后多 = 它跟不上。
        if self.sequence_trigger:
            if seq_ack is None:
                text += f" | 序号模式 已发 {seq_sent} 条（PLC 未回报 ack，检查 DB1.146）"
            else:
                # seq_now 是**序号值**（从时间戳起算），seq_sent 是**发送条数** ——
                # 两者不是一回事，显示时要分清，否则看起来像 "已发 403 / PLC 939123"。
                lag = (seq_now - seq_ack) & 0x7FFFFFFF
                text += f" | 序号模式 已发 {seq_sent} 条 / PLC 落后 {lag}"
                if lag > 3:
                    text += "  ← PLC 跟不上，指令在积压"
        # ★★ 用**实测周期**做安全自检 —— 不要用 --plc-rate 名义值。
        #    实测常只有 60~70Hz（S7 往返耗时主导），而名义可能是 200：
        #    用名义值算会把 slew 12 误判成 2400mm/s（制动 192mm）而狂报警，
        #    用实测值算才是真实的 840mm/s（制动 24mm）。
        #    限速器的单位是 mm/**周期**，所以"周期多快"直接决定指令速度。
        rate_hz = cycles / elapsed
        warn = self._safety_warning_text(rate_hz)
        if warn:
            text += warn
        return text

    def _safety_warning_text(self, rate_hz):
        """把"余量够不够刹得住"挂到诊断行上（有实测周期才算得准）。

        返回 ""（安全）或一段警告。只在**不安全**时输出，安全时不占地方。
        """
        slew = self.target_slew_mm
        if not slew or slew <= 0.0 or rate_hz <= 0.0:
            return ""
        # 超调按 TIA 减速度估算（见安全自检说明），不用软件加速度限幅
        try:
            warnings = safety_warnings(slew, rate_hz)
        except Exception:
            return ""
        if not warnings:
            return ""
        brake, speed = braking_distance_mm(slew, rate_hz)
        # 包络到限位开关之间的余量就是 MOTION_MARGIN_MM（包络 = TRAVEL_* 内缩它）
        margin = MOTION_MARGIN_MM
        return (f" | [!]安全自检 {speed:.0f}mm/s 制动{brake:.0f}mm > 余量{margin:.0f}mm"
                f" —— 冲向限位会刹不住，降 --target-slew-mm 或加 --motion-margin-mm")


    @property
    def connect_attempted(self):
        return self._connect_attempted

    @property
    def ever_connected(self):
        return self._ever_connected

    @property
    def feedback(self):
        with self._lock:
            return self._feedback

    @property
    def armed(self):
        """True only after explicit enable_axes and a healthy fresh PLC report."""
        with self._lock:
            return (self._armed and self._client.connected and not self._stop_event.is_set()
                    and self._motion_safe(self._feedback)
                    and time.perf_counter() - self._feedback.stamp <= TARGET_MAX_AGE)

    def game_to_physical(self, x, y):
        return _game_to_physical(self.zone, x, y)

    def physical_to_game(self, x, y):
        return _physical_to_game(self.zone, x, y)

    @property
    def reachable_bounds(self):
        """AI 可请求的命令范围（游戏坐标），由 COMMAND_* 而非 TRAVEL_* 决定。

        这个属性是给 AI 的 recovery_bounds 用的：告诉它"哪些位置是真能到的"。

        [!] 若用比命令包络更宽的范围（例如机械行程本身），AI 以为够得到、一直瞄外面，而 PLC 在下发时
        默默把目标夹回包络 —— 表现就是"机械臂长期顶在包络角点上"，
        而且 AI 的 target 和 PLC 实际下发的 target 对不上（有测试守着这个往返一致）。

        判据很简单：**"AI 能瞄的范围" 必须等于 "PLC 会原样接受的范围"。**
        这里读的是模块全局 COMMAND_*，所以 set_motion_margin() 之后自动跟着变。
        """
        left, top = self.physical_to_game(COMMAND_X_MIN, COMMAND_Y_MIN)
        right, bottom = self.physical_to_game(COMMAND_X_MAX, COMMAND_Y_MAX)
        return left, right, top, bottom

    def latency_report(self):
        """延迟排查汇总（一行文本, 结论提示列表），调用后清零统计窗口。"""
        return self.latency.report()

    def motion_snapshot(self):
        """给运行记录用的轴状态快照（游戏坐标）；时间用"距今多久"表示，与时钟源无关。"""
        now = time.perf_counter()
        with self._lock:
            feedback, sent, armed = self._feedback, self._sent_snapshot, self._armed
        if not feedback.valid:
            return None
        snapshot = {
            "age": now - feedback.stamp, "x": feedback.x, "y": feedback.y,
            "busy": bool(feedback.busy), "armed": bool(armed),
            "seq_ack": feedback.seq_ack, "remaining": feedback.mc_remaining, "sent": None,
        }
        if sent is not None:
            try:
                sent_x, sent_y = self.physical_to_game(sent[1], sent[2])
                snapshot["sent"] = {"age": now - sent[0], "x": sent_x, "y": sent_y}
            except (ValueError, OverflowError):
                pass
        return snapshot

    def set_target(self, game_x, game_y):
        try:
            x, y = _validate_game_point(self.zone, game_x, game_y)
        except (ValueError, OverflowError):
            return False
        with self._lock:
            if self._stop_event.is_set():
                return False
            self._target = x, y
            self._target_at = time.perf_counter()
        return True

    def clear_target(self):
        with self._lock:
            self._target = None
            self._target_at = 0.0

    def reset_tracking(self):
        return self._run_on_worker("清死区", self._client.reset_tracking)

    def enable_axes(self):
        return self._run_on_worker("使能", self._enable_on_worker)

    # ---- 单轴操作 ----------------------------------------------------------
    # 现场需要：某个轴报 801 / 撞限位之后**只把那个轴**撤下来，另一个继续用；
    # 以及单独回零 / 单独复位。
    #
    # ★ 单轴使能不会让程序开始发运动 —— _motion_safe 要求 x_en 和 y_en 同时为真，
    #   所以只有一个轴使能时是安全的。
    def set_axis_enable(self, axis, enabled):
        """axis 为 "x" / "y"。使能走 _run_on_worker（要靠新状态才 arm）。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        if enabled:
            return self._run_on_worker(f"{axis.upper()} 轴使能",
                                       lambda: self._enable_axis_on_worker(axis))
        return self._queue_disarmed_command(f"{axis.upper()} 轴撤销",
                                           lambda: self._client.set_axis_enable(axis, False))

    def _enable_on_worker(self):
        """双轴使能（E 键 / 按钮）。单轴版本见 _enable_axis_on_worker。"""
        if not self._client.enable_axes():
            return False
        # 报警/回零可能中断旧目标；人工重新授权后不能把未完成目标当作已到达。
        self._client.reset_tracking()
        with self._lock:
            if not self._stop_event.is_set():
                # Arming awaits a healthy fresh status, never a cached snapshot.
                # A pre-enable target must not become the first motion command.
                self._target = None
                self._target_at = 0.0
                self._arm_pending = True
        return True

    def _enable_axis_on_worker(self, axis):
        if not self._client.set_axis_enable(axis, True):
            return False
        # 目标跟踪作废：重新使能后不能把旧目标当作"已到达"（和双轴使能一致）。
        self._client.reset_tracking()
        with self._lock:
            if not self._stop_event.is_set():
                self._target = None
                self._target_at = 0.0
                self._arm_pending = True
        return True

    def toggle_axis_enable(self, axis):
        """按当前反馈状态取反。返回 (是否已发出, 切换后的期望状态)。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False, None
        feedback = self.feedback
        currently = feedback.x_en if axis == "x" else feedback.y_en
        wanted = not bool(currently)
        return self.set_axis_enable(axis, wanted), wanted

    def home_axis(self, axis):
        """单轴回零。[!] 回零会让该轴动，是刻意的人工动作。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        return self._queue_disarmed_command(f"{axis.upper()} 轴回零",
                                           lambda: self._client.home_axis(axis))

    def reset_axis(self, axis):
        """单轴复位（清该轴报警）。"""
        axis = str(axis).lower()
        if axis not in AXIS_NAMES:
            return False
        return self._queue_disarmed_command(f"{axis.upper()} 轴复位",
                                           lambda: self._client.reset_axis(axis))

    def home_axes(self):
        return self._queue_disarmed_command("回零", self._client.home_axes)

    def axes_reset(self):
        return self._queue_disarmed_command("轴复位", self._client.axes_reset)

    def drain_messages(self):
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def _run_on_worker(self, name, func):
        with self._lock:
            if self._stop_event.is_set() or not self._client.connected:
                return False
            self._commands.put((name, func))
            return True

    def _queue_disarmed_command(self, name, func):
        with self._lock:
            if self._stop_event.is_set() or not self._client.connected:
                return False
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
            self._commands.put((name, lambda: self._execute_disarmed(func)))
            return True

    def _execute_disarmed(self, func):
        try:
            return func()
        finally:
            with self._lock:
                self._target = None
                self._target_at = 0.0
                self._armed = False
                self._arm_pending = False

    def _drain_commands(self):
        while not self._stop_event.is_set():
            try:
                name, func = self._commands.get_nowait()
            except queue.Empty:
                return
            if not self._client.connected:
                continue
            try:
                if not func():
                    self._messages.put(f"{name}指令发送失败")
            except Exception as exc:
                self._messages.put(f"{name}异常：{exc}")

    def _invalidate_feedback(self):
        with self._lock:
            self._target = None
            self._target_at = 0.0
            self._armed = False
            self._arm_pending = False
            self._trigger_high_at = None
            if self._feedback.valid:
                self._feedback = PLCFeedback()
            # Commands from the old connection must not be replayed on reconnect.
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break

    def _retry_connect(self, now):
        if not HAS_SNAP7 and self._requires_snap7:
            if not self._snap7_warned:
                self._snap7_warned = True
                self._connect_attempted = True
                self._messages.put(SNAP7_ERROR)
            return
        if now - self._last_retry < RECONNECT_INTERVAL or self._stop_event.is_set():
            return
        self._last_retry = now
        was_connected = self._ever_connected
        self._connect_attempted = True
        if self._client.connect(quiet=True):
            self._ever_connected = True
            self._last_plc_heartbeat = None
            self._echo_change_at = now
            self._echo_seen = False
            self._messages.put("PLC 已重新连接" if was_connected else "PLC 已连接")
        elif self._requires_snap7:
            self._messages.put(f"PLC 连接失败：{self._client.last_error}")

    def _publish(self, status):
        now = time.perf_counter()
        try:
            x, y = self.physical_to_game(status["x_pos"], status["y_pos"])
        except (ValueError, OverflowError):
            self._invalidate_feedback()
            return None
        heartbeat = status["plc_heartbeat"]
        if self._last_plc_heartbeat is None:
            self._last_plc_heartbeat = heartbeat
            self._echo_change_at = now
        elif heartbeat != self._last_plc_heartbeat:
            self._last_plc_heartbeat = heartbeat
            self._echo_change_at = now
            self._echo_seen = True
        echo_ok = self._echo_seen and now - self._echo_change_at < ECHO_TIMEOUT
        with self._lock:
            previous = self._feedback
            dt = now - previous.stamp
            vx = (x - previous.x) / dt if previous.valid and dt > 1e-6 else 0.0
            vy = (y - previous.y) / dt if previous.valid and dt > 1e-6 else 0.0
            self._feedback = PLCFeedback(
                stamp=now, valid=True, x=x, y=y, vx=vx, vy=vy,
                x_en=status["x_en"], y_en=status["y_en"],
                x_home=status["x_home"], y_home=status["y_home"],
                x_err=status["x_err"], y_err=status["y_err"],
                x_errid=status["x_errid"], y_errid=status["y_errid"],
                mc_error=status["mc_error"], mc_errid=status["mc_errid"],
                busy=status["busy"], comm_lost=status["comm_lost"],
                axes_ready=status["axes_ready"], plc_echo_ok=echo_ok,
                group_stop=status["group_stop"],
                comm_lost_latch=status["comm_lost_latch"],
                kinematics_error=status["kinematics_error"],
                kin_reaction=status["kin_reaction"],
                seq_ack=status["seq_ack"],
                warning_word=status["warning_word"],
                mc_remaining=status["mc_remaining"],
                ob1_ns=status["ob1_ns"],
                rtinfo_status=status["rtinfo_status"],
            )
            return self._feedback

    @staticmethod
    def _motion_safe(feedback):
        return (feedback.valid and feedback.plc_echo_ok and feedback.axes_ready
                and feedback.x_en and feedback.y_en
                and not feedback.axis_fault and not feedback.comm_lost
                and not feedback.comm_lost_latch and not feedback.group_stop
                and feedback.kinematics_error == 0)

    def _cycle(self):
        client = self._client
        if not self.sequence_trigger:
            # ---- 脉冲模式的脉宽纪律（序号模式完全不需要）----
            if self._trigger_high_at is not None:
                remaining_high = self.trigger_low_time - (time.perf_counter() - self._trigger_high_at)
                if remaining_high > 0.0 and self._stop_event.wait(remaining_high):
                    return
            try:
                client.clear_trigger()
            except Exception as exc:
                self._messages.put(f"PLC 清除运动触发失败：{exc}")
                client.disconnect()
                self._invalidate_feedback()
                return
            self._trigger_high_at = None
        # 状态读取与心跳的耗时已算在低电平内，只补剩余的脉宽。
        cleared_at = time.perf_counter()
        status = client.read_status_and_position()
        if status is None or not client.connected:
            self._invalidate_feedback()
            return
        client.keep_alive()
        status_done = time.perf_counter()
        if not client.connected or self._stop_event.is_set():
            self._invalidate_feedback()
            return
        previous = self.feedback
        feedback = self._publish(status)
        if feedback is None:
            return
        # ★ 把轴的实际位置交给 client —— 死区判据要用它判断"轴到没到"。
        #   不更新的话一旦轴没走到上次发的目标，代码会永远跳过（假死）。
        try:
            client.actual_game = (feedback.x, feedback.y)
        except Exception:
            pass
        if isinstance(client, PLCInterface):
            client.actual_busy = feedback.busy
        safe = self._motion_safe(feedback)
        revoked = False
        interrupted_target = None
        with self._lock:
            # Serious faults revoke even a pending arm request. During the initial
            # heartbeat/enable ramp, hold pending without allowing any movement.
            # 行程越界由硬件限位开关负责（触发后 TIA 报轴故障，走 axis_fault 撤销）。
            fault = (feedback.axis_fault or feedback.comm_lost or feedback.comm_lost_latch
                     or feedback.group_stop or feedback.kinematics_error != 0
                     or (not feedback.plc_echo_ok
                         and time.perf_counter() - self._echo_change_at >= ECHO_TIMEOUT)
                     or (self._armed and not safe))
            if fault:
                revoked = self._armed or self._arm_pending
                interrupted_target = self._target
                self._armed = False
                self._arm_pending = False
                self._target = None
                self._target_at = 0.0
            elif safe and self._arm_pending:
                self._armed = True
                self._arm_pending = False
            target = (self._target if safe and self._armed
                      and not self._stop_event.is_set()
                      and time.perf_counter() - self._target_at <= TARGET_MAX_AGE else None)
            armed_now = self._armed
        # 延迟排查统计（非必要功能，任何异常都不能影响通信）
        try:
            self.latency.on_status(
                status_done, (status_done - cleared_at) * 1000.0,
                (float(status["x_pos"]), float(status["y_pos"])), armed=armed_now,
                busy=feedback.busy, seq_ack=feedback.seq_ack,
                remaining=feedback.mc_remaining, ob1_ms=feedback.ob1_ms)
        except Exception:
            pass
        if revoked:
            def _limit_note(px, py):
                """把原始位置和行程比一下 —— 故障出现时一眼看出是不是碰到了限位开关。"""
                notes = []
                for axis, value, low, high, margin_low, margin_high in (
                        ("X", px, TRAVEL_X_MIN, TRAVEL_X_MAX, COMMAND_X_MIN, COMMAND_X_MAX),
                        ("Y", py, TRAVEL_Y_MIN, TRAVEL_Y_MAX, COMMAND_Y_MIN, COMMAND_Y_MAX)):
                    if value < low or value > high:
                        notes.append(f"{axis}={value:.2f} 已超出机械行程[{low},{high}] ← 限位开关")
                    elif value < margin_low or value > margin_high:
                        notes.append(f"{axis}={value:.2f} 在命令范围外（行程内），离限位开关 "
                                     f"{min(abs(value - low), abs(value - high)):.2f}mm")
                    else:
                        notes.append(f"{axis}={value:.2f} 在命令范围内")
                return " | ".join(notes)

            self._messages.put(
                f"运动撤销快照: kin={feedback.kinematics_error} "
                f"en=({feedback.x_en},{feedback.y_en}) ready={feedback.axes_ready} "
                f"err=({feedback.x_err},{feedback.y_err}) busy={feedback.busy} "
                f"echo={feedback.plc_echo_ok} comm_lost={feedback.comm_lost} "
                f"stop={feedback.group_stop} latch={feedback.comm_lost_latch} "
                f"physical=({status['x_pos']:.3f},{status['y_pos']:.3f}) "
                + _limit_note(status["x_pos"], status["y_pos"]) + " "
                + feedback.fault_detail() + " "
                f"prev_valid={previous.valid} prev_kin={previous.kinematics_error} "
                f"prev_en=({previous.x_en},{previous.y_en}) prev_ready={previous.axes_ready} "
                f"prev_err=({previous.x_err},{previous.y_err}) prev_busy={previous.busy} "
                f"intent={interrupted_target} "
                f"last_sent=({getattr(client, '_last_phys_x', None)},{getattr(client, '_last_phys_y', None)}) "
                f"mode={getattr(client, 'last_buffer_mode', None)} "
                f"step_mm={getattr(client, 'last_send_dist_mm', None)}"
            )
        # Network I/O is never under the lock. Cancellation of an already executing
        # db_write cannot be guaranteed; no subsequent target remains queued.
        if target is not None:
            # 把"触发低电平"的剩余时间补满，然后才抬起来下发。
            remaining_low = self.trigger_low_time - (time.perf_counter() - cleared_at)
            if remaining_low > 0.0 and self._stop_event.wait(remaining_low):
                return
            # 等待脉宽期间生产者可能更新、清除目标或撤销授权，发送前取最新意图。
            with self._lock:
                target = (self._target if self._armed and not self._stop_event.is_set()
                          and time.perf_counter() - self._target_at <= TARGET_MAX_AGE
                          and time.perf_counter() - feedback.stamp <= TARGET_MAX_AGE else None)
                target_at = self._target_at
            if target is None:
                self.cycle_count += 1
                return
            send_start = time.perf_counter()
            sent_sequence = None
            if self.sequence_trigger:
                # 序号模式：写目标块 + 递增的序号，PLC 比较后执行，不需要脉宽。
                # ★ 序号只在**真的下发成功**时推进 —— 死区跳过不能推进，
                #   否则 PLC 会拿同一条目标重复触发。
                candidate = self._sequence + 1
                client.send_linear_move_sequence(target[0], target[1], candidate)
                if client.connected and client.last_send_state == "sent":
                    self._sequence = candidate
                    self._sequence_sent += 1
                    sent_sequence = candidate
            else:
                client.send_linear_move(*target)
                if client.connected and client.last_send_state == "sent":
                    self._trigger_high_at = time.perf_counter()
            send_done = time.perf_counter()
            try:
                sent_x = getattr(client, "_last_phys_x", None)
                sent_y = getattr(client, "_last_phys_y", None)
                if client.connected and client.last_send_state == "sent" and sent_x is not None:
                    with self._lock:
                        self._sent_snapshot = (send_done, sent_x, sent_y)
                    self.latency.on_send(send_done, (send_done - send_start) * 1000.0,
                                         send_start - target_at, sent_x, sent_y, seq=sent_sequence)
            except Exception:
                pass
            # 现场诊断累计。整个累计块必须容错 —— 诊断是"非必要"的，
            # 绝不能让它的任何异常把 worker 线程打死（那会让后续命令全部失效）。
            try:
                state = client.last_send_state
                with self._lock:
                    # ★ "deadzone-resend" 也是真发出去了（目标没变、但轴没到位，
                    #   所以重发）—— 必须算进"下发 Hz"。原来它落到下面的 else
                    #   分支，于是现场日志出现"下发 0.0Hz"却其实一直在发，
                    #   把"轴到底动没动"这个判断彻底带偏。
                    if state in ("sent", "deadzone-resend"):
                        self._stats_counters["sent"] += 1
                        mode = getattr(client, "last_buffer_mode", None)
                        if mode in (1, 5):
                            self._stats_counters["mode1" if mode == 1 else "mode5"] += 1
                            self._stats_dists.append(
                                float(getattr(client, "last_send_dist_mm", 0.0) or 0.0))
                    elif state == "deadzone":
                        self._stats_counters["deadzone"] += 1
                    elif state == "pending":
                        self._stats_counters["pending"] += 1
                    else:
                        self._stats_counters["other"] += 1
            except Exception:
                pass
        if not client.connected:
            self._invalidate_feedback()
        self.cycle_count += 1

    def _run(self):
        client = self._client
        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                try:
                    if client.connected:
                        self._drain_commands()
                        if client.connected and not self._stop_event.is_set():
                            self._cycle()
                    else:
                        self._invalidate_feedback()
                        self._retry_connect(cycle_start)
                except Exception as exc:
                    self._messages.put(f"PLC 通信异常：{exc}")
                    self._invalidate_feedback()
                    client.disconnect()
                remaining = self.period - (time.perf_counter() - cycle_start)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        finally:
            self._invalidate_feedback()
            if client.connected:
                try:
                    client.clear_trigger()
                except Exception as exc:
                    self._messages.put(f"PLC 清除运动触发失败，请现场确认轴状态：{exc}")
                if self._disable_on_stop:
                    try:
                        if not client.disable_axes():
                            self._messages.put("PLC 撤销使能失败，请现场确认轴状态")
                    except Exception as exc:
                        self._messages.put(f"PLC 撤销使能失败，请现场确认轴状态：{exc}")
            elif self._disable_on_stop and self._ever_connected:
                self._messages.put("PLC 已断线，无法确认撤销使能；请现场确认轴状态")
            try:
                client.disconnect()
            except Exception:
                pass
