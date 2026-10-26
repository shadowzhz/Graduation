"""S7-1200（CPU 1211C，工程 1211kongqi）DB1 / DB18 的变量布局（偏移、位）与 python-snap7 的兼容导入。

[!] 这里的每个偏移都必须和 TIA 里【实际排布】一致 —— 读错偏移会静默返回垃圾值。

snap7 / Areas / WordLen / HAS_SNAP7 由 interface.py、link.py 通过 `protocol.xxx`
模块属性访问（不要 from-import），测试用 patch.multiple(protocol, ...) 换成内存传输。
"""

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
# 两个 DB 都是 Standard（非优化）布局，偏移按 TIA 排布规则从 2026-10-26 工程档案算出：
#   Bool 按位连续排；其他类型从偶数字节开始；Real 4 字节，LReal 8 字节。
#   DB1 共 120 字节，DB18 共 54 字节（档案里工作内存 124 / 58，正好多 4 字节块头）。
#
# DB1（数据块）
#   0.0~0.7  X点正 X点反 Y点正 Y点反 X复位 Y复位 X启动 Y启动
#   1.0~1.5  X回原点 Y回原点 X使能Status Y使能Status X回原点完成 Y回原点完成
#   2.0 / 6.0  X/Y轴实时位置 Real
#   10.0 XY轴直线busy（= X 或 Y 的 MC_MoveAbsolute.Busy）
#   12.0 XY轴直线绝对 Array[1..4] of Real（FC1 网络 6 的两个 MC_MoveAbsolute.Position）
#   32.0 XY轴直线绝对启动
#   88.2 停止所有运动触发  88.3 XY系统复位  88.4 通信丢失锁存
#   90.0 指令序号 DInt   94.0 上次序号 DInt   118.0 气缸启动
MOTION_CMD_BYTE = 0
HOMING_BYTE = 1
AXIS_ENABLE_BITS = {"x": 6, "y": 7}
AXIS_RESET_BITS = {"x": 4, "y": 5}
AXIS_HOME_BITS = {"x": 0, "y": 1}
AXIS_NAMES = ("x", "y")
MOTION_STATUS_BYTE = 1
X_POS_OFFSET = 2
Y_POS_OFFSET = 6
MOTION_BUSY_BYTE = 10
MOTION_BLOCK_START = 12
TRIGGER_BYTE = 32
TRIGGER_LINEAR_BIT = 0
GROUP_CTRL_BYTE = 88
GROUP_STOP_BIT = 2
GROUP_RESET_BIT = 3
GROUP_COMM_LOST_LATCH_BIT = 4
# 序号握手（FC1 网络 5）：Python 先写目标块再写指令序号，PLC 发现序号变了就执行并记到上次序号。
COMM_SEQ_OFFSET = 90
COMM_SEQ_ACK_OFFSET = 94
CYLINDER_BYTE = 118
CYLINDER_START_BIT = 0
STATUS_READ_START = 0
STATUS_READ_SIZE = 98   # 读到上次序号（94~97）为止

# DB18（通讯块）
#   0.0 Python心跳  0.1 plc心跳  0.2 通信丢失  0.3 轴已就绪  0.4 X轴ERROR
#   2.0 X轴ERRORID Word  4.0 Y轴ERROR  6.0 Y轴ERRORID Word
#   32.0 XY轴ErrorDetail.Number UInt  34.3 error  36.0 errorid Word
#   38.0 XY轴ErrorDetail.Reaction DInt  46.0 XY轴RemainingDistance LReal
# [!] 32 以后的诊断字段在 1211C 程序里还没人写，读出来恒为 0。
COMM_READ_START = 0
COMM_READ_SIZE = 54
COMM_BYTE = 0
COMM_HEARTBEAT_BIT = 0
COMM_PLC_HEARTBEAT_BIT = 1
COMM_LINK_LOST_BIT = 2
COMM_AXES_READY_BIT = 3
COMM_X_ERR_BIT = 4
COMM_X_ERRID_OFFSET = 2
COMM_Y_ERR_OFFSET = 4
COMM_Y_ERR_BIT = 0
COMM_Y_ERRID_OFFSET = 6
COMM_KIN_ERROR_OFFSET = 32
COMM_MC_ERR_OFFSET = 34
COMM_MC_ERR_BIT = 3
COMM_MC_ERRID_OFFSET = 36
COMM_KIN_REACTION_OFFSET = 38
COMM_MC_REMAINING_OFFSET = 46
