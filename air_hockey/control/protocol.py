"""S7-1500 DB1 / DB18 的变量布局（偏移、位）与 python-snap7 的兼容导入。

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
# [!] 不能超过 DB1 实际长度（现场 DB1：TransitionParameter[1..5] LReal @150~189，气缸启动 @190.0）。
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
# （之后是 TransitionParameter @150.0~189，气缸启动 @190.0）
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

# TransitionParameter[1]（精磨距离 = 混合起点离目标多远）在 DB1 里的字节偏移。
# 现场在 DB1 建了 `TransitionParameter : Array[1..5] of LReal` @ 150.0，
# FC1 用指针 P#DB1.DBX150.0 传给 MC_MoveLinearAbsolute 的 TransitionParameter。
COMM_TRANSITION_OFFSET = 150
