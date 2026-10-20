"""PLCLink 工作线程发布的只读反馈快照。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
