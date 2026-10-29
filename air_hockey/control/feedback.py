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
    x_err: bool = False
    y_err: bool = False
    # 下面可为 None 的字段：None = 还没有有效反馈。
    # 轴自己的故障码（DB18.2 / DB18.6），撞限位、跟随误差等都在这里给出 TIA 里看到的代码
    x_errid: Optional[int] = None
    y_errid: Optional[int] = None
    # DB18.32 以后的诊断字段 1211C 程序还没写，读到恒为 0
    mc_error: Optional[bool] = None         # 运动指令 Error / ErrorId（DB18.34.3 / DB18.36）
    mc_errid: Optional[int] = None
    busy: bool = False
    comm_lost: bool = False
    axes_ready: bool = False
    plc_echo_ok: bool = True
    group_stop: bool = False
    comm_lost_latch: bool = False
    kinematics_error: int = 0               # ErrorDetail.Number：报了什么
    kin_reaction: Optional[int] = None      # ErrorDetail.Reaction：TO 因此做了什么
    seq_ack: Optional[int] = None           # PLC 已处理到的序号（DB1.94），和本地序号一比就是"落后几条"
    mc_remaining: Optional[float] = None    # 当前运动还差多远（RemainingDistance）

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
