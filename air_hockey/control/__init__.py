"""控制适配层：AI 半场目标 -> S7-1500T 运动学闭环。

    protocol   DB1 / DB18 变量布局与 snap7 兼容导入
    limits     机械行程、命令包络、TIA 动力学限值（MotionLimits）、坐标映射
    feedback   PLCFeedback 反馈快照
    interface  PLCInterface：阻塞式 S7 传输 + 运动指令规划
    link       PLCLink：后台工作线程（心跳 / 授权互锁 / 下发 / 诊断）
"""

from .adapter import PlcControlAdapter, PlcTarget
from .feedback import PLCFeedback
from .limits import MotionLimits
from .link import PLCLink

__all__ = ["MotionLimits", "PlcControlAdapter", "PlcTarget", "PLCLink", "PLCFeedback"]
