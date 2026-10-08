"""控制适配层：AI 半场目标 -> S7-1500T 运动学闭环。"""

from .adapter import PlcControlAdapter, PlcTarget
from .plc import PLCLink, PLCFeedback

__all__ = ["PlcControlAdapter", "PlcTarget", "PLCLink", "PLCFeedback"]
