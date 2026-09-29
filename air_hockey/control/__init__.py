"""控制适配层：AI 目标 -> 安全限幅的 PlcWriteRequest -> PLCInterface。"""

from .adapter import PlcControlAdapter, PlcWriteRequest
from .output import PlcOutputWorker

__all__ = ["PlcControlAdapter", "PlcOutputWorker", "PlcWriteRequest"]
