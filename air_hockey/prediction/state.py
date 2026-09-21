"""预测结果数据层。

PredictionState 打包一次轨迹预测的完整结果（采样轨迹、终点、时长、初始状态），
业务层（AI、VisionRuntime）只与 PredictionState 交互，不再直接传递裸的 list[(x, y)]。

为方便渲染与遍历，PredictionState 实现了序列协议，等价于其 trajectory 本身。
"""

from __future__ import annotations

from dataclasses import dataclass

from game_state import CurlingState


@dataclass(eq=False)
class PredictionState:
    """一次轨迹预测的结果。

    trajectory        按时间顺序采样的轨迹点
    endpoint          预测终点（轨迹最后一点）
    duration          从初始状态到终点的模拟时长（秒）
    source_state      预测所依据的初始冰壶状态
    """

    trajectory: list[tuple[float, float]]
    endpoint: tuple[float, float]
    duration: float
    source_state: CurlingState

    def __len__(self) -> int:
        return len(self.trajectory)

    def __iter__(self):
        return iter(self.trajectory)

    def __getitem__(self, index):
        return self.trajectory[index]

    def __bool__(self) -> bool:
        return bool(self.trajectory)

    def __eq__(self, other) -> bool:
        # 轨迹相同即视为同一预测结果；也允许直接与裸点列表比较。
        if isinstance(other, PredictionState):
            return self.trajectory == other.trajectory
        if isinstance(other, list):
            return self.trajectory == other
        return NotImplemented

    __hash__ = None
