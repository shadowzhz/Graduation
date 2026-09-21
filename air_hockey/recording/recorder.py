"""真实运行数据记录。

RuntimeRecorder 从实时 VisionResult 中抽取 CurlingState / PredictionState / timestamp / FPS，
按统一格式累积并在结束时写出 JSON。仿真数据可用 record_simulation_result 写成同一格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from game_state import CurlingState

from ..simulation import SimulationResult
from .schema import build_document, build_frame


class RuntimeRecorder:
    """实时运行记录器：累积帧记录并写出统一 JSON。"""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        source: str = "runtime",
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.path = None if path is None else Path(path)
        self.source = source
        self.meta = dict(meta or {})
        self._frames: list[dict] = []
        self._closed = False

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def closed(self) -> bool:
        return self._closed

    def record(self, result, *, ai_decision=None, plc_request=None) -> None:
        """记录一帧 VisionResult（取 curling_state / prediction / timestamp / fps 及控制输出）。"""
        self.record_frame(
            timestamp=result.frame.timestamp,
            fps=result.fps,
            curling_state=result.curling_state,
            prediction=result.prediction,
            ai_decision=ai_decision,
            plc_request=plc_request,
        )

    def record_frame(
        self,
        timestamp: float,
        fps: float,
        curling_state: Optional[CurlingState] = None,
        prediction=None,
        ai_decision=None,
        plc_request=None,
    ) -> None:
        """记录一帧原始字段。"""
        if self._closed:
            raise RuntimeError("recorder already closed")
        self._frames.append(
            build_frame(
                len(self._frames),
                timestamp,
                fps,
                curling_state,
                prediction,
                ai_decision,
                plc_request,
            )
        )

    def to_dict(self) -> dict:
        return build_document(self.source, self._frames, self.meta)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def flush(self, indent: int = 2) -> Optional[Path]:
        """把当前记录写入文件（原子替换），未设置 path 时返回 None。"""
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(self.to_json(indent=indent), encoding="utf-8")
        temporary.replace(self.path)
        return self.path

    def close(self) -> Optional[Path]:
        """结束记录并落盘，重复调用安全。"""
        if self._closed:
            return None
        self._closed = True
        return self.flush()

    def __enter__(self) -> "RuntimeRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _simulation_states(result: SimulationResult) -> list[CurlingState]:
    """由滤波轨迹重建逐帧 CurlingState（速度用相邻差分）。"""
    points = result.filtered_trajectory
    timestamps = result.timestamps
    states: list[CurlingState] = []
    for index, (point_x, point_y) in enumerate(points):
        if len(points) > 1:
            if index < len(points) - 1:
                previous, following = index, index + 1
            else:
                previous, following = index - 1, index
            dt = timestamps[following] - timestamps[previous]
            vx = 0.0 if dt <= 0.0 else (points[following][0] - points[previous][0]) / dt
            vy = 0.0 if dt <= 0.0 else (points[following][1] - points[previous][1]) / dt
        else:
            vx = vy = 0.0
        states.append(CurlingState(x=point_x, y=point_y, vx=vx, vy=vy, timestamp=timestamps[index]))
    return states


def record_simulation_result(
    result: SimulationResult,
    path: Optional[str | Path] = None,
    meta: Optional[dict[str, Any]] = None,
) -> RuntimeRecorder:
    """把一次仿真结果写成与真实运行一致的记录格式。"""
    recorder = RuntimeRecorder(path, source="simulation", meta=meta)
    states = _simulation_states(result)

    dt = 0.0
    if len(result.timestamps) > 1:
        dt = result.timestamps[1] - result.timestamps[0]
    fps = 0.0 if dt <= 0.0 else 1.0 / dt

    prediction = result.prediction
    predict_index = len(states) - 1
    source = prediction.source_state
    if isinstance(source, CurlingState):
        for index, timestamp in enumerate(result.timestamps):
            if abs(timestamp - source.timestamp) < 1e-9:
                predict_index = index
                break

    for index, state in enumerate(states):
        recorder.record_frame(
            timestamp=result.timestamps[index],
            fps=fps,
            curling_state=state,
            prediction=prediction if index == predict_index else None,
        )
    return recorder
