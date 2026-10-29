"""运行记录格式。

所有记录统一为一个版本化 JSON 文档，帧级字段一致：
    format / version / source / meta / frames
每帧记录 timestamp、fps、curling_state、prediction、ai_decision、plc_request、plc_state。
plc_state 是接 PLC 时的轴状态快照（实际位置、最近一次下发点、busy、seq_ack 等），
其中的 age 字段表示"距记录时多久以前"，用来和帧时间对齐计算延迟。
"""

from __future__ import annotations

from typing import Any, Optional

from ..game_state import CurlingState
from ..prediction import PredictionState

RECORDING_FORMAT = "curling_recording"
# 4：2026-10-29 场地坐标改为各向同性（Y 拉长到 968 单位），与 3 及以前的坐标不可直接比较
RECORDING_VERSION = 4


def curling_state_to_dict(state: CurlingState) -> dict:
    """CurlingState -> 帧级字典。"""
    return {
        "x": float(state.x),
        "y": float(state.y),
        "vx": float(state.vx),
        "vy": float(state.vy),
        "timestamp": float(state.timestamp),
        "confidence": float(state.confidence),
        "radius": float(state.radius),
    }


def prediction_to_dict(pred: PredictionState) -> dict:
    """PredictionState -> 帧级字典。"""
    source = pred.source_state
    return {
        "trajectory": [[float(point_x), float(point_y)] for point_x, point_y in pred.trajectory],
        "endpoint": [float(pred.endpoint[0]), float(pred.endpoint[1])],
        "duration": float(pred.duration),
        "source_state": curling_state_to_dict(source) if isinstance(source, CurlingState) else None,
    }


def ai_decision_to_dict(decision) -> Optional[dict]:
    """AIDecision -> 帧级字典；None 时返回 None。"""
    if decision is None:
        return None
    return {
        "target_x": float(decision.target_x),
        "target_y": float(decision.target_y),
        "stalled_stone_phase": str(getattr(decision, "stalled_stone_phase", "")),
        "reaction_timer": float(getattr(decision, "reaction_timer", 0.0)),
    }


def plc_request_to_dict(request) -> Optional[dict]:
    """PLC 半场目标 -> 帧级字典（含 timestamp）；None 时返回 None。"""
    if request is None:
        return None
    return request.to_dict()


def build_frame(
    index: int,
    timestamp: float,
    fps: float,
    curling_state: Optional[CurlingState] = None,
    prediction: Optional[PredictionState] = None,
    ai_decision=None,
    plc_request=None,
    plc_state: Optional[dict] = None,
) -> dict:
    """构建单帧记录。"""
    return {
        "index": int(index),
        "timestamp": float(timestamp),
        "fps": float(fps),
        "curling_state": None if curling_state is None else curling_state_to_dict(curling_state),
        "prediction": None if prediction is None else prediction_to_dict(prediction),
        "ai_decision": ai_decision_to_dict(ai_decision),
        "plc_request": plc_request_to_dict(plc_request),
        "plc_state": None if plc_state is None else dict(plc_state),
    }


def build_document(source: str, frames: list[dict], meta: Optional[dict[str, Any]] = None) -> dict:
    """构建完整记录文档。"""
    return {
        "format": RECORDING_FORMAT,
        "version": RECORDING_VERSION,
        "source": source,
        "meta": dict(meta or {}),
        "frames": list(frames),
    }
