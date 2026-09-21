"""真实运行数据记录模块。"""

from .recorder import RuntimeRecorder, record_simulation_result
from .schema import (
    RECORDING_FORMAT,
    RECORDING_VERSION,
    ai_decision_to_dict,
    build_document,
    build_frame,
    curling_state_to_dict,
    plc_request_to_dict,
    prediction_to_dict,
)

__all__ = [
    "RECORDING_FORMAT",
    "RECORDING_VERSION",
    "RuntimeRecorder",
    "ai_decision_to_dict",
    "build_document",
    "build_frame",
    "curling_state_to_dict",
    "plc_request_to_dict",
    "prediction_to_dict",
    "record_simulation_result",
]
