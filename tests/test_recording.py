"""真实运行数据记录测试。

覆盖：
- 统一格式的字段序列化（CurlingState / PredictionState / AIDecision / PLC request）
- RuntimeRecorder 记录 VisionResult 并写出 JSON
- 仿真数据写入同一格式（真实/仿真格式统一）
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from air_hockey.ai import AIDecision
from air_hockey.camera.types import Frame
from air_hockey.app.vision_runtime import VisionResult, VisionRuntime
from air_hockey.control import PlcWriteRequest
from air_hockey.prediction import PredictionState
from air_hockey.recording import (
    RECORDING_FORMAT,
    RECORDING_VERSION,
    RuntimeRecorder,
    ai_decision_to_dict,
    build_frame,
    curling_state_to_dict,
    plc_request_to_dict,
    prediction_to_dict,
    record_simulation_result,
)
from air_hockey.simulation import MotionSimulator
from air_hockey.vision.types import Detection
from game_state import CurlingState

CALIB_FILE = Path(__file__).resolve().parents[1] / "calibration" / "camera_calibration.npz"

DOCUMENT_KEYS = {"format", "version", "source", "meta", "frames"}
FRAME_KEYS = {"index", "timestamp", "fps", "curling_state", "prediction", "ai_decision", "plc_request"}
STATE_KEYS = {"x", "y", "vx", "vy", "timestamp", "confidence", "radius"}
PREDICTION_KEYS = {"trajectory", "endpoint", "duration", "source_state"}
DECISION_KEYS = {"target_x", "target_y", "stalled_stone_phase", "reaction_timer"}


def _curling_state():
    return CurlingState(x=100.0, y=200.0, vx=10.0, vy=-20.0, timestamp=1.0, confidence=0.9, radius=14.0)


def _prediction(source=None):
    return PredictionState(
        trajectory=[(100.0, 200.0), (110.0, 180.0)],
        endpoint=(110.0, 180.0),
        duration=0.5,
        source_state=source if source is not None else _curling_state(),
    )


def test_curling_state_to_dict_fields():
    payload = curling_state_to_dict(_curling_state())
    assert set(payload.keys()) == STATE_KEYS
    assert payload["x"] == 100.0 and payload["y"] == 200.0
    assert payload["confidence"] == 0.9 and payload["radius"] == 14.0


def test_prediction_to_dict_fields():
    payload = prediction_to_dict(_prediction())
    assert set(payload.keys()) == PREDICTION_KEYS
    assert payload["trajectory"] == [[100.0, 200.0], [110.0, 180.0]]
    assert payload["endpoint"] == [110.0, 180.0]
    assert payload["duration"] == 0.5
    assert payload["source_state"]["x"] == 100.0


def _decision():
    return AIDecision(target_x=120.0, target_y=340.0, stalled_stone_phase="idle", reaction_timer=0.02)


def _plc_request():
    return PlcWriteRequest(
        ai_target_x=120.0, ai_target_y=340.0, ai_x=300.0, ai_y=600.0,
        stone_x=100.0, stone_y=200.0, stone_vx=10.0, stone_vy=-20.0,
        player_score=1, ai_score=2, timestamp=3.5,
    )


def test_ai_decision_to_dict_fields():
    payload = ai_decision_to_dict(_decision())
    assert set(payload.keys()) == DECISION_KEYS
    assert payload["target_x"] == 120.0 and payload["target_y"] == 340.0
    assert payload["stalled_stone_phase"] == "idle"
    assert payload["reaction_timer"] == 0.02
    assert ai_decision_to_dict(None) is None


def test_plc_request_to_dict_includes_timestamp():
    payload = plc_request_to_dict(_plc_request())
    assert payload["ai_target_x"] == 120.0
    assert payload["ai_target_y"] == 340.0
    assert payload["timestamp"] == 3.5
    assert plc_request_to_dict(None) is None


def test_build_frame_includes_control_outputs():
    frame = build_frame(0, 1.0, 60.0, _curling_state(), _prediction(), _decision(), _plc_request())
    assert set(frame.keys()) == FRAME_KEYS
    assert frame["ai_decision"]["target_x"] == 120.0
    assert frame["plc_request"]["timestamp"] == 3.5


def test_recorder_records_control_outputs_and_defaults_to_none():
    recorder = RuntimeRecorder(None)
    recorder.record_frame(0.0, 30.0, _curling_state(), _prediction(), _decision(), _plc_request())
    recorder.record_frame(0.1, 30.0, _curling_state())

    frames = recorder.to_dict()["frames"]
    assert set(frames[0].keys()) == FRAME_KEYS
    assert frames[0]["ai_decision"]["stalled_stone_phase"] == "idle"
    assert frames[0]["plc_request"]["ai_target_y"] == 340.0
    assert frames[1]["ai_decision"] is None
    assert frames[1]["plc_request"] is None


def test_recorder_record_accepts_control_outputs():
    result = VisionResult(
        frame=Frame(image=None, timestamp=2.5, sequence=7),
        curling_state=_curling_state(),
        prediction=_prediction(),
        fps=58.0,
    )
    recorder = RuntimeRecorder(None)
    recorder.record(result, ai_decision=_decision(), plc_request=_plc_request())
    frame = recorder.to_dict()["frames"][0]
    assert frame["ai_decision"]["target_y"] == 340.0
    assert frame["plc_request"]["ai_x"] == 300.0


def test_build_frame_shape_with_empty_values():
    frame = build_frame(3, 1.5, 59.9)
    assert set(frame.keys()) == FRAME_KEYS
    assert frame["index"] == 3
    assert frame["curling_state"] is None and frame["prediction"] is None


def test_recorder_records_vision_result_and_writes_json():
    result = VisionResult(
        frame=Frame(image=None, timestamp=2.5, sequence=7),
        curling_state=_curling_state(),
        prediction=_prediction(),
        fps=58.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        with RuntimeRecorder(path, meta={"mode": "headless"}) as recorder:
            recorder.record(result)
            assert recorder.frame_count == 1
        assert recorder.closed

        document = json.loads(path.read_text(encoding="utf-8"))
        assert set(document.keys()) == DOCUMENT_KEYS
        assert document["format"] == RECORDING_FORMAT
        assert document["version"] == RECORDING_VERSION
        assert document["source"] == "runtime"
        assert document["meta"] == {"mode": "headless"}

        frame = document["frames"][0]
        assert set(frame.keys()) == FRAME_KEYS
        assert frame["timestamp"] == 2.5
        assert frame["fps"] == 58.0
        assert frame["curling_state"]["x"] == 100.0
        assert frame["prediction"]["endpoint"] == [110.0, 180.0]


def test_recorder_manual_frames_and_close_is_idempotent():
    recorder = RuntimeRecorder(None)
    recorder.record_frame(0.0, 30.0, _curling_state())
    recorder.record_frame(0.1, 30.0, None, _prediction())
    assert recorder.frame_count == 2

    document = recorder.to_dict()
    assert document["frames"][0]["prediction"] is None
    assert document["frames"][1]["curling_state"] is None

    recorder.close()
    recorder.close()  # 重复 close 安全
    assert recorder.closed

    try:
        recorder.record_frame(0.2, 30.0)
        assert False, "closed 后记录应抛 RuntimeError"
    except RuntimeError:
        pass


def test_empty_recorder_serializes_valid_document():
    recorder = RuntimeRecorder(None, source="runtime")
    document = json.loads(recorder.to_json())
    assert set(document.keys()) == DOCUMENT_KEYS
    assert document["frames"] == []


def test_simulation_recording_matches_runtime_format():
    simulation = MotionSimulator.default(steps=60).simulate(predict_step=-1)
    recorder = record_simulation_result(simulation)
    simulation_document = recorder.to_dict()

    runtime_document = RuntimeRecorder(None).to_dict()

    assert set(simulation_document.keys()) == set(runtime_document.keys()) == DOCUMENT_KEYS
    assert simulation_document["source"] == "simulation"
    assert simulation_document["version"] == runtime_document["version"] == RECORDING_VERSION
    assert len(simulation_document["frames"]) == len(simulation.timestamps)

    for frame in simulation_document["frames"]:
        assert set(frame.keys()) == FRAME_KEYS
        assert set(frame["curling_state"].keys()) == STATE_KEYS
        if frame["prediction"] is not None:
            assert set(frame["prediction"].keys()) == PREDICTION_KEYS


def test_simulation_recording_attaches_prediction_at_source_frame():
    simulation = MotionSimulator.default(steps=60).simulate(predict_step=10)
    document = record_simulation_result(simulation).to_dict()
    with_prediction = [frame["index"] for frame in document["frames"] if frame["prediction"] is not None]
    assert with_prediction == [10]


def test_simulation_recording_writes_file():
    simulation = MotionSimulator.default(steps=40).simulate()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sim.json"
        recorder = record_simulation_result(simulation, path)
        recorder.close()
        document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document.keys()) == DOCUMENT_KEYS
    assert document["source"] == "simulation"
    assert len(document["frames"]) > 0


class _MockDetector:
    def __init__(self):
        self._count = 0

    def detect(self, frame, dynamic_roi=None):
        self._count += 1
        return Detection(
            center_x=500.0 + 6.0 * self._count,
            center_y=300.0 + 3.0 * self._count,
            radius=25.0,
            area=1960.0,
            timestamp=frame.timestamp,
            score=0.8,
        )


def test_runtime_pipeline_records_in_unified_format():
    """真实运行链路（VisionRuntime -> recorder）产出的记录与仿真格式一致。"""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    runtime = VisionRuntime(
        table_roi=(350, 0, 580, 650),
        calibration_file=str(CALIB_FILE),
        table_calibration_file=None,
        disable_undistort=True,
        detector=_MockDetector(),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        with RuntimeRecorder(path, source="runtime") as recorder:
            for index in range(9):
                result = runtime.process_frame(
                    Frame(image=image, timestamp=1.0 + 0.033 * index, sequence=index + 1)
                )
                recorder.record(result)
        assert recorder.frame_count == 9
        document = json.loads(path.read_text(encoding="utf-8"))

    assert set(document.keys()) == DOCUMENT_KEYS
    assert document["source"] == "runtime"
    assert len(document["frames"]) == 9
    for frame in document["frames"]:
        assert set(frame.keys()) == FRAME_KEYS
        assert set(frame["curling_state"].keys()) == STATE_KEYS
        assert set(frame["prediction"].keys()) == PREDICTION_KEYS
        assert frame["curling_state"]["radius"] == 25.0
        assert frame["curling_state"]["confidence"] == 0.8

