"""真实运行数据记录。

RuntimeRecorder 从实时 VisionResult 中抽取 CurlingState / PredictionState / timestamp / FPS，
逐帧同步日志，结束时流式写出统一 JSON；异常退出的日志可恢复。
未设置输出路径时仅在内存中记录，供有限长度的离线仿真使用。
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Optional

from game_state import CurlingState

from ..simulation import SimulationResult
from .schema import RECORDING_FORMAT, RECORDING_VERSION, build_document, build_frame


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_document(journal, path: Path, indent: int) -> int:
    header = json.loads(journal.readline())
    if (set(header) != {"format", "version", "source", "meta"}
            or header["format"] != RECORDING_FORMAT or header["version"] != RECORDING_VERSION):
        raise ValueError("invalid recording journal header")
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(header, ensure_ascii=False, indent=indent)[:-1] + ',"frames":[\n')
        for line in journal:
            if not line.endswith("\n"):
                break  # 崩溃只可能留下最后一帧的残片；完整行中的错误必须报出。
            frame = json.loads(line)
            if frame["index"] != count:
                raise ValueError("recording journal frame index is out of order")
            if count:
                output.write(",\n")
            output.write(line.rstrip("\n"))
            count += 1
        output.write("\n]}\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)
    return count


class RuntimeRecorder:
    """单写者记录器：有路径时逐帧落盘，内存不随运行时长增长。"""

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
        self._frame_count = 0
        self._closed = False
        self._error = None
        self._journal = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            journal_path = self.path.with_name(self.path.name + ".journal")
            self._journal = journal_path.open("x", encoding="utf-8")
            try:
                fcntl.flock(self._journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
                header = build_document(self.source, [], self.meta)
                del header["frames"]
                self._journal.write(json.dumps(header, ensure_ascii=False) + "\n")
                self._journal.flush()
                os.fsync(self._journal.fileno())
                _sync_directory(self.path.parent)
            except Exception:
                self._journal.close()
                raise

    @property
    def frame_count(self) -> int:
        return self._frame_count

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
        if self._error is not None:
            raise RuntimeError("journal write failed; close or recover before recording") from self._error
        frame = build_frame(
            self._frame_count,
            timestamp,
            fps,
            curling_state,
            prediction,
            ai_decision,
            plc_request,
        )
        if self._journal is None:
            self._frames.append(frame)
        else:
            line = json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"
            try:
                self._journal.write(line)
                self._journal.flush()
                os.fsync(self._journal.fileno())
            except OSError as exc:
                self._error = exc
                raise
        self._frame_count += 1

    def to_dict(self) -> dict:
        """显式导出完整文档时才把磁盘帧读入内存。"""
        if self.path is not None:
            if self._closed:
                return json.loads(self.path.read_text(encoding="utf-8"))
            with Path(self._journal.name).open(encoding="utf-8") as journal:
                document = json.loads(journal.readline())
                document["frames"] = [json.loads(line) for line in journal if line.endswith("\n")]
                return document
        return build_document(self.source, self._frames, self.meta)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def flush(self, indent: int = 2) -> Optional[Path]:
        """流式导出快照（原子替换），保留日志供继续记录或失败恢复。"""
        if self.path is None:
            return None
        if not self._closed:
            self._journal.flush()
            os.fsync(self._journal.fileno())
            with Path(self._journal.name).open(encoding="utf-8") as journal:
                self._frame_count = _write_document(journal, self.path, indent)
        return self.path

    def close(self) -> Optional[Path]:
        """结束记录并落盘，重复调用安全。"""
        if self._closed:
            return None
        path = self.flush()
        if self._journal is not None:
            self._journal.close()
        self._closed = True
        if self._journal is not None:
            Path(self._journal.name).unlink()
            _sync_directory(self.path.parent)
        return path

    @classmethod
    def recover(cls, path: str | Path) -> Path:
        """恢复已退出进程的日志；拒绝恢复仍由其他记录器持有的文件。"""
        path = Path(path)
        journal_path = path.with_name(path.name + ".journal")
        with journal_path.open(encoding="utf-8") as journal:
            fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_document(journal, path, 2)
            journal_path.unlink()
            _sync_directory(path.parent)
        return path

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
