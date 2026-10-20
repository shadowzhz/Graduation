"""实时运行的性能统计：阶段耗时样本、[延迟] 分段汇总与 headless 性能报告。"""

import math
import threading
from collections import deque


class LatencySamples:
    """保留最近的阶段样本；全部时间均为主机单调时钟，不代表曝光到执行器延迟。"""

    def __init__(self, window=4096):
        self._lock = threading.Lock()
        self._values = {}
        self.window = window

    def add(self, stage, elapsed_ms):
        if elapsed_ms is None:
            return
        with self._lock:
            self._values.setdefault(stage, deque(maxlen=self.window)).append(float(elapsed_ms))

    @staticmethod
    def distribution(values):
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "max": ordered[-1],
        }

    def snapshot(self):
        with self._lock:
            return {key: self.distribution(values) for key, values in self._values.items()}


def print_latency_report(decision_ages, plc):
    """[延迟] 分段延迟汇总：视觉段 + PC→PLC→轴 段（见 air_hockey/control/latency.py）。"""
    ages = sorted(decision_ages)
    decision_ages.clear()
    if ages:
        vision = (f"视觉 帧读出→目标交给PLC线程 p50 {ages[len(ages) // 2]:.1f}ms "
                  f"p95 {ages[min(len(ages) - 1, int(len(ages) * 0.95))]:.1f}ms")
    else:
        vision = "视觉 本窗口无目标"
    if plc is None:
        print(f"[延迟] {vision}")
        return
    text, hints = plc.latency_report()
    print(f"[延迟] {vision} | {text}")
    for hint in hints:
        print(f"[延迟] [!] {hint}")


def build_perf_report(camera, runtime, samples, skipped_frames, gst_pts_count):
    """退出时的性能报告（--perf-json）。"""
    info = camera.info
    return {
        "time_reference": "Host perf_counter: frame timestamp is after backend read, not sensor exposure; raw Gst PTS (when present) has a different clock domain",
        "sample_scope": "Stage distributions cover the latest processed frames only; capture FPS counts all reads. PLC write completion is not motor acknowledgement.",
        "camera": None if info is None else {
            "device": info.device,
            "backend": info.backend,
            "width": info.width,
            "height": info.height,
            "requested_fps": info.requested_fps,
            "negotiated_fps": info.negotiated_fps,
            "source_format": info.source_format,
            "output_format": info.output_format,
        },
        "capture": vars(camera.get_stats()),
        "processed_frames": runtime.frame_index,
        "skipped_sequence_frames": skipped_frames,
        "raw_gst_pts_samples": gst_pts_count,
        "sample_window": samples.window,
        "stages_ms": samples.snapshot(),
    }
