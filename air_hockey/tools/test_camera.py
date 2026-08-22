"""无 GUI、无检测的摄像头采集 FPS 基准工具。

示例::

    python3 tools/test_camera.py --benchmark
    python3 tools/test_camera.py --benchmark --backend v4l2
    python3 tools/test_camera.py --benchmark --backend gstreamer --duration 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# 允许从项目根目录直接运行本文件。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera import CameraConfig, CameraManager  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试纯摄像头采集 FPS，不启动 GUI 和检测")
    parser.add_argument("--benchmark", action="store_true", help="运行 FPS 基准测试")
    parser.add_argument(
        "--backend",
        choices=("auto", "gstreamer", "v4l2"),
        default="auto",
        help="选择摄像头后端，默认自动尝试 GStreamer 后再尝试 V4L2",
    )
    parser.add_argument("--device", default=None, help="摄像头设备，例如 /dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=200.0, dest="requested_fps")
    parser.add_argument("--pixel-format", choices=("MJPG", "YUYV", "YUY2"), default="MJPG")
    parser.add_argument("--duration", type=float, default=10.0, help="测试时长，单位为秒")
    return parser


def run_benchmark(args: argparse.Namespace) -> int:
    if args.duration <= 0:
        print("--duration 必须大于 0", file=sys.stderr)
        return 2

    config = CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        requested_fps=args.requested_fps,
        pixel_format=args.pixel_format,
        backend=args.backend,
    )
    camera = CameraManager(config)
    try:
        camera.start(timeout=3.0)
        info = camera.info
        print(
            f"基准测试开始：后端={info.backend if info else '未知'}，"
            f"模式={info.width if info else '?'}x{info.height if info else '?'}，"
            f"协商 FPS={info.negotiated_fps if info else 0:.2f}，"
            f"输入格式={info.source_format if info else '?'}"
        )
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            time.sleep(0.25)
        stats = camera.get_stats()
        print(
            f"benchmark: current={stats.current_fps:.1f} FPS, "
            f"average={stats.average_fps:.1f} FPS, "
            f"max={stats.max_fps:.1f} FPS, frames={stats.frame_count}, "
            f"elapsed={stats.elapsed:.1f} s"
        )
        return 0
    except Exception as exc:
        print(f"摄像头基准测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.stop()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.benchmark:
        print("请添加 --benchmark 运行纯摄像头 FPS 测试")
        return 2
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
