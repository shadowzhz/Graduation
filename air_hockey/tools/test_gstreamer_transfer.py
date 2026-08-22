"""测量 Jetson GStreamer appsink 到 NumPy 的图像传输开销。

本工具不使用 CameraManager、GUI 或检测逻辑，用于区分以下阶段的性能：

* map: 只映射 appsink 缓冲区，不访问像素数据；
* bgrx-copy: 复制完整的四通道 BGRx 图像；
* bgr-copy: 将 BGRx 裁为三通道 BGR 并复制，与当前 Camera 层行为一致。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 GStreamer appsink 到 NumPy 的传输 FPS")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--mode",
        choices=("map", "bgrx-copy", "bgr-copy"),
        default="bgr-copy",
        help="map=仅映射；bgrx-copy=完整四通道复制；bgr-copy=复制为三通道 BGR",
    )
    return parser


def load_gst():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def pipeline_description(args: argparse.Namespace) -> str:
    return (
        f"v4l2src device={args.device} io-mode=2 ! "
        f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "appsink name=benchmark_sink drop=true max-buffers=1 sync=false"
    )


def run(args: argparse.Namespace) -> int:
    if args.duration <= 0:
        print("--duration 必须大于 0", file=sys.stderr)
        return 2
    Gst = load_gst()
    pipeline = Gst.parse_launch(pipeline_description(args))
    appsink = pipeline.get_by_name("benchmark_sink")
    if appsink is None:
        print("找不到 appsink", file=sys.stderr)
        return 1
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("GStreamer 管道无法启动", file=sys.stderr)
        return 1

    count = 0
    try:
        result, state, _pending = pipeline.get_state(5 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
            print(f"GStreamer 管道状态异常: {state.value_nick}", file=sys.stderr)
            return 1
        print(f"传输基准开始：mode={args.mode}，时长={args.duration:g} 秒")
        started_at = time.perf_counter()
        deadline = started_at + args.duration
        while time.perf_counter() < deadline:
            sample = appsink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                continue
            buffer = sample.get_buffer()
            success, mapped = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue
            try:
                if args.mode != "map":
                    raw = np.frombuffer(mapped.data, dtype=np.uint8)
                    if args.mode == "bgrx-copy":
                        image = raw.copy()
                    else:
                        image = raw.reshape((args.height, args.width, 4))[:, :, :3].copy()
                    # 防止解释器把数组计算优化为无效操作。
                    if image.size == 0:
                        raise RuntimeError("空图像")
                count += 1
            finally:
                buffer.unmap(mapped)
        elapsed = time.perf_counter() - started_at
        print(f"transfer benchmark: {count / elapsed:.1f} FPS, frames={count}, elapsed={elapsed:.1f} s")
        return 0
    finally:
        pipeline.set_state(Gst.State.NULL)


def main(argv: Optional[List[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
