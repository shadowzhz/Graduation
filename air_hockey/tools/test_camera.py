"""纯摄像头采集 FPS 基准，不开 GUI 也不做检测。

用法: python3 air_hockey/tools/test_camera.py --benchmark --duration 10
"""


import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from air_hockey.camera import CameraConfig, CameraManager


def build_parser():
    parser = argparse.ArgumentParser(description="测试纯摄像头采集 FPS，不启动 GUI 和检测")
    parser.add_argument("--benchmark", action="store_true", help="运行 FPS 基准测试")
    parser.add_argument("--device", default=None, help="Linux 摄像头路径或 Windows 摄像头编号，例如 0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=200.0, dest="requested_fps")
    parser.add_argument("--duration", type=float, default=10.0, help="测试时长，单位为秒")
    return parser


def report_samples(name, samples):
    if not samples:
        print(f"{name}: 无样本")
        return
    ordered = sorted(samples)

    def percentile(p):
        position = (len(ordered) - 1) * p
        lower = int(position)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)

    print(
        f"{name}: n={len(ordered)} min={ordered[0]:.3f} "
        f"p50={percentile(0.5):.3f} p95={percentile(0.95):.3f} max={ordered[-1]:.3f} ms"
    )


def run_benchmark(args):
    if args.duration <= 0:
        print("--duration 必须大于 0", file=sys.stderr)
        return 2

    config = CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        requested_fps=args.requested_fps,
    )
    camera = CameraManager(config)
    try:
        camera.start(timeout=3.0)
        info = camera.info
        requested_mode = (
            f"{config.width}x{config.height} @ {config.requested_fps:g} FPS，MJPG"
            if sys.platform == "win32"
            else f"{config.width}x{config.height} @ {config.requested_fps:g} FPS"
        )
        print(
            f"基准测试开始：设备={info.device if info else config.device}，"
            f"后端={info.backend if info else '未知'}，"
            f"请求模式={requested_mode}，"
            f"模式={info.width if info else '?'}x{info.height if info else '?'}，"
            f"驱动报告 FPS={info.negotiated_fps if info else 0:.2f}，"
            f"输入格式={info.source_format if info else '?'}"
        )
        read_samples = []
        convert_samples = []
        last_sequence = -1
        last_frame = None
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            if camera.error is not None:
                raise RuntimeError(f"摄像头采集失败：{camera.error}") from camera.error
            frame = camera.get_latest_frame()
            if frame is not None and frame.sequence != last_sequence:
                last_sequence = frame.sequence
                last_frame = frame
                read_samples.append(frame.capture_read_ms)
                convert_samples.append(frame.color_convert_ms)
            time.sleep(0.001)
        if camera.error is not None:
            raise RuntimeError(f"摄像头采集失败：{camera.error}") from camera.error
        stats = camera.get_stats()
        print(
            f"实际采集 FPS={stats.average_fps:.1f}（采集线程记录的 {stats.frame_count} 帧 / "
            f"{stats.elapsed:.2f} s；请求/协商 FPS 不是实测采集 FPS）"
        )
        print(f"阶段耗时来自读取最新帧的去重样本 {len(read_samples)} 帧；覆盖不保证每一采集帧。")
        report_samples("后端读取总耗时", read_samples)
        report_samples("后端单独报告的颜色转换耗时", convert_samples)
        if last_frame is not None:
            pts = (f"Gst PTS={last_frame.gst_pts_ns} ns (原始管道时钟域)"
                   if last_frame.gst_pts_ns is not None else "此后端不提供 Gst PTS")
            print(f"最近样本 Frame.timestamp={last_frame.timestamp:.6f} (host perf_counter，后端读取后)，{pts}")
        print("读取耗时来自后端 read；解码和颜色处理未必能单独计时，不包含传感器曝光前的耗时。")
        print("Gst PTS 未映射主机时钟；缺少该值时不能据此推算曝光/端到端延迟。")
        return 0
    except Exception as exc:
        print(f"摄像头基准测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.stop()


def main():
    args = build_parser().parse_args()
    if not args.benchmark:
        print("请添加 --benchmark 运行纯摄像头 FPS 测试")
        return 2
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
