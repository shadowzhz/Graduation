"""Real-time StoneDetector + StoneTracker test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera import CameraConfig, CameraManager
from vision import StoneDetector
from vision.tracker import StoneTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test StoneDetector and StoneTracker with a camera."
    )

    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=(350, 0, 580, 650),
    )

    parser.add_argument(
        "--lower",
        type=int,
        nargs=3,
        default=(170, 100, 80),
    )

    parser.add_argument(
        "--upper",
        type=int,
        nargs=3,
        default=(179, 255, 255),
    )

    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--min-radius", type=float, default=25.0)
    parser.add_argument("--min-circularity", type=float, default=0.65)

    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-missed-frames", type=int, default=5)

    parser.add_argument("--duration", type=float, default=0.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = CameraManager(
        CameraConfig(
            device=args.device,
            width=1280,
            height=720,
            requested_fps=200.0,
            pixel_format="MJPG",
            backend=args.backend,
        )
    )

    detector = StoneDetector(
        roi=tuple(args.roi),
        color_space="hsv",
        lower=tuple(args.lower),
        upper=tuple(args.upper),
        min_area=args.min_area,
        min_radius=args.min_radius,
        min_circularity=args.min_circularity,
    )

    tracker = StoneTracker(
        max_distance=args.max_distance,
        max_missed_frames=args.max_missed_frames,
    )

    camera.start()

    print("Tracker test started.")
    print("Press Ctrl+C to stop.")
    print()

    start_time = time.monotonic()
    last_sequence = -1

    try:
        while True:
            frame = camera.get_latest_frame()

            if frame is None:
                time.sleep(0.001)
                continue

            # FrameBuffer 只保留最新帧，避免重复处理同一帧。
            if frame.sequence == last_sequence:
                time.sleep(0.0005)
                continue

            last_sequence = frame.sequence

            detection = detector.detect(frame)
            tracks = tracker.update(detection)

            if tracks:
                track = tracks[0]

                print(
                    "\r"
                    "ID={:<3d} "
                    "state={:<6s} "
                    "x={:7.1f} "
                    "y={:7.1f} "
                    "r={:6.1f} "
                    "vx={:8.1f} "
                    "vy={:8.1f} "
                    "age={:<5d} "
                    "miss={:<2d}"
                    .format(
                        track.track_id,
                        track.state.value,
                        track.center_x,
                        track.center_y,
                        track.radius,
                        track.vx,
                        track.vy,
                        track.age,
                        track.missed_frames,
                    ),
                    end="",
                    flush=True,
                )
            else:
                print(
                    "\rNo active track.                         ",
                    end="",
                    flush=True,
                )

            if args.duration > 0.0:
                if time.monotonic() - start_time >= args.duration:
                    break

    except KeyboardInterrupt:
        print()

    finally:
        camera.stop()

    stats = camera.get_stats()

    print()
    print()
    print("=== Camera statistics ===")
    print("Capture FPS: {:.1f}".format(stats.capture_fps))


if __name__ == "__main__":
    main()
