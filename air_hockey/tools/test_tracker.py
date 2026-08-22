"""Real-time StoneDetector + StoneTracker test with visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

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
        "--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        default=(350, 0, 580, 650),
    )
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80))
    parser.add_argument("--upper", type=int, nargs=3, default=(179, 255, 255))
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--min-radius", type=float, default=25.0)
    parser.add_argument("--min-circularity", type=float, default=0.65)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--max-missed-frames", type=int, default=5)
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def draw_overlay(image, roi, detection, track, fps):
    x, y, w, h = roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    if detection is not None:
        cx = int(round(detection.center_x))
        cy = int(round(detection.center_y))
        radius = max(1, int(round(detection.radius)))
        cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(
            image, "DETECTION", (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    if track is not None:
        cx = int(round(track.center_x))
        cy = int(round(track.center_y))
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"ID={track.track_id} {track.state.value} "
            f"v=({track.vx:.0f},{track.vy:.0f}) px/s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            image, "NO TRACK", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

    cv2.putText(
        image, f"FPS: {fps:.1f}", (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        image, "Q / ESC: quit", (20, image.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


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
    print("Tracker visualization started.")
    print("Press Q or ESC to stop.")

    start_time = time.monotonic()
    last_sequence = -1
    display_fps = 0.0
    last_time = time.monotonic()

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            if frame.sequence == last_sequence:
                time.sleep(0.0005)
                continue
            last_sequence = frame.sequence

            now = time.monotonic()
            dt = now - last_time
            if dt > 0.0:
                instant_fps = 1.0 / dt
                display_fps = (
                    instant_fps if display_fps == 0.0
                    else display_fps * 0.9 + instant_fps * 0.1
                )
            last_time = now

            detection = detector.detect(frame)
            tracks = tracker.update(detection)
            track = tracks[0] if tracks else None

            image = frame.image.copy()
            draw_overlay(image, tuple(args.roi), detection, track, display_fps)
            cv2.imshow("Stone Tracker", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if args.duration > 0.0:
                if time.monotonic() - start_time >= args.duration:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    stats = camera.get_stats()
    print()
    print("=== Camera statistics ===")
    print("Capture FPS: {:.1f}".format(stats.capture_fps))


if __name__ == "__main__":
    main()
