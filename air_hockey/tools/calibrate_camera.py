"""交互式棋盘格相机标定工具。

运行示例：
    python3 air_hockey/tools/calibrate_camera.py --cols 9 --rows 6 --square-size 25

窗口中：
    SPACE 采集当前棋盘格
    C     使用已采集样本执行标定并保存
    Q/ESC 退出
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# 允许直接执行 air_hockey/tools/calibrate_camera.py。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.calibration import CameraCalibrator


def parse_args():
    parser = argparse.ArgumentParser(description="棋盘格相机标定")
    parser.add_argument("--cols", type=int, default=9, help="棋盘格横向内角点数量")
    parser.add_argument("--rows", type=int, default=6, help="棋盘格纵向内角点数量")
    parser.add_argument("--square-size", type=float, default=25.0, help="棋盘格单格尺寸，单位自定")
    parser.add_argument("--device", default=None, help="摄像头设备，例如 /dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--detect-fps", type=float, default=10.0, help="棋盘格检测频率，默认 10 FPS")
    parser.add_argument("--backend", choices=("gstreamer", "v4l2", "auto"), default="gstreamer")
    parser.add_argument("--output", default="calibration/camera_calibration.npz", help="标定结果保存路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.detect_fps <= 0:
        raise ValueError("detect-fps 必须大于 0")

    calibrator = CameraCalibrator((args.cols, args.rows), args.square_size)
    config = CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        requested_fps=args.fps,
        backend=args.backend,
    )
    camera = CameraManager(config)

    print(f"棋盘格内角点：{args.cols} x {args.rows}")
    print(f"单格尺寸：{args.square_size}")
    print(f"摄像头 FPS：{args.fps:.0f}，棋盘格检测 FPS：{args.detect_fps:.1f}")
    print("SPACE=采集，C=标定并保存，Q/ESC=退出")

    detect_interval = 1.0 / args.detect_fps
    last_detect_time = 0.0
    latest_corners = None
    latest_frame = None

    try:
        camera.start()
        cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)

        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                latest_frame = frame.image.copy()
                now = time.monotonic()
                if now - last_detect_time >= detect_interval:
                    latest_corners = calibrator.detect(latest_frame)
                    last_detect_time = now

            if latest_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            image = latest_frame.copy()
            if latest_corners is not None:
                cv2.drawChessboardCorners(
                    image, calibrator.pattern_size, latest_corners, True
                )

            detected = latest_corners is not None
            cv2.putText(
                image,
                f"Samples: {calibrator.sample_count} | SPACE capture | C calibrate | Q quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if detected else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            status = "Chessboard detected" if detected else "Chessboard not detected"
            cv2.putText(
                image,
                status,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if detected else (0, 0, 255),
                2,
            )
            cv2.imshow("Camera Calibration", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if latest_corners is None:
                    print("当前画面未检测到完整棋盘格，未采集。")
                elif calibrator.add_sample(latest_frame, latest_corners):
                    print(f"已采集第 {calibrator.sample_count} 组样本。")
            elif key in (ord("c"), ord("C")):
                if calibrator.sample_count < 5:
                    print("样本不足：至少 5 组，建议采集 15~25 组。")
                    continue
                result = calibrator.calibrate()
                output = Path(args.output)
                if not output.is_absolute():
                    output = ROOT / output
                calibrator.save(result, output)
                print(
                    f"标定完成：RMS={result.rms:.6f}, "
                    f"平均重投影误差={result.reprojection_error:.6f}"
                )
                print(f"已保存：{output}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
