"""棋盘格相机自动标定工具。"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.calibration import CameraCalibrator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cols', type=int, default=10)
    p.add_argument('--rows', type=int, default=7)
    p.add_argument('--square-size', type=float, default=25)
    p.add_argument('--fps', type=float, default=200)
    p.add_argument('--detect-fps', type=float, default=5)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--auto', action='store_true')
    p.add_argument('--output', default='calibration/camera_calibration.npz')
    return p.parse_args()


def main():
    args = parse_args()
    calibrator = CameraCalibrator((args.cols, args.rows), args.square_size)
    camera = CameraManager(CameraConfig(requested_fps=args.fps))

    print(f'棋盘格内角点: {args.cols} x {args.rows}')
    print(f'目标样本: {args.samples}')

    last_detect = 0
    corners = None
    frame_img = None
    last_center = None

    camera.start()
    try:
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                frame_img = frame.image.copy()

            if frame_img is None:
                continue

            now = time.monotonic()
            if now - last_detect > 1 / args.detect_fps:
                corners = calibrator.detect(frame_img)
                last_detect = now

            view = frame_img.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, calibrator.pattern_size, corners, True)

            cv2.putText(view, f'Samples {calibrator.sample_count}/{args.samples}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Calibration', view)

            if corners is not None and (args.auto or True):
                center = corners.mean(axis=0)[0]
                if last_center is None or ((center-last_center)**2).sum() > 500:
                    if calibrator.add_sample(frame_img, corners):
                        last_center = center
                        print(f'采集 {calibrator.sample_count}/{args.samples}')

            if calibrator.sample_count >= args.samples:
                result = calibrator.calibrate()
                output = Path(args.output)
                if not output.is_absolute():
                    output = ROOT / output
                calibrator.save(result, output)
                print(f'完成 RMS={result.rms}, error={result.reprojection_error}')
                break

            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
