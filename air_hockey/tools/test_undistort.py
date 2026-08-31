"""畸变校正测试工具。

无需连接摄像头，可使用已有图片验证标定文件。
"""

import argparse
from pathlib import Path

import cv2

from calibration import Undistorter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--calibration", default="camera_calibration.npz")
    parser.add_argument("--output", default="undistorted.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("无法读取图片")

    undistorter = Undistorter(args.calibration)
    result = undistorter.apply_image(image)

    cv2.imwrite(args.output, result)
    print(f"输出: {Path(args.output)}")


if __name__ == "__main__":
    main()
