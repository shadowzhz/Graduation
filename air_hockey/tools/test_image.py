"""离线跑一次 StoneDetector 并输出标注图。

用法: python3 tools/test_image.py 图片路径 --output 结果图路径
"""


import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.types import Frame
from vision import ROI, StoneDetector


def _int_triplet(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three integer values")
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold values must be integers") from exc


def build_parser():
    parser = argparse.ArgumentParser(description="Run StoneDetector on one image")
    parser.add_argument("image", type=Path, help="input BGR image")
    parser.add_argument("--output", type=Path, help="annotated output image path")
    parser.add_argument("--color-space", choices=("hsv", "lab"), default="hsv")
    parser.add_argument("--lower", nargs=3, metavar=("C1", "C2", "C3"), default=(0, 40, 30), type=int)
    parser.add_argument("--upper", nargs=3, metavar=("C1", "C2", "C3"), default=(180, 255, 255), type=int)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--min-circularity", type=float, default=0.55)
    return parser


def annotate(image, detection, roi):
    """在原图上画 ROI 和检测结果。"""
    output = image.copy()
    height, width = output.shape[:2]
    if roi is not None:
        bounded = roi.clamp(width, height)
        if bounded is not None:
            cv2.rectangle(
                output,
                (bounded.x, bounded.y),
                (bounded.x + bounded.width - 1, bounded.y + bounded.height - 1),
                (255, 180, 0),
                2,
            )
    if detection is None:
        # 没检到也输出结果图，方便看阈值是不是太严
        cv2.putText(output, "No detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return output

    center = (round(detection.center_x), round(detection.center_y))
    radius = round(detection.radius)
    cv2.circle(output, center, radius, (0, 255, 0), 2)
    cv2.drawMarker(output, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    label = f"r={detection.radius:.1f} score={detection.score:.3f}"
    text_origin = (max(0, center[0] - radius), max(24, center[1] - radius - 8))
    cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return output


def main():
    args = build_parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取图像: {args.image}", file=sys.stderr)
        return 2
    roi = ROI(*args.roi) if args.roi else None
    detector = StoneDetector(
        roi=roi,
        color_space=args.color_space,
        lower=args.lower,
        upper=args.upper,
        min_area=args.min_area,
        min_radius=args.min_radius,
        min_circularity=args.min_circularity,
    )
    detection = detector.detect(Frame(image))
    output_path = args.output or args.image.with_name(f"{args.image.stem}_detected{args.image.suffix}")
    annotated = annotate(image, detection, roi)
    if not cv2.imwrite(str(output_path), annotated):
        print(f"无法写入结果图像: {output_path}", file=sys.stderr)
        return 3
    if detection is None:
        print(f"Detection: None\n结果图: {output_path}")
        return 1
    print(
        "Detection: "
        f"center=({detection.center_x:.1f}, {detection.center_y:.1f}), "
        f"radius={detection.radius:.1f}, area={detection.area:.1f}, "
        f"circularity={detection.circularity:.3f}, score={detection.score:.3f}"
    )
    print(f"结果图: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
