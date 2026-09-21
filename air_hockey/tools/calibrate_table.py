"""球台四点透视（Homography）标定工具。

流程：
  1. 从相机抓取一帧原始图像（或使用 --image 提供的静态图片）；
  2. 依次点击球台四角：左上(TL) -> 右上(TR) -> 右下(BR) -> 左下(BL)，坐标为 raw pixel；
  3. 使用 CameraGeometry.raw_to_undistorted() 把四点转换到 undistorted pixel；
  4. cv2.getPerspectiveTransform() 求解 undistorted pixel -> table/rink 的 3x3 单应矩阵；
  5. 保存 homography_matrix / raw_points / undistorted_points 到 npz。

用法: python3 air_hockey/tools/calibrate_table.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from air_hockey import core_config as core
from air_hockey.camera import CameraConfig, CameraManager
from air_hockey.vision.geometry import CameraGeometry

# 固定的点击顺序：左上 -> 右上 -> 右下 -> 左下
CORNER_LABELS = ("TL", "TR", "BR", "BL")
RINK_CORNERS = np.array(
    [
        [core.RINK_LEFT, core.RINK_TOP],
        [core.RINK_RIGHT, core.RINK_TOP],
        [core.RINK_RIGHT, core.RINK_BOTTOM],
        [core.RINK_LEFT, core.RINK_BOTTOM],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="球台四点 Homography 标定（undistorted -> table）")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机内参标定文件")
    parser.add_argument("--output", default="calibration/table_homography.npz", help="输出单应矩阵文件")
    parser.add_argument("--backend", default="auto", choices=("auto", "gstreamer", "v4l2"))
    parser.add_argument("--device", default=None, help="摄像头设备，例如 /dev/video0")
    parser.add_argument("--image", default=None, help="改用静态图片而非相机")
    return parser.parse_args()


def resolve_output(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[2] / output
    return output


def grab_frame(args) -> np.ndarray:
    """获取一帧原始 BGR 图像。"""
    if args.image:
        image = cv2.imread(str(Path(args.image)), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"无法读取图片: {args.image}")
        return image

    camera = CameraManager(CameraConfig(device=args.device, backend=args.backend))
    try:
        camera.start(timeout=3.0)
    except Exception as exc:
        raise SystemExit(f"摄像头启动失败：{exc}") from exc
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            frame = camera.get_latest_frame()
            if frame is not None:
                return frame.image.copy()
            time.sleep(0.01)
        raise SystemExit("未能在超时前获取相机帧")
    finally:
        camera.stop()


class CornerCollector:
    """在静态帧上收集 4 个 raw pixel 角点。"""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.points = []
        self.window = "Table Homography Calibration"

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < len(CORNER_LABELS):
            self.points.append((float(x), float(y)))

    def render(self) -> np.ndarray:
        view = self.image.copy()
        for index, (px, py) in enumerate(self.points):
            center = (round(px), round(py))
            cv2.circle(view, center, 6, (0, 255, 0), -1)
            cv2.putText(view, CORNER_LABELS[index], (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(self.points) < len(CORNER_LABELS):
            tip = f"click {CORNER_LABELS[len(self.points)]}  ({len(self.points)}/4)   r: reset   q/ESC: quit"
        else:
            tip = "4 corners done   s: save   r: reset   q/ESC: quit"
        cv2.putText(view, tip, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(view, f"order: {' -> '.join(CORNER_LABELS)}", (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return view


def save_homography(geometry: CameraGeometry, raw_points, image_size, output: Path) -> None:
    """把 4 个 raw 角点转换到 undistorted 并求解 undistorted -> table 的 Homography。"""
    if image_size is None:
        raise SystemExit("缺少标定图像分辨率，无法保存 Homography")
    raw = np.asarray(raw_points, dtype=np.float64)
    undistorted = np.array(
        [geometry.raw_to_undistorted(px, py) for px, py in raw],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(undistorted, RINK_CORNERS)
    if not np.isfinite(homography).all():
        raise SystemExit("四点退化，无法求解 Homography，请重新点击四角")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        homography_matrix=homography.astype(np.float64),
        raw_points=raw,
        undistorted_points=undistorted.astype(np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
    )
    print(f"已保存: {output}")
    print(f"标定分辨率: {int(image_size[0])}x{int(image_size[1])}")
    print("homography_matrix (undistorted -> table):")
    print(homography)


def main():
    args = parse_args()
    image = grab_frame(args)
    height, width = image.shape[:2]

    geometry = CameraGeometry.from_calibration_file(
        args.calibration,
        rink_bounds=(core.RINK_LEFT, core.RINK_RIGHT, core.RINK_TOP, core.RINK_BOTTOM),
        enabled=True,
    )
    geometry.set_image_size((width, height))

    output = resolve_output(args.output)
    collector = CornerCollector(image)
    cv2.namedWindow(collector.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(collector.window, collector.on_mouse)

    print("依次点击球台四角：左上(TL) -> 右上(TR) -> 右下(BR) -> 左下(BL)")
    print("点击完成后按 s 保存，r 重置，q / ESC 退出")
    while True:
        cv2.imshow(collector.window, collector.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            collector.points.clear()
        elif key == ord("s"):
            if len(collector.points) != len(CORNER_LABELS):
                print(f"还需要点击 {len(CORNER_LABELS) - len(collector.points)} 个角点")
                continue
            save_homography(geometry, collector.points, geometry.image_size, output)
            break
        if cv2.getWindowProperty(collector.window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
