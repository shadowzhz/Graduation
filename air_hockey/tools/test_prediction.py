"""轨迹预测模块离线测试。无需摄像头和台面。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "air_hockey", ROOT / "冰壶仿真"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prediction import TrajectoryPredictor


def main():
    predictor = TrajectoryPredictor()
    trajectory = predictor.predict(
        100.0,
        200.0,
        120.0,
        -50.0,
    )
    print(f"预测点数: {len(trajectory)}")
    for i, (px, py) in enumerate(trajectory):
        print(f"[{i:02d}] x={px:.2f}, y={py:.2f}")


if __name__ == "__main__":
    main()
