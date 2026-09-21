"""无摄像头运行运动轨迹仿真，输出真实/观测/滤波轨迹与预测终点。

用法:
    python3 air_hockey/simulation/run_simulation.py
    python3 air_hockey/simulation/run_simulation.py --seed 3 --steps 220 --noise 4.0 --predict-step 20
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_hockey.simulation import MotionSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="无摄像头运动轨迹仿真：真实->观测->Kalman->预测")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（观测噪声）")
    parser.add_argument("--steps", type=int, default=220, help="仿真步数")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="时间步长（秒）")
    parser.add_argument("--noise", type=float, default=3.0, help="视觉位置噪声标准差")
    parser.add_argument("--predict-step", type=int, default=-1, help="从第几步滤波状态发起预测，-1 表示最后一步")
    parser.add_argument("--stride", type=int, default=20, help="轨迹打印采样间隔")
    parser.add_argument("--random", action="store_true", help="随机生成初始位置与速度")
    return parser


def print_trajectory(name: str, points, stride: int) -> None:
    sampled = points[::stride]
    body = "  ".join(f"({x:7.1f},{y:7.1f})" for x, y in sampled)
    print(f"{name}({len(points)}):")
    print(f"  {body}")


def main() -> int:
    args = build_parser().parse_args()
    overrides = {"steps": args.steps, "dt": args.dt, "position_noise": args.noise}
    if args.random:
        simulator = MotionSimulator.random(seed=args.seed, **overrides)
    else:
        simulator = MotionSimulator.default(seed=args.seed, **overrides)

    predict_step = None if args.predict_step < 0 else args.predict_step
    result = simulator.simulate(predict_step=predict_step)

    config = simulator.config
    print("=== 仿真配置 ===")
    print(
        f"初始位置 ({config.initial_x:.1f}, {config.initial_y:.1f}) | "
        f"初始速度 ({config.initial_vx:.1f}, {config.initial_vy:.1f}) | "
        f"dt {config.dt:.4f}s | steps {config.steps} | noise {config.position_noise:.1f}"
    )
    print("\n=== 轨迹输出 ===")
    print_trajectory("真实轨迹", result.true_trajectory, args.stride)
    print_trajectory("观测轨迹", result.observed_trajectory, args.stride)
    print_trajectory("滤波轨迹", result.filtered_trajectory, args.stride)
    print("\n=== 预测终点 ===")
    print(f"预测终点 ({result.predicted_endpoint[0]:.1f}, {result.predicted_endpoint[1]:.1f})")
    print(f"真实终点 ({result.true_endpoint[0]:.1f}, {result.true_endpoint[1]:.1f})")
    print(f"\n{result.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
