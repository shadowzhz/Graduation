"""运行多组算法评估实验，输出控制台摘要与 JSON 报告。

用法:
    python3 air_hockey/evaluation/run_evaluation.py
    python3 air_hockey/evaluation/run_evaluation.py --runs 8 --predict-step 30 --json report.json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_hockey.evaluation import Experiment, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="状态估计与轨迹预测算法评估")
    parser.add_argument("--runs", type=int, default=5, help="实验组数")
    parser.add_argument("--seed", type=int, default=0, help="起始随机种子")
    parser.add_argument("--predict-step", type=int, default=20, help="从第几步滤波状态发起预测，-1 表示最后一步")
    parser.add_argument("--steps", type=int, default=220, help="每组仿真步数")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="时间步长（秒）")
    parser.add_argument("--noise", type=float, default=3.0, help="视觉位置噪声标准差")
    parser.add_argument("--fixed-initial", action="store_true", help="使用固定初始条件（默认随机初始条件）")
    parser.add_argument("--json", default=None, help="JSON 报告输出路径")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        runs=args.runs,
        seed=args.seed,
        predict_step=None if args.predict_step < 0 else args.predict_step,
        steps=args.steps,
        dt=args.dt,
        position_noise=args.noise,
        random_initial=not args.fixed_initial,
    )
    report = Experiment(config).run()
    print(report.console_summary())

    if args.json:
        report.to_json(args.json)
        print(f"\nJSON 报告已写入: {args.json}")
    else:
        print("\n--- JSON ---")
        print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
