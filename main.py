"""项目统一入口：解析命令行并分发到实时视觉回合或仿真游戏。

用法：
    python main.py                      # 默认：实时视觉演示（摄像头 -> 检测 -> 追踪 -> AI）
    python main.py --headless           # 视觉模式：无显示性能基准测试
    python main.py --sim                # 仿真模式：启动冰壶仿真游戏（也可写作 --game）

实时运行的线程编排见 air_hockey/app/live.py。
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # 找到 main.py 所在文件夹
SIM_ROOT = PROJECT_ROOT / "sim_gui"              # 仿真代码（仅 --sim 子进程使用）

# Windows 控制台默认 GBK，打印编不出的字符（² ³ 等）会抛 UnicodeEncodeError 打断程序；
# 改成 replace，编不出的字符显示成 "?"。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  # 被重定向到不支持 reconfigure 的对象：忽略

from air_hockey.app.live import run_vision
from air_hockey.control.limits import (
    DEFAULT_MOTION_MARGIN_MM, TRAVEL_X_MAX, TRAVEL_X_MIN, TRAVEL_Y_MAX, TRAVEL_Y_MIN,
)


def run_game(plc_ip=None, plc_rate=30.0):
    """启动仿真子进程，显式设置 cwd 避免相对路径资源加载报错。"""
    command = [sys.executable, str(SIM_ROOT / "game.py")]
    if plc_ip:
        command.extend(("--plc", plc_ip, "--plc-rate", str(plc_rate)))
    raise SystemExit(subprocess.run(command, cwd=str(SIM_ROOT)).returncode)


def build_parser():
    parser = argparse.ArgumentParser(description="空气冰壶项目入口")

    # 模式开关：默认运行视觉模式；传入 --sim / --game 运行仿真游戏
    parser.add_argument("--sim", "--game", dest="sim", action="store_true", help="启动冰壶仿真游戏（默认运行实时视觉演示）")

    # 视觉模式相关参数
    parser.add_argument("--headless", action="store_true", help="无显示性能测试模式")
    parser.add_argument("--benchmark-seconds", type=float, default=0.0, metavar="SEC", help="headless 模式自动停止时间；默认持续运行")
    parser.add_argument("--perf-json", default=None, metavar="PATH", help="退出时保存最近 4096 样本的阶段耗时分布")
    parser.add_argument("--record", default=None, metavar="PATH", help="记录运行数据到 JSON 文件（CurlingState/PredictionState/timestamp/FPS）")
    parser.add_argument("--dry-run", action="store_true", help="完整回合影子运行：显示/记录请求目标，不发送 PLC 运动或现场轴命令")
    parser.add_argument("--plc", default=None, metavar="IP", help="连接已配置 DB1/DB18 的 S7-1200（1211C）；轴需现场主动使能")
    parser.add_argument("--plc-rate", type=float, default=40.0, help="PLC 通信周期频率（Hz，默认 40）")
    parser.add_argument("--mallet-speed", type=float, default=500.0, metavar="UNITS_PER_SEC",
                        help="球槌可达速度估计（球台单位/秒，默认 500）。用来算提前量："
                             "目标瞄冰壶再走 lead_time 之后的位置，而不是它现在的位置 —— "
                             "不然轴总是慢一拍、球走了才追到。设 0 关闭提前量。"
                             "这不是 PLC 轴速度设置，需要在现场实测后调准")
    parser.add_argument("--motion-margin-mm", type=float, default=DEFAULT_MOTION_MARGIN_MM, metavar="MM",
                        help=f"命令余量：下发目标离机械行程端点（硬件限位开关）再内缩多少毫米，"
                             f"默认 {DEFAULT_MOTION_MARGIN_MM:g}（冲到边上可能碰限位开关停轴，频繁发生就调大）。"
                             f"机械行程 X {TRAVEL_X_MIN:g}..{TRAVEL_X_MAX:g} / Y {TRAVEL_Y_MIN:g}..{TRAVEL_Y_MAX:g}，"
                             "包络 = 行程内缩它，X/Y 另有固定附加余量。程序不另设软件行程保护")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--camera-device", default=None, help="相机路径（如 /dev/video2）；不给则依次尝试 /dev/video*")
    parser.add_argument("--calibration", default="calibration/camera_calibration.npz", help="相机内参标定文件")
    parser.add_argument("--table-calibration", default="calibration/table_homography.npz", help="球台四点 Homography 标定文件")
    parser.add_argument("--disable-undistort", action="store_true", help="关闭相机畸变校正（必须同时 --disable-homography）")
    parser.add_argument("--disable-homography", action="store_true", help="关闭 Homography，回退到旧 ROI 线性映射（调试用）")
    parser.add_argument("--roi", type=int, nargs=4, default=(4, 10, 1216, 710), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--lower", type=int, nargs=3, default=(170, 100, 80), metavar=("C1", "C2", "C3"))
    parser.add_argument("--upper", type=int, nargs=3, default=(10, 255, 255), metavar=("C1", "C2", "C3"))
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sim and args.dry_run:
        parser.error("--dry-run 只适用于实时视觉回合；仿真不接受该安全开关")

    if args.plc and (not math.isfinite(args.plc_rate) or args.plc_rate <= 0.0):
        parser.error("--plc-rate 必须为有限正数")

    if args.plc and not args.sim and (args.disable_undistort or args.disable_homography):
        parser.error("实机 --plc 必须启用相机内参及球台 Homography 标定")

    if not math.isfinite(args.benchmark_seconds) or args.benchmark_seconds < 0 or (args.benchmark_seconds and not args.headless):
        parser.error("--benchmark-seconds 必须为非负数，且只适用于 --headless")

    if args.sim and (args.benchmark_seconds or args.perf_json):
        parser.error("--benchmark-seconds 和 --perf-json 只适用于视觉模式")

    if not args.sim and args.disable_undistort and not args.disable_homography:
        parser.error(
            "--disable-undistort 必须与 --disable-homography 一起使用："
            "Homography 基于去畸变坐标标定，二者不能只关一半"
        )

    return args


def main(argv=None):
    args = parse_args(argv)
    if args.sim:
        run_game(args.plc, args.plc_rate)
    else:
        run_vision(args)


if __name__ == "__main__":
    main()
