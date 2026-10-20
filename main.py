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

# ★★★ Windows 控制台默认是 GBK(936)，打印一个它编不出的字符（² ³ [!] OK …）
#     就会抛 UnicodeEncodeError 把整个程序打断 —— 现场踩过：
#         UnicodeEncodeError: 'gbk' codec can't encode character '\xb2'
#     这里把 stdout/stderr 的编码错误策略改成 replace：编不出的字符显示成 "?"，
#     但**永远不会再因为一句日志而崩**。改任何提示文案都不用再担心这件事。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  # 老解释器 / 被重定向到不支持的对象：忽略，不影响运行

from air_hockey.app.live import run_vision
from air_hockey.app.rally import DEFAULT_DEFENSE_FORWARD_LIMIT


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
    parser.add_argument("--plc", default=None, metavar="IP", help="连接已配置学弟版 DB1/DB18 的 S7-1500T；轴需现场主动使能")
    parser.add_argument("--plc-rate", type=float, default=30.0, help="PLC 通信周期频率（Hz，默认 30）")
    parser.add_argument("--mallet-speed", type=float, default=500.0, metavar="UNITS_PER_SEC",
                        help="球槌可达速度估计（球台单位/秒，默认 500）。用来算提前量："
                             "目标瞄冰壶再走 lead_time 之后的位置，而不是它现在的位置 —— "
                             "不然轴总是慢一拍、球走了才追到。设 0 关闭提前量。"
                             "这不是 PLC 轴速度设置，需要在现场实测后调准")
    parser.add_argument("--defense-forward-limit", type=float,
                        default=DEFAULT_DEFENSE_FORWARD_LIMIT, metavar="UNITS",
                        help="冰壶在机器半场还在运动时，机械臂最多离开防守线往前多少"
                             "（球台单位，1 单位≈1.317mm）；冰壶停住（死球）时不受此限。"
                             "**默认不限制**（= ai.py 自身的上限，现场选的是「全部追球」）。"
                             "想保守再显式给值：0 = 钉死在防守线（Y 完全不动），"
                             "50 ≈ 往前 66mm，100 ≈ 往前 132mm（数值越大越主动）")
    parser.add_argument("--blend-threshold-mm", type=float, default=12.0, metavar="MM",
                        help="相邻两条目标之间步距超过它就改用 BufferMode=1（附加停稳），"
                             "否则用 5（混合）。默认 12，与 --target-slew-mm 默认值相同 —— "
                             "步距不超阈值时保持混合。"
                             "★ 太小会频繁附加停稳；不修改 PLC 的实际速度或加速度")
    parser.add_argument("--target-slew-mm", type=float, default=12.0, metavar="MM",
                        help="目标限速（mm/周期），默认 12。"
                             "★★★ 这个治的是「顿」：rally 状态机切换时目标会**瞬间跳变**"
                             "（现场实测 p95 步距 192.8mm），大步会使用 "
                             "BufferMode=1，完成当前运动并停稳后再执行下一条。"
                             "★★★ 但它决定**指令速度**（= slew × 实测周期），"
                             "所以直接决定刹不刹得住：超调 = v^2/(2×TIA 减速度) 必须小于 "
                             "--motion-margin-mm，否则全速冲向边缘时可能碰到限位开关。"
                             "默认 12 @30Hz = 360mm/s，超调约 2mm，配默认余量 5mm 是安全的；"
                             "调大 slew 时看启动自检，必要时同步加大余量")
    parser.add_argument("--motion-margin-mm", type=float, default=5.0, metavar="MM",
                        help="命令余量：下发目标离机械行程端点（硬件限位开关）再内缩多少毫米，"
                             "默认 5。机械行程 X -192..224 / Y -129..187，"
                             "包络 = 行程内缩它。程序不另设软件行程保护；"
                             "★ 它应大于启动自检打印的超调距离，否则全速冲向边缘时可能碰到限位开关")
    parser.add_argument("--target-accel-mm-s2", type=float, default=12000.0, metavar="MM_S2",
                        help="目标加速度限幅（mm/s^2），默认 12000；0 = 关闭。"
                             "★★★ 治的是**跟随误差 801**：只限步长（--target-slew-mm）"
                             "管不住「目标突然反向」——限速器只看步长大小、不看方向。"
                             "★★ 必须**低于 TIA「运动机构运动 → 加速度」的预设值**"
                             "（现在是 15000），否则等于没限。"
                             "现场 TIA 工艺对象 Y轴「状态位和错误位」实测："
                             "错误=跟随误差(红)、警告=动态限值(黄)、已归位=绿、驱动装置=灰，"
                             "快照 physical Y=110.49（行程正中间、prev_busy=True）——"
                             "**轴在正常位置自由运动途中跟不上指令，不是撞限位**。"
                             "算下来 40mm/周期 @65Hz 需要约 169,000 mm/s^2，"
                             "而轴的能力只有 106N/3kg ≈ 35,000 mm/s^2，差约 5 倍。"
                             "限加速度后急转时目标自己拐大弯、直线时仍可全速。"
                             "★ 调大更跟手但更容易报跟随误差；调小更稳但急转变慢")
    parser.add_argument("--tia-decel-mm-s2", type=float, default=None, metavar="MM_S2",
                        help="同步 TIA「运动机构运动 → 减速度」的预设值（默认按 35000）。"
                             "★★★ 它决定**制动距离** = v^2/(2x减速度)，也就决定"
                             "「这个 slew 会不会撞限位」。默认值是按现场已改好的 35000；"
                             "**如果你其实没改减速度，就加 --tia-decel-mm-s2 15000** —— "
                             "否则自检会说安全而实际会撞")
    parser.add_argument("--tia-accel-mm-s2", type=float, default=None, metavar="MM_S2",
                        help="同步 TIA「运动机构运动 → 加速度」的预设值（默认按 35000）。"
                             "用来判断 --target-accel-mm-s2 有没有超过机器能力")
    parser.add_argument("--tia-speed-mm-s", type=float, default=None, metavar="MM_S",
                        help="同步 TIA「运动机构运动 → 速度」的预设值（默认按 3750）。"
                             "用来判断指令速度会不会被 TO 截断")
    parser.add_argument("--preview-fps", type=float, default=20.0, help="预览刷新率上限")
    parser.add_argument("--camera-device", default=None, help="Linux 相机路径（如 /dev/video2）；Windows 摄像头编号（默认 0）")
    parser.add_argument("--camera-slow", action="store_true",
                        help="不在打开时协商 MJPG（旧行为）。这台相机在 Windows 上会停在 YUY2 10 FPS，仅排查用")
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
