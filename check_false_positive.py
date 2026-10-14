"""抓很多帧，统计每个检出位置**重复出现**的次数 —— 揪出静态误检。

★ 判据：真球在被玩的时候位置一直在变；静态误检（高光/反光/外壳边缘/脏点）
  会在**同一个位置反复复现**。所以"出现次数最高的那个位置"就是嫌疑犯。

★ 现场日志：误检固定在 game(550.8, 49.6)，而真球在 (315,375)→(240,583)→(143,695)
  之间移动 —— 正是这个特征。

不碰 PLC、不动轴，只读相机。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "冰壶仿真"))

import cv2

from air_hockey.camera.manager import CameraConfig, CameraManager
from air_hockey.vision.detector import StoneDetector

FRAMES = 150          # 抓多少帧
CLUSTER_PX = 12.0     # 多近算"同一个位置"

print("=" * 74)
print(f"抓 {FRAMES} 帧，统计检出位置的复现次数")
print("=" * 74)
print("★ 让球处在会被打来打去的状态最好；静止也能看出哪个位置最顽固。")
print()

detector = StoneDetector()
camera = CameraManager(CameraConfig(device=0))
camera.start()
try:
    hits = []          # 每次检出的图像坐标
    first_image = None
    t0 = time.monotonic()
    for i in range(FRAMES):
        frame = camera.wait_for_frame(timeout=2.0)
        if frame is None:
            continue
        if first_image is None:
            first_image = frame.image
        det = detector.detect(frame)
        if det is not None:
            hits.append((float(det.x), float(det.y)))
        if (i + 1) % 30 == 0:
            print(f"  ...已抓 {i+1} 帧，检出 {len(hits)} 次，"
                  f"{time.monotonic() - t0:.1f}s")
    if camera.error is not None:
        print("  采集报错：", camera.error)
finally:
    camera.stop()

print()
print(f"  共 {FRAMES} 帧，检出 {len(hits)} 次（{100.0*len(hits)/max(1,FRAMES):.0f}% 的帧有检出）")
if not hits:
    print("  一次都没检出 —— 球不在画面里？或者阈值/ROI 不对。")
    raise SystemExit(0)

# 简单聚类：按"离已有簇心多近"归并，统计每个簇的出现次数
clusters = []
for x, y in hits:
    for c in clusters:
        if abs(x - c["x"]) <= CLUSTER_PX and abs(y - c["y"]) <= CLUSTER_PX:
            c["n"] += 1
            c["x"] += (x - c["x"]) / c["n"]
            c["y"] += (y - c["y"]) / c["n"]
            break
    else:
        clusters.append({"x": x, "y": y, "n": 1})

clusters.sort(key=lambda c: -c["n"])
print()
print(f"  归并成 {len(clusters)} 个位置簇，按复现次数排序：")
print()
print(f"  {'次数':>5} {'占比':>6} {'图像坐标':>16}   说明")
print(f"  {'-'*5} {'-'*6} {'-'*16}   ----")
for c in clusters[:12]:
    pct = 100.0 * c["n"] / len(hits)
    note = ""
    if c["n"] >= 0.5 * len(hits):
        note = "★ 复现超过一半 —— 极度可能是静态误检（或球一直没动）"
    elif c["n"] >= 0.2 * len(hits):
        note = "复现较多"
    print(f"  {c['n']:>5} {pct:>5.0f}% ({c['x']:>6.0f},{c['y']:>6.0f})   {note}")

if first_image is not None:
    marked = first_image.copy()
    for i, c in enumerate(clusters):
        color = (0, 0, 255) if i < 3 else (0, 160, 0)
        cv2.circle(marked, (int(c["x"]), int(c["y"])), 26, color, 4)
        cv2.putText(marked, str(c["n"]), (int(c["x"]) + 28, int(c["y"]) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(str(ROOT / "camera_recurring_positions.png"), marked)
    print()
    print("  已保存：camera_recurring_positions.png")
    print("    （前 3 个位置画红圈并标了复现次数，其余绿圈）")

print()
print("★ 怎么读：真球在被玩时会散成很多小簇；")
print("  某个位置**独占大多数帧**，那就是静态误检 —— 对着那张图看它落在什么上面。")
