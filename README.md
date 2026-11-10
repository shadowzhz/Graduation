# 毕业设计：基于计算机视觉与 PLC 的空气冰壶实时检测与智能对弈系统 🥌

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20%7C%20Linux-76B900.svg?logo=nvidia&logoColor=white" alt="Platform: Jetson">
  <img src="https://img.shields.io/badge/Industrial%20PLC-Siemens%20S7--1200-006487.svg?logo=siemens&logoColor=white" alt="Siemens S7-1200">
  <img src="https://img.shields.io/badge/Vision-OpenCV%20%7C%20GStreamer-blue.svg?logo=opencv&logoColor=white" alt="Vision">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" alt="Status: Completed">
</p>

> **Air Hockey: Real-Time Vision Tracking, Trajectory Prediction and Intelligent Control**  
> 毕业设计工程项目 | 结合嵌入式边缘计算（NVIDIA Jetson）与工业控制（西门子 S7-1200 PLC）的空气冰壶实时对弈系统。

---

## 📌 课题简介 (Overview)

空气冰壶（Air Hockey）是一项典型的高速桌面竞技运动。冰壶滑行速度快、碰撞反弹频繁，对视觉感知系统的采集帧率、图像处理时效性、弹墙轨迹预判精度以及下位机控制响应均有极高要求。

本课题结合嵌入式边缘计算与工业自动化控制技术，开发了一套完整的空气冰壶目标检测、运动追踪、状态估计、轨迹预测、控制规划与智能对弈系统：
- 依托 **NVIDIA Jetson** 边缘计算平台实现高帧率视觉捕获与微步物理弹墙预测；
- 驱动 AI 算法进行攻防对战决策；
- 通过 **西门子 S7-1200 PLC** 工业通信接口输出控制指令驱动电机轴定位，实现软硬件深度闭环。

---

## ✨ 核心特性与技术指标 (Features)

- ⚡ **超高帧率采集管线**：Linux / Jetson 原生 GStreamer appsink 采集解码与单帧覆盖式缓存，支持最高 200 FPS 高速工业相机输入。
- 🎯 **自适应 ROI 与机动卡尔曼滤波**：HSV + 几何约束快速识别，动态局部 ROI 显著压缩计算量；引入机动门限检测，弹墙与击球突变 50ms 内即可收敛收敛。
- 📐 **各向同性物理投影**：`raw → undistorted → table` 三层几何解耦，支持球台四点单应性透视变换（Homography），消除光学畸变与视角倾斜。
- 🔮 **微步物理轨迹预测**：微步积分阻尼滑行、边墙弹性碰撞、球门圆弧柱反弹与进球判定，视觉、仿真、AI 三端共用统一物理核心。
- 🤖 **纯连续导引 AI 决策**：全面采用纯连续物理导引生成击球与防守目标，状态机退化为纯观察器，杜绝边界模式切换跳变。
- 🔒 **工业级 S7-1200 PLC 闭环通信**：通过 Snap7 实现 DB1/DB18 读写、心跳检测、序号握手与轴安全行程硬件保护互锁。

---

## 🏗️ 系统软件架构 (Architecture)

### 1. 端到端数据链路

实时链路使用 `CurlingState` / `PredictionState` / `PlcTarget` 与实际轴位置反馈闭环，PLC 仅接收安全限幅后的目标位置：

```text
CameraManager（Jetson / Linux: GStreamer appsink）
  │  Frame(raw BGR)
  ▼
StoneDetector            HSV/Lab 阈值 + 轮廓几何 + 动态 ROI
  │  CurlingState(raw 像素, confidence)
  ▼
StoneTracker             常速关联 + 漏检维持（仅服务关联与 ROI）
  │  Track(raw)
  ▼
CameraGeometry           raw → undistorted → table（支持四点 Homography）
  │  table 坐标
  ▼
KalmanFilter             实时默认常量速度；离线可选摩擦 / 墙碰撞感知 EKF
  │  CurlingState / StoneState(table)
  ▼
TrajectoryPredictor      微步物理：摩擦减速 + 弹墙 + 门柱 + 进球
  │  PredictionState(trajectory / endpoint / duration / source_state)
  ▼
AirHockeyAI              守门 / 前压 / 回防决策
  │  AIDecision(target_x, target_y, ...)
  ▼
RallyController         纯连续物理导引（架空模式决策）：
  │  直接由物理连续导引生成目标；WAITING/DEFENDING/ATTACKING/RECOVERING 退化为纯观察器标签
  ▼
PlcControlAdapter        AI 半场限幅 + 时间戳
  │  PlcTarget（固定 600×1060 场地逻辑坐标，各向同性：1 单位 ≈ 0.909mm）
  ▼
PLCLink(后台周期线程)      反馈/心跳/安全互锁/序号握手 → DB1.12 Real 终点 + DB1.90 序号
  │  DB1/DB18 状态位置反馈（mm → 场地逻辑坐标）
  └─────────────────────────────────────────────────→ VisionRuntime / AirHockeyAI / RallyController
```

### 2. 模块依赖方向

视觉运行链单向依赖，共享物理/AI/状态核心均归位于 `air_hockey/` 包内，**视觉核心不依赖 `sim_gui/` 目录**：

```text
Camera -> VisionRuntime
            -> Detector
            -> Tracker
            -> KalmanFilter
            -> CameraGeometry
            -> TrajectoryPredictor
            -> AirHockeyAI
            -> PlcControlAdapter (可选)
```

> *注：仿真端（`sim_gui/`）复用同一套 `physics` / `ai` / `prediction` / `game_state` 和 `air_hockey/control`（PLCLink），仅保留独立的 Tkinter GUI 与交互配置。*

### 3. 场地坐标系定义

游戏与实机坐标系采用**各向同性**映射：
- 真实台面墙内 480mm × 880mm 映射为逻辑坐标 `RINK_LEFT..RINK_RIGHT = 36..564`、`RINK_TOP..RINK_BOTTOM = 46..1014`；
- X、Y 轴均为 1 单位 = 480/528 ≈ 0.909mm，球台中线为 `y=530`；
- 保证圆形半径、速度模长、摩擦系数及碰撞距离计算在几何上各向同性。

### 4. 统一数据契约（`air_hockey/game_state.py`）

| 数据结构 | 核心职责 |
|---|---|
| `CurlingState` | 统一冰壶状态：`x/y`（坐标）、`vx/vy`（速度）、`timestamp`、`confidence`、`radius` |
| `StoneState` | `CurlingState` + `tracking_state`（视觉追踪路径状态） |
| `GameState` | AI 决策快照：球槌位置 + 冰壶状态 + 开球/难度等 |
| `PredictionState` | 预测结果：`trajectory` / `endpoint` / `duration` / `source_state` |

---

## ⚡ 快速复现与运行 (Quick Start)

### 1. 标定与运行命令

统一通过根目录入口 `main.py` 启动：

```bash
# 1. 相机内参标定 (棋盘格)
python3 air_hockey/tools/calibrate_camera.py --cols 10 --rows 7 --square-size 25 --output calibration/camera_calibration.npz

# 2. 球台四角单应性标定
python3 air_hockey/tools/calibrate_table.py --output calibration/table_homography.npz

# 3. 启动实时视觉演示（Camera -> Detection -> AI）
python3 main.py

# 4. 启动虚拟仿真对战游戏（无摄像头纯软件算法验证）
python3 main.py --sim

# 5. 纯算法性能基准测试（Headless 模式）
python3 main.py --headless --benchmark-seconds 30 --perf-json benchmark.json

# 6. 联动西门子 S7-1200 实机对弈
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --record play.json
```

### 2. 实机现场调试辅助工具

```bash
# 不开相机的人工五点对齐与电机行程测试
python3 air_hockey/tools/probe_rally_points.py --plc 192.168.0.64

# 预演模式（只做视觉识别与控制意图解算，不下发运动，安全验证）
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --dry-run --record shadow.json

# 无摄像头运动仿真与控制评估
python3 air_hockey/simulation/run_simulation.py
python3 air_hockey/evaluation/run_evaluation.py --runs 20 --noise 0 --seed 42 --compare-estimators

# 全量自动化回归测试
python3 tests/run_tests.py
```

---

## 💡 核心技术与系统实现 (Core Implementation)

1. **高速图像采集**：GStreamer 原生 appsink 解码 + 线程安全单帧覆盖缓存；不以请求/协商 FPS 替代实测帧率。
2. **目标检测与动态局部 ROI**：HSV 颜色阈值 + 面积/半径/圆形度几何筛选，基于上一帧预测位置的动态 ROI 降低计算量，输出统一 `CurlingState`。
3. **单目标追踪**：原始像素坐标下的常速关联与漏检维持，负责目标关联与动态 ROI 预测。
4. **卡尔曼状态估计**：实时默认使用常量速度模型（`[px, py, vx, vy]`），并启用机动检测：归一化新息平方超过门限时从最近两次观测拟合速度。仿真中被击打后 50ms 内速度估计收敛到 92%（传统滤波需 420ms）。
5. **坐标系几何映射**：`raw → undistorted → table` 三层解耦；支持**球台四点 Homography**（透视校正），速度采用空间微元有限差分换算。
6. **微步物理轨迹预测**：微步积分处理摩擦阻尼、边墙弹性碰撞、门柱圆弧反弹与进球穿透判定；视觉、仿真、AI 共用同一物理预测核心。
7. **AI 智能防守与对弈决策**：实时模式复用预测轨迹计算门线截距；身后球按侧移、后退、对齐、击球分段回收，杜绝自家门乌龙。
8. **PLC 通信与安全互锁**：按 S7-1200（CPU 1211C）DB1/DB18 协议收发；只下发 X/Y 终点（Real）和递增序号，通过心跳与到位状态形成安全互锁。

---

## 📊 实验数据与统一日志格式 (Experiments & Logs)

通过 `--record <path>` 选项可流式落盘版本化（Version 4）的结构化 JSON 运行实验记录：

```json
{
  "format": "curling_recording", "version": 4, "source": "runtime", "meta": {...},
  "frames": [
    {
      "index": 0, "timestamp": 1.0, "fps": 59.8,
      "curling_state": {"x":0,"y":0,"vx":0,"vy":0,"timestamp":0,"confidence":0,"radius":0},
      "prediction": {"trajectory":[[x,y],...],"endpoint":[x,y],"duration":0,"source_state":{...}},
      "ai_decision": {"target_x":0,"target_y":0,"stalled_stone_phase":"idle","reaction_timer":0},
      "plc_request": {"target_x":0,"target_y":0,"timestamp":0}
    }
  ]
}
```

- **日志容灾与文件锁**：逐帧写入 `.journal` 临时文件，带 `fcntl.flock` 进程锁保护，进程异常退出可调用 `RuntimeRecorder.recover('run.json')` 零丢失恢复。

---

## 📁 项目工程目录 (Project Structure)

```text
Graduation/
├── main.py                     # 统一命令行入口：参数解析与实时/仿真/测试调度
├── HANDOFF.md                  # 现场调试与技术交接手册
├── air_hockey/                 # 核心算法与系统闭环包
│   ├── game_state.py           # 共享数据契约（CurlingState / StoneState / GameState）
│   ├── core_config.py          # 场地几何、动力学参数、难度等级配置
│   ├── physics.py              # StoneMotion 物理仿真与碰撞模型
│   ├── ai.py                   # AI 攻防决策引擎
│   ├── camera/                 # 采集层（GStreamer / 最新帧缓存 / FPS 统计）
│   ├── vision/                 # 目标检测 + 追踪 + 坐标几何映射
│   ├── estimation/             # 状态估计与机动检测卡尔曼滤波
│   ├── prediction/             # 微步物理轨迹预测核心 (TrajectoryPredictor)
│   ├── control/                # PLC 闭环：协议封装、行程限幅、PLCLink 周期通讯
│   ├── recording/              # 结构化运行实验日志记录器
│   ├── simulation/             # 纯软件运动仿真引擎
│   ├── evaluation/             # 多场景误差量化评估工具
│   ├── calibration/            # 相机内参标定与去畸变模块
│   ├── app/                    # 实时运行调度、回合控制与渲染预览
│   └── tools/                  # 标定辅助与硬件诊断工具
├── calibration/                # 相机内参与球台单应性矩阵存储目录
├── sim_gui/                    # 独立桌面仿真 GUI
├── docs/                       # 硬件选型、平台调研与历史交接文档
└── tests/                      # 自动化单元测试套件
```

---

## 🎯 课题研究进展与展望 (Roadmap)

### 已完成工作
* [x] Jetson / Linux GStreamer 高帧率相机采集后端
* [x] HSV 阈值与几何约束快速目标检测与动态局部 ROI 加速
* [x] 原始像素单目标常速追踪与漏检维持
* [x] **卡尔曼滤波状态估计层**（常量速度模型 + 机动门限检测）
* [x] 单目内参标定与**球台四点 Homography 透视校正**
* [x] 视觉、仿真、AI 三端统一的微步物理轨迹预测核心
* [x] 西门子 S7-1200 DB1/DB18 闭环控制与安全限幅机制
* [x] 桌面仿真环境、多组实验评估与统一日志容灾记录器
* [x] 自动化功能回归测试套件

### 后续展望与改进方向
* [ ] **实台物理参数辨识**：在真实气浮台采集滑行与反弹数据，进一步精细拟合摩擦与恢复系数。
* [ ] **视觉 AI → PLC 实机深度调优**：持续优化通信往返延迟与机械运动加速度参数。
* [ ] **多目标协同跟踪**：拓展为多个冰壶与击球槌的同场识别与追踪。

---

## 📑 技术交接与开发文档 (Handoff & Docs)

- **现场调试与状态交接手册**：[HANDOFF.md](HANDOFF.md)
- **底层模块接口与几何推导**：[docs/archive/交接文档-2026-10-10.md](docs/archive/交接文档-2026-10-10.md)
- **Jetson Xavier NX 硬件平台调研报告**：[docs/Jetson系统.md](docs/Jetson系统.md)

---

## 📄 开源许可证与版权声明 (License)

本项目为毕业设计学术与工程研究课题成果。版权归属原作者所有，代码仅供学术交流与非商业研究使用。
