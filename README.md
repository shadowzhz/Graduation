# 毕业设计：基于计算机视觉与 PLC 的空气冰壶实时检测与智能对弈系统

> **Air Hockey: Real-Time Vision Tracking, Trajectory Prediction and Intelligent Control**
> 毕业设计项目 (Graduation Project) | 嵌入式视觉与工业控制应用

---

## 📌 课题简介

空气冰壶（Air Hockey）是一项典型的高速桌面竞技运动。冰壶滑行速度快、碰撞反弹频繁，对视觉感知系统的采集帧率、图像处理时效性、弹墙轨迹预判精度以及下位机控制响应均有较高要求。

本毕业设计结合嵌入式边缘计算与工业自动化控制技术，开发了一套完整的空气冰壶目标检测、运动追踪、状态估计、轨迹预测、控制规划与智能对弈系统。系统依托 NVIDIA Jetson 边缘计算平台实现高帧率视觉捕获与弹墙物理预测，驱动 AI 算法进行防守反击决策，并通过西门子 PLC 工业通信接口输出控制指令，具备良好的软硬件工程实用性。

---

## 🏗️ 系统软件架构

### 1. 端到端数据链路

实时链路使用 `CurlingState` / `PredictionState` / `PlcWriteRequest` 单向贯通；离线轨迹规划仍保留独立的 `ControlCommand`，但现有 PLC DB 不消费其方向和速度：

```text
Camera (仅 GStreamer appsink，线程安全最新帧缓存)
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
KalmanFilter             常量速度模型状态估计，抑制视觉抖动
  │  CurlingState / StoneState(table)
  ▼
TrajectoryPredictor      微步物理：摩擦减速 + 弹墙 + 门柱 + 进球
  │  PredictionState(trajectory / endpoint / duration / source_state)
  ▼
AirHockeyAI              守门 / 前压 / 回防决策
  │  AIDecision(target_x, target_y, ...)
  ▼
PlcControlAdapter        目标坐标检查 + 输出限幅 + 时间戳
  │  PlcWriteRequest
  ▼
PlcOutputWorker(非阻塞)   新请求唤醒 + 限频 + 覆盖式最新请求 → 后台写入
  │
  ▼
PLCInterface             S7-1500 DB1 写入（可选，--plc）
```

### 2. 模块依赖方向

视觉运行链单向依赖，共享物理/AI/状态核心均归位于 `air_hockey/` 包内，**视觉核心不依赖 `冰壶仿真/` 目录**：

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

> 仿真端（`冰壶仿真/`）复用同一 `physics` / `ai` / `prediction` / `game_state`，仅保留仿真 GUI、GUI 配置与 PLC 通信模块。

### 3. 统一数据契约（`game_state.py`）

| 类型 | 职责 |
|---|---|
| `CurlingState` | 统一冰壶状态：`x/y`(position)、`vx/vy`(velocity)、`timestamp`、`confidence`、`radius` |
| `StoneState` | `CurlingState` + `tracking_state`（视觉追踪路径） |
| `GameState` | AI 决策快照：球槌位置 + 冰壶状态 + 开球/难度等 |
| `PredictionState` | 预测结果：`trajectory` / `endpoint` / `duration` / `source_state` |
| `ControlCommand` | 独立轨迹规划模型：`target_position` / `direction` / `speed`；当前 PLC DB 不接收方向/速度 |

---

## 💡 核心技术与系统实现

1. **图像采集**：GStreamer 原生 appsink 解码 + 线程安全单帧覆盖缓存；实际采集 FPS 和阶段耗时需在目标设备实测，不以请求/协商 FPS 替代。
2. **目标检测与动态局部 ROI**：HSV 颜色阈值 + 面积/半径/圆形度几何筛选，基于上一帧预测位置的动态 ROI 降低计算量，输出统一 `CurlingState`。
3. **单目标追踪**：原始像素坐标下的常速关联与漏检维持，负责目标关联与动态 ROI 预测。
4. **卡尔曼状态估计**：常量速度模型（`[px, py, vx, vy]`）融合观测序列，输出平滑的 `CurlingState`，显著降低检测抖动（观测误差 ≈3.8 → 滤波误差 ≈1.7）。
5. **单目去畸变与坐标系几何映射**：`raw → undistorted → table` 三层解耦；`undistorted → table` 支持**球台四点 Homography**（透视校正），也保留 ROI 线性映射回退；速度用空间微元有限差分换算。
6. **统一微步物理轨迹预测核心**：`TrajectoryPredictor` 涵盖摩擦阻尼、边墙弹性碰撞、门柱圆弧反弹与进球穿透判定；视觉、仿真、AI **共用同一物理核心**。
7. **AI 智能防守与对战决策**：实时模式复用本帧预测轨迹计算门线截距，覆盖前压击球、死球处理与自动回中状态机；独立仿真调用仍可自行预测。
8. **轨迹规划与控制适配层**：独立 `TrajectoryPlanner` 可生成 `ControlCommand`；实时 `PlcControlAdapter` 按现有 DB 协议直接将 AI 目标与状态封装成 `PlcWriteRequest`，`PlcOutputWorker` 在新请求到达时唤醒并限频写入。
9. **仿真 / 评估 / 记录工具链**：无摄像头运动仿真（真实轨迹→观测→Kalman→预测）、多组实验误差评估（JSON + 控制台报告）、真实/仿真统一格式运行日志（含 CurlingState / PredictionState / AIDecision / PLC request）。
10. **西门子 PLC 工业通信**：`PLCInterface` 通过 snap7 将 AI 目标、球槌位置、冰壶状态与比分写入 S7 DB 块。

---

## 🚀 系统运行与复现操作

统一通过根目录 [main.py](file:///run/media/shadowemperor/游戏/Ubuntu/Project/Python/Graduation/main.py) 启动：

默认视觉模式须先在目标摄像头/分辨率下生成 `calibration/camera_calibration.npz`（内参）及 `calibration/table_homography.npz`（球台四角）。文件缺失/格式错误会报错，不会静默使用无畸变或线性 ROI。仅排查采集、不使用标定时可显式同时加 `--disable-undistort --disable-homography`；`--sim` 不使用相机标定。真实启动要求目标系统提供 `/dev/video*`、PyGObject/GStreamer 及 Jetson MJPEG 硬件解码元件；没有 V4L2/OpenCV 回退。

```bash
python3 air_hockey/tools/calibrate_camera.py --cols 10 --rows 7 --square-size 25 --output calibration/camera_calibration.npz
python3 air_hockey/tools/calibrate_table.py --output calibration/table_homography.npz
python3 main.py                       # 实时视觉演示（Camera -> ... -> AI）
python3 main.py --headless            # 无显示性能基准测试
python3 main.py --headless --benchmark-seconds 30 --perf-json benchmark.json  # 有标定文件时的分段耗时报告
python3 main.py --sim                 # 虚拟仿真对战（也可写作 --game）
python3 main.py --record run.json     # 记录运行数据（统一 JSON 日志）
python3 main.py --plc 192.168.0.1 --plc-rate 30   # 启用 AI->PLC 非阻塞输出
```

常用参数：`--preview-fps`、`--calibration`、`--table-calibration`、`--disable-undistort`、`--disable-homography`、`--roi`、`--lower/--upper`、`--record`、`--plc`、`--plc-rate`。

采集诊断：`python3 air_hockey/tools/test_camera.py --benchmark --duration 10` 给出设备、请求与真实 caps 协商模式、采集线程实测 FPS、独立取样的 read/convert 耗时分布（read 总耗时包含取样等待和转换，不与 convert 相加）；`python3 air_hockey/tools/test_gstreamer_transfer.py --mode bgr-cvt --duration 10` 在独立管道比较与实际后端相同的 OpenCV BGRx→BGR 转换，`--mode bgr-copy` 则只测 NumPy 三通道裁剪复制。后者不是实际转换路径，独立管道的 FPS 也不是应用采集 FPS。`main.py --headless --benchmark-seconds 30 --perf-json benchmark.json` 报告视觉链路分段耗时；请求/协商 FPS、实测采集 FPS、处理 FPS 与各阶段耗时不可混称。`Frame.timestamp` 是 host `perf_counter` 的读取/转换完成时刻，Gst PTS 是未映射的管道时钟域原值，二者均不是曝光时间；不能据此声称曝光到显示或 PLC 端到端延迟。本机无 Jetson，尚未取得本次变更的实机测量数据。

预览链路先将原始帧缩到 640 像素宽，再按相同比例绘制 ROI、观测、轨迹和 AI 目标；`render()` 全分辨率输出仍保持原有行为。此路径不再为预览复制、绘制整张原始分辨率图像。本机 1280×720 合成画面、各预热 3 次后各测 50 次的单次对比：预览编码耗时 p50 从约 1.97 ms 降为约 0.96 ms；这不是 Jetson 或真实相机的性能结论。GStreamer 后端仍保留 cv2 的 BGRx→BGR 转换，不在缺少 Jetson 实测时改动硬件协商管道。

辅助工具：

```bash
python3 air_hockey/simulation/run_simulation.py     # 无摄像头轨迹仿真
python3 air_hockey/evaluation/run_evaluation.py     # 算法评估报告（JSON + 控制台）
python3 air_hockey/tools/calibrate_table.py         # 球台四点 Homography 标定
python3 tests/run_tests.py                          # 全量自动化测试
```

---

## 📊 运行日志与实验数据

`--record <path>` 输出统一格式的版本化 JSON 运行日志，真实运行与仿真格式一致，每帧包含：

```json
{
  "format": "curling_recording", "version": 2, "source": "runtime", "meta": {...},
  "frames": [
    {
      "index": 0, "timestamp": 1.0, "fps": 59.8,
      "curling_state": {"x":0,"y":0,"vx":0,"vy":0,"timestamp":0,"confidence":0,"radius":0},
      "prediction": {"trajectory":[[x,y],...],"endpoint":[x,y],"duration":0,"source_state":{...}},
      "ai_decision": {"target_x":0,"target_y":0,"stalled_stone_phase":"idle","reaction_timer":0},
      "plc_request": {"ai_target_x":0,"ai_target_y":0,"ai_x":0,"ai_y":0,"stone_x":0,"stone_y":0,
                       "stone_vx":0,"stone_vy":0,"player_score":0,"ai_score":0,"timestamp":0}
    }
  ]
}
```

---

## 📂 项目工程目录

```text
Graduation/
├── main.py                     # 统一入口（视觉 / headless / 仿真 / 记录 / PLC 输出）
├── game_state.py               # 共享数据契约（CurlingState / StoneState / GameState）
├── README.md                   # 本文档
├── Jetson系统.md               # Jetson Xavier NX 硬件平台调研报告
├── air_hockey/                 # 核心算法包
│   ├── core_config.py          # 场地几何、动力学参数、难度等级
│   ├── physics.py              # StoneMotion 物理实体与碰撞规则
│   ├── ai.py                   # AI 决策器（依赖注入 TrajectoryPredictor）
│   ├── camera/                 # 采集层（GStreamer / 线程安全最新帧缓存 / FPS 统计）
│   ├── vision/                 # 检测 + 追踪 + 坐标几何 + 预处理
│   ├── estimation/             # KalmanFilter 状态估计层
│   ├── prediction/             # PredictionState + TrajectoryPredictor（统一物理核心）
│   ├── planning/               # ControlCommand + TrajectoryPlanner
│   ├── control/                # PlcControlAdapter + PlcOutputWorker（AI->PLC 适配）
│   ├── recording/              # 统一运行日志（JSON schema + RuntimeRecorder）
│   ├── simulation/             # 无摄像头运动仿真测试环境
│   ├── evaluation/             # 多组实验误差评估与报告
│   ├── calibration/            # 相机内参标定与去畸变工具
│   ├── app/                    # VisionRuntime 运行核心 + 渲染 + FPS GUI
│   ├── tools/                  # 标定 / 检测 / 追踪 / 预测 等诊断脚本
│   └── 交接文档.md             # 面向开发者的详细技术与工程交接文档
├── calibration/                # 相机内参在库；table_homography.npz 须在实台标定后生成
├── 冰壶仿真/                   # 桌面仿真 GUI、GUI 配置与 PLC 通信模块
└── tests/                      # 自动化测试套件（154 项，无需 pytest）
```

---

## 🎯 课题研究进展与展望

### 已完成工作
* [x] GStreamer-only 相机采集与阶段诊断接口（待 Jetson 实机验证）
* [x] 颜色 + 几何约束检测算法与动态局部 ROI 加速
* [x] 原始像素单目标常速追踪与漏检维持
* [x] **卡尔曼滤波状态估计层**（常量速度模型，抑制视觉抖动）
* [x] 单目内参标定与点级去畸变；**球台四点 Homography 透视校正**
* [x] 视觉、仿真、AI 三端统一的微步物理轨迹预测核心（`PredictionState`）
* [x] 独立 `ControlCommand` 轨迹规划与目标坐标直达 PLC 的非阻塞软件输出链（实机执行机构尚未联调）
* [x] 仿真测试环境、多组实验评估与统一格式运行日志
* [x] 西门子 S7 PLC 通信底层封装
* [x] 154 项自动化功能回归测试套件

### 后续展望与改进方向
* [ ] **实台物理参数辨识**：在真实气浮台采集滑行/碰撞数据，回归摩擦与恢复系数。
* [ ] **实机 AI → PLC 联调**：接入真实击球机构，补充端到端时延测量与前瞻补偿。
* [ ] **多目标跟踪**：扩展为多冰壶/球槌协同跟踪与状态机。
* [ ] **自动化 CI**：配置 GitHub Actions 执行 `compileall` 与全部单元测试。

---

## 📑 技术交接与开发文档

关于更底层的模块接口、坐标系数学定义、Jetson 手动锁频脚本与技术债务分析，请参考：
👉 [Air Hockey 项目交接文档 (air_hockey/交接文档.md)](air_hockey/交接文档.md)
