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

实时链路使用 `CurlingState` / `PredictionState` / `PlcTarget` 与实际轴位置反馈闭环；离线轨迹规划仍保留独立的 `ControlCommand`，PLC 只接收目标位置：

```text
CameraManager（Linux / Jetson: GStreamer appsink；Windows: OpenCV DirectShow）
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
  │  PlcTarget（固定 600×760 场地逻辑坐标）
  ▼
PLCLink(后台周期线程)      反馈/心跳/安全互锁/触发脉冲 → DB1.20 LREAL 运动目标
  │  DB1/DB18 状态位置反馈（mm → 场地逻辑坐标）
  └─────────────────────────────────────────────────→ VisionRuntime / AirHockeyAI / RallyController
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

> 仿真端（`冰壶仿真/`）复用同一 `physics` / `ai` / `prediction` / `game_state` 和 `air_hockey/control/plc.py`，仅保留 GUI 与 GUI 配置。

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
4. **卡尔曼状态估计**：实时默认使用常量速度模型（`[px, py, vx, vy]`）；离线可选 `friction` 或 `collision_aware`（只处理墙法向反弹，不处理门柱），用 `--compare-estimators` 在同一组位置观测上比较三者。墙模型只用估计状态和已知球台几何，不接收真值碰撞事件；Q/R/P0 与摩擦 baseline 一致。
5. **单目去畸变与坐标系几何映射**：`raw → undistorted → table` 三层解耦；`undistorted → table` 支持**球台四点 Homography**（透视校正），也保留 ROI 线性映射回退；速度用空间微元有限差分换算。
6. **统一微步物理轨迹预测核心**：`TrajectoryPredictor` 涵盖摩擦阻尼、边墙弹性碰撞、门柱圆弧反弹与进球穿透判定；视觉、仿真、AI **共用同一物理核心**。
7. **AI 智能防守与对战决策**：实时模式复用本帧预测轨迹计算门线截距，实机视觉控制关闭随机瞄准误差，避免每次反应刷新都横向跳目标；仿真难度与误差不变。身后冰壶按侧移、后退、对齐、击球分段回收，避免斜退穿球造成乌龙。进入球门侧安全退让线（当前逻辑 y≤91）的球不主动向自家球门推，不能保证救回已经越过可安全击球区域的快球；独立仿真调用仍可自行预测。
8. **轨迹规划与控制适配层**：独立 `TrajectoryPlanner` 可生成 `ControlCommand`；实时 `PlcControlAdapter` 限制 AI 半场目标，`PLCLink` 周期读取轴状态及位置，安全状态成立才下发运动目标。
9. **仿真 / 评估 / 记录工具链**：无摄像头运动仿真（真实轨迹→观测→Kalman→预测）、多组实验误差评估（JSON + 控制台报告）、真实/仿真统一格式运行日志（含 CurlingState / PredictionState / AIDecision / PLC request）。
10. **西门子 PLC 工业通信**：按 S7-1500T/TO_Kinematics 联调版的 DB1/DB18 协议收发；只下发 `[X,0,Y,0]` 运动学位置和触发脉冲，冰壶状态与比分不再写入 PLC。

计划性跳过检测的帧只外推位置，保留追踪状态和漏检次数；真正执行检测却没找到冰壶时才进入 `lost`，重新检测到后恢复 `active`。运行时漏检会把状态估计推进到当前帧时间；追踪 ID 改变时清空旧估计，新目标不继承旧速度。预测前先限制当前及目标速度，完整预测到停止、进球或障碍接触，不再以 18 个绘制点或反弹次数截断物理过程。`endpoint` / `duration` 是终态位置及完整时长；默认最多 4000 个微步（约 16.67 秒），耗尽预算会明确报错，不返回假终点。

离线仿真真值处理门柱与进球，得分后保持位置直到配置末帧。仿真和评估的 `dt` 必须为有限正数，`noise` 必须为有限非负数；NaN / Inf 会报错。

---

## 🚀 系统运行与复现操作

统一通过根目录 [main.py](file:///run/media/shadowemperor/游戏/Ubuntu/Project/Python/Graduation/main.py) 启动：

Windows 使用已经能运行 PLC 软件的同一个 Python 环境；若它缺视觉依赖，安装 `numpy` 和 `opencv-python`（`python -m pip install numpy opencv-python`），不要为此更换已有可用的 `python-snap7` 版本。
默认视觉模式须先在目标摄像头/分辨率下生成 `calibration/camera_calibration.npz`（内参）及 `calibration/table_homography.npz`（球台四角）。文件缺失/格式错误会报错，不会静默使用无畸变或线性 ROI。仅排查采集、不使用标定时可显式同时加 `--disable-undistort --disable-homography`；`--sim` 不使用相机标定。相机采集按系统选择：Linux / Jetson 用 V4L2 GStreamer（Jetson 有 NVIDIA 元件时硬解码，普通 Linux 软件解码）；Windows 用 OpenCV DirectShow，默认摄像头编号为 `0`，可用 `--camera-device 1` 切换。Windows 请求 MJPG、1280×720 和配置帧率（默认 200 FPS）；启动状态报告实际尺寸、驱动报告帧率和格式，实测采集速度以帧计数为准。200 FPS 是否可达、视野是否完整取决于相机支持的模式。先确认完整球台都在画面内；视野变化后，即使尺寸仍为 1280×720，也必须重新做相机内参和球台标定，验证完成前不要接 PLC。

```bash
python3 air_hockey/tools/calibrate_camera.py --cols 10 --rows 7 --square-size 25 --output calibration/camera_calibration.npz
python3 air_hockey/tools/calibrate_table.py --output calibration/table_homography.npz
python3 main.py                       # 实时视觉演示（Camera -> ... -> AI）
python3 main.py --camera-device /dev/video2 --disable-undistort --disable-homography  # Linux 相机调试，非实台物理坐标
python3 main.py --headless            # 无显示性能基准测试
python3 main.py --headless --benchmark-seconds 30 --perf-json benchmark.json  # 有标定文件时的分段耗时报告
python3 main.py --sim                 # 虚拟仿真对战（也可写作 --game）
python3 main.py --record run.json     # 记录运行数据（统一 JSON 日志）
python3 main.py --plc 192.168.0.64 --plc-rate 30  # 仅适用已配置对应 DB1/DB18 的 S7-1500T
python3 main.py --sim --plc 192.168.0.64          # 仿真游戏/实机轴反馈联动
# Windows PowerShell：先验证相机画面；PLC 运行时用已安装 Snap7 的同一 Python 环境
python main.py --camera-device 0
# 确认完整球台都在预览中，并在当前相机画面下完成标定后，才连接 PLC
python main.py --camera-device 0 --plc 192.168.0.64 --plc-rate 30
```

仿真游戏现支持 `--plc-rate`（默认 30 Hz，与视觉模式一致）；例如 `python3 main.py --sim --plc 192.168.0.64 --plc-rate 40`。它只改变上位机请求通信周期，不改变 PLC 轴的速度、加速度或 AI 决策间隔。游戏默认「普通」难度每 85 ms 刷新 AI 目标，「困难」每 35 ms 刷新；若游戏目标变化快而实机位置跟随慢，先核对屏幕「目标/实际」与 PLC 运动参数。频率过高可能导致读写赶不上周期、反馈失效；应在现场安全条件下测量往返耗时和位置反馈后再调整，不要将软件调频当作机械安全改造。

`main.py --sim` 传播游戏子进程的退出状态；窗口启动失败不再被报告为成功。棋盘格标定始终自动采样，不再接受原本无效的 `--auto`；重复帧不重复采样，采集出错立即退出并清理相机。

游戏重新开局/换局会在 PLC 工作线程重置上次发送位置的死区缓存；该操作不写 PLC。此前将成功的无返回值操作误报为「清死区指令发送失败」，现已修复。历史视频里的这条提示不能单独作为通信故障证据。

**实机使用前提**：确认设备区域安全、硬件急停有效、TIA DB 布局匹配，完成相机内参与球台四点标定；不要用 `--disable-homography` / `--disable-undistort` 驱动实机。连接本身不使能轴；图形界面 E/「轴使能」、H/「轴回零」、C/「轴复位」由现场操作员操作。按 H 或 C 时停止游戏目标；操作完成、轴状态恢复后需重新按 E 授权，仿真游戏还需手动继续。PLC 响应异常、轴未就绪、心跳超时和反馈失效时禁止下发；断线重连也需重新人工使能。机械限位和急停仍由 PLC/硬件负责，软件限幅不能替代。`--headless --plc` 无现场轴使能按钮，不用于首次联调。

**现场安全边界**：实时默认首帧检测就接受并产生动作意图，不再等第二次硬检测。保留现有漏检宽限；轨迹删除后，仅在现有一秒窗口和球槌邻近条件内保持最后的有效目标，超时回 READY；状态过期/非有限值/越界则清目标。遮挡保持由显式 `held_target` 决策发布，不靠状态栏文字判断运动资格。所有目标仍受 PLC 人工授权、报警/心跳/反馈互锁和机械命令范围约束。上位机控制位采用同步 S7 BIT 写入，保留 PLC/HMI 的相邻故障、停止、锁存和圆弧位。运动数据先写 `DB1.20..55` 的 36 字节 `[X,0,Y,0] + BufferMode`；当前默认 `--trigger-mode sequence` 再更新 `DB1.DBD142` 序号，要求 PLC 已安装对应握手逻辑；`--trigger-mode pulse` 则置 `DB1.56.0`。连接须先成功初始化心跳位并清除直线触发位，才允许后续授权。部分写入失败不能撤回 PLC 已接受的命令；仍须现场验证 PLC 扫描、触发脉宽和独立失联停止。软件清目标或撤销授权只禁止后续下发，不能撤销已经接受的运动，**不得作为急停使用**。

### 直接对打运行方式

实时默认每帧检测，默认 tracker 首次硬检测即为 ACTIVE；不增加连续两帧确认。局部 ROI 没找到球时，同一帧立即在配置的全台 ROI 重找，防止回击后速度突变把球裁掉。显式传入更大的 detection_interval 时仍保留 80 ms 时间补检；计划预测帧不会伪造硬检测。单帧误检现在也可能产生运动意图，应确认绿色检测标记跟随的是冰壶。

视觉候选关联：默认检测器保留同帧候选，优先从预测位置或上次观测的关联范围内，结合形状质量、位置和半径连续性选球；无匹配时回退形状最佳候选，首次检测仍立即响应。自适应 ROI 保留原 140 像素基准搜索框，并覆盖前次观测、按速度与帧间隔外推的位置和球半径余量，最终裁切到图像边界。局部无检测仍同帧全台重找；漏检宽限、匀速估计、PLC 控制和颜色阈值均未改变。合成多目标、掉帧及反向案例已验证；关联属于启发式选择，不能保证遮挡或相似目标交叉时身份不切换。

- 真人半场（y>380）：首帧速度未知时先按当前位置横向对齐防守线；来球复用 Predictor 的轨迹截距。不等额外确认、回 READY 或 busy 清除。
- 机器半场（y≤380）：直接发送现有 AI 的击球目标，包括静止球、分界线附近的球和回位途中收到的球。身后球继续使用侧移、后退、对齐、向真人方向击打的现有流程，不从前面把球推入自家球门。
- 球返回真人半场：退回防守线；方向明确远离机器时请求 READY `(300,139.52)`。没有回球连帧计数、重新武装或五秒回合等待；下一球可立即更新目标。

实时 AI 关闭仿真反应延迟与随机误差，使用完整预测横坐标，攻击范围覆盖机器半场；仿真难度、Estimator/Predictor 算法和物理参数不变。已架空 RallyController 状态机的离散模式裁决，控制链路全面转由纯连续物理导引驱动，WAITING/DEFENDING/ATTACKING/RECOVERING 彻底退化为观察器显示标签，绝不作为截断或篡改运动目标的门禁。PLC 新鲜实际位置优先，无 PLC 时使用现有软件球槌位置。机械不可达或已经进入球门的球仍不能保证救回。

接 PLC 时，回收点和到位判断使用 `COMMAND_*` 映射的可请求范围，并保留侧移、回撤、对齐、击打流程；侧向接触不要求先退开完整的两半径加额外间距。确实无法在冰壶朝自家门一侧接触时清目标并提示人工移球，不盲目直穿冰壶回 READY。命令余量、序号握手、AI 提前量和物理参数本轮不变；实际位置越过已配置的 `PHYSICAL_*` 软限位才触发越界撤销，不能把仍在软限位内的旧位置误判为失控。

`TransitionParameter[1]` 可选写入失败会记录并打印实际异常，停用该写入，保留 TO 的既有混合距离设置；不会因为日志队列属性错误把连接误标为断线。运动数据、触发位或序号写入失败仍按原通信失败路径处理。

限速开启时，所有斜坡步均由软件规划保证连续平滑，统一保持 `BufferMode=5`（混合滤波），彻底杜绝中间步停稳造成的卡顿；只有未开启限速（slew<=0）且真实超过阈值的大步才发 `BufferMode=1`。

状态机增加中线迟滞带（进攻态需退至 392 才解除）与人类半场远离速度阈值（vy>20 才退回 READY），并在状态切换间增加 80ms 最小驻留时间，消除边界零点高频横跳。

PLC 反馈仍为 Busy、且请求目标与上次成功下发点相差小于现有 3 mm 死区时，不再追加相同运动作业。比较在目标限速前完成，变化目标和仍在推进的限速斜坡不等待 Busy 清除；PLC 空闲但实际位置仍未到位时保留重发。日志增加「执行中同点跳过」，并按实际发送的 BufferMode 1/5 统计「附加停稳/混合」，不再漏掉模式 1。发送周期、限速参数、PLC 协议和人工授权不变。

运动目标现按剩余距离提前减速，并对最终中间指令检查物理软限位；首次运动及到位判断使用未裁切的实际位置。轴在命令范围外、软限位内时，允许按限速步长逐步返回，不再跳到命令边界。启动与运行诊断纳入软件目标加速度限幅；这些检查不能替代 PLC 限位、伺服保护或硬件急停。

AI 提前量沿既有微步物理模型推进，计入弹墙和摩擦，不再将无界直线外推结果直接裁到台边。拦截时间仍采用原有常速球槌估计，须现场校准球槌速度。

```bash
# 不开相机的人工五点测试；e/h/c/n/q 各自输入后回车，不自动使能。
python3 air_hockey/tools/test_rally_points.py --plc 192.168.0.64
# 只看识别/目标，不发运动，确认坐标与检测对象后再退出重启实机模式。
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --dry-run --record shadow.json
# 实际对打：GUI E 人工使能，H 回零，C 复位；无需速度估计/可达性参数。
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --record play.json
```

Windows 将 `python3` 改为 `python`，相机用 `--camera-device 0`。标定和轴位置方向必须正确；实机机械速度/加速度仍由 PLC 配置，软件不设置或保证其值。每次记录使用未用过的文件名；`plc_request` 是意图不是执行确认，meta 的 control 为 `direct_play`，并标明 `dry_run/live/no_plc`。

记录器启动时拒绝已存在的最终 JSON 文件并报 `FileExistsError`，避免重用文件名覆盖旧实验。同次运行的快照更新、关闭及崩溃日志恢复保持原有行为。

GUI/控制台显示当前动作、track_valid、球速度、目标、球槌实际位置、身后球击打阶段及 PLC armed/safe。动作/追踪/授权变化立即打印，不按每帧刷。没有反应时核对检测标记是否跟球、目标是否变化、PLC 是否授权及实际位置是否跟上；不能用仿真成功替代实机接触验证。

`--trigger-mode pulse` 下，`--trigger-low-ms`（默认 10）同时保证运动触发位的高、低电平最小保持时间；状态读取和心跳耗时计入低电平，不额外重复等待。只保证低电平时，高通信频率可能把高电平缩短到一次 S7 往返，导致 PLC 扫描漏掉上升沿。等待结束后只发送最新有效目标，期间被清除的目标不会继续下发。实际频率可能低于 `--plc-rate` 请求值；未测 PLC 扫描周期不要缩短脉宽。DB 地址、BIT、BufferMode、轴速度/加速度和报警互锁不变。

运动中报警撤销授权时，工作线程只打印一条 `运动撤销快照`，包含前后使能/就绪/轴故障/busy 状态、当前运动学错误码、原始机械位置、最后已发送的物理目标及 BufferMode。801 表示参与运动系统的轴未就绪，具体原因仍需结合这条快照和 TIA 诊断；不会自动复位或重新授权。人工 E 重新授权会清除本地已发送目标的死区缓存，避免把报警中断的目标当作已完成；报警清除本身不重发目标。Windows 现场可用 `python -u main.py --camera-device 0 --camera-fast --plc 192.168.0.64 --plc-rate 30 --trigger-low-ms 10 2>&1 | Tee-Object -FilePath play_console.log` 保留控制台；故障时提供这条快照而不是仅截取“禁止运动”。

PLC 模式需要 `python-snap7`，不再强制降级到 1.3；已有可工作的现场环境应保留。已用本地 S7 服务验证 1.3、2.0.2、3.0.0、3.2.1 的连接、心跳、反馈、人工使能及运动下发，尚未在 Windows 或实机复测。1.x/2.x 使用对应的 Snap7 原生库；3.x 是纯 Python 实现，需要 Python 3.10 及以上。Python 3.8 使用 1.3 时仍需 `pkg_resources`：`python3 -m pip install "python-snap7==1.3" "setuptools<81"`。离线测试不需要 Snap7。导入或连接失败会输出具体错误。按位寻址采用 [S7 WriteArea / S7WLBit 协议](https://snap7.sourceforge.net/sharp7.html)，不回退整字节读改写；旧版 3.0 的单 BIT 长度编码在发送前修正，应答仍须成功才接受连接。

常用参数：`--camera-device`、`--preview-fps`、`--calibration`、`--table-calibration`、`--disable-undistort`、`--disable-homography`、`--roi`、`--lower/--upper`、`--record`、`--plc`、`--plc-rate`。冰壶 HSV 默认阈值为 `--lower 170 100 80 --upper 10 255 255`，会同时覆盖红色在 HSV 色调 0 和 179 两端的范围；灯光、曝光或壶颜色不同需用实拍图重新调。

默认检测区域为原始像素矩形 `4 10 1216 710`，覆盖当前 1280×720 标定台面；主程序及追踪诊断工具无需再传 `--roi`。换相机位置或画幅后可用 `--roi X Y W H` 覆盖；全台面搜索也会包含红色球槌，须确认未误识别后再联动 PLC。

Jetson 的 Python 3.8 运行时需使用已包含延迟类型注解修复的最新代码，并确保该解释器安装了 `numpy`、`cv2`、`gi` 等运行依赖；无标定开关只跳过标定，不会跳过摄像头和 GStreamer 依赖。

Jetson 上若看到 `unknown type GstFraction`，需更新到通过 `Gst.Structure.get_fraction("framerate")` 读取协商帧率的版本。`/dev/video*` 中可能有非采集设备；用 `v4l2-ctl --device=/dev/video0 --list-formats-ext` 核对 MJPEG 分辨率/帧率，再用 `python3 air_hockey/tools/test_camera.py --device /dev/video0 --benchmark --duration 10` 单独诊断该采集节点。

采集诊断：`python3 air_hockey/tools/test_camera.py --benchmark --duration 10` 显示实际图像尺寸、驱动报告 FPS、实测采集 FPS 和采集耗时；Linux GStreamer 另提供 appsink 转换计时，Windows DirectShow 的解码/颜色处理计入 `read()`。`python3 air_hockey/tools/test_gstreamer_transfer.py` 只用于 Linux GStreamer 管道；独立管道的 FPS 不是应用采集 FPS。`main.py --headless --benchmark-seconds 30 --perf-json benchmark.json` 报告视觉链路分段耗时；请求/驱动报告 FPS、实测采集 FPS、处理 FPS 与各阶段耗时不可混称。`Frame.timestamp` 是 host `perf_counter` 的后端读取完成时刻，Gst PTS（仅 GStreamer 提供）是未映射的管道时钟域原值，不能声称曝光到显示或 PLC 端到端延迟。本机无 Jetson；Windows 相机分支尚未接实机测量。

预览链路先将原始帧缩到 640 像素宽，再按相同比例绘制 ROI、观测、轨迹和 AI 目标；`render()` 全分辨率输出仍保持原有行为。预览按稀疏采样的全画面亮度做缓慢校正，以减轻整幅画面的明暗闪烁；只修改显示图像，不改检测、追踪或 PLC 输入，局部条纹仍需稳定照明。本机 1280×720 合成画面、各预热 3 次后各测 50 次的单次对比：预览编码耗时 p50 从约 1.97 ms 降为约 0.96 ms；这不是 Jetson 或真实相机的性能结论。GStreamer 后端仍保留 cv2 的 BGRx→BGR 转换，不在缺少 Jetson 实测时改动硬件协商管道。

等价计算优化：预览轨迹通过 `CameraGeometry.table_to_raw_batch` 批量投影；物理微步跳过不可能发生的墙/门柱碰撞检查，并复用速度长度计算。检测频率、物理步长、记录落盘和 PLC 时序不变。本机同一 205 点轨迹、预热后各测 100 次，中位投影耗时由 1.287 ms 降至 0.070 ms；横向速度 1100 的完整预测由 3.894 ms 降至 1.984 ms。保存画面渲染像素差为零，1000 组状态各推进 10 步的最大数值差约 2.3e-13；不代表 Jetson 或机械端到端性能。

预览状态栏使用固定宽高容器并换行：识别到冰壶后较长的位置、速度和预测文本不会撑宽 Tk 窗口，也不会使居中的预览图像左右跳动。

辅助工具：

```bash
python3 air_hockey/simulation/run_simulation.py     # 无摄像头轨迹仿真
python3 air_hockey/evaluation/run_evaluation.py     # 算法评估报告（JSON + 控制台）
python3 air_hockey/evaluation/run_evaluation.py --runs 20 --noise 0 --seed 42 --compare-estimators
python3 air_hockey/tools/calibrate_table.py         # 球台四点 Homography 标定
python3 tests/run_tests.py                          # 全量自动化测试
```

---

## 📊 运行日志与实验数据

`--record <path>` 输出统一格式的版本化 JSON 运行日志，真实运行与仿真格式一致，每帧包含：

`plc_request` 记录本帧经 AI 半场限幅的目标（场地逻辑坐标），并非 PLC 接收或电机到位确认；`version=3` 不兼容旧的冰壶状态/比分载荷记录解析。

Linux 和 Windows 上，设置路径后逐帧写入 `<path>.journal` 并执行 `fsync`，内存不随帧数增长；正常关闭时流式生成上述版本 3 JSON，再删除日志。导出失败保留日志且允许重试关闭。输出 JSON 与日志会短暂共存，磁盘须留出导出空间；完整预测也会增加每帧日志大小。只有显式调用 `to_dict()` / `to_json()` 才把全部帧读入内存，未设置路径的有限长度离线记录仍使用内存。

文件锁使用系统标准库：Linux 用 `fcntl.flock`，Windows 用 [msvcrt.locking](https://docs.python.org/3/library/msvcrt.html#msvcrt.locking)，不需要安装 `fcntl`。Windows 快照复用持锁句柄，恢复后先关闭文件再删除日志。Linux 另同步目录；Windows 标准库不能 `fsync` 目录，不承诺突然断电后的目录项持久性。Windows DirectShow 分支在本机通过模拟摄像头的回归测试，但未在 Windows 相机或 PLC 实机验证；Jetson 实时采集仍需 Linux/NVIDIA GStreamer 环境。

异常退出后，先确认旧进程已退出，再恢复；仅丢弃最后一个未完成行，完整行损坏会报错并保留日志。正在运行的日志由文件锁保护，同路径的新记录器也拒绝覆盖未恢复日志：

```bash
python3 -c "from air_hockey.recording import RuntimeRecorder; RuntimeRecorder.recover('run.json')"
```

完整预测与逐帧同步落盘的处理耗时须在 Jetson 实测，不能从本机结果保证 200 FPS。

```json
{
  "format": "curling_recording", "version": 3, "source": "runtime", "meta": {...},
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
│   ├── control/                # PlcControlAdapter + PLCLink（AI/轴反馈闭环）
│   ├── recording/              # 统一运行日志（JSON schema + RuntimeRecorder）
│   ├── simulation/             # 无摄像头运动仿真测试环境
│   ├── evaluation/             # 多组实验误差评估与报告
│   ├── calibration/            # 相机内参标定与去畸变工具
│   ├── app/                    # VisionRuntime 运行核心 + 渲染 + FPS GUI
│   ├── tools/                  # 标定 / 检测 / 追踪 / 预测 等诊断脚本
│   └── 交接文档.md             # 面向开发者的详细技术与工程交接文档
├── calibration/                # 相机内参在库；table_homography.npz 须在实台标定后生成
├── 冰壶仿真/                   # 桌面仿真 GUI 与 GUI 配置
└── tests/                      # 自动化测试套件
```

---

## 🎯 课题研究进展与展望

### 已完成工作
* [x] Linux / Jetson GStreamer 与 Windows DirectShow 相机采集后端（尚未做 Windows / Jetson 实机测量）
* [x] 颜色 + 几何约束检测算法与动态局部 ROI 加速
* [x] 原始像素单目标常速追踪与漏检维持
* [x] **卡尔曼滤波状态估计层**（常量速度模型，抑制视觉抖动）
* [x] 单目内参标定与点级去畸变；**球台四点 Homography 透视校正**
* [x] 视觉、仿真、AI 三端统一的微步物理轨迹预测核心（`PredictionState`）
* [x] 独立 `ControlCommand` 轨迹规划与 DB1/DB18 目标/实际位置闭环软件链（当前项目的视觉→实机路径仍须现场验证）
* [x] 仿真测试环境、多组实验评估与统一格式运行日志
* [x] 西门子 S7 PLC 通信底层封装
* [x] 自动化功能回归测试套件

### 后续展望与改进方向
* [ ] **实台物理参数辨识**：在真实气浮台采集滑行/碰撞数据，回归摩擦与恢复系数。
* [ ] **视觉 AI → PLC 实机复核**：结合机械联调成果，逐项确认 DB 偏移、坐标方向、保护状态和端到端执行延迟。
* [ ] **多目标跟踪**：扩展为多冰壶/球槌协同跟踪与状态机。
* [ ] **自动化 CI**：配置 GitHub Actions 执行 `compileall` 与全部单元测试。

---

## 📑 技术交接与开发文档

关于更底层的模块接口、坐标系数学定义、Jetson 手动锁频脚本与技术债务分析，请参考：
👉 [Air Hockey 项目交接文档 (air_hockey/交接文档.md)](air_hockey/交接文档.md)
