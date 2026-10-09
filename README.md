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
RallyController         实时单回合：WAITING / DEFENDING / RETURNED / RECOVERING
  │  固定防守线拦截 / READY（保留原 AI 内部计算，不下发主动进攻目标）
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

**现场安全边界**：视觉计划性抽帧仍可预测目标；一次实际检测漏检后立即清除动态拦截目标并结束本回合，随后只请求固定 READY 回位。READY 不依赖冰壶确认，但仍必须满足 PLC 人工授权与全部安全互锁；动态拦截仍要求 confirmed track、有效状态和 AI 输出。所有上位机控制位采用同步 S7 BIT 写入，不再读改写整个共享字节，保留 PLC/HMI 的相邻故障、停止、锁存和圆弧位。运动数据先写 `DB1.20..55` 的 36 字节 `[X,0,Y,0] + BufferMode`，成功后单独置 `DB1.56.0`；DB 地址不变。连接必须先成功初始化心跳位并清除直线触发位，才允许后续授权。单个位原子写入不等于整组命令原子完成，部分失败不能撤回 PLC 已接受的位；仍须现场验证 PLC 扫描、触发脉宽和独立失联停止。软件暂停只清除后续目标，不能撤销已经进入网络发送过程的运动指令，**不得作为急停使用**。

### 单回合现场联调（先观察，再运动）

实时模式现在只做基础挡球/回击：上半场是机器侧，`vy < -25` 为来球；READY 复用原 AI HOME `(300, 139.52)`。只有确认轨迹的真实检测、有效新鲜状态及 READY 到位才启动防守；慢速抖动、候选轨迹和 outgoing 不启动。用预测轨迹第一次向上的 `y=184.52` 前接触线交点确定横向位置，球槌保持 `y=139.52`，不追预测终点。到达时间采用路径长度/速度上界的保守估计（扣除状态年龄），不是采样点序号推算；新移动需比估计到达时间早 85 ms。已经覆盖拦截点则保持，不在接触前因时间裕量不足撤位。

回球确认要求近期观测/观测线段靠近实际球槌（接触距离 45 + 容差 14），随后至少两次不同时间戳的真实检测持续 `vy > 25`，跨度至少 40 ms、向人方位移至少 7；predict 帧不计数，反向/低速硬检测会重置计数。确认前只保持已到位的拦截位置（未到位则清目标），不追 outgoing，也不因单帧符号变化撤位；RETURNED 才立即请求 READY。实机回位须新鲜实际反馈、非 busy、距 READY ≤14；反馈失效不会靠计时假装回位成功。5 秒未返回、丢失/换 ID 或穿过防守线判失败并回位；下次须旧轨迹失效、新确认 ID，或再次从人方半场入射。

```bash
# A：不放球；离线先查看映射，随后连接现场 PLC（IP 按设备替换）
python3 air_hockey/tools/test_rally_points.py --dry-run
python3 air_hockey/tools/test_rally_points.py --plc 192.168.0.64
# 终端逐次输入 e / h / c / n / q 并回车；回零/复位后须重新 e。
# 每个 n 只请求一处：READY → LEFT → CENTER → RIGHT → READY，ARRIVED 后再 n。

# B：观察模式；PLC 只读反馈/心跳，不发运动，也不提供轴使能/回零/复位操作。
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --dry-run --record shadow.json

# C–F：标定、坐标方向、速度与安全隔离确认后才运行。GUI E 人工使能，H 回零，C 复位。
python3 -u main.py --camera-device /dev/video2 --plc 192.168.0.64 --mallet-speed 500 --record rally.json
```

Windows 用现场已有 Python 环境，将 `python3` 改为 `python`、相机改为 `--camera-device 0`。不得使用关闭标定的开关驱动实机。`--mallet-speed` 单位是场地逻辑单位/秒，只影响可达性，不设置 PLC 速度；500 是软件估计，不是实测机械保证。A 的相邻横向点相隔 132 单位，可结合输出的 host request-to-arrival 时间估计保守速度，再在 C 验证；该时间含通信/轮询，不是纯电机运动时间。没有 PLC 时使用标明为 software 的球槌位置，只能证明软件状态逻辑。

B 连续观察 10 球；若要检查 RETURNED，在轴禁止运动并确认硬件安全后使用被动挡板/手动回球，不能把 dry-run 当作球槌已经击球。C 使用可靠隔离/接球装置防止实际接触，核对预测点、时间与实际位置；D 去除装置后只试一次真实接触；E 完成返回→回位→READY；F 不重启连续 10 次。每次转换打印原因，GUI/周期控制台显示 confirmed、方向、速度、拦截点、到达时间下界、球槌旅行时间、目标、实际位置及 PLC armed/safe。

控制台/退出输出 `incoming_detected / defense_entered / intercept_attempted / successful_return / failed_intercept / recovered`；前两个是已接受的确认来球/防守次数，attempted 是可达请求，不是电机到位，returned 是视觉返回证据，不是碰撞传感器证明。`--record` 的 `plc_request` 也是请求而非执行确认，meta 标明 `dry_run / live / no_plc`。状态转换/计数保存在控制台，需持久保存可用终端日志或 Linux `| tee rally-console.log`。

故障归类：无 track→Detection；tentative/lost→Tracker；速度方向不符→Direction；无交点/状态不匹配→Predictor/intercept；`reachable=False`→Unreachable；`armed=False`→PLC authorization；实际位置赶不上且旅行时间超过到达时间→Motion too slow；A 的方向/坐标不符→Mapping；5 秒未确认返回→Return detection/missed intercept；回位超过 5 秒或反馈失效→Recovery。先定位阶段，不调 estimator/物理参数；本版不保证回击角度、复杂进攻、高难度来球或门柱特殊碰撞。

PLC 模式需要 `python-snap7`，不再强制降级到 1.3；已有可工作的现场环境应保留。已用本地 S7 服务验证 1.3、2.0.2、3.0.0、3.2.1 的连接、心跳、反馈、人工使能及运动下发，尚未在 Windows 或实机复测。1.x/2.x 使用对应的 Snap7 原生库；3.x 是纯 Python 实现，需要 Python 3.10 及以上。Python 3.8 使用 1.3 时仍需 `pkg_resources`：`python3 -m pip install "python-snap7==1.3" "setuptools<81"`。离线测试不需要 Snap7。导入或连接失败会输出具体错误。按位寻址采用 [S7 WriteArea / S7WLBit 协议](https://snap7.sourceforge.net/sharp7.html)，不回退整字节读改写；旧版 3.0 的单 BIT 长度编码在发送前修正，应答仍须成功才接受连接。

常用参数：`--camera-device`、`--preview-fps`、`--calibration`、`--table-calibration`、`--disable-undistort`、`--disable-homography`、`--roi`、`--lower/--upper`、`--record`、`--plc`、`--plc-rate`。冰壶 HSV 默认阈值为 `--lower 170 100 80 --upper 10 255 255`，会同时覆盖红色在 HSV 色调 0 和 179 两端的范围；灯光、曝光或壶颜色不同需用实拍图重新调。

默认检测区域为原始像素矩形 `4 10 1216 710`，覆盖当前 1280×720 标定台面；主程序及追踪诊断工具无需再传 `--roi`。换相机位置或画幅后可用 `--roi X Y W H` 覆盖；全台面搜索也会包含红色球槌，须确认未误识别后再联动 PLC。

Jetson 的 Python 3.8 运行时需使用已包含延迟类型注解修复的最新代码，并确保该解释器安装了 `numpy`、`cv2`、`gi` 等运行依赖；无标定开关只跳过标定，不会跳过摄像头和 GStreamer 依赖。

Jetson 上若看到 `unknown type GstFraction`，需更新到通过 `Gst.Structure.get_fraction("framerate")` 读取协商帧率的版本。`/dev/video*` 中可能有非采集设备；用 `v4l2-ctl --device=/dev/video0 --list-formats-ext` 核对 MJPEG 分辨率/帧率，再用 `python3 air_hockey/tools/test_camera.py --device /dev/video0 --benchmark --duration 10` 单独诊断该采集节点。

采集诊断：`python3 air_hockey/tools/test_camera.py --benchmark --duration 10` 显示实际图像尺寸、驱动报告 FPS、实测采集 FPS 和采集耗时；Linux GStreamer 另提供 appsink 转换计时，Windows DirectShow 的解码/颜色处理计入 `read()`。`python3 air_hockey/tools/test_gstreamer_transfer.py` 只用于 Linux GStreamer 管道；独立管道的 FPS 不是应用采集 FPS。`main.py --headless --benchmark-seconds 30 --perf-json benchmark.json` 报告视觉链路分段耗时；请求/驱动报告 FPS、实测采集 FPS、处理 FPS 与各阶段耗时不可混称。`Frame.timestamp` 是 host `perf_counter` 的后端读取完成时刻，Gst PTS（仅 GStreamer 提供）是未映射的管道时钟域原值，不能声称曝光到显示或 PLC 端到端延迟。本机无 Jetson；Windows 相机分支尚未接实机测量。

预览链路先将原始帧缩到 640 像素宽，再按相同比例绘制 ROI、观测、轨迹和 AI 目标；`render()` 全分辨率输出仍保持原有行为。预览按稀疏采样的全画面亮度做缓慢校正，以减轻整幅画面的明暗闪烁；只修改显示图像，不改检测、追踪或 PLC 输入，局部条纹仍需稳定照明。本机 1280×720 合成画面、各预热 3 次后各测 50 次的单次对比：预览编码耗时 p50 从约 1.97 ms 降为约 0.96 ms；这不是 Jetson 或真实相机的性能结论。GStreamer 后端仍保留 cv2 的 BGRx→BGR 转换，不在缺少 Jetson 实测时改动硬件协商管道。

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
