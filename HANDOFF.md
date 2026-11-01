# 项目交接与当前状态 (HANDOFF.md)

## 1. 当前系统架构与控制模式
- **PLC 已换成 S7-1200（CPU 1211C，TIA 工程 1211kongqi）**（2026-10-26）：X/Y 各一个 `MC_MoveAbsolute`，没有运动机构、没有 BufferMode / 舍入间距。Python 只写 DB1.12 目标（Real：X 写 [1]，Y 同写 [2][3]）+ DB1.90 指令序号。全部偏移见 `control/protocol.py` 顶部注释（按 Standard 布局从工程档案算出，DB1 共 120 字节、DB18 共 54 字节）。
- **纯连续物理导引**：控制链路由连续物理导引与门线预判生成目标；状态机标签仅用于监控展示。
- **只有终点模式**：只下发 AI 终点；新序号会让 MC_MoveAbsolute 重新触发 = 中止当前定位改去新终点，已取消轴忙时的10mm目标变化门槛，仅对 REAL 编码相同的终点去重；原有实际到位3mm死区保留。
- **冲门快球强制门线堵截**：冲门且穿线点在球门内时，Y 钉在 181.5 门线（若球已越过门线则退守球与门之间）、X 提前横移堵截；判据带迟滞（`ai.py` 的 `FAST_SHOT_*`）。

- **场地坐标改为各向同性**（2026-10-29）：场地 528×968 单位（原 528×668），1 单位 ≈ 0.909mm，中线 y=530、READY (300,181.5)。`calibration/table_homography.npz` 已按存档的去畸变四角重算；旧坐标的记录（version ≤ 3）不能和新记录直接比较。
- **AI 换球/丢轨时清记忆**：`AirHockeyAI.reset()` 由 `VisionRuntime` 在换身份和丢轨时调用，回收阶段与冲门迟滞不再带到下一颗球。
- **漏检宽限与遮挡保持按物理时间**（2026-10-31）：追踪器宽限 = min(8 帧, 0.25s)；遮挡保持从最后真实观测起算 1.0s，漏检预测帧不续期。
- **Rally 记忆绑定轨迹代号**（2026-10-31）：换身份、无效状态、AI 目标缺失/非有限、PLC 撤权/回零/重连后清空接触与保持记忆，旧目标不会复活。
- **PLC 授权代号驱动重置**（2026-11-01）：`PLCLink.arm_generation` 在真正撤权/复权时各 +1；LiveSession 比对代号，复权那一帧先清 Rally 记忆再用本帧观测重建；撤权期间攒下的记忆不会带进来，两帧之间的撤权再复权也认得出。
- **目标绑定授权代号**（2026-11-01）：视觉在处理帧前捕获 `arm_generation` 并随目标传递；PLC 在接收与最终发送时都检查代号。等待授权期间发布、或跨越撤权/复权计算的旧目标不能进入新授权；工具未指定代号时绑定发布时的代号。此门禁不能撤回已进入网络写入或 PLC 已接受的运动，不能代替独立停止。
- **80ms 驻留只锁纵向策略线**（2026-11-01）：ATTACKING/DEFENDING 切换的驻留期内继续采用最新 AI 目标的 X，只保持旧目标的 Y；驻留结束才切 Y，X 不会跟着二次跳。
- **回零 = MC_Home Mode 0**：轴不动，直接把当前位置设为原点；轴不在参考位置时按 H / Shift+X / Shift+Y 会让坐标系错位（Caps Lock 下小写 x/y 也会触发）。

## 2. 核心文件
- `air_hockey/control/protocol.py`：DB1/DB18 偏移（1211C 版，已清理旧 `TRIGGER_BYTE` 触发位并兼容 `snap7.type` 多变量读取）。
- `air_hockey/control/interface.py`：`_prepare_move` 坐标换算 + 3mm 死区；`_write_payload` 写 4 个 Real；`_read_multi_vars` 兼容 `Cli_ReadMultiVars` 与 `read_multi_vars`。命令余量在 `control/limits.py`：机械行程 X -190..222 / Y -129..187，两轴内缩 5mm，X 再多收 6mm、Y 再多收 10mm，命令范围 X -179..211 / Y -114..172；`PlcControlAdapter` 已同步采用 `plc.reachable_bounds`。
- `air_hockey/ai.py`：冲门判断 `_goal_guard_decision`（含越过防守线退守）、提前量 `_lead_target`、迎击 `_strike_through`、中线进攻迟滞 `ATTACK_HYSTERESIS`、`reaction_timer` 重置修复。
- `air_hockey/camera/`：非 Jetson 软件解码直接协商 `BGR` 免除每帧四通道转换拷贝；`_stop_locked` 异常退出确保清理线程与后端引用。

## 3. 测试与验证状态
- `python3 tests/run_tests.py`：349 项完整回归通过（无多变量读取回退告警）。已清理废弃空目录 `air_hockey/planning`。
- 本机用 snap7 3.2.1 自带服务端走过一遍：状态读取、目标块、序号、气缸位都写到了正确地址；旧长度 58 读 DB18 会报 Invalid address（说明旧代码连这台 PLC 必然显示未连接）。**尚未连实机。**

## 4. PLC 侧待确认 / 待补（TIA 里改，Python 管不到）
- [ ] FC1 网络 6 两个 MC_MoveAbsolute 的 Position 接的是 `XY轴直线绝对` 哪个下标；确认后把 `_write_payload` 里多写的 Y 去掉。
- [ ] 两个 MC_MoveAbsolute 的 **Velocity 没接线**，S7-1200 默认 10.0（单位/s），轴会极慢；要接到一个速度值（如工艺对象最大速度）。
- [ ] OB30 心跳超时会置 `停止所有运动触发`（DB1.88.2），但 FC1 里没人用它 -> 通信丢失时轴不会停。要加 MC_Halt 或撤 MC_Power。
- [ ] MC_Home 的 Mode 没接 = 0（直接把当前位置设为原点，不找参考点）；按现场需要定。
- [ ] DB18 32 以后的诊断字段（ErrorDetail.Number / error / RemainingDistance）PLC 没写，Python 读到恒 0，不影响安全互锁。

## 5. 挡不住球的根因（2026-10-21 play2 回放，仍有效）
- 球台标定点已重做（提交 395ab80），新点 (55,641)(56,95)(1059,72)(1084,622)。
- 墙弹性 `WALL_RESTITUTION=0.65`、`WALL_TANGENT_RETENTION=0.6`；摩擦已改为 230 单位/s²（≈210mm/s²，play2 只有 3 段干净滑行：225/318/561mm/s²），需更多 `--record` 数据复核。

## 6. 下一步
- [ ] 连 1211C 实机：先看界面能否显示已连接、位置是否跟 TIA 一致，再使能小范围走点，确认 X/Y 方向和下标。
- [ ] 新标定 + `collision_aware` 卡尔曼上实机，带 `--record` 打侧墙反弹球，看是否跟得上。
- [ ] `STONE_RADIUS`=26 用贴墙球心复核；球台视野内不要有红色杂物。
