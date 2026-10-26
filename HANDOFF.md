# 项目交接与当前状态 (HANDOFF.md)

## 1. 当前系统架构与控制模式
- **PLC 已换成 S7-1200（CPU 1211C，TIA 工程 1211kongqi）**（2026-10-26）：X/Y 各一个 `MC_MoveAbsolute`，没有运动机构、没有 BufferMode / 舍入间距。Python 只写 DB1.12 目标（Real：X 写 [1]，Y 同写 [2][3]）+ DB1.90 指令序号。全部偏移见 `control/protocol.py` 顶部注释（按 Standard 布局从工程档案算出，DB1 共 120 字节、DB18 共 54 字节）。
- **纯连续物理导引**：控制链路由连续物理导引与门线预判生成目标；状态机标签仅用于监控展示。
- **只有终点模式**：只下发 AI 终点；新序号会让 MC_MoveAbsolute 重新触发 = 中止当前定位改去新终点，轴忙时新终点变化 < 10mm 不发（`ENDPOINT_RESEND_MM`）。
- **冲门快球强制门线堵截**：冲门且穿线点在球门内时，Y 钉在 140 门线、X 提前横移堵截；判据带迟滞（`ai.py` 的 `FAST_SHOT_*`）。

## 2. 核心文件
- `air_hockey/control/protocol.py`：DB1/DB18 偏移（1211C 版）。
- `air_hockey/control/interface.py`：`_prepare_move` 坐标换算 + 3mm 死区；`_write_payload` 写 4 个 Real；位置按 Real 读 DB1.2/6。命令余量在 `control/limits.py`。
- `air_hockey/ai.py`：冲门判断 `_goal_guard_decision`、提前量 `_lead_target`、迎击 `_strike_through`。
- `air_hockey/app/rally.py`：连续物理导引、二次追打放手规则与防抖。

## 3. 测试与验证状态
- `python3 tests/run_tests.py`：342 项全过（删了 TransitionParameter 写失败用例）。测试假 PLC 的 DB 长度已改为 120 / 54，读超长会报错，和真 PLC 一致。
- 本机用 snap7 3.2.1 自带服务端走过一遍：状态读取、目标块、序号、气缸位都写到了正确地址；旧长度 58 读 DB18 会报 Invalid address（说明旧代码连这台 PLC 必然显示未连接）。**尚未连实机。**

## 4. PLC 侧待确认 / 待补（TIA 里改，Python 管不到）
- [ ] FC1 网络 6 两个 MC_MoveAbsolute 的 Position 接的是 `XY轴直线绝对` 哪个下标；确认后把 `_write_payload` 里多写的 Y 去掉。
- [ ] 两个 MC_MoveAbsolute 的 **Velocity 没接线**，S7-1200 默认 10.0（单位/s），轴会极慢；要接到一个速度值（如工艺对象最大速度）。
- [ ] OB30 心跳超时会置 `停止所有运动触发`（DB1.88.2），但 FC1 里没人用它 -> 通信丢失时轴不会停。要加 MC_Halt 或撤 MC_Power。
- [ ] MC_Home 的 Mode 没接 = 0（直接把当前位置设为原点，不找参考点）；按现场需要定。
- [ ] DB18 32 以后的诊断字段（ErrorDetail.Number / error / RemainingDistance）PLC 没写，Python 读到恒 0，不影响安全互锁。

## 5. 挡不住球的根因（2026-10-21 play2 回放，仍有效）
- 球台标定点已重做（提交 395ab80），新点 (55,641)(56,95)(1059,72)(1084,622)。
- 墙弹性 `WALL_RESTITUTION=0.65`、`WALL_TANGENT_RETENTION=0.6`；直线摩擦实测约 156（模型 72），暂不改。

## 6. 下一步
- [ ] 连 1211C 实机：先看界面能否显示已连接、位置是否跟 TIA 一致，再使能小范围走点，确认 X/Y 方向和下标。
- [ ] 新标定 + `collision_aware` 卡尔曼上实机，带 `--record` 打侧墙反弹球，看是否跟得上。
- [ ] `STONE_RADIUS`=26 用贴墙球心复核；球台视野内不要有红色杂物。
