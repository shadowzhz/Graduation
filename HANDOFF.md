# 项目交接与当前状态 (HANDOFF.md)

## 1. 当前系统架构与控制模式
- **纯连续物理导引**：已架空 `RallyController` 的离散模式切换决策，控制链路全面转由连续物理导引生成目标；`WAITING / DEFENDING / ATTACKING / RECOVERING` 彻底退化为观察器显示标签，不截断或篡改运动目标。
- **状态防抖与空间迟滞**：引入中线迟滞带（进攻态需退至 392 才解除）与速度死区（vy>20 才退回 READY），并在状态切换间增加 80ms 最小驻留时间，消除零点与分界线高频横跳。
- **全时段平滑混合下发**：在开启目标限速斜坡（`target_slew_mm > 0`）时，PLC 指令 100% 保持在 `BufferMode=5`（混合滤波），杜绝西门子 TO_Kinematics 中途停稳顿挫。
- **视觉自适应搜索与候选关联**：`StoneDetector` 支持多候选旧轨迹优先软关联；`VisionRuntime` 动态 ROI 沿运动轴自适应延展并包含反向包络；预览轨迹改用 `table_to_raw_batch` 批量投影。

## 2. 核心文件与改动位置
- `air_hockey/app/rally.py`：连续物理导引实现，消灭模式截断。
- `air_hockey/control/plc.py`：纯平滑混合下发，消除中间步停稳。
- `air_hockey/vision/geometry.py`：OpenCV 批量坐标逆投影。
- `air_hockey/physics.py`：门柱与边墙粗筛，复用速度模长。
- `air_hockey/recording/recorder.py`：已有最终记录文件防覆写保护。

## 3. 测试与验证状态
- 全量自动化测试套件 `python3 tests/run_tests.py`：**342 项全部通过，0 失败**。
- 静态死代码扫描已核实，无阻碍核心运行的死分支。

## 4. 下一步待办事项
- [ ] 实机低速连接 S7-1500T 验证消除 BufferMode=1 后的连续对打顺滑度。
- [ ] 结合现场机械臂实际遮挡情况，验证长时遮挡后的轨迹接回与防乌龙球策略。
- [ ] 视需要清理历史未用符号（如 `COMM_READ_SIZE_LEGACY`）。
