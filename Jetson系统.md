# Jetson 系统报告

> 勘察时间：2026-08-19，约 16:52-17:06（Asia/Shanghai）  
> 勘察方式：只读命令检查。除本报告文件外，没有安装、卸载、升级软件，没有修改配置，没有重启系统，也没有执行修复操作。

## 1. 机器身份

### 1.1 已确认身份

| 项目 | 结果 | 主要命令依据 |
|---|---|---|
| Jetson 型号 | **NVIDIA Jetson Xavier NX Developer Kit** | `cat /proc/device-tree/model` |
| SoC | **Tegra194** | `cat /proc/device-tree/compatible`、`jetson_clocks --show` |
| 设备树兼容标识 | `nvidia,p3449-0000+p3668-0000`、`nvidia,p3509-0000+p3668-0000`、`nvidia,tegra194` | `/proc/device-tree/compatible` |
| L4T | **R35.6.5** | `cat /etc/nv_tegra_release`、`dpkg-query` |
| 主机名 | `zdh-desktop` | `hostname` |
| 架构 | `aarch64`，ARM 64 位 | `uname -m`、`lscpu` |

型号不是根据外观或经验推测，而是由设备树直接报告。`p3449`/`p3668` 是 Xavier NX Developer Kit 相关的板级标识。

### 1.2 JetPack 版本的边界

已确认 L4T 为 `35.6.5`，并且 `nvidia-l4t-*` 组件完整存在。系统中没有安装以下 JetPack 元包：

```text
nvidia-jetpack
nvidia-jetpack-runtime
nvidia-jetpack-dev
```

因此本次不能从本机软件包直接确认一个精确的 `JetPack 5.x.y` 元版本。可以确定这是 **JetPack 5 / L4T R35 系列环境**，但报告不把具体 JetPack 小版本当成已确认事实。后续学习时优先以本机 L4T `R35.6.5` 和各组件实际版本为准。

## 2. 硬件规格

### 2.1 CPU

通过 `lscpu` 和 `/proc/cpuinfo` 得到：

- ARMv8 `aarch64`。
- NVIDIA 实现，CPU implementer `0x4e`，CPU part `0x004`。
- 系统枚举到 6 个 CPU：`0-5`。
- 当前真正 online 的只有 `CPU 0` 和 `CPU 1`，`/sys/devices/system/cpu/online` 为 `0-1`。
- `lscpu` 报告 1 个 socket、2 个 core、每个 core 1 个 thread；其余 4 个 CPU 当前 offline。
- CPU 频率范围约为 `115.2 MHz` 到 `1907.2 MHz`；当前策略为 `schedutil`。

当前运行的 Jetson 电源模式是：

```text
NV Power Mode: MODE_20W_2CORE
```

依据：`nvpmodel -q`、`tegrastats`、`/sys/devices/system/cpu/online`。

这解释了为什么芯片本身枚举为 6 CPU，但 `tegrastats` 显示后四个 CPU 为 `off`。这不是“CPU 硬件只有双核”，而是当前 nvpmodel 功耗模式只启用了两个核心。

### 2.2 内存

依据：`free -h`、`tegrastats`。

| 项目 | 当前值 |
|---|---:|
| Linux 可见总内存 | 6833 MiB，约 6.7 GiB |
| 已使用 | 约 4.2 GiB |
| 可直接 free | 约 100-148 MiB，随采样变化 |
| buff/cache | 约 2.3 GiB |
| available | 约 2.2 GiB |
| Swap 总量 | 约 3.3 GiB |
| Swap 已使用 | 约 14 MiB，采样时很少 |

`free` 中的 `free` 很小不一定表示内存耗尽，因为 Linux 会把空闲内存用于缓存；本机更应关注 `available`，当前仍有约 2.2 GiB。`tegrastats` 同时报告 RAM 约 `4439/6833 MB`。

### 2.3 GPU

确认到的 GPU 设备节点信息：

- SoC GPU 设备树节点：`/gv11b`。
- GPU 驱动标识：`gk20a`。
- 设备路径：`/sys/devices/platform/17000000.gv11b`。
- `gpu_powered_on`：`on`。
- `runtime_status`：`active`。
- 采样时 GPU load：sysfs 约 `15`，`tegrastats` 的 `GR3D_FREQ` 为 `0%`（采样时没有 GPU 3D 负载）。
- GPU 频率请求值：`1109250000`，约 1.109 GHz。
- GPU 温度：约 38.5-39.5°C。

Jetson 通常使用 Tegra 的 `tegrastats`、sysfs 和 nvgpu 接口，不依赖桌面服务器常见的 `nvidia-smi`。本机 `nvidia-smi` 命令不存在，因此不能用它查看 GPU。

### 2.4 存储

依据：`lsblk`、`df -hT`、`/sys/block/mmcblk0/device/*`。

- 主块设备：`/dev/mmcblk0`，约 `119.1G`。
- 设备信息：名称 `SN128`，内核报告类型 `SD`，制造日期 `03/2021`。
- 根分区：`/dev/mmcblk0p1`，ext4，约 `118.4G` 分区容量。
- 根文件系统实际可用容量：约 `92.7-93G`，使用率约 `16-17%`。
- 根文件系统挂载点：`/`。
- 另外存在多个小容量启动/固件分区（`mmcblk0p2` 到 `mmcblk0p22`），这是 Jetson 启动介质常见布局。
- `zram0` 和 `zram1` 各约 `1.7G`，用于压缩 Swap，不是额外物理磁盘。

本次没有发现已挂载的 NVMe 或 USB 存储设备。USB 枚举命令在当前检查上下文中因 libusb 权限/设备访问错误未能完成，因此不能据此断言没有 USB 设备。

### 2.5 其他硬件观察

- 有线网卡 `eth0` 的 MAC 地址为 `48:b0:2d:3d:89:27`。
- `wlan0` 存在，但当前 `DOWN` 且 `NO-CARRIER`。
- `usb0`、`rndis0`、`l4tbr0` 存在，但当前没有链路。
- 当前系统配置了 `nvargus-daemon`，说明 Jetson 相机栈服务存在并运行。
- 内核日志报告两个 `imx219` 相机探测失败，见“当前存在的问题”。

## 3. 操作系统

### 3.1 发行版与内核

依据：`cat /etc/os-release`、`uname -a`。

- 发行版：**Ubuntu 20.04.6 LTS (Focal Fossa)**。
- 内核：`5.10.216-tegra`。
- 内核架构：`aarch64`。
- 内核构建时间信息：`Wed Jul 1 02:07:52 PDT 2026`。
- 启动根设备：`/dev/mmcblk0p1`。
- 启动参数包含 `rootwait`、`rootfstype=ext4`、Jetson 的 `ttyTCU0` 控制台参数。

### 3.2 systemd 与启动方式

在宿主机层面：

- PID 1 是 `/sbin/init`，实际链接到 `/usr/lib/systemd/systemd`。
- `systemctl is-system-running`：`running`。
- systemd 版本：`245`。
- 本次 `systemctl --failed` 没有发现失败的 systemd unit。
- 当前启用了并运行的关键服务包括：`NetworkManager`、`systemd-resolved`、`docker`、`nvfancontrol`、`nvargus-daemon` 和 `ssh`。

说明：部分早期命令是在只读隔离 shell 中执行的，该 shell 自己的 PID 1 是隔离器，且不能访问宿主机 systemd 总线；后续使用宿主机只读权限复核后，以上 systemd 结论以宿主机结果为准。

### 3.3 用户、权限、Shell

当前用户：

```text
uid=1000(zdh) gid=1000(zdh)
```

宿主机用户组包括：

```text
zdh adm cdrom sudo audio dip video plugdev render i2c lpadmin gdm
sambashare weston-launch gpio jtop
```

- 当前 Shell：`/bin/bash`。
- 用户属于 `sudo`、`video`、`render`、`i2c`、`gpio`、`jtop` 等组，具备较多开发和硬件访问相关权限。
- 用户**不在 `docker` 组**；`getent group docker` 显示该组存在但没有 `zdh` 成员。

### 3.4 时间、启动时长、负载

采样时：

- 当前时间约为 `2026-08-19 17:02 CST`。
- 已运行约 `1 小时 27 分钟`。
- 登录用户数：2。
- load average 约为 `1.82, 2.39, 2.65`。
- 宿主机总进程数约 339；没有发现 zombie 进程。

由于当前只有两个 CPU online，load average 约 1.8 意味着系统已经接近两核容量，不能用“6 核”作为当前负载分母。

## 4. NVIDIA / JetPack 环境

### 4.1 L4T / Jetson 系统组件

`/etc/nv_tegra_release` 原始信息：

```text
# R35 (release), REVISION: 6.5, GCID: 46196569,
BOARD: t186ref, EABI: aarch64,
DATE: Wed Jul 1 09:42:50 UTC 2026
```

已安装的主要 `nvidia-l4t-*` 组件版本为：

```text
35.6.5-20260701022834
```

包括 `nvidia-l4t-core`、`nvidia-l4t-cuda`、`nvidia-l4t-kernel`、`nvidia-l4t-camera`、`nvidia-l4t-gstreamer`、`nvidia-l4t-multimedia`、`nvidia-l4t-jetson-multimedia-api`、`nvidia-l4t-nvpmodel`、`nvidia-l4t-nvfancontrol` 等。

### 4.2 CUDA

依据：`/usr/local/cuda/version.json`、`/usr/local/cuda/bin/nvcc --version`、CUDA dpkg 包。

- CUDA SDK：**11.4.19**。
- `nvcc`：**11.4.315**。
- CUDA 安装目录：`/usr/local/cuda-11.4`。
- `/usr/local/cuda` 通过 alternatives 指向 CUDA 11.4。
- 当前普通 shell 的 `PATH` 没有直接包含 `/usr/local/cuda/bin`，所以直接输入 `nvcc` 报“command not found”；使用绝对路径 `/usr/local/cuda/bin/nvcc` 可以正常执行。

这说明 CUDA 工具包实际安装了，但当前用户环境变量没有完整配置到 CUDA 编译器路径。

### 4.3 cuDNN

依据：`dpkg-query`、`ldconfig -p`。

- `libcudnn8`：**8.6.0.166-1+cuda11.4**。
- `libcudnn8-dev`：**8.6.0.166-1+cuda11.4**。
- cuDNN runtime、infer、train 等动态库可被 linker cache 找到。

### 4.4 TensorRT

依据：`dpkg-query`、`/usr/src/tensorrt/bin/trtexec --help`。

- TensorRT native runtime：**8.5.2**。
- `libnvinfer8`、`libnvinfer-dev`、`libnvinfer-plugin8` 等已安装。
- `trtexec` 存在于 `/usr/src/tensorrt/bin/trtexec`。
- `trtexec` 启动信息为 `TensorRT v8502`。
- Python 3.8 中没有发现可导入的 `tensorrt` 模块。

因此当前是“TensorRT C++/native 栈已安装，Python TensorRT 绑定未确认存在”的状态。

### 4.5 VPI

依据：`dpkg-query`、Python import。

- VPI：**2.4.8**。
- 已安装 `libnvvpi2`、`vpi2-dev`、`vpi2-samples`、`vpi2-demos`。
- `python3.8-vpi2` 和 `python3.9-vpi2` 存在。
- 宿主机 Python 可以导入 VPI 并报告版本 `2.4.8`。
- 导入时出现：`PVA is not available and may be oversubscribed in the system`。

这条警告表明本次导入时 PVA 加速设备不可用或资源状态不满足要求；它不等同于 VPI 软件包未安装，后续需要用实际 VPI 算子和硬件状态进一步验证。

### 4.6 OpenCV 与 GStreamer

依据：`pkg-config`、`python3 -c 'import cv2'`、`gst-launch-1.0 --version`、`gst-inspect-1.0`。

- OpenCV：**4.5.4**。
- Python `cv2`：**4.5.4**。
- OpenCV 构建包含：NEON/FP16、FFmpeg、GStreamer、V4L2、GTK2、TBB。
- GStreamer：**1.16.3**。
- Jetson 关键插件在宿主机检查中存在：`nvarguscamerasrc`、`nvv4l2decoder`、`nvv4l2h264enc`、`nvvidconv`。
- `nvargus-daemon` 进程正在运行。

早期隔离 shell 中加载 OpenCV/GStreamer 设备插件时出现了 `NvRmMemInit` 和段错误提示，这是因为该检查 shell 没有完整映射 Jetson 设备节点；宿主机层面重新检查后，Jetson GStreamer 插件均可被发现。插件“存在”不代表相机一定已连接或可正常采集，当前相机 I2C 探测错误仍需单独关注。

## 5. 开发环境

### 5.1 编译和版本控制工具

依据：命令版本和 dpkg 包状态。

| 工具 | 状态 |
|---|---|
| gcc | 9.4.0，可用 |
| g++ | 9.4.0，可用 |
| make | 4.2.1，可用 |
| cmake | 当前命令不存在；dpkg 中也未发现已安装的 `cmake` 包 |
| git | 2.25.1，可用 |
| ninja | 不存在 |

OpenCV 的构建信息显示它过去构建时使用过 CMake 3.16.3，但这不表示当前系统仍安装 CMake；当前 `cmake --version` 的结果是 `command not found`。

### 5.2 Python

依据：`python3 --version`、`python3 -m pip list`、Python import 探测。

- Python：**3.8.10**，解释器 `/usr/bin/python3`。
- Python prefix 和 base prefix 都是 `/usr`，当前不是 Conda 或 venv 环境。
- pip 命令：**25.0.1**，位置 `/usr/local/lib/python3.8/dist-packages/pip`。
- dpkg 的 `python3-pip` 包版本为 `20.0.2-5ubuntu1.11`，但实际执行的 pip 是 `/usr/local` 中的 25.0.1；这说明系统包版本和当前命令版本不一致。
- Conda、Mamba、Micromamba、uv、virtualenv、ninja、node、npm 均未发现。
- `venv` 作为 Python 模块/命令没有单独发现；本次未扫描每个项目目录中的自带虚拟环境。

### 5.3 已发现的 Python AI/CV 库

| 库 | 版本/状态 |
|---|---|
| NumPy | 1.17.4 |
| SciPy | 1.3.3 |
| OpenCV | 4.5.4 |
| Pillow | 7.0.0 |
| Matplotlib | 3.1.2 |
| Pandas | 0.25.3 |
| Jetson.GPIO | 2.1.9 |
| VPI | 2.4.8，宿主机可导入但有 PVA 警告 |
| PyGObject / `gi` | 3.36.0 |
| PyTorch | 未发现 |
| TorchVision | 未发现 |
| TensorFlow | 未发现 |
| ONNX / ONNX Runtime | 未发现 |
| PyCUDA | 未发现 |
| `jetson_inference` | 未发现 |
| Python TensorRT binding | 未发现 |

### 5.4 VS Code、Docker 和远程开发

- VS Code：**1.133.0，arm64**，可执行文件来自 `/home/zdh/.local/`。
- VS Code Server 正在运行，说明当前有远程开发会话或服务器端组件。
- Docker CLI：**26.1.3**。
- Docker 服务：systemd 显示 `enabled` 且 `active`。
- NVIDIA Container Runtime：**3.9.0**。
- NVIDIA Container Toolkit：**1.11.0~rc.1**。
- Docker daemon 本身未能以当前用户查询：`permission denied while trying to connect to /var/run/docker.sock`。
- 原因是当前用户不在 `docker` 组；本次没有使用 sudo，也没有改变组权限。

## 6. 网络环境

### 6.1 接口、地址和路由

依据：宿主机 `ip -brief link`、`ip -brief address`、`ip route`。

| 接口 | 状态 | 地址/说明 |
|---|---|---|
| `eth0` | UP，LOWER_UP | `10.42.0.217/24`，主网络接口 |
| `wlan0` | DOWN，NO-CARRIER | 无当前无线链路 |
| `usb0` | DOWN，NO-CARRIER | USB 网络接口未连接 |
| `rndis0` | DOWN，NO-CARRIER | USB RNDIS 未连接 |
| `l4tbr0` | DOWN | Jetson USB/桥接相关接口未启用 |
| `docker0` | DOWN，NO-CARRIER | `172.17.0.1/16`，当前没有活动容器链路 |
| `lo` | UP | `127.0.0.1`、`::1` |

默认路由：

```text
default via 10.42.0.1 dev eth0 proto dhcp metric 100
```

### 6.2 DNS 和互联网

- `/etc/resolv.conf` 使用 `127.0.0.53`，这是 systemd-resolved 的本地 stub。
- `resolvectl status` 显示 `eth0` 的实际 DNS 为 `10.42.0.1`。
- 宿主机 `getent ahosts example.com` 成功解析到 IPv4/IPv6 地址。
- 宿主机 `wget --spider https://example.com` 成功建立 HTTPS 连接并收到 `HTTP/1.1 200 OK`。

结论：当前宿主机的有线网络、默认路由、DNS 和基本互联网访问均正常。隔离检查 shell 中的网络命令曾显示“无法打开 netlink socket”或 DNS 失败，那是检查环境的网络隔离结果，不代表宿主机网络故障。

### 6.3 SSH

- `ssh.service`：`enabled`、`active (running)`。
- SSHD 监听 `0.0.0.0:22` 和 `[::]:22`，即所有 IPv4/IPv6 接口。
- 日志中有来自 `10.42.0.1` 的成功 SSH 登录，也有一次失败密码后成功登录的记录。

这对远程开发是方便的，但也意味着 SSH 暴露在所有可达接口上。报告只记录现状，不自动修改 SSH 配置或防火墙。

## 7. 当前资源状态

### 7.1 CPU 与负载

依据：宿主机 `top`、`uptime`、`tegrastats`、进程列表。

采样时宿主机 `top` 显示：

```text
CPU: 64.5% user, 28.9% system, 2.6% idle
load average: 1.82, 2.39, 2.65
```

这是一个短时间采样，并且勘察命令本身也会增加少量负载；但进程列表确认存在明显的实际 CPU 消耗：

```text
/usr/bin/python /home/zdh/Desktop/my-project/FPS_test.py
```

该进程采样时约占 `92.9%` CPU。由于当前只有 2 个 CPU online，它大约占用了一个上线 CPU 核心，导致系统整体接近当前双核配置的容量。

### 7.2 内存与 Swap

- 总内存约 6.7 GiB。
- available 约 2.2 GiB。
- zram Swap 总量约 3.3 GiB。
- Swap 使用约 14 MiB，当前没有明显的 Swap 压力。
- 没有 zombie 进程。

### 7.3 GPU、温度和功耗

最近一次 `tegrastats` 采样大致为：

```text
RAM 4439/6833MB
SWAP 3/3417MB
CPU [约 69-75%@1907, 约 2 个核心在线]
GR3D_FREQ 0%
AUX 约 38.5C
CPU 约 39.0-39.5C
GPU 约 38.5-39.0C
AO 约 38.0C
PMIC 约 50.0C
```

温度目前不高，没有看到过热迹象。功耗方面：

- 本次 `tegrastats` 没有提供可用的 VDDIN 功耗读数。
- `/sys/class/power_supply` 没有读到可用的电池/电源容量字段。
- `nvpmodel -q` 能读出 `MODE_20W_2CORE`，但读取 EMC 上限和电流限制文件时出现权限错误。

所以当前不能给出可靠的实时瓦数；不能把 `20W` 模式名称当作当前瞬时功耗。

### 7.4 重要进程

宿主机进程列表中值得注意的进程包括：

- `FPS_test.py`：当前最高 CPU 消耗，约 92.9%。
- `gnome-shell`、`Xorg`：桌面图形会话。
- VS Code 和 VS Code Server：当前开发环境。
- `nvargus-daemon`：Jetson 相机服务。
- `nvfancontrol`：风扇控制服务。
- `nvgpu_channel_p`：NVIDIA GPU 内核线程。
- `NetworkManager`、`systemd-resolved`、`sshd`、`dockerd`：网络、DNS、SSH、容器服务。

## 8. 已安装的重要软件

下面列出与 Jetson 学习和开发直接相关的代表性软件包。完整包列表没有全部复制进报告，以免把桌面依赖和系统库淹没在其中。

| 软件 | 已确认版本/状态 |
|---|---|
| Ubuntu | 20.04.6 LTS |
| Linux kernel | 5.10.216-tegra |
| L4T packages | 35.6.5 |
| CUDA toolkit | 11.4.19 / nvcc 11.4.315 |
| cuDNN | 8.6.0.166 |
| TensorRT | 8.5.2，native runtime/trtexec |
| VPI | 2.4.8 |
| OpenCV | 4.5.4 |
| GStreamer | 1.16.3 |
| Jetson Multimedia API | L4T 35.6.5 包已安装 |
| Jetson.GPIO | 2.1.9 |
| gcc/g++ | 9.4.0 |
| make | 4.2.1 |
| git | 2.25.1 |
| Python | 3.8.10 |
| pip | 实际命令 25.0.1 |
| VS Code | 1.133.0 arm64 |
| Docker | 26.1.3，服务运行 |
| NVIDIA Container Runtime | 3.9.0 |
| NVIDIA Container Toolkit | 1.11.0~rc.1 |

## 9. 当前存在的问题

以下分为“已观察到的问题”和“待确认项”，不代表本次自动修复过任何一个问题。

### 9.1 已观察到或值得关注

1. **当前只有 2 个 CPU online。**  
   `nvpmodel -q` 为 `MODE_20W_2CORE`，这会限制并行计算能力。若学习多线程、编译或 AI 推理性能，需要理解 nvpmodel 模式与 CPU online 状态的关系。

2. **`FPS_test.py` CPU 占用较高。**  
   采样时约 92.9% CPU；在双核模式下会明显影响系统响应和其他程序。仅从当前采样看，这是最主要的资源消耗者。

3. **当前 shell 没有把 CUDA bin 放进 PATH。**  
   `/usr/local/cuda/bin/nvcc` 存在，但直接输入 `nvcc` 找不到。CUDA 库已安装，问题是命令路径可见性，不是 CUDA 包缺失。

4. **当前没有 CMake。**  
   gcc、g++、make 已有，但 `cmake` 命令不存在。很多 CUDA、TensorRT、OpenCV 示例工程会因此无法直接按常见流程构建。

5. **TensorRT Python 模块和 PyTorch 等常见 AI 框架未发现。**  
   TensorRT native 库和 `trtexec` 存在，但 Python `tensorrt`、PyTorch、TensorFlow、ONNX Runtime 等没有发现。当前更适合从 CUDA/C++、OpenCV、GStreamer、TensorRT native 工具链开始学习。

6. **VPI 导入出现 PVA 不可用警告。**  
   VPI 2.4.8 本身可导入，但 PVA 加速路径在本次状态下不可用或资源不足。需要用具体 VPI 程序验证哪些后端可用。

7. **两个 IMX219 相机探测失败。**  
   内核日志中有：`imx219_board_setup: error during i2c read probe (-121)`，发生在 I2C 总线 9 和 10。常见含义是相机模块未连接、供电/排线问题、总线通信失败，或者设备树配置与实际硬件不匹配。如果当前没有接 IMX219 相机，这些日志可能是预配置硬件节点探测不到设备，并不一定是故障。

8. **SSH 对所有接口监听并使用过密码登录。**  
   服务已启用、端口 22 在所有地址监听，日志有来自 `10.42.0.1` 的密码登录。若设备接入不受信任网络，这是需要后续安全评估的配置点。

9. **当前用户不能访问 Docker daemon。**  
   服务是 active，但 `docker.sock` 查询被拒绝，且用户不在 docker 组。这是权限状态，不是 Docker 服务停止。

### 9.2 内核日志中的次要告警

- `IRQ267: set affinity failed(-22)`：中断 affinity 设置失败，可能与当前 CPU offline/平台中断约束有关。
- `hdmi: can't get adapter for ddc bus 3`：HDMI DDC 读取告警，可能与显示器连接或显示控制器初始化时机有关。
- `mmc0: host does not support reading read-only switch`：驱动无法读取存储卡只读开关，因而假设可写，是常见平台提示。
- `nvidia: loading out-of-tree module taints kernel`：NVIDIA 外部模块使内核 taint；在 Jetson 官方驱动栈中并不自动等于故障。
- 系统启动时还有一个 snapd 配置项兼容性提示，以及 BPF/cgroup firewall 能力提示。

### 9.3 本次无法完整确认的项目

- `nvidia-smi` 不存在；Jetson 使用 Tegra 专用接口，因此没有桌面 NVIDIA GPU 那种标准输出。
- 实时功耗未能读取。
- USB 设备完整枚举受 libusb 访问错误影响。
- 某些隔离 shell 中无法访问宿主机 netlink、systemd 总线和 GPU 设备节点；宿主机网络和服务状态已通过只读宿主机检查补齐。

## 10. 值得我学习的内容

### 10.1 Linux 观察系统的方法

建议先掌握这些信息源的区别：

- `/proc`：内核提供的运行时信息，例如 `/proc/cpuinfo`、`/proc/loadavg`、`/proc/cmdline`。
- `/sys`：设备、驱动、电源、CPU online 状态、thermal zone 等结构化接口。
- `systemctl`/`journalctl`：服务状态与启动日志。
- `dpkg-query`：Ubuntu/Debian 软件包“已安装版本”，和“命令是否在 PATH 中”是两件事。
- `ps`/`top`/`free`/`swapon`：进程、CPU、内存和 Swap 的不同观察角度。

### 10.2 Jetson 的功耗和性能模型

重点理解：

- `nvpmodel` 决定功耗模式、CPU online 数量、部分频率上限。
- `jetson_clocks` 用于查看/锁定时钟，但通常需要 root 权限。
- `tegrastats` 是 Jetson 日常观察 CPU、EMC、GPU、内存、温度的核心工具。
- 物理上有 6 个 CPU 不代表当前模式下 6 个都可用。
- 温度、频率、GPU 利用率和功耗是不同指标，不能互相替代。

### 10.3 NVIDIA 软件栈层次

可以按以下层次理解：

1. L4T：Jetson 的 Linux、内核、驱动和平台组件。
2. CUDA：GPU 通用计算平台和 CUDA C/C++ 编译工具。
3. cuDNN：深度学习常用神经网络算子库。
4. TensorRT：模型解析、优化和推理运行时。
5. VPI：面向视觉处理的硬件加速 API。
6. OpenCV：通用计算机视觉 API；可以调用 CPU、V4L2、GStreamer 等能力。
7. GStreamer：媒体 pipeline 框架；Jetson 的摄像头和硬件编解码通常通过 NVIDIA 插件接入。

理解这些边界后，遇到“程序能 import 但硬件加速不可用”或“库已安装但命令找不到”时，就能更快定位问题属于软件包、环境变量、设备权限还是硬件状态。

### 10.4 摄像头开发

建议学习链路：

```text
摄像头传感器 -> I2C/设备树 -> nvargus-daemon -> nvarguscamerasrc
-> GStreamer pipeline -> OpenCV / 编码器 / TensorRT
```

其中 I2C 探测、设备树、Argus、V4L2 和 GStreamer 是不同层次。相机没有图像时，不应只检查 Python 的 `cv2` 是否安装。

### 10.5 容器与权限

本机 Docker 和 NVIDIA Container Toolkit 都已安装，但当前用户不能访问 Docker socket。值得学习：

- Unix socket 的属主/属组权限。
- `docker` 组的安全含义。
- NVIDIA Container Runtime 如何把 GPU、CUDA 库和设备映射到容器。
- Jetson 容器通常需要和宿主机 L4T 版本兼容。


### 总结

这是一台型号和平台信息明确的 **NVIDIA Jetson Xavier NX Developer Kit**，运行 Ubuntu 20.04.6、L4T R35.6.5，CUDA/cuDNN/TensorRT/VPI/OpenCV/GStreamer 的 Jetson 软件栈基本齐全。当前最突出的运行事实是设备处于 `20W_2CORE` 模式、只有两个 CPU online，且 `FPS_test.py` 正在消耗约一个 CPU 核心；内存、磁盘和温度总体正常，有线网络和 SSH 正常。开发环境缺少当前可用的 CMake，Python AI 框架较少，Docker 服务虽运行但当前用户无 daemon 访问权限。报告中列出的相机探测、VPI PVA、SSH 暴露和 CPU 负载是后续最值得理解的几个方向。
