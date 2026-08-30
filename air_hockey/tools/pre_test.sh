#!/bin/bash
# 实验前性能检查与手动锁频（在 Jetson 上以 root 运行）：
#   sudo bash air_hockey/tools/pre_test.sh
#
# 背景：本机 L4T 35.6.5 的 jetson_clocks 脚本会静默失效
# （NVPM ERROR: emc_cap / VDDIN_OC_LIMIT 读取失败），所以这里
# 直接写 sysfs 手动锁频，不依赖该脚本。
#
# 用法顺序：先用 nvpmodel 选好功耗模式，再跑本脚本。
#   sudo nvpmodel -m 8     # 例：20W_6CORE
#   sudo bash air_hockey/tools/pre_test.sh

set -u

echo "== 当前功耗模式 =="
nvpmodel -q

echo
echo "== 手动锁频 =="
# CPU：全部核心切 performance（频率自动顶到当前模式的上限）
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
# GPU / EMC / DLA / PVA 等 devfreq 设备一并切 performance（EMC 对拷贝带宽最关键）
for g in /sys/class/devfreq/*/governor; do
    echo performance > "$g" 2>/dev/null
done

echo
echo "== 验证 =="
echo "在线核心:        $(cat /sys/devices/system/cpu/online)"
echo "cpu0 调度器:     $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "cpu0 当前频率:   $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
echo
echo "调度器必须全部为 performance，频率必须等于当前模式的上限，然后才能开始采数据。"
