# QNN DLC Android 设备运行与性能优化手册

> 版本：2026-07-31  
> 当前验证环境：QAIRT/QNN 2.47、Android、HTP backend  
> 目标：下次拿到任意 QNN DLC 后，按本文完成“能运行、结果正确、性能可信、过程可复现”

---

## 0. 最短流程

```text
检查 metadata
  → 确认手机和 SoC
  → 找到 HTP 架构与 soc_model
  → 准备输入 RAW
  → 推送工具/模型/依赖
  → CPU 或 HTP 在线 prepare 跑通
  → 生成 baseline context
  → 查看 graphName 和 context 元数据
  → 最大 VTCM + O3 重新生成 context
  → 单进程多轮 benchmark
  → 拉回输出并与参考结果对齐
  → 保存 SDK/配置/context/profile/精度结果
```

---

## 1. 环境准备

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601
source "$QNN_SDK_ROOT/bin/envsetup.sh"
```

注意：`envsetup.sh` 可能把 `QNN_SDK_ROOT` 解析到真实安装路径。后续脚本建议使用：

```bash
QNN_SDK_ROOT=${QAIRT_SDK_ROOT:-$QNN_SDK_ROOT}
```

确认 SDK：

```bash
echo "$QNN_SDK_ROOT"
find "$QNN_SDK_ROOT/bin" -name qnn-net-run
find "$QNN_SDK_ROOT/lib" -name libQnnHtp.so
```

确认设备：

```bash
adb devices -l
adb shell getprop ro.product.model
adb shell getprop ro.soc.model
adb shell getprop ro.board.platform
adb shell getprop ro.build.version.release
```

如有多台设备：

```bash
export ANDROID_SERIAL=<adb-serial>
adb -s "$ANDROID_SERIAL" get-state
```

---

## 2. 建立模型工作目录

推荐目录：

```text
model-qnn-dlc-precision/
├── model.dlc
├── metadata.json
├── prepare_input.py
├── decode_output.py
├── run_on_device.sh
├── htp_netrun_config.json
├── htp_graph_config.json
├── .qnn_work/
├── context/
└── results/
```

必须记录：

```bash
sha256sum model.dlc
stat -c '%s' model.dlc
cat metadata.json
```

metadata 中重点提取：

- DLC 生成 QAIRT 版本；
- precision；
- 输入/输出 tensor 名；
- shape；
- dtype；
- layout；
- quantization scale；
- zero point。

不要根据文件名推测真实精度和 I/O 协议。

---

## 3. 确认 SoC、HTP 架构和 soc_model

设备侧：

```bash
SOC=$(adb shell getprop ro.soc.model | tr -d '\r')
echo "$SOC"
```

SDK 中查架构：

```bash
rg -n "$SOC" \
  "$QNN_SDK_ROOT/docs/QAIRT-Docs/QNN/general/overview.html" \
  "$QNN_SDK_ROOT/include/QNN/QnnTypes.h"
```

当前已验证示例：

```text
SM8850
  soc_model = 87
  HTP arch  = v81
  Stub      = libQnnHtpV81Stub.so
  Skel      = libQnnHtpV81Skel.so
```

不要拿错误版本的 Stub/Skel 混用。SoC 或 HTP 架构不明确时，先停止生成 optimized context。

---

## 4. 准备输入数据

### 4.1 float 输入

默认情况下，`qnn-net-run` 把输入文件按 float32 读取，即使图的原生输入是量化类型，它也可以在主机侧转换。

```python
array = np.asarray(image, dtype=np.float32) / 255.0
array.tofile("input_float32.raw")
```

### 4.2 原生量化输入

若 metadata 声明量化输入：

$$
q=clip(round(x/scale+zeroPoint),q_{min},q_{max})
$$

uint16 示例：

```python
native = np.clip(
    np.rint(float_array / scale + zero_point),
    0,
    65535,
).astype(np.uint16)
native.tofile("input_uint16.raw")
```

运行时添加：

```bash
--use_native_input_files
```

### 4.3 input_list

单输入：

```text
input_tensor_name:=input.raw
```

多输入同一轮：

```text
input_a:=a.raw input_b:=b.raw
```

多轮输入写多行。也可通过 `--num_inferences` 循环同一个 input list。

### 4.4 原生输出反量化

运行时：

```bash
--use_native_output_files
```

反量化：

$$
x=(q-zeroPoint)\times scale
$$

注意 QNN 输出旁边的 JSON 有时把 offset 写成负数；优先统一使用下载包 `metadata.json` 的 zero point 语义。

---

## 5. 推送公共工具和依赖

示例变量：

```bash
SERIAL=${ANDROID_SERIAL:-$(adb devices | awk 'NR==2 {print $1}')}
REMOTE=/data/local/tmp/qnn_model_test
ADB="adb -s $SERIAL"

$ADB shell "mkdir -p '$REMOTE/context'"
```

公共工具：

```bash
$ADB push "$QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/bin/aarch64-android/qnn-context-binary-generator" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/bin/aarch64-android/qnn-context-binary-utility" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/bin/aarch64-android/qnn-profile-viewer" "$REMOTE/"
```

DLC loader 和系统库：

```bash
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnModelDlc.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$REMOTE/"
```

HTP backend，以 v81 为例：

```bash
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$REMOTE/"
```

如果还要验证 CPU/GPU：

```bash
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnCpu.so" "$REMOTE/"
$ADB push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnGpu.so" "$REMOTE/"
```

模型和输入：

```bash
$ADB push model.dlc "$REMOTE/model.dlc"
$ADB push input.raw "$REMOTE/input.raw"
$ADB push input_list.txt "$REMOTE/input_list.txt"
$ADB shell "chmod 755 '$REMOTE'/qnn-*"
```

推荐按文件大小或 SHA256 判断是否需要重复 push，避免每次传大模型。

---

## 6. 设备运行环境

```bash
REMOTE_LD="$REMOTE:/vendor/lib64"
REMOTE_ADSP="$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

设备命令前统一：

```bash
cd "$REMOTE"
export LD_LIBRARY_PATH="$REMOTE_LD"
export ADSP_LIBRARY_PATH="$REMOTE_ADSP"
```

注意：

- `LD_LIBRARY_PATH` 使用冒号；
- `ADSP_LIBRARY_PATH` 使用分号；
- Skel 必须出现在 DSP 可搜索路径中。

---

## 7. 先跑通 baseline

### 7.1 CPU

CPU 适合验证 DLC loader 和数据协议，不适合大模型性能结论：

```bash
./qnn-net-run \
  --backend libQnnCpu.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --input_list input_list.txt \
  --output_dir output_cpu \
  --profiling_level basic
```

### 7.2 GPU

```bash
./qnn-net-run \
  --backend libQnnGpu.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --input_list input_list.txt \
  --output_dir output_gpu \
  --profiling_level basic
```

### 7.3 HTP 在线 prepare

```bash
./qnn-net-run \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --input_list input_list.txt \
  --output_dir output_htp_online \
  --profiling_level basic \
  --log_level info
```

在线 prepare 的目标是证明图可被 HTP 接受。不要把 compose/finalize 时间当作稳定推理延迟。

---

## 8. 生成 baseline context

```bash
./qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --output_dir context \
  --binary_file model_baseline \
  --profiling_level basic \
  --log_level info
```

注意：`--binary_file` 建议传不带 `.bin` 的 stem，工具会生成 `model_baseline.bin`。

运行 baseline context：

```bash
./qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context context/model_baseline.bin \
  --input_list input_list.txt \
  --output_dir output_baseline \
  --profiling_level basic
```

---

## 9. 查看 graphName 和 context 元数据

```bash
./qnn-context-binary-utility \
  --context_binary context/model_baseline.bin \
  --json_file context/model_baseline_info.json
```

重点查看：

```bash
rg -n 'graphName|spillFillBufferSize|optimizationLevel|vtcmSize|htpDlbc|numHvxThreads|dspArch|socModel' \
  context/model_baseline_info.json
```

这些字段回答：

- 实际图名是什么；
- 使用多少 VTCM；
- 优化等级是什么；
- 使用多少 HVX 线程；
- spill/fill buffer 多大；
- context 为哪个 SoC 和 HTP 架构生成。

不要只相信输入 JSON；以生成后的 context 元数据为准。

---

## 10. 生成 optimized context

### 10.1 外层 NetRun config

`htp_netrun_config.json`：

```json
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_graph_config.json"
  }
}
```

### 10.2 HTP graph config 模板

`htp_graph_config.json`：

```json
{
  "graphs": [
    {
      "graph_names": [
        "<GRAPH_NAME_FROM_CONTEXT_INFO>"
      ],
      "vtcm_mb": 0,
      "O": 3
    }
  ],
  "devices": [
    {
      "soc_model": 87,
      "dsp_arch": "v81",
      "cores": [
        {
          "perf_profile": "burst",
          "rpc_polling_time": 9999,
          "rpc_control_latency": 100
        }
      ]
    }
  ]
}
```

需要替换：

- `graph_names`；
- `soc_model`；
- `dsp_arch`。

`vtcm_mb=0` 表示请求目标 SoC 对该 QNN 图支持的最大 VTCM，不表示 0 MB。生成后必须通过 context utility 查看实际值。

### 10.3 生成

```bash
./qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --config_file htp_netrun_config.json \
  --output_dir context \
  --binary_file model_maxvtcm_o3 \
  --profiling_level basic \
  --log_level info
```

### 10.4 验证优化写入 context

```bash
./qnn-context-binary-utility \
  --context_binary context/model_maxvtcm_o3.bin \
  --json_file context/model_maxvtcm_o3_info.json
```

确认：

```text
optimizationLevel = 3
vtcmSize          = 目标实际值
numHvxThreads     = 预期值
socModel/dspArch  = 当前设备
```

若仍是默认值，不要继续做性能结论。

---

## 11. 正确的稳态 benchmark

```bash
./qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context context/model_maxvtcm_o3.bin \
  --config_file htp_netrun_config.json \
  --input_list input_list_native.txt \
  --output_dir output_benchmark \
  --profiling_level basic \
  --perf_profile burst \
  --num_inferences 50 \
  --keep_num_outputs 1 \
  --use_native_input_files \
  --use_native_output_files \
  --shared_buffer \
  --log_level warn
```

原则：

1. 单进程多轮，减少进程启动影响；
2. 只保存少量输出，避免磁盘污染；
3. 至少报告平均、最小、最大；
4. 同时记录 accelerator、QNN execute、NetRun 和总体 IPS；
5. 记录设备温度和系统负载；
6. 第一轮和稳态可分开报告。

温度示例：

```bash
adb shell cat /sys/class/power_supply/battery/temp
```

---

## 12. 解析 profiling

```bash
./qnn-profile-viewer \
  --input_log output_benchmark/qnn-profiling-data_0.log
```

主要指标：

| 指标 | 含义 |
|---|---|
| Init/Load | context、backend 和 RPC 初始化 |
| Compose | 在线构图；加载 context 时通常为 0 |
| Finalize | 在线 prepare/编译 |
| NetRun Execute | qnn-net-run 观察到的执行时间 |
| RPC Execute | AP 与 HTP RPC 区间 |
| QNN accelerator Execute | 设备侧图执行主指标 |
| Accelerator excluding wait | 排除部分等待后的设备计算区间 |
| Overall IPS | 包含 I/O 和其它 NetRun 开销的吞吐 |

报告必须明确使用哪一个数字。不要只写“延迟 20 ms”而不说明计时层级。

---

## 13. 性能消融模板

按以下顺序逐项变化：

| 序号 | 配置 | 目的 |
|---:|---|---|
| 1 | baseline context | 建立基线 |
| 2 | `burst` | 判断频率影响 |
| 3 | native I/O | 判断主机转换影响 |
| 4 | 最大 VTCM | 判断中间特征驻留影响 |
| 5 | shared buffer | 判断 I/O copy 影响 |
| 6 | O3 | 判断编译调度影响 |
| 7 | DLBC/SLC | 判断带宽压缩/缓存收益 |
| 8 | 精度切换 | 建立速度-质量 Pareto |

每次只改变一个变量，并保存：

- 完整命令；
- context 哈希；
- config；
- profile log；
- 输出 raw；
- 温度；
- 结果 JSON。

---

## 14. 正确性验证

### 14.1 先建立参考输出

参考优先级：

1. 原始框架 checkpoint；
2. ONNX Runtime CPU；
3. QNN CPU float；
4. 已验证的其它 backend。

### 14.2 指标

```python
diff = candidate - reference
mae = np.mean(np.abs(diff))
max_abs = np.max(np.abs(diff))
mse = np.mean(diff.astype(np.float64) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
```

至少报告：

- MAE；
- 最大绝对误差；
- MSE/PSNR；
- NaN/Inf；
- 输出最小/最大值；
- 可视化差异图。

### 14.3 优化一致性

对于只改变 context 优化的实验，优先检查原生输出是否逐元素一致：

```python
np.array_equal(before, after)
```

### 14.4 数据集质量

部署数值一致性不等于模型任务质量。正式验收还需要：

- 任务数据集；
- GT；
- PSNR/SSIM/mAP/IoU 等任务指标；
- 域外和失败样本；
- 业务指标。

---

## 15. context 缓存命名规则

不要只使用 `model.bin`。推荐：

```text
<model>_<precision>_<soc>_<htp>_<optimization>_<model-hash>_<config-hash>.bin
```

示例：

```text
nafnet_w8a16_sm8850_v81_maxvtcm_o3_ffa61c0e_a1309c8d.bin
```

context 应视为以下组合的派生产物：

```text
model DLC
+ DLC生成版本
+ runtime SDK
+ SoC
+ HTP架构
+ graph config
+ backend extension
```

任意关键项变化都应重新生成或至少重新验证 context。

---

## 16. 常见故障矩阵

| 现象 | 高概率原因 | 检查动作 |
|---|---|---|
| `adb unauthorized` | 手机未授权 | 解锁手机并接受 RSA |
| 找不到 backend `.so` | `LD_LIBRARY_PATH` 错 | 检查远端目录和权限 |
| 找不到 Skel | `ADSP_LIBRARY_PATH` 错或架构错误 | 检查分号和 HTP 版本 |
| DLC 无法作为 model 加载 | 缺 DLC loader | 使用 `libQnnModelDlc.so --dlc_path` |
| 输入尺寸错误 | shape/layout 不匹配 | 查 metadata 和 raw 字节数 |
| 输出全黑/全白 | dtype/量化参数错误 | 检查 native flag、scale、zero point |
| 首次运行几十秒 | 在线 prepare | 生成 context cache |
| context 加载快但 execute 慢 | sub-optimal context | 查 VTCM、O3、spill/fill |
| `burst` 无收益 | 非频率瓶颈 | 查内存、回退、图调度 |
| CPU/GPU 正常，HTP 失败 | 算子或架构不支持 | detailed log、SDK兼容、Stub/Skel |
| context 无法加载 | SDK/SoC/context 不兼容 | 重生成并检查 context metadata |
| 多轮后变慢 | 热降频 | 记录温度和频率，增加冷却间隔 |
| 本地远慢于官网 | 精度/设备/统计口径不同 | 先做同口径对比 |
| 优化后结果改变 | 配置、I/O 或量化错误 | 对比原生 raw 和命令 |

---

## 17. 推荐自动化脚本能力

一个可维护的 `run_on_device.sh` 至少应：

1. 自动加载 SDK 环境；
2. 自动选择或校验 adb 设备；
3. 检查 SoC 与目标 HTP 架构；
4. 从 metadata 读取输入输出；
5. 生成 float 或 native RAW；
6. 按大小/哈希增量 push；
7. 推送正确 Stub/Skel/Prepare/Extensions；
8. 用模型与配置哈希命名 context；
9. 自动复用或重建 context；
10. 支持 `NUM_INFERENCES`；
11. 只保留指定数量输出；
12. 自动 pull profile 和 raw；
13. 自动运行 profile viewer；
14. 自动反量化和保存预览；
15. 输出完整运行目录。

建议环境变量：

```text
QNN_SDK_ROOT
ANDROID_SERIAL
REMOTE_DIR
HTP_ARCH
SOC_MODEL_ID
NUM_INFERENCES
KEEP_NUM_OUTPUTS
REBUILD_CONTEXT
PERF_PROFILE
USE_SHARED_BUFFER
```

---

## 18. 每个模型应保存的报告

推荐生成 `benchmark_results.json`：

```json
{
  "date": "YYYY-MM-DD",
  "device": {
    "model": "...",
    "soc": "...",
    "htp_arch": "...",
    "android": "..."
  },
  "software": {
    "dlc_generation_qairt": "...",
    "runtime_qairt": "..."
  },
  "model": {
    "sha256": "...",
    "precision": "...",
    "inputs": [],
    "outputs": []
  },
  "context": {
    "sha256": "...",
    "optimization_level": 3,
    "vtcm_mb": 8,
    "hvx_threads": 8,
    "spill_fill_bytes": 0
  },
  "performance": [],
  "accuracy": {},
  "notes": []
}
```

---

## 19. 性能结果最小报告模板

```markdown
### 环境

- 设备：
- SoC：
- Android：
- HTP：
- DLC 生成 QAIRT：
- 运行 QAIRT：
- 模型 SHA256：
- Context SHA256：

### 输入

- Shape：
- Layout：
- Dtype：
- Quantization：

### 性能

| 配置 | Init | Accelerator Avg | Min | Max | NetRun Avg | IPS |
|---|---:|---:|---:|---:|---:|---:|

### 正确性

- Reference：
- MAE：
- Max abs：
- PSNR：
- NaN/Inf：

### 结论

- 事实：
- 推测：
- 建议：
```

---

## 20. 最终验收清单

### 能运行

- [ ] adb 状态为 `device`
- [ ] SoC/HTP 架构已确认
- [ ] DLC loader 正确
- [ ] Stub/Skel 匹配
- [ ] 输入 shape/layout/dtype 正确
- [ ] 输出可解码且无 NaN/Inf

### 结果正确

- [ ] 有可靠参考输出
- [ ] 数值误差已记录
- [ ] 量化参数来自 metadata
- [ ] 优化前后输出已对比
- [ ] 有任务数据集或业务样本验证

### 性能可信

- [ ] prepare 与 execute 分开
- [ ] 使用 context cache
- [ ] context 元数据已检查
- [ ] 单进程多轮测试
- [ ] 记录平均/最小/最大
- [ ] 记录温度
- [ ] 官网比较使用同精度、同分辨率、同设备等级

### 可复现

- [ ] 保存完整命令
- [ ] 保存 SDK 版本
- [ ] 保存模型和 context 哈希
- [ ] 保存 backend config
- [ ] 保存 profile 原始日志
- [ ] 保存输出 raw 和精度结果
- [ ] 更新 README/benchmark JSON

---

## 21. NAFNet 参考实现

本手册对应的完整可运行例子：

```text
/media/code/tools/naf/nafnet_deblur-qnn_dlc-float
/media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16
```

其中 w8a16 脚本展示了：

- 原生 uint16 输入输出；
- SM8850/v81 校验；
- 最大 VTCM；
- O3 context；
- burst；
- shared buffer；
- 多轮推理；
- profile 解析；
- 输出反量化。

遇到新 DLC 时，优先复制目录结构和脚本框架，再替换 metadata、模型文件、图名、SoC 和预处理逻辑。
