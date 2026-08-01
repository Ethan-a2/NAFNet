# NAFNet w8a16：PC ONNX、Android ONNX QNN EP 与 QNN DLC 完整运行手册

> 最后核对日期：2026-08-01  
> 仓库根目录：`/media/code/tools/naf/NAFNet`  
> 已连接设备：Xiaomi `2512BPNDAC` / `SM8850` / Android 16 / SDK 36 / `arm64-v8a`  
> adb serial：`c495c2c3`  
> HTP：v81，最大 VTCM 8 MB  
> 文档目标：从零理解并复现 PC ONNX、手机 ONNX QNN EP、手机 QNN DLC 三条运行链路，并正确比较模型格式、数值与性能。

---

## 1. 结论先行

### 1.1 PC 和手机使用的是不是同一个 ONNX

是。本项目 PC 和 Android 使用完全相同的两个模型文件：

```text
nafnet_deblur.onnx
nafnet_deblur.data
```

其中 `.onnx` 保存图结构和外部权重引用，约 67.9M 参数对应的大部分权重存放在 `.data` 中。完整部署时必须同时复制这两个文件，并保持文件名和相对目录关系不变。

当前主机和手机上的模型哈希完全一致：

```text
48c5ae6afff38988efe88c1275bce704d025d729762f75e095ec9434a1ae12c5  nafnet_deblur.onnx
742aaccef608e9f4380dbb940994498dc0c50ca4bb9938b89e0109513321d004  nafnet_deblur.data
```

PC 和手机之间的区别不在 ONNX 文件，而在执行后端：

```text
PC:
ONNX -> ONNX Runtime CPUExecutionProvider -> x86 CPU

Android:
ONNX -> ONNX Runtime Android -> QNN EP 插件 -> QNN HTP -> FastRPC -> HTP/NPU
```

### 1.2 当前设备是否已经具备运行条件

具备。2026-08-01 实际检查结果：

- 手机已连接并授权，状态为 `device`。
- 主机和手机上的 ONNX、外部权重、Android runner、DLC 哈希一致。
- Android ONNX 工作目录已经部署完整。
- Android DLC 工作目录已经部署完整。
- DLC 的最大 VTCM、O3 HTP context 已经生成并保留。
- Android runner 动态链接正常，可以输出 usage。
- PC ONNX Runtime 环境可运行。

当前没有阻塞 adb benchmark 的必需环境缺口。仍然存在的产品化缺口是：

1. ONNX QNN EP context cache 尚未接入，导致每次创建 Session 在线编译约 54 秒。
2. ORT Android AAR 和 QNN EP AAR 当前位于 `/tmp`，应移动到稳定目录。
3. QNN EP 2.4.0 的验证组合偏向 QNN 2.45，而本机使用 QNN 2.47，当前依赖 `skip_qnn_version_check=1`。
4. 当前方式是 `/data/local/tmp` 下的原生 benchmark，不是 APK/JNI 产品集成。

### 1.3 三条链路的实测性能

以下结果来自同一模型、同一张 640x360 图片，并且只统计模型执行，不包含图片解码、resize、PNG 保存和 adb 传输：

| 路径 | Session/Context | 稳态平均延迟 | FPS |
|---|---:|---:|---:|
| 主机 ORT CPU，24 线程 | 453.81 ms | 2180.27 ms | 0.46 |
| Android ORT QNN EP，无 profiler | 53.96 s 在线编译 | 45.876 ms | 21.80 |
| Android QNN DLC，缓存 context | 复用离线 context | 44.742 ms | 22.35 |

主要结论：

- Android ONNX 稳态比主机 CPU 快约 47.5 倍。
- Android ONNX 相比 DLC 只慢 1.134 ms，约 2.53%。
- Android ONNX 当前最大的缺点不是稳态推理，而是约 54 秒的在线 QNN graph Finalize。
- profiler 会把 Android ONNX 端到端延迟从约 45.88 ms 增加到约 52.25 ms。

---

## 2. 当前设备与环境快照

### 2.1 手机

```text
manufacturer=Xiaomi
model=2512BPNDAC
device=nezha
soc=SM8850
android=16
sdk=36
abi=arm64-v8a
battery_temperature=27.5°C
adb_serial=c495c2c3
```

检查命令：

```bash
adb devices -l

SERIAL=c495c2c3

adb -s "$SERIAL" shell '
  printf "manufacturer="; getprop ro.product.manufacturer
  printf "model="; getprop ro.product.model
  printf "device="; getprop ro.product.device
  printf "soc="; getprop ro.soc.model
  printf "android="; getprop ro.build.version.release
  printf "sdk="; getprop ro.build.version.sdk
  printf "abi="; getprop ro.product.cpu.abi
'
```

### 2.2 主机软件

```text
Python                         3.14.4
onnxruntime                    1.26.0
onnx                           1.21.0
numpy                          2.3.5
Pillow                         12.3.0
Android Debug Bridge           35.0.1
Android NDK                    27.0.12077973
ONNX Runtime Android           1.24.3
ONNX Runtime QNN EP plugin     2.4.0
QAIRT/QNN Runtime              2.47.0.260601
模型生成 QAIRT                 2.45.0.260326154327
```

### 2.3 主机路径

```text
仓库：
/media/code/tools/naf/NAFNet

Android NDK：
/media/ext/opt/Android/Sdk/ndk/27.0.12077973

QAIRT/QNN SDK：
/opt/qcom/aistack/qairt/2.47.0.260601

ORT Android AAR：
/tmp/onnxruntime-android-1.24.3.aar

QNN EP AAR：
/tmp/onnxruntime-android-qnn-2.4.0.aar

ORT Android 解压目录：
/tmp/ort-aar

QNN EP 解压目录：
/tmp/qnn-ep-aar
```

### 2.4 手机远端目录

```text
Android ONNX：
/data/local/tmp/nafnet_ort_qnn_w8a16

Android DLC：
/data/local/tmp/nafnet_deblur_qnn_w8a16
```

当前远端目录已经包含完整运行文件，因此可以直接跳到第 6 节运行 Android ONNX，或第 10 节运行 DLC。

---

## 3. 模型规格

### 3.1 ONNX w8a16

| Tensor | Shape | Layout | Dtype | Scale | Zero point |
|---|---|---|---|---:|---:|
| `image` | `[1,3,360,640]` | NCHW | `uint16` | `1.5259021893143654e-05` | 0 |
| `deblurred_image` | `[1,3,360,640]` | NCHW | `uint16` | `1.9742114091059193e-05` | 8480 |

模型规模：

```text
IR version       9
Opset            21
节点数           3617
Initializer      2603
Q/DQ 节点        2804
外部权重         272,800,256 bytes
```

主要算子：

| 算子 | 数量 |
|---|---:|
| `DequantizeLinear` | 1549 |
| `QuantizeLinear` | 1255 |
| `Conv` | 226 |
| `Mul` | 180 |
| `Transpose` | 144 |
| `Add` | 77 |
| `LayerNormalization` | 72 |
| `Split` | 72 |
| `GlobalAveragePool` | 36 |
| `DepthToSpace` | 4 |

Q/DQ 节点约占图节点的 77.5%，这是 Android ONNX 在线编译时间很长的主要原因之一。

### 3.2 QNN DLC w8a16

| Tensor | Shape | Layout | Dtype | Scale | Zero point |
|---|---|---|---|---:|---:|
| `image` | `[1,360,640,3]` | NHWC | `uint16` | `1.5259021893143654e-05` | 0 |
| `deblurred_image` | `[1,360,640,3]` | NHWC | `uint16` | `1.9742114091059193e-05` | 8480 |

ONNX 和 DLC 的语义、分辨率、dtype、量化 scale 和 zero point 相同，但布局不同：

```text
ONNX: NCHW
DLC:  NHWC
```

两个 raw 文件的字节数都为：

```text
1 × 3 × 360 × 640 × 2 = 1,382,400 bytes
```

因此仅检查文件大小无法发现布局错误。ONNX raw 和 DLC raw 不能直接互用，必须转置。

### 3.3 模型文件大小

| 格式 | 文件 | 总大小 |
|---|---|---:|
| ONNX | `.onnx + .data` | 274,102,496 bytes |
| DLC | `.dlc` | 78,193,020 bytes |

当前 ONNX 模型文件约为 DLC 的 3.5 倍。

---

## 4. 三条执行链路

### 4.1 PC ONNX CPU

```text
输入图片
  -> Pillow resize/RGB
  -> uint16 NCHW 量化
  -> ONNX Runtime CPUExecutionProvider
  -> uint16 NCHW 输出
  -> 反量化
  -> RGB PNG
```

### 4.2 Android ONNX QNN EP

```text
nafnet_deblur.onnx + nafnet_deblur.data
  -> libonnxruntime.so
  -> libonnxruntime_providers_qnn.so
  -> libQnnHtp.so
  -> libQnnHtpV81Stub.so
  -> libcdsprpc.so / FastRPC
  -> libQnnHtpV81Skel.so
  -> HTP v81 / HVX / HMX / 8 MB VTCM
```

Android runner 做了以下关键设置：

```text
session.disable_cpu_ep_fallback=1
backend_path=/data/local/tmp/nafnet_ort_qnn_w8a16/libQnnHtp.so
skip_qnn_version_check=1
htp_performance_mode=burst
htp_graph_finalization_optimization_mode=3
vtcm_mb=8
qnn_context_priority=high
enable_htp_shared_memory_allocator=1
offload_graph_io_quantization=0
qnn.perf_mode=burst
qnn.rpc_control_latency=100
```

### 4.3 Android QNN DLC

```text
nafnet_deblur.dlc
  -> libQnnModelDlc.so
  -> qnn-context-binary-generator
  -> SM8850/v81 HTP context binary
  -> qnn-net-run --retrieve_context
  -> libQnnHtp.so
  -> FastRPC
  -> HTP v81
```

DLC 不是可直接执行的 HTP binary，也不能写成：

```bash
qnn-net-run --model nafnet_deblur.dlc
```

正确方式是通过 DLC loader：

```text
--model libQnnModelDlc.so
--dlc_path nafnet_deblur.dlc
```

或者先生成 context，再使用 `--retrieve_context`。

---

## 5. PC ONNX 运行

### 5.1 安装 Python 环境

当前主机已经安装，可以直接运行。新环境推荐使用 venv：

```bash
cd /media/code/tools/naf/NAFNet

python3 -m venv .venv-onnx
source .venv-onnx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r onnx/requirements.txt
```

检查：

```bash
python3 - <<'PY'
import onnxruntime
import numpy
import PIL

print("onnxruntime", onnxruntime.__version__)
print("numpy", numpy.__version__)
print("Pillow", PIL.__version__)
print("providers", onnxruntime.get_available_providers())
PY
```

### 5.2 执行 PC CPU benchmark

```bash
cd /media/code/tools/naf/NAFNet
ROOT=$(pwd -P)

mkdir -p /tmp/nafnet_pc

python3 "$ROOT/nafnet_deblur-onnx-w8a16/benchmark_ort.py" \
  "$ROOT/onnx/input.jpg" \
  --warmup 1 \
  --runs 5 \
  --threads 24 \
  --input-raw /tmp/nafnet_pc/input_nchw_uint16.raw \
  --output-raw /tmp/nafnet_pc/output_nchw_uint16.raw \
  --output /tmp/nafnet_pc/output.png \
  --report /tmp/nafnet_pc/benchmark.json
```

输出：

```text
/tmp/nafnet_pc/input_nchw_uint16.raw
/tmp/nafnet_pc/output_nchw_uint16.raw
/tmp/nafnet_pc/output.png
/tmp/nafnet_pc/benchmark.json
```

### 5.3 与 DLC 输出比较

如果已有 DLC 原生 NHWC 输出，可以增加：

```bash
--qnn-output /path/to/deblurred_image_native.raw
```

脚本会把 DLC NHWC 转为 NCHW，再计算：

- uint16 MAE。
- uint16 最大绝对误差。
- 精确相同比例。
- 反量化 MAE/RMSE。
- 单位动态范围 PSNR。

---

## 6. 当前手机直接运行 ONNX

当前手机 `/data/local/tmp/nafnet_ort_qnn_w8a16` 已经部署完成，可以直接执行，不需要重新 push。

### 6.1 无 profiler 稳态 benchmark

```bash
SERIAL=c495c2c3
REMOTE=/data/local/tmp/nafnet_ort_qnn_w8a16

adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE' &&
  export ADSP_LIBRARY_PATH='$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' &&
  ./ort_qnn_android_benchmark \
    nafnet_deblur.onnx \
    input_ort_nchw_uint16.raw \
    output_ort_qnn_current.raw \
    '$REMOTE/libonnxruntime_providers_qnn.so' \
    '$REMOTE/libQnnHtp.so' \
    - \
    5 \
    30
"
```

参数：

| 参数 | 含义 |
|---|---|
| `nafnet_deblur.onnx` | 模型图，外部权重从同目录加载 |
| `input_ort_nchw_uint16.raw` | NCHW uint16 输入 |
| `output_ort_qnn_current.raw` | NCHW uint16 输出 |
| `libonnxruntime_providers_qnn.so` | 动态 QNN EP 插件绝对路径 |
| `libQnnHtp.so` | QNN HTP backend 绝对路径 |
| `-` | 关闭 QNN profiler |
| `5` | warmup 次数 |
| `30` | 正式统计次数 |

首次创建 Session 时约 54 秒没有推理输出是正常现象。当前 runner 尚未加载 context cache，QNN EP 正在将 3617 节点 QDQ 图在线转换和 Finalize。

### 6.2 启用 profiler

把第 6 个 runner 参数从 `-` 换成 CSV 路径：

```bash
SERIAL=c495c2c3
REMOTE=/data/local/tmp/nafnet_ort_qnn_w8a16

adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE' &&
  export ADSP_LIBRARY_PATH='$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' &&
  ./ort_qnn_android_benchmark \
    nafnet_deblur.onnx \
    input_ort_nchw_uint16.raw \
    output_ort_qnn_profiled.raw \
    '$REMOTE/libonnxruntime_providers_qnn.so' \
    '$REMOTE/libQnnHtp.so' \
    '$REMOTE/qnn_profile_current.csv' \
    5 \
    50
"
```

拉取 profiler：

```bash
adb -s "$SERIAL" pull \
  "$REMOTE/qnn_profile_current.csv" \
  /tmp/qnn_profile_current.csv
```

Profiler 仅用于拆分 QNN、RPC 和 accelerator 时间，不能作为产品端到端延迟。

### 6.3 拉取输出

```bash
adb -s "$SERIAL" pull \
  "$REMOTE/output_ort_qnn_current.raw" \
  /tmp/output_ort_qnn_current.raw
```

### 6.4 解码 ONNX NCHW 输出

```bash
python3 - \
  /tmp/output_ort_qnn_current.raw \
  /tmp/output_ort_qnn_current.png <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

raw_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

height = 360
width = 640
channels = 3
scale = 0.000019742114091059193
zero_point = 8480

native = np.fromfile(raw_path, dtype=np.uint16)
expected = channels * height * width
if native.size != expected:
    raise ValueError(f"Expected {expected} elements, got {native.size}")

native = native.reshape(1, channels, height, width)
image = (native.astype(np.float32) - zero_point) * scale
image = np.transpose(image[0], (1, 2, 0))
image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

Image.fromarray(image).save(output_path)
print(output_path)
PY
```

---

## 7. Android ONNX 从零部署

本节适合远端目录被清理、换手机或重建 runner 后使用。

### 7.1 设置变量

```bash
cd /media/code/tools/naf/NAFNet
ROOT=$(pwd -P)

export SERIAL=c495c2c3
export REMOTE=/data/local/tmp/nafnet_ort_qnn_w8a16

export NDK=/media/ext/opt/Android/Sdk/ndk/27.0.12077973
export QNN=/opt/qcom/aistack/qairt/2.47.0.260601
export ORT=/tmp/ort-aar
export QNN_EP=/tmp/qnn-ep-aar

export CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++"
export LIBCPP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
```

如果有多台设备，必须明确指定 `SERIAL`：

```bash
adb devices -l
```

### 7.2 解压 Android AAR

如果 `/tmp/ort-aar` 或 `/tmp/qnn-ep-aar` 被清理：

```bash
rm -rf /tmp/ort-aar /tmp/qnn-ep-aar
mkdir -p /tmp/ort-aar /tmp/qnn-ep-aar

unzip -q /tmp/onnxruntime-android-1.24.3.aar \
  -d /tmp/ort-aar

unzip -q /tmp/onnxruntime-android-qnn-2.4.0.aar \
  -d /tmp/qnn-ep-aar
```

检查：

```bash
test -f "$ORT/headers/onnxruntime_cxx_api.h"
test -f "$ORT/jni/arm64-v8a/libonnxruntime.so"
test -f "$QNN_EP/jni/arm64-v8a/libonnxruntime_providers_qnn.so"
```

### 7.3 编译 runner

```bash
"$CXX" -std=c++17 -O3 -fPIE -pie \
  -I"$ORT/headers" \
  "$ROOT/nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark.cpp" \
  -L"$ORT/jni/arm64-v8a" \
  -lonnxruntime \
  -Wl,-rpath,'$ORIGIN' \
  -o "$ROOT/nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark"
```

检查架构：

```bash
file "$ROOT/nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark"
```

期望：

```text
ELF 64-bit LSB pie executable, ARM aarch64
interpreter /system/bin/linker64
for Android 26
```

当前仓库 runner 哈希：

```text
7cc176e593aa5e3c0b0a3a89cee40158eff999a91f47d6313a6361448ae621a1
```

### 7.4 准备 ONNX 输入

已有输入可以直接使用：

```bash
INPUT_RAW="$ROOT/nafnet_deblur-onnx-w8a16/input_ort_nchw_uint16.raw"
```

对新图片，可以运行 PC benchmark 生成输入 raw：

```bash
mkdir -p /tmp/nafnet_prepare

python3 "$ROOT/nafnet_deblur-onnx-w8a16/benchmark_ort.py" \
  /path/to/input.jpg \
  --warmup 0 \
  --runs 1 \
  --threads 24 \
  --input-raw /tmp/nafnet_prepare/input_ort_nchw_uint16.raw \
  --output-raw /tmp/nafnet_prepare/output_cpu.raw \
  --output /tmp/nafnet_prepare/output_cpu.png \
  --report /tmp/nafnet_prepare/benchmark_cpu.json

INPUT_RAW=/tmp/nafnet_prepare/input_ort_nchw_uint16.raw
```

检查字节数：

```bash
stat -c '%s' "$INPUT_RAW"
```

期望：

```text
1382400
```

### 7.5 推送文件

```bash
adb -s "$SERIAL" shell "rm -rf '$REMOTE' && mkdir -p '$REMOTE'"

push_file() {
  adb -s "$SERIAL" push "$1" "$REMOTE/"
}

push_file "$ROOT/nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark"
push_file "$ORT/jni/arm64-v8a/libonnxruntime.so"
push_file "$QNN_EP/jni/arm64-v8a/libonnxruntime_providers_qnn.so"
push_file "$LIBCPP"

push_file "$QNN/lib/aarch64-android/libQnnHtp.so"
push_file "$QNN/lib/aarch64-android/libQnnHtpV81Stub.so"
push_file "$QNN/lib/aarch64-android/libQnnHtpPrepare.so"
push_file "$QNN/lib/aarch64-android/libQnnSystem.so"
push_file "$QNN/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"

push_file "$ROOT/nafnet_deblur-onnx-w8a16/nafnet_deblur.onnx"
push_file "$ROOT/nafnet_deblur-onnx-w8a16/nafnet_deblur.data"
push_file "$INPUT_RAW"

adb -s "$SERIAL" shell \
  "chmod 755 '$REMOTE/ort_qnn_android_benchmark'"
```

如果 `INPUT_RAW` 的 basename 不是 `input_ort_nchw_uint16.raw`，运行命令中必须使用实际 basename。

### 7.6 远端必需文件

```text
ort_qnn_android_benchmark
libonnxruntime.so
libonnxruntime_providers_qnn.so
libc++_shared.so
libQnnHtp.so
libQnnHtpV81Stub.so
libQnnHtpPrepare.so
libQnnSystem.so
libQnnHtpV81Skel.so
nafnet_deblur.onnx
nafnet_deblur.data
input_ort_nchw_uint16.raw
```

### 7.7 执行

```bash
adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE' &&
  export ADSP_LIBRARY_PATH='$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' &&
  ./ort_qnn_android_benchmark \
    nafnet_deblur.onnx \
    input_ort_nchw_uint16.raw \
    output_ort_qnn.raw \
    '$REMOTE/libonnxruntime_providers_qnn.so' \
    '$REMOTE/libQnnHtp.so' \
    - \
    5 \
    30
"
```

---

## 8. 如何证明 Android ONNX 真的运行在 HTP

不能只看“命令成功”或“速度变快”，应至少使用以下证据。

### 8.1 QNN EP device 被发现

runner 应输出：

```text
EP device: QNNExecutionProvider
```

### 8.2 强制关闭 CPU fallback

当前 runner 设置：

```text
session.disable_cpu_ep_fallback=1
```

在此条件下 Session 创建和推理仍成功，说明全部所需节点已被 QNN EP 接受，没有 CPU 混跑。

### 8.3 使用 HTP backend 绝对路径

必须传入：

```text
/data/local/tmp/nafnet_ort_qnn_w8a16/libQnnHtp.so
```

仅使用：

```text
backend_type=htp
```

在本次 ORT/QNN 组合中曾出现没有形成 HTP 分区、实际回到 CPU 路径的问题。

### 8.4 Profiler 出现 HTP accelerator

已记录的 profiler 包含：

```text
HVX threads: 8
QNN accelerator execute
RPC execute
Accelerator execute excluding wait
```

### 8.5 性能符合 HTP 区间

当前目标区间：

```text
无 profiler：约 45.7-46.5 ms
带 profiler：约 52 ms
```

如果推理约 2-3 秒，应优先怀疑 QNN EP 未加载、版本检查失败或 CPU fallback。

---

## 9. Android ONNX 关键环境规则

### 9.1 ONNX 路径不要加入 `/vendor/lib64`

正确：

```bash
export LD_LIBRARY_PATH="$REMOTE"
```

错误：

```bash
export LD_LIBRARY_PATH="$REMOTE:/vendor/lib64"
```

在当前 Android 16 设备上，错误设置会让 ORT 依赖的系统 `libandroid.so` 错误链接到 vendor 侧不匹配的图形库，出现 `SurfaceComposerClient` 相关符号缺失。

系统 linker 能自行找到系统和 vendor 依赖，不要改变其优先级。

### 9.2 ADSP_LIBRARY_PATH 使用分号

正确：

```bash
export ADSP_LIBRARY_PATH="$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

不要使用 Linux 风格冒号连接 DSP 路径。

### 9.3 Stub 和 Skel 必须匹配 HTP 架构

当前设备：

```text
SM8850 -> HTP v81
```

因此使用：

```text
libQnnHtpV81Stub.so
libQnnHtpV81Skel.so
```

换手机前必须确认 SoC、HTP 架构和最大 VTCM，不能机械复用 v81 和 `vtcm_mb=8`。

### 9.4 QNN 版本检查

当前组合：

```text
QNN EP plugin  2.4.0
ORT Android    1.24.3
QNN Runtime    2.47.0
```

runner 使用：

```text
skip_qnn_version_check=1
```

它表示“本模型在当前设备上已经实测可运行”，不表示所有模型和接口都无条件兼容。产品版本应优先使用插件明确验证的 QNN Runtime，或者重新完成回归验证。

---

## 10. 当前手机直接运行 DLC

### 10.1 推荐：一键脚本

当前手机已经存在 O3/max-VTCM context，可以直接复用：

```bash
cd /media/code/tools/naf/NAFNet
ROOT=$(pwd -P)

export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601
export ANDROID_SERIAL=c495c2c3

NUM_INFERENCES=50 \
  "$ROOT/nafnet_deblur-qnn_dlc-w8a16/run_on_device.sh" \
  "$ROOT/onnx/input.jpg" \
  "$ROOT/nafnet_deblur-qnn_dlc-w8a16/output_qnn_w8a16.png"
```

脚本自动完成：

1. 加载 QAIRT 环境。
2. 检查 adb 设备数量。
3. 检查 SoC 是否为 SM8850/SM8850L。
4. 将图片转换为 NHWC uint16 native 输入。
5. 按文件大小判断是否需要 push。
6. 首次生成或后续复用 HTP context。
7. 使用 burst、shared buffer 和 native I/O 运行。
8. 拉取结果和 profiler。
9. 将 NHWC uint16 输出解码为 PNG。

### 10.2 强制重建 DLC context

```bash
REBUILD_CONTEXT=1 \
NUM_INFERENCES=50 \
  "$ROOT/nafnet_deblur-qnn_dlc-w8a16/run_on_device.sh" \
  "$ROOT/onnx/input.jpg" \
  "$ROOT/nafnet_deblur-qnn_dlc-w8a16/output_qnn_w8a16.png"
```

首次生成 O3/max-VTCM context 约需一分钟。

### 10.3 当前已缓存 context

当前设备上已经存在：

```text
/data/local/tmp/nafnet_deblur_qnn_w8a16/context_htp/
  nafnet_w8a16_htp_v81_maxvtcm_o3_ffa61c0ee24afd41.bin
```

文件大小约 83 MB。

### 10.4 DLC 手工生成 context

```bash
SERIAL=c495c2c3
REMOTE=/data/local/tmp/nafnet_deblur_qnn_w8a16

adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE:/vendor/lib64' &&
  export ADSP_LIBRARY_PATH='$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' &&
  ./qnn-context-binary-generator \
    --backend libQnnHtp.so \
    --model libQnnModelDlc.so \
    --dlc_path nafnet_deblur.dlc \
    --config_file htp_netrun_o3_config.json \
    --output_dir context_htp \
    --binary_file nafnet_w8a16_htp_v81_maxvtcm_o3_manual \
    --profiling_level basic \
    --log_level info
"
```

### 10.5 DLC 手工运行缓存 context

```bash
SERIAL=c495c2c3
REMOTE=/data/local/tmp/nafnet_deblur_qnn_w8a16

adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE:/vendor/lib64' &&
  export ADSP_LIBRARY_PATH='$REMOTE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' &&
  ./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context context_htp/nafnet_w8a16_htp_v81_maxvtcm_o3_ffa61c0ee24afd41.bin \
    --config_file htp_netrun_o3_config.json \
    --input_list input_list_native.txt \
    --output_dir output_htp_manual \
    --profiling_level basic \
    --perf_profile burst \
    --num_inferences 50 \
    --keep_num_outputs 1 \
    --use_native_input_files \
    --use_native_output_files \
    --shared_buffer \
    --log_level warn
"
```

注意 DLC 的 `LD_LIBRARY_PATH` 可以包含 `/vendor/lib64`，因为 `qnn-net-run` 不具有本次 ORT `libandroid.so` 冲突。不要把 DLC 的环境设置直接复制到 ONNX runner。

---

## 11. PC ONNX、Android ONNX 与 DLC 的异同

### 11.1 格式与可移植性

| 维度 | PC/Android ONNX | QNN DLC |
|---|---|---|
| 格式 | 开放 ONNX 标准 | Qualcomm 专有容器 |
| 模型文件 | `.onnx + .data` | `.dlc` |
| 跨平台 | 可使用 CPU/CUDA/QNN 等 EP | 面向 Qualcomm QNN |
| 本项目布局 | NCHW | NHWC |
| 本项目 dtype | uint16 | uint16 |
| 量化参数 | 与 DLC 相同 | 与 ONNX 相同 |
| 手机运行时 | ORT + QNN EP + QNN HTP | QNN loader + QNN HTP |
| HTP 最终执行 | 是 | 是 |
| 当前 context cache | 未接入 | 已接入并复用 |

### 11.2 ONNX 和 DLC 是不是同一个模型

两者属于同一个 NAFNet-DeBlur w8a16 模型，输入输出语义、固定分辨率和量化参数一致。但它们不是同一个二进制容器，也不能认为内部节点一定一一对应：

- ONNX 显式保存标准算子和 Q/DQ 节点。
- DLC 保存 QAIRT 转换后的 Qualcomm 图表示。
- QNN EP 在手机上还会再次把 ONNX 转换成 QNN graph。
- Context binary 是针对特定 HTP 架构和优化配置 Finalize 后的进一步部署产物。

可以把它们理解为：

```text
同一模型语义
  -> ONNX 标准图
  -> QNN DLC 容器
  -> 特定设备 HTP context
```

### 11.3 数值差异

Android ORT QNN EP 与 PC ORT CPU：

| 指标 | 结果 |
|---|---:|
| uint16 MAE | 11.679 LSB |
| uint16 最大绝对误差 | 319 LSB |
| 反量化 MAE | 0.0002306 |
| 反量化 RMSE | 0.0003240 |
| PSNR | 69.79 dB |

Android ORT QNN EP 与 QNN DLC：

| 指标 | 结果 |
|---|---:|
| uint16 MAE | 6.000 LSB |
| uint16 最大绝对误差 | 81 LSB |
| 反量化 MAE | 0.0001184 |
| 反量化 RMSE | 0.0001621 |
| PSNR | 75.81 dB |

这些差异来自后端 kernel、图变换、融合和计算顺序，不表示布局或量化协议错误。

### 11.4 性能差异

| 指标 | Android ONNX QNN EP | Android DLC | 差值 |
|---|---:|---:|---:|
| 最佳端到端平均 | 45.876 ms | 44.742 ms | +1.134 ms / +2.53% |
| HTP accelerator 平均 | 44.963 ms | 43.819 ms | +1.144 ms / +2.61% |

稳态差异很小，说明主要 HTP 计算已经接近。Android ONNX 多出的时间主要可能来自：

- ORT 到 QNN 的 glue。
- 输入输出 tensor 管理。
- QNN EP 调度。
- RPC 控制。
- 同步边界。

### 11.5 冷启动差异

如果只运行一张图片并把初始化计入：

```text
PC ORT CPU:
约 0.454 s Session + 2.180 s 推理 = 2.63 s

Android ORT QNN EP 当前路径:
约 53.956 s Session + 0.046 s 推理 = 54.00 s
```

因此当前未缓存的手机 ONNX 对一次性任务反而更慢。手机优势只出现在 Session 创建完成后的重复推理阶段。

DLC 已通过 context cache 将一分钟级准备移出后续运行路径，因此当前 DLC 更接近产品部署方式。

---

## 12. ONNX QNN EP context cache 待办

当前 runner 的 provider options 没有启用：

```text
ep.context_enable=1
ep.context_file_path=/path/to/nafnet_w8a16_ctx.onnx
ep.context_embed_mode=0
```

目标流程：

```text
第一次/构建阶段：
原始 ONNX -> QNN EP 在线编译 -> context ONNX + QNN binary

产品运行阶段：
context ONNX -> 加载已有 QNN binary -> HTP 推理
```

预期收益是移除当前约 54 秒的在线 graph composition/Finalize，而不是显著降低 45.88 ms 稳态 execute。

当前仓库尚未完成该功能的最终实测，因此这里属于明确的后续开发项，不应把预期当成已测结果。

---

## 13. 性能测试口径

正确比较必须满足：

1. 同一输入图片。
2. 同一固定分辨率 640x360。
3. 相同 warmup 数量。
4. 相同正式统计次数。
5. 不把图片解码和 PNG 保存混入模型延迟。
6. profiler 开启和关闭分别测试。
7. Session/context 初始化单独统计。
8. 同一进程连续执行多次。
9. 记录设备温度和电源状态。
10. 确认没有 CPU fallback。

延迟层级：

| 指标 | 含义 |
|---|---|
| Session init | ORT 加载、图优化、QNN graph composition/Finalize |
| NetRun/ORT wall | 上层调用观察到的单次端到端执行 |
| QNN execute | QNN runtime 区间 |
| RPC execute | AP 到 HTP 的 RPC 区间 |
| QNN accelerator | HTP 图执行主指标 |
| Accelerator excluding wait | 尽量排除等待的设备执行时间 |

不能将官网数字、不同设备、不同精度或不同计时边界直接相除。

---

## 14. 常见错误与排障

### 14.1 `adb unauthorized`

原因：手机尚未接受 RSA 调试授权。

处理：

```bash
adb kill-server
adb start-server
adb devices -l
```

解锁手机并接受授权弹窗。

### 14.2 多台设备

显式指定：

```bash
export SERIAL=设备序列号
adb -s "$SERIAL" shell getprop ro.product.model
```

### 14.3 ONNX 找不到外部数据

症状：Session 创建时报外部 initializer 或文件不存在。

检查：

```bash
adb -s "$SERIAL" shell \
  "ls -l '$REMOTE/nafnet_deblur.onnx' '$REMOTE/nafnet_deblur.data'"
```

两个文件必须在同一目录。

### 14.4 找不到 `libonnxruntime.so`

检查：

```bash
adb -s "$SERIAL" shell "
  cd '$REMOTE' &&
  export LD_LIBRARY_PATH='$REMOTE' &&
  ./ort_qnn_android_benchmark
"
```

如果动态链接正常，应打印 usage，而不是 shared library 错误。

### 14.5 找不到 QNN EP device

检查：

- `libonnxruntime_providers_qnn.so` 是否为 `arm64-v8a`。
- 插件是否使用绝对路径。
- ORT Android 和插件接口是否匹配。
- 是否出现 QNN 版本检查错误。

### 14.6 Session 只初始化 1-2 秒，推理约 2-3 秒

历史上这通常表示 QNN 图没有在 HTP 上形成，实际走了 CPU fallback。当前 runner 已禁用 CPU fallback，正常情况下会直接失败而不是静默回退。

检查：

- 是否传入 `libQnnHtp.so` 绝对路径。
- 是否启用 `skip_qnn_version_check=1`。
- 是否看到了 `EP device: QNNExecutionProvider`。
- 是否误用了不同版本的 ORT 或 QNN EP。

### 14.7 `SurfaceComposerClient` 或 `libandroid.so` 符号错误

删除 ONNX 运行环境中的 `/vendor/lib64`：

```bash
export LD_LIBRARY_PATH="$REMOTE"
```

### 14.8 找不到 HTP Skel

检查：

```bash
adb -s "$SERIAL" shell \
  "ls -l '$REMOTE/libQnnHtpV81Stub.so' '$REMOTE/libQnnHtpV81Skel.so'"
```

并确认 `ADSP_LIBRARY_PATH` 使用分号。

### 14.9 HTP 架构不匹配

当前配置只面向 SM8850/v81。换设备时需要重新确认：

- `ro.soc.model`。
- HTP 架构版本。
- 对应 Stub/Skel。
- 最大 VTCM。
- context 是否需要重新生成。

### 14.10 输出颜色错乱或图像不可识别

优先检查布局：

```text
ONNX output: NCHW
DLC output:  NHWC
```

两个 raw 字节数相同，reshape 错误不会自动报错。

### 14.11 首次运行一分钟

- ONNX：在线 QNN graph composition/Finalize，当前每个新 Session 都会发生。
- DLC：首次生成 context，后续脚本会复用缓存。

不要把首次 prepare 时间当成稳态推理时间。

### 14.12 多轮后延迟升高

可能原因：

- 温控降频。
- burst 状态变化。
- 调度波动。
- RPC control latency。
- profiler 开销。

应同步记录温度、频率和每轮延迟，不要只记录平均值。

---

## 15. 哈希与部署一致性校验

### 15.1 主机

```bash
cd /media/code/tools/naf/NAFNet

sha256sum \
  nafnet_deblur-onnx-w8a16/nafnet_deblur.onnx \
  nafnet_deblur-onnx-w8a16/nafnet_deblur.data \
  nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark \
  nafnet_deblur-qnn_dlc-w8a16/nafnet_deblur.dlc
```

### 15.2 手机

```bash
SERIAL=c495c2c3

adb -s "$SERIAL" shell '
  sha256sum \
    /data/local/tmp/nafnet_ort_qnn_w8a16/nafnet_deblur.onnx \
    /data/local/tmp/nafnet_ort_qnn_w8a16/nafnet_deblur.data \
    /data/local/tmp/nafnet_ort_qnn_w8a16/ort_qnn_android_benchmark \
    /data/local/tmp/nafnet_deblur_qnn_w8a16/nafnet_deblur.dlc
'
```

期望：

```text
48c5ae6afff38988efe88c1275bce704d025d729762f75e095ec9434a1ae12c5  nafnet_deblur.onnx
742aaccef608e9f4380dbb940994498dc0c50ca4bb9938b89e0109513321d004  nafnet_deblur.data
7cc176e593aa5e3c0b0a3a89cee40158eff999a91f47d6313a6361448ae621a1  ort_qnn_android_benchmark
ffa61c0ee24afd411fdd8fdf6dca82efcb8a3d8f6632d2345fba973160a97099  nafnet_deblur.dlc
```

---

## 16. 验收清单

### 16.1 PC ONNX

- [ ] Python 依赖安装成功。
- [ ] ONNX 和 `.data` 在同一目录。
- [ ] 输入生成的 raw 为 1,382,400 bytes。
- [ ] Session 使用 `CPUExecutionProvider`。
- [ ] 输出 shape 为 `[1,3,360,640]`。
- [ ] 输出 dtype 为 `uint16`。
- [ ] PNG 能正常解码。
- [ ] benchmark JSON 已保存。

### 16.2 Android ONNX

- [ ] `adb devices -l` 状态为 `device`。
- [ ] 设备是 arm64-v8a。
- [ ] SoC/HTP 与 Stub/Skel 匹配。
- [ ] runner 能打印 usage，无动态链接错误。
- [ ] 手机和主机模型哈希一致。
- [ ] `EP device: QNNExecutionProvider` 出现。
- [ ] CPU fallback 已禁用。
- [ ] Session 创建成功。
- [ ] 无 profiler 延迟约 46 ms。
- [ ] 输出 raw 为 1,382,400 bytes。
- [ ] NCHW 输出可以正常解码。

### 16.3 Android DLC

- [ ] `libQnnModelDlc.so` 已部署。
- [ ] DLC 和主机哈希一致。
- [ ] 输入是 NHWC uint16。
- [ ] context 与当前模型 hash、HTP 架构匹配。
- [ ] 使用最大 VTCM、O3、burst、shared buffer。
- [ ] 使用 native input/output flags。
- [ ] accelerator 平均约 43.8 ms。
- [ ] NetRun 平均约 44.7 ms。
- [ ] NHWC 输出可以正常解码。

### 16.4 产品化

- [ ] ORT/QNN AAR 移到稳定依赖目录。
- [ ] QNN Runtime 与插件版本正式对齐。
- [ ] ONNX QNN EP context cache 完成实测。
- [ ] APK/JNI 集成完成。
- [ ] context 按 SoC/HTP/runtime 版本管理。
- [ ] 完成持续运行温控测试。
- [ ] 完成 REDS val300 正式精度验收。

---

## 17. 仓库证据与入口

### Android ONNX

```text
nafnet_deblur-onnx-w8a16/nafnet_deblur.onnx
nafnet_deblur-onnx-w8a16/nafnet_deblur.data
nafnet_deblur-onnx-w8a16/metadata.json
nafnet_deblur-onnx-w8a16/benchmark_ort.py
nafnet_deblur-onnx-w8a16/ort_qnn_android_benchmark.cpp
nafnet_deblur-onnx-w8a16/benchmark_ort_cpu.json
nafnet_deblur-onnx-w8a16/android_qnn_ep_results/benchmark_android_qnn_ep.json
nafnet_deblur-onnx-w8a16/android_qnn_ep_results/qnn_profile_50.csv
nafnet_deblur-onnx-w8a16/android_qnn_ep_results/output_comparison.json
```

### QNN DLC

```text
nafnet_deblur-qnn_dlc-w8a16/nafnet_deblur.dlc
nafnet_deblur-qnn_dlc-w8a16/metadata.json
nafnet_deblur-qnn_dlc-w8a16/prepare_input.py
nafnet_deblur-qnn_dlc-w8a16/decode_output.py
nafnet_deblur-qnn_dlc-w8a16/run_on_device.sh
nafnet_deblur-qnn_dlc-w8a16/benchmark_results.json
nafnet_deblur-qnn_dlc-w8a16/accuracy_w8a16_vs_float_htp.json
nafnet_deblur-qnn_dlc-w8a16/context_maxvtcm_o3_info.json
```

### 相关工程报告

```text
docs/NAFNET_ONNX_W8A16_AND_DIRECT_HTP_REPORT.md
docs/NAFNET_DETAILED_ENGINEERING_REPORT.md
docs/QNN_DLC_DEVICE_RUNBOOK.md
docs/BLOG_NAFNET_FROM_313MS_TO_43MS.md
```

---

## 18. 推荐后续工作顺序

1. 立即复跑当前 Android ONNX 30 次无 profiler benchmark，确认约 46 ms。
2. 复跑 DLC 50 次，确认约 44.7 ms。
3. 在同一轮测试中同步记录温度和频率。
4. 实现并验证 ONNX QNN EP context cache。
5. 将 ORT 和 QNN EP AAR 从 `/tmp` 固化到第三方依赖目录。
6. 对齐 QNN EP 与 QNN Runtime 的正式支持版本。
7. 将原生 runner 逻辑封装为 JNI，完成 APK 集成。
8. 对 REDS val300 做正式质量验收，而不是只比较单张图片。
