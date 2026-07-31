# 从 313 ms 到 43.8 ms：一次 NAFNet 手机 NPU 部署的完整排障记录

> 日期：2026-07-31  
> 设备：Xiaomi 2512BPNDAC，Snapdragon SM8850，HTP v81  
> 模型：NAFNet-REDS-width64，640×360 去模糊

模型第一次在手机上跑起来时，我看到的数字是 313 ms。

Qualcomm AI Hub 页面上却有一个很醒目的“约 17 ms”。表面看，本地慢了将近二十倍。最自然的第一反应是：

- 是不是模型跑到了 CPU？
- 是不是量化模型才是 17 ms？
- 是不是手机没有进入高性能模式？
- 是不是文件 I/O 和 adb 把时间算进去了？
- 是不是 QNN 版本不兼容？

最后的答案比“开个 burst”复杂，但也更有价值：**问题主要不在时钟频率，而在 HTP 图如何使用片上内存，以及 context 是怎样被编译出来的。**

这篇文章记录从理解 NAFNet，到运行 DLC，再到把 w8a16 从 327.6 ms 优化到 43.8 ms 的完整过程。

---

## 一、先理解模型：NAFNet 到底在做什么

去模糊不是普通的图像滤镜。

相机曝光期间，如果手、相机或物体发生运动，同一物体会在多个位置留下能量。一个简化的观测模型是：

$$
B=\mathcal{K}(S)+N
$$

`S` 是我们希望得到的清晰图，`B` 是拍到的模糊图，`K` 是运动和曝光形成的空间变化模糊过程，`N` 是噪声、JPEG 和 ISP 误差。

问题在于，这个过程通常不可逆。模糊后的一个像素可能混合了多个清晰位置的信息，而已经消失的高频细节不可能靠数学除法原样找回。模型真正做的是：根据训练数据学到自然图像和运动模糊的先验，选择一个最合理的清晰解释。

### NAFNet 的核心思路

NAFNet 的名字来自 Nonlinear Activation Free Network。它没有使用 ReLU、GELU、Sigmoid 或 Softmax 这样的传统激活函数，但这不意味着它是线性网络。

它的核心 NAFBlock 使用了一个非常简单的门控：

$$
SimpleGate(x)=x_1\odot x_2
$$

特征沿通道分成两半，再逐元素相乘。乘法本身就会产生二阶交互，因此仍然提供了强非线性。

一个 NAFBlock 大致是：

```text
LayerNorm
  → 1×1 Conv
  → 3×3 Depthwise Conv
  → SimpleGate
  → Simplified Channel Attention
  → 1×1 Conv
  → Residual

LayerNorm
  → 1×1 Conv
  → SimpleGate
  → 1×1 Conv
  → Residual
```

整个网络则是一个四层 U-Net：编码器逐步降低分辨率、扩大感受野，解码器逐步恢复分辨率，同时用 skip connection 保留高频细节。最后网络输出与原始模糊图相加，让网络学习“应该修改多少”，而不是从零生成一张图。

当前 AI Hub 模型是 `NAFNet-REDS-width64`：基础宽度 64，共 36 个 NAFBlock、约 67.89M 参数。它是一个相当重的图像恢复网络。

---

## 二、先把 DLC 正确加载起来

拿到的目录里只有：

```text
nafnet_deblur.dlc
metadata.json
```

metadata 给了两个非常重要的信息。

第一，模型精度是 `w8a16`：

- 权重 8 bit；
- 激活 16 bit；
- 输入输出都是 uint16。

第二，DLC 的生成版本是：

```json
"qairt": "2.45.0.260326154327"
```

这不是从运行日志猜出来的，而是包自带的生成元数据。手机运行时使用的是 QAIRT 2.47，所以要明确区分：**模型生成版本是 2.45，运行版本是 2.47。**

### DLC 不是普通的 model.so

最开始容易犯的错误是直接这样传：

```bash
qnn-net-run --model nafnet_deblur.dlc
```

正确方式是使用 `libQnnModelDlc.so` 作为 loader：

```bash
qnn-net-run \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path nafnet_deblur.dlc \
  --input_list input_list.txt
```

运行 HTP 还需要对应架构的 Stub、Skel 和 Prepare 库。当前手机的 SoC 是 SM8850，对应：

- QNN soc model 87；
- HTP v81；
- `libQnnHtpV81Stub.so`；
- `libQnnHtpV81Skel.so`。

把这些组件通过 adb 推到 `/data/local/tmp`，设置 `LD_LIBRARY_PATH` 和 `ADSP_LIBRARY_PATH` 后，模型成功运行。

但第一次在线 prepare 花了约 59 秒，Execute 仍然在 313 ms 左右。

---

## 三、第一次误判：是不是没有开 burst

看到 NPU 慢，最常见的动作是打开高性能模式：

```bash
--perf_profile burst
```

结果是：

| 配置 | Accelerator 平均延迟 |
|---|---:|
| 默认 | 327.597 ms |
| burst | 327.558 ms |

几乎没有变化。

这非常重要。它告诉我们：**当前瓶颈不是因为 HTP 频率没有拉高。**

接着换成原生 uint16 输入输出，希望减少量化转换和文件处理：

| 配置 | Accelerator 平均延迟 |
|---|---:|
| float 文件输入 | 327.558 ms |
| 原生 uint16 输入 | 327.253 ms |

仍然没有实质变化。

这又排除了一个方向：模型的 300 多毫秒并不是主机 I/O 或输入量化造成的，因为 profiling 中真正的 accelerator execute 本身就已经很慢。

---

## 四、真正的突破：VTCM

HTP 有一块片上高速内存 VTCM。可以把它理解成厨师手边的工作台：

- VTCM 是工作台，容量小但访问快；
- LPDDR 是仓库，容量大但搬运慢；
- 中间特征放不进工作台时，就要不断在工作台和仓库之间往返。

QNN SDK 文档显示，如果没有特殊配置，图 context 默认写入 4 MB VTCM。

而 NAFNet 是图像复原网络。即使输入只有 640×360，它仍然会产生大量高分辨率、多通道中间特征。67.89M 参数只是模型大小的一部分，真正影响执行的还有激活和中间张量的数据移动。

配置中把：

```json
"vtcm_mb": 0
```

设置为 0 不是禁用 VTCM，而是请求目标 SoC 的最大值。随后通过 `qnn-context-binary-utility` 检查生成的 context，得到：

```json
{
  "spillFillBufferSize": 61276160,
  "optimizationLevel": 3,
  "vtcmSize": 8,
  "numHvxThreads": 8
}
```

也就是说，SM8850 上这个图最终使用了 8 MB VTCM。

只做这一项修改，延迟发生了数量级变化：

| 配置 | Accelerator 平均延迟 |
|---|---:|
| 默认 4 MB | 327.597 ms |
| 最大 8 MB | 64.619 ms |

约 5.1 倍加速。

这说明最初真正的瓶颈是片上内存驻留与 spill/fill，而不是算力或频率。

---

## 五、第二个关键：O3 图优化

最大 VTCM 把延迟降到了 64 ms，但官网同精度仍然更快。

接下来启用 HTP finalize optimization level 3：

```json
{
  "graphs": [
    {
      "graph_names": ["graph_adnms07w"],
      "vtcm_mb": 0,
      "O": 3
    }
  ]
}
```

O3 不是运行时开关，而是在生成 context 时改变图的准备结果。它会尝试更完整的编译优化、分块和调度，因此 prepare 更慢，context 也更大。

生成时间从约 22 秒增加到约 54 秒，但 Execute 继续下降：

| 配置 | Accelerator 平均延迟 |
|---|---:|
| 最大 VTCM + shared buffer | 62.916 ms |
| 最大 VTCM + O3 + shared buffer | **43.819 ms** |

O3 又带来了约 1.44 倍加速。

从默认配置到最终配置，总加速比是：

$$
327.597/43.819\approx7.48
$$

更重要的是，优化前后的 uint16 输出逐元素完全一致。它不是通过降低精度换来的速度，而是同一张计算图获得了更好的执行计划。

---

## 六、为什么最终还是没有达到 17 ms

这里出现了第二个容易误解的地方：官网同时展示多种精度。

当前页面在 Samsung Galaxy S26 上的数据大致是：

| 精度 | QNN DLC 延迟 |
|---|---:|
| w8a8 | 17.199 ms |
| w8a16 | 39.251 ms |
| float | 44.648 ms |

我们下载的模型是 w8a16，不是 w8a8。因此正确比较应该是：

```text
本机 w8a16：43.819 ms
页面 w8a16：39.251 ms
```

差距约 11.6%，而不是二十倍。

float 模型也做了相同优化：

```text
本机 float：49.192 ms
页面 float：44.648 ms
```

差距约 10.2%。两个精度的差距比例非常接近，这反而说明本地优化已经基本对齐正确口径。

剩余差距可能来自 Samsung 与 Xiaomi 的固件、LPDDR 策略、电源管理、HTP 驱动、SDK 版本和 benchmark 统计方式。没有完全相同的设备和底层计数器，不能把其中任何一个写成确定根因。

---

## 七、量化到底带来了什么

w8a16 的 DLC 从约 260 MiB 降到约 75 MiB，这是非常明确的存储收益。

但延迟只从 float 的 49.192 ms 降到 w8a16 的 43.819 ms，提升约 12.3%。为什么不是 2 倍甚至 4 倍？

因为 w8a16 只把权重降到 8 bit，激活仍是 16 bit。对于 NAFNet 这种高分辨率图像恢复网络，中间特征的数据移动非常重要。当执行已经受 VTCM、DDR 和分块影响时，权重位宽并不是唯一决定因素。

这也是一个通用教训：

> 模型量化减少了什么，不等于端到端瓶颈就在哪里。

---

## 八、如何证明输出仍然正确

性能数字必须和正确性一起看。

首先，将 float QNN 后端与 ONNX Runtime 输出对齐：

| 后端 | 相对 ONNX PSNR |
|---|---:|
| QNN CPU | 136.47 dB |
| QNN GPU | 139.16 dB |
| QNN HTP | 73.85 dB |

CPU/GPU 基本逐元素一致；HTP 存在很小的低精度误差，但仍高度一致。

然后比较 w8a16 和 float HTP：

- MAE：0.001278；
- 最大绝对误差：0.02567；
- PSNR：54.69 dB。

这些结果证明部署格式和优化链路是正确的。但它们不能证明模型已经达到 REDS checkpoint 的完整质量指标。真正的模型验收还需要 REDS val300、有 GT 的 PSNR/SSIM，以及真实手机模糊样本。

---

## 九、这次排障最有价值的工具

### adb

它不只是推文件，还用于确认设备状态、SoC、Android 版本、温度、库路径和执行环境。

### qnn-net-run

用于加载 DLC 或 context、执行图、设置性能模式、原生 I/O、shared buffer 和多轮推理。

### qnn-context-binary-generator

这是整个优化过程的关键工具。它不仅消除在线 prepare，还把 VTCM 和 O3 编译决策固化进 context。

### qnn-profile-viewer

它把笼统的“程序花了多久”拆成 context load、RPC、QNN accelerator 和 NetRun 时间，帮助排除 adb 和文件 I/O 干扰。

### qnn-context-binary-utility

它最终证明 optimized context 中：

- VTCM 是 8 MB；
- optimization level 是 3；
- HVX 使用 8 个线程；
- spill/fill buffer 约 61.28 MB。

### metadata.json

它是数据协议的唯一可信入口：shape、dtype、scale、zero point 和 DLC 生成版本都来自这里。

### Python、NumPy、Pillow、ONNX Runtime

它们负责预处理、量化、反量化、参考输出和数值误差分析。

---

## 十、可以复用到其它模型的经验

1. **先核对比较口径。**同一精度、设备、分辨率和统计区间再比较。
2. **先证明能跑，再证明跑对，再证明跑快。**不要颠倒顺序。
3. **把 prepare 和 execute 分开。**首次运行慢不等于每次推理慢。
4. **不要迷信 burst。**频率只解决频率瓶颈。
5. **检查 context，而不只是命令。**真正的 VTCM、优化等级和线程数写在 context 元数据里。
6. **图像模型优先怀疑中间特征搬运。**尤其是高分辨率 U-Net、超分和去噪网络。
7. **每个优化都做消融。**否则无法知道收益来自哪里。
8. **性能优化必须做输出一致性。**更快但结果错了没有意义。
9. **保存完整环境。**SDK、SoC、Stub/Skel、配置和模型哈希缺一不可。
10. **最终目标不是一个数字，而是可重复的工程流程。**

---

## 十一、最终命令

当前 w8a16 已经封装成一键脚本：

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601
source "$QNN_SDK_ROOT/bin/envsetup.sh"

cd /media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16

NUM_INFERENCES=50 ./run_on_device.sh \
  /media/code/tools/naf/nafnet_deblur-onnx-float/input.jpg \
  output_qnn_w8a16.png
```

首次生成 O3/max-VTCM context，后续自动复用。模型或 SDK 变化时：

```bash
REBUILD_CONTEXT=1 NUM_INFERENCES=50 ./run_on_device.sh input.jpg output.png
```

---

## 结语

这次优化过程中，最容易做的是打开 `burst`，最有价值的却是证明 `burst` 不解决问题。

真正的转折来自把一个模糊的“为什么 NPU 慢”，拆成：

- 模型有没有回退；
- 慢在准备还是执行；
- 慢在计算还是数据搬运；
- context 实际用了多少 VTCM；
- O3 是否真正写入；
- 比较的是不是同一种精度。

最终，313 ms 并不是 NAFNet 在这台手机上的能力上限。它只是一个 sub-optimal context 的结果。

当图编译、片上内存和计时口径被正确处理后，答案变成了 43.8 ms。
