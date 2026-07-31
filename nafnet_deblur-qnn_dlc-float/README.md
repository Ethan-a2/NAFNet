# NAFNet Deblur QNN DLC

该模型下载自 Qualcomm AI Hub 的 `nafnet_deblur` 页面。页面说明其权重来自 `NAFNet-REDS-width64` checkpoint，用于去除运动模糊和 JPEG 伪影。

模型接口：

- 输入：`image`，`float32`，NHWC `[1, 360, 640, 3]`，RGB `[0, 1]`；
- 输出：`deblurred_image`，`float32`，NHWC `[1, 360, 640, 3]`；
- DLC 生成版本：QAIRT 2.45；
- 本次运行版本：QAIRT 2.47。

## 一键运行

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601
source "$QNN_SDK_ROOT/bin/envsetup.sh"

./run_on_device.sh input.jpg output_qnn.png htp
```

也可以选择 CPU 或 GPU：

```bash
./run_on_device.sh input.jpg output_cpu.png cpu
./run_on_device.sh input.jpg output_gpu.png gpu
```

脚本会自动完成：

1. 将图片缩放到 640×360；
2. 生成 NHWC float32 RAW；
3. 推送 DLC、QNN 工具和后端依赖；
4. 识别 SM8850 使用 HTP v81；
5. 首次生成 HTP context binary；
6. 运行推理并拉取结果；
7. 解析 profiling；
8. 将 RAW 输出保存为 PNG。

HTP 首次生成 `O=3`、最大 VTCM context cache 大约需要两分钟，缓存大小约 145 MiB。之后脚本会复用缓存，端到端启动明显缩短。模型或 SDK 变化后可强制重建：

```bash
REBUILD_CONTEXT=1 ./run_on_device.sh input.jpg output_qnn.png htp
```

## 已验证设备

- Xiaomi `2512BPNDAC`；
- Android 16；
- Qualcomm SM8850；
- HTP v81；
- adb serial：测试时为 `c495c2c3`。

## 实测性能

单张 640×360，profiling level 为 basic：

| 后端 | 图准备/加载 | Execute | 主机观察端到端 |
|---|---:|---:|---:|
| CPU | 388.7 ms | 4323.9 ms | 约 6.19 s |
| GPU | 3001.4 ms | 2027.3 ms | 约 6.48 s |
| HTP 在线准备 | 58972.0 ms | 313.1 ms | 约 61.3 s |
| HTP context cache | 156.0 ms | 320.2 ms | 约 1.20 s |
| HTP 最大 VTCM + O3 + shared buffer，50 次平均 | 152.9 ms | **49.2 ms** | 单进程约 17.2 inf/s（含 I/O） |

默认 HTP 图只使用 4 MB VTCM，导致大量中间特征写回 DDR；改为目标 SoC 最大 VTCM，并在离线准备阶段启用 `O=3` 后，accelerator 平均延迟从约 317 ms 降至 49.2 ms。首次运行慢的主要原因仍是 67.9M 参数大图的离线准备。

完整结构化结果见 `benchmark_results.json`。

## 数值一致性

同一输入与 ONNX Runtime CPU 输出比较：

| 后端 | MAE | 最大绝对误差 | 与 ONNX 的 PSNR |
|---|---:|---:|---:|
| QNN CPU | 9.43e-8 | 2.21e-6 | 136.47 dB |
| QNN GPU | 7.00e-8 | 1.55e-6 | 139.16 dB |
| QNN HTP | 1.48e-4 | 2.61e-3 | 73.85 dB |

CPU/GPU 与 ONNX 基本逐元素一致。HTP 存在较小的低精度计算误差，但图像质量差异很小。

## 手工运行核心命令

DLC 不能直接作为 `--model`；需要 `libQnnModelDlc.so`：

```bash
./qnn-net-run \
  --backend libQnnCpu.so \
  --model libQnnModelDlc.so \
  --dlc_path nafnet_deblur.dlc \
  --input_list input_list.txt \
  --output_dir output_cpu
```

HTP 推荐先生成 context：

```bash
./qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path nafnet_deblur.dlc \
  --config_file htp_netrun_o3_config.json \
  --output_dir context_htp \
  --binary_file nafnet_htp_v81_maxvtcm_o3

./qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context context_htp/nafnet_htp_v81_maxvtcm_o3.bin \
  --config_file htp_netrun_o3_config.json \
  --input_list input_list.txt \
  --perf_profile burst \
  --shared_buffer \
  --output_dir output_htp
```

HTP 运行前需要：

```bash
export LD_LIBRARY_PATH=/data/local/tmp/nafnet_deblur_qnn:/vendor/lib64
export ADSP_LIBRARY_PATH="/data/local/tmp/nafnet_deblur_qnn;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

## 用户可能需要执行的操作

当前设备已经授权并完成运行，不需要额外操作。换手机时只需：

1. 开启开发者选项；
2. 开启 USB 调试；
3. USB 连接电脑；
4. 手机上接受 RSA 调试授权；
5. 确认 `adb devices -l` 显示状态为 `device`；
6. 非 SM8850 设备设置正确的 `HTP_ARCH`，例如 `HTP_ARCH=v79`。
