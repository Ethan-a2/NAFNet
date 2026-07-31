# NAFNet Deblur QNN w8a16

该目录包含 Qualcomm AI Hub 导出的 `w8a16` DLC。输入和输出均为 NHWC `uint16`，形状为 `1x360x640x3`；量化参数以 `metadata.json` 为准。

## 一键运行

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.47.0.260601
source "$QNN_SDK_ROOT/bin/envsetup.sh"

cd /media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16
./run_on_device.sh \
  /media/code/tools/naf/nafnet_deblur-onnx-float/input.jpg \
  output_qnn_w8a16.png
```

首次运行会生成针对 SM8850/v81 的 `O=3`、最大 VTCM HTP 上下文，约需一分钟；后续直接加载缓存。

单进程连续跑 50 次：

```bash
NUM_INFERENCES=50 ./run_on_device.sh \
  /media/code/tools/naf/nafnet_deblur-onnx-float/input.jpg \
  output_qnn_w8a16.png
```

强制重建上下文：

```bash
REBUILD_CONTEXT=1 ./run_on_device.sh input.jpg output.png
```

## 本机结果

设备为 SM8850、HTP v81，运行时为 QAIRT `2.47.0.260601`，DLC 由 QAIRT `2.45.0.260326154327` 生成。

| 配置 | HTP accelerator 平均延迟 |
|---|---:|
| 默认 4 MB VTCM | 327.597 ms |
| 最大 VTCM | 64.619 ms |
| 最大 VTCM + shared buffer | 62.916 ms |
| 最大 VTCM + O3 + shared buffer | **43.819 ms** |

量化输入与原 float 输入的最大重建误差为 `5.96e-8`。最佳配置输出与未优化量化上下文逐元素完全相同；相对已验证的 float HTP 输出，MAE 为 `0.001278`，PSNR 为 `54.69 dB`。

完整原始结果见 `results/`、`benchmark_results.json` 和 `accuracy_w8a16_vs_float_htp.json`。
