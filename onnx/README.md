# NAFNet Deblur ONNX 推理

此目录包含固定输入尺寸为 `1x3x360x640` 的 NAFNet 去模糊 ONNX 模型。模型使用 RGB、`float32`、`[0, 1]` 范围输入，并依赖同目录下的外部权重文件 `nafnet_deblur.data`。

完整的源码拆解、ONNX 逆向、设计决策、论文消融、本地验证和 Mermaid 知识地图见：

- [`docs/NAFNET_ONNX_DESIGN_ANALYSIS.md`](docs/NAFNET_ONNX_DESIGN_ANALYSIS.md)

## 安装

```bash
python3 -m pip install -r requirements.txt
```

如需 NVIDIA GPU 推理，请将 `onnxruntime` 替换为与本机 CUDA 环境匹配的 `onnxruntime-gpu`。

## 运行

快速推理会把输入缩放到模型固定尺寸，并输出 `640x360` 图片：

```bash
python3 inference.py input.jpg -o output.png
```

保持原图分辨率时，脚本会按 `640x360` 分块推理：

```bash
python3 inference.py input.jpg -o output.png --mode tile
```

可增加重叠来减轻分块边界，但会增加推理次数：

```bash
python3 inference.py input.jpg -o output.png --mode tile --overlap 32
```

默认自动使用 CUDA（可用时），否则回退到 CPU。也可以显式指定：

```bash
python3 inference.py input.jpg -o output.png --provider cpu
python3 inference.py input.jpg -o output.png --provider cuda
```

查看全部参数：

```bash
python3 inference.py --help
```
