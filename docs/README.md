# NAFNet 与 QNN 文档索引

## 推荐阅读顺序

1. `NAFNET_ONE_PAGE_REPORT.md`：快速了解模型、部署、优化和结论。
2. `BLOG_NAFNET_FROM_313MS_TO_43MS.md`：按排障故事理解为什么从 313 ms 优化到 43.8 ms。
3. `NAFNET_DETAILED_ENGINEERING_REPORT.md`：完整原理、项目拆解、验证、消融、局限和测试题。
4. `QNN_DLC_DEVICE_RUNBOOK.md`：运行下一个 DLC 时直接复制执行的通用手册。

## 原始工程入口

- NAFNet 源码：`/media/code/tools/naf/NAFNet`
- ONNX：`/media/code/tools/naf/nafnet_deblur-onnx-float`
- QNN float：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-float`
- QNN w8a16：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16`

## 原始证据

- ONNX 逆向设计分析：`/media/code/tools/naf/nafnet_deblur-onnx-float/docs/NAFNET_ONNX_DESIGN_ANALYSIS.md`
- float 基准：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-float/benchmark_results.json`
- w8a16 消融：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16/benchmark_results.json`
- w8a16 精度：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16/accuracy_w8a16_vs_float_htp.json`
- Context 元数据：`/media/code/tools/naf/nafnet_deblur-qnn_dlc-w8a16/context_maxvtcm_o3_info.json`
