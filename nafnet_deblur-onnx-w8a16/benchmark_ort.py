#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the NAFNet w8a16 ONNX model.")
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("nafnet_deblur.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path(__file__).with_name("metadata.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output_ort_cpu.png"))
    parser.add_argument("--input-raw", type=Path, default=Path(__file__).with_name("input_ort_nchw_uint16.raw"))
    parser.add_argument("--output-raw", type=Path, default=Path(__file__).with_name("output_ort_cpu_uint16.raw"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("benchmark_ort_cpu.json"))
    parser.add_argument("--qnn-output", type=Path, help="Optional native uint16 NHWC QNN output")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1 or args.threads < 1:
        raise ValueError("warmup must be >= 0; runs and threads must be >= 1")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    model_info = metadata["model_files"][args.model.name]
    input_info = model_info["inputs"]["image"]
    output_info = model_info["outputs"]["deblurred_image"]
    _, channels, height, width = input_info["shape"]

    with Image.open(args.input) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    input_scale = input_info["quantization_parameters"]["scale"]
    input_zero_point = input_info["quantization_parameters"]["zero_point"]
    input_nhwc = np.asarray(image, dtype=np.float32) / 255.0
    input_nchw = np.transpose(input_nhwc, (2, 0, 1))[None, ...]
    quantized_input = np.clip(
        np.rint(input_nchw / input_scale + input_zero_point),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    args.input_raw.parent.mkdir(parents=True, exist_ok=True)
    quantized_input.tofile(args.input_raw)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = args.threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    init_start = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    output = None
    for _ in range(args.warmup):
        output = session.run(None, {"image": quantized_input})[0]

    run_times_ms = []
    for run_index in range(args.runs):
        start = time.perf_counter()
        output = session.run(None, {"image": quantized_input})[0]
        duration_ms = (time.perf_counter() - start) * 1000.0
        run_times_ms.append(duration_ms)
        print(f"run {run_index + 1}: {duration_ms:.3f} ms")

    if output is None:
        raise RuntimeError("No inference output was produced")
    if output.shape != (1, channels, height, width) or output.dtype != np.uint16:
        raise ValueError(f"Unexpected output: shape={output.shape}, dtype={output.dtype}")

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output_raw)

    output_scale = output_info["quantization_parameters"]["scale"]
    output_zero_point = output_info["quantization_parameters"]["zero_point"]
    output_float = (output.astype(np.float32) - output_zero_point) * output_scale
    output_nhwc = np.transpose(output_float[0], (1, 2, 0))
    output_image = np.rint(np.clip(output_nhwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_image).save(args.output)

    result = {
        "model": str(args.model),
        "input": str(args.input),
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "threads": args.threads,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "session_init_ms": init_ms,
        "run_times_ms": run_times_ms,
        "average_ms": float(np.mean(run_times_ms)),
        "median_ms": float(np.median(run_times_ms)),
        "minimum_ms": float(np.min(run_times_ms)),
        "maximum_ms": float(np.max(run_times_ms)),
        "output_native_min": int(output.min()),
        "output_native_max": int(output.max()),
    }

    if args.qnn_output:
        qnn_nhwc = np.fromfile(args.qnn_output, dtype=np.uint16).reshape(1, height, width, channels)
        qnn_nchw = np.transpose(qnn_nhwc, (0, 3, 1, 2))
        native_delta = output.astype(np.int32) - qnn_nchw.astype(np.int32)
        float_delta = native_delta.astype(np.float64) * output_scale
        rmse = float(np.sqrt(np.mean(float_delta**2)))
        result["qnn_comparison"] = {
            "qnn_output": str(args.qnn_output),
            "native_mae": float(np.mean(np.abs(native_delta))),
            "native_max_abs_error": int(np.max(np.abs(native_delta))),
            "native_exact_fraction": float(np.mean(native_delta == 0)),
            "dequantized_mae": float(np.mean(np.abs(float_delta))),
            "dequantized_rmse": rmse,
            "dequantized_max_abs_error": float(np.max(np.abs(float_delta))),
            "psnr_for_unit_range_db": float(20.0 * np.log10(1.0 / max(rmse, 1e-12))),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
