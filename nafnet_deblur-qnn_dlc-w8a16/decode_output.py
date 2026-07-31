#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode native uint16 output from the NAFNet w8a16 DLC."
    )
    parser.add_argument("raw", type=Path, help="deblurred_image_native.raw")
    parser.add_argument("output", type=Path, help="Output image path")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    model = metadata["model_files"]["nafnet_deblur.dlc"]
    output_info = model["outputs"]["deblurred_image"]
    if output_info["dtype"] != "uint16":
        raise ValueError(f"Expected uint16 output, got {output_info['dtype']}")

    _, height, width, channels = output_info["shape"]
    expected_elements = height * width * channels
    native_array = np.fromfile(args.raw, dtype=np.uint16)
    if native_array.size != expected_elements:
        raise ValueError(
            f"Expected {expected_elements} uint16 values, got {native_array.size}"
        )

    quantization = output_info["quantization_parameters"]
    scale = quantization["scale"]
    zero_point = quantization["zero_point"]
    array = (native_array.reshape(height, width, channels).astype(np.float32) - zero_point) * scale
    stats = {
        "shape": list(array.shape),
        "native_dtype": "uint16",
        "scale": scale,
        "zero_point": zero_point,
        "native_min": int(native_array.min()),
        "native_max": int(native_array.max()),
        "dequantized_min": float(array.min()),
        "dequantized_max": float(array.max()),
        "finite": bool(np.isfinite(array).all()),
    }
    if not stats["finite"]:
        raise ValueError("Output contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image).save(args.output)
    stats_path = args.output.with_suffix(args.output.suffix + ".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
