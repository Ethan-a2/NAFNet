#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare native uint16 NHWC input for the NAFNet w8a16 DLC."
    )
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("--raw", type=Path, default=Path("input_uint16.raw"))
    parser.add_argument("--preview", type=Path, default=Path("input_640x360.png"))
    parser.add_argument(
        "--input-list", type=Path, default=Path("input_list_native.txt")
    )
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
    input_info = model["inputs"]["image"]
    if input_info["dtype"] != "uint16":
        raise ValueError(f"Expected uint16 input, got {input_info['dtype']}")

    _, height, width, channels = input_info["shape"]
    if channels != 3:
        raise ValueError(f"Expected RGB input, got {channels} channels")
    quantization = input_info["quantization_parameters"]
    scale = quantization["scale"]
    zero_point = quantization["zero_point"]

    with Image.open(args.input) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    float_array = np.asarray(image, dtype=np.float32) / 255.0
    native_array = np.clip(
        np.rint(float_array / scale + zero_point), 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.input_list.parent.mkdir(parents=True, exist_ok=True)
    native_array.tofile(args.raw)
    image.save(args.preview)
    args.input_list.write_text(f"image:={args.raw.name}\n", encoding="utf-8")

    reconstructed = (native_array.astype(np.float32) - zero_point) * scale
    max_error = float(np.max(np.abs(reconstructed - float_array)))
    print(f"Prepared {args.raw}: shape={native_array.shape}, dtype=uint16")
    print(f"Quantized range: [{native_array.min()}, {native_array.max()}]")
    print(f"Input quantization max error: {max_error:.9g}")


if __name__ == "__main__":
    main()
