#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a float32 NHWC input for the NAFNet QNN DLC."
    )
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("--raw", type=Path, default=Path("input.raw"))
    parser.add_argument("--preview", type=Path, default=Path("input_640x360.png"))
    parser.add_argument("--input-list", type=Path, default=Path("input_list.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Image.open(args.input) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((640, 360), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.input_list.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.raw)
    image.save(args.preview)
    args.input_list.write_text(f"image:={args.raw.name}\n", encoding="utf-8")

    print(f"Prepared {args.raw}: shape={array.shape}, dtype={array.dtype}")
    print(f"Range: [{array.min():.6f}, {array.max():.6f}]")


if __name__ == "__main__":
    main()
