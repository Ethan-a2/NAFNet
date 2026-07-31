#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode NAFNet QNN float output.")
    parser.add_argument("raw", type=Path, help="deblurred_image.raw")
    parser.add_argument("output", type=Path, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    array = np.fromfile(args.raw, dtype=np.float32)
    expected_elements = 360 * 640 * 3
    if array.size != expected_elements:
        raise ValueError(
            f"Expected {expected_elements} float32 values, got {array.size}"
        )

    array = array.reshape(360, 640, 3)
    stats = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
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
