#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NAFNet image deblurring with ONNX Runtime."
    )
    parser.add_argument("input", type=Path, help="Path to the blurry input image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output.png"),
        help="Output image path (default: output.png)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("nafnet_deblur.onnx"),
        help="ONNX model path",
    )
    parser.add_argument(
        "--mode",
        choices=("resize", "tile"),
        default="resize",
        help="Resize to the model input or tile at the original resolution",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Tile overlap in pixels when --mode tile is used (default: 0)",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime execution provider (default: auto)",
    )
    return parser.parse_args()


def choose_providers(requested: str) -> list[str]:
    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu "
                "with a compatible CUDA runtime, or use --provider cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path: Path, provider: str) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=choose_providers(provider),
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("Expected a model with exactly one input and one output")
    return session


def model_image_size(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise RuntimeError(f"Expected NCHW input [1, 3, H, W], got {shape}")
    if not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise RuntimeError(f"Expected fixed input dimensions, got {shape}")
    return shape[2], shape[3]


def infer_tile(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
    result = session.run([output_name], {input_name: tensor})[0]
    if result.shape[0] != 1 or result.shape[1] != 3:
        raise RuntimeError(f"Expected NCHW RGB output, got {result.shape}")
    return result[0].transpose(1, 2, 0)


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def blend_axis(starts: list[int], index: int, tile_size: int) -> np.ndarray:
    weights = np.ones(tile_size, dtype=np.float32)
    if index > 0:
        leading_overlap = starts[index - 1] + tile_size - starts[index]
        if leading_overlap > 0:
            weights[:leading_overlap] *= np.linspace(
                0.0, 1.0, leading_overlap + 2, dtype=np.float32
            )[1:-1]
    if index + 1 < len(starts):
        trailing_overlap = starts[index] + tile_size - starts[index + 1]
        if trailing_overlap > 0:
            weights[-trailing_overlap:] *= np.linspace(
                1.0, 0.0, trailing_overlap + 2, dtype=np.float32
            )[1:-1]
    return weights


def infer_tiled(
    session: ort.InferenceSession,
    image: np.ndarray,
    tile_height: int,
    tile_width: int,
    overlap: int,
) -> np.ndarray:
    if overlap < 0 or overlap >= min(tile_height, tile_width):
        raise ValueError(
            f"Overlap must be between 0 and {min(tile_height, tile_width) - 1}"
        )

    image_height, image_width = image.shape[:2]
    padded_height = max(image_height, tile_height)
    padded_width = max(image_width, tile_width)
    padded = np.pad(
        image,
        ((0, padded_height - image_height), (0, padded_width - image_width), (0, 0)),
        mode="edge",
    )

    y_starts = tile_starts(padded_height, tile_height, overlap)
    x_starts = tile_starts(padded_width, tile_width, overlap)
    output = np.zeros_like(padded, dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width, 1), dtype=np.float32)

    tile_count = len(y_starts) * len(x_starts)
    tile_number = 0
    for y_index, y_start in enumerate(y_starts):
        y_weights = blend_axis(y_starts, y_index, tile_height)
        for x_index, x_start in enumerate(x_starts):
            tile_number += 1
            print(f"Processing tile {tile_number}/{tile_count}", flush=True)
            x_weights = blend_axis(x_starts, x_index, tile_width)
            weights = y_weights[:, None] * x_weights[None, :]
            tile = padded[
                y_start : y_start + tile_height,
                x_start : x_start + tile_width,
            ]
            result = infer_tile(session, tile)
            output[
                y_start : y_start + tile_height,
                x_start : x_start + tile_width,
            ] += result * weights[:, :, None]
            weight_sum[
                y_start : y_start + tile_height,
                x_start : x_start + tile_width,
            ] += weights[:, :, None]

    output /= np.maximum(weight_sum, np.finfo(np.float32).eps)
    return output[:image_height, :image_width]


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    with Image.open(path) as image:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32) / 255.0


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8_image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(uint8_image).save(path)


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    session = create_session(args.model, args.provider)
    tile_height, tile_width = model_image_size(session)
    image = load_image(args.input)

    if args.mode == "resize":
        resized = Image.fromarray(np.rint(image * 255.0).astype(np.uint8)).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        model_input = np.asarray(resized, dtype=np.float32) / 255.0
        result = infer_tile(session, model_input)
    else:
        result = infer_tiled(
            session,
            image,
            tile_height,
            tile_width,
            args.overlap,
        )

    save_image(result, args.output)
    elapsed = time.perf_counter() - started_at
    print(
        f"Saved {args.output} ({result.shape[1]}x{result.shape[0]}) "
        f"using {session.get_providers()[0]} in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
