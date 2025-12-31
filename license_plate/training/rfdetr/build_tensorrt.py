#!/usr/bin/env python3
"""Build TensorRT engines from ONNX models and test inference."""

from pathlib import Path
from typing import Literal

import numpy as np
import tensorrt as trt  # type: ignore[import-untyped]

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"
ONNX_DIR = OUTPUT_DIR / "exported_models"
TRT_DIR = OUTPUT_DIR / "tensorrt_export"

Precision = Literal["fp32", "fp16", "int8"]


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: Precision = "fp16",
    workspace_mb: int = 1024,
) -> bool:
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.INFO)  # type: ignore[attr-defined]
    builder = trt.Builder(logger)  # type: ignore[attr-defined]
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # type: ignore[attr-defined]
    parser = trt.OnnxParser(network, logger)  # type: ignore[attr-defined]

    # Parse ONNX
    print(f"Parsing {onnx_path.name}...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            return False

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)  # type: ignore[attr-defined]

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
            print("  Using FP16 precision")
        else:
            print("  FP16 not supported, falling back to FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)  # type: ignore[attr-defined]
            print("  Using INT8 precision")
        else:
            print("  INT8 not supported, falling back to FP32")

    # Build engine
    print(f"Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("  Failed to build engine")
        return False

    # Save engine
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {engine_path.name} ({size_mb:.1f} MB)")
    return True


def test_engine(engine_path: Path) -> bool:
    """Load and test TensorRT engine with dummy input."""
    logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
    runtime = trt.Runtime(logger)  # type: ignore[attr-defined]

    print(f"Loading {engine_path.name}...")
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("  Failed to load engine")
        return False

    context = engine.create_execution_context()

    # Get I/O tensor info
    print("  Tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"    {name}: {list(shape)} ({dtype}) [{mode}]")

    # Allocate buffers and run inference
    import torch

    device = torch.device("cuda")
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        # Create tensor
        tensor = torch.zeros(tuple(shape), dtype=torch.from_numpy(np.zeros(1, dtype=dtype)).dtype, device=device)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # type: ignore[attr-defined]
            tensor = torch.randn_like(tensor)
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())

    # Run inference
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Print output shapes
    print("  Inference successful:")
    for name, tensor in buffers.items():
        if engine.get_tensor_mode(engine.get_tensor_name(list(buffers.keys()).index(name))) == trt.TensorIOMode.OUTPUT:  # type: ignore[attr-defined]
            print(f"    {name}: {list(tensor.shape)}")

    return True


def benchmark_engine(engine_path: Path, warmup: int = 10, iterations: int = 100) -> None:
    """Benchmark TensorRT engine inference speed."""
    import time

    import torch

    logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
    runtime = trt.Runtime(logger)  # type: ignore[attr-defined]

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    device = torch.device("cuda")

    # Allocate buffers
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        tensor = torch.zeros(tuple(shape), dtype=torch.from_numpy(np.zeros(1, dtype=dtype)).dtype, device=device)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # type: ignore[attr-defined]
            tensor = torch.randn_like(tensor)
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    fps = iterations / elapsed
    print(f"  {engine_path.name}: {avg_ms:.2f} ms/inference, {fps:.1f} FPS")


def main():
    TRT_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        ("lp_detection.onnx", "lp_detection.engine"),
        ("char_detection.onnx", "char_detection.engine"),
    ]

    precision: Precision = "fp16"

    print("=" * 60)
    print("Building TensorRT Engines")
    print("=" * 60)

    for onnx_name, engine_name in models:
        onnx_path = ONNX_DIR / onnx_name
        engine_path = TRT_DIR / engine_name

        if not onnx_path.exists():
            print(f"ONNX not found: {onnx_path}")
            continue

        if not build_engine(onnx_path, engine_path, precision):
            continue

    print("\n" + "=" * 60)
    print("Testing Engines")
    print("=" * 60)

    for _, engine_name in models:
        engine_path = TRT_DIR / engine_name
        if engine_path.exists():
            test_engine(engine_path)

    print("\n" + "=" * 60)
    print("Benchmarking Engines")
    print("=" * 60)

    for _, engine_name in models:
        engine_path = TRT_DIR / engine_name
        if engine_path.exists():
            benchmark_engine(engine_path)


if __name__ == "__main__":
    main()
