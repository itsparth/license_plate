#!/usr/bin/env python3
"""Compare TensorRT vs ONNX vs PyTorch model accuracy.

Tests all three backends on test datasets and compares:
- Numerical precision (max/mean difference in outputs)
- Detection consistency across backends
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import tensorrt as trt  # type: ignore[import-untyped]
import torch
from PIL import Image

from rfdetr.detr import RFDETRNano

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"

# Model paths
LP_DETECTION_MODEL = OUTPUT_DIR / "lp_detection_training" / "checkpoint_best_ema.pth"
CHAR_DETECTION_MODEL = OUTPUT_DIR / "rfdetr_training" / "checkpoint_best_ema.pth"
LP_ONNX = OUTPUT_DIR / "exported_models" / "lp_detection.onnx"
CHAR_ONNX = OUTPUT_DIR / "exported_models" / "char_detection.onnx"
LP_TRT = OUTPUT_DIR / "tensorrt_export" / "lp_detection.engine"
CHAR_TRT = OUTPUT_DIR / "tensorrt_export" / "char_detection.engine"

# Test datasets
LP_TEST_DIR = OUTPUT_DIR / "lp_detection_combined" / "test"
CHAR_TEST_DIR = OUTPUT_DIR / "training_data" / "test"


def load_pytorch_model(checkpoint_path: Path) -> tuple[torch.nn.Module, int]:
    """Load PyTorch model and return (model, input_size)."""
    rfdetr = RFDETRNano(pretrain_weights=str(checkpoint_path))
    model = rfdetr.model.model  # type: ignore[union-attr]
    model.cpu()  # type: ignore[union-attr]
    model.eval()  # type: ignore[union-attr]
    if hasattr(model, "export"):
        model.export()  # type: ignore[union-attr]
    return model, rfdetr.model.resolution  # type: ignore[return-value]


def load_onnx_model(onnx_path: Path) -> ort.InferenceSession:
    """Load ONNX model."""
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


class TensorRTInference:
    """TensorRT inference wrapper."""

    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[attr-defined]
        runtime = trt.Runtime(logger)  # type: ignore[attr-defined]

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Allocate buffers
        self.buffers: dict[str, torch.Tensor] = {}
        self.input_name = ""
        self.output_names: list[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            tensor = torch.zeros(
                tuple(shape),
                dtype=torch.from_numpy(np.zeros(1, dtype=dtype)).dtype,
                device="cuda",
            )
            self.buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # type: ignore[attr-defined]
                self.input_name = name
            else:
                self.output_names.append(name)

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on input image."""
        # Copy input to GPU
        self.buffers[self.input_name].copy_(torch.from_numpy(image).cuda())

        # Run inference
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Get outputs (boxes, logits)
        boxes = self.buffers[self.output_names[0]].cpu().numpy()
        logits = self.buffers[self.output_names[1]].cpu().numpy()
        return boxes, logits


def preprocess_image(image_path: Path, input_size: int) -> np.ndarray:
    """Load and preprocess image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # Add batch dimension
    return arr


def pytorch_inference(model: torch.nn.Module, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run PyTorch inference."""
    tensor = torch.from_numpy(image).float()
    with torch.no_grad():
        boxes, logits, _ = model(tensor)
    return boxes.numpy(), logits.numpy()


def onnx_inference(session: ort.InferenceSession, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX inference."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image})
    return outputs[0], outputs[1]  # type: ignore[return-value]


def compare_outputs(
    ref_boxes: np.ndarray,
    ref_logits: np.ndarray,
    test_boxes: np.ndarray,
    test_logits: np.ndarray,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compare reference and test outputs."""
    boxes_diff = np.abs(ref_boxes - test_boxes)
    logits_diff = np.abs(ref_logits - test_logits)

    # Compare high-confidence predictions
    ref_scores = 1 / (1 + np.exp(-ref_logits))
    test_scores = 1 / (1 + np.exp(-test_logits))
    ref_max_scores = np.max(ref_scores, axis=-1)[0]
    test_max_scores = np.max(test_scores, axis=-1)[0]

    ref_num_detections = int(np.sum(ref_max_scores > conf_threshold))
    test_num_detections = int(np.sum(test_max_scores > conf_threshold))

    high_conf_mask = (ref_max_scores > conf_threshold) | (test_max_scores > conf_threshold)
    num_high_conf = int(np.sum(high_conf_mask))

    if num_high_conf > 0:
        ref_classes = np.argmax(ref_logits[0, high_conf_mask], axis=-1)
        test_classes = np.argmax(test_logits[0, high_conf_mask], axis=-1)
        class_match_rate = float(np.mean(ref_classes == test_classes))
        high_conf_boxes_diff = boxes_diff[0, high_conf_mask]
        high_conf_logits_diff = logits_diff[0, high_conf_mask]
    else:
        class_match_rate = 1.0
        high_conf_boxes_diff = np.array([0.0])
        high_conf_logits_diff = np.array([0.0])

    return {
        "boxes_max_diff": float(np.max(boxes_diff)),
        "boxes_mean_diff": float(np.mean(boxes_diff)),
        "logits_max_diff": float(np.max(logits_diff)),
        "logits_mean_diff": float(np.mean(logits_diff)),
        "high_conf_boxes_max": float(np.max(high_conf_boxes_diff)),
        "high_conf_boxes_mean": float(np.mean(high_conf_boxes_diff)),
        "high_conf_logits_max": float(np.max(high_conf_logits_diff)),
        "high_conf_logits_mean": float(np.mean(high_conf_logits_diff)),
        "num_high_conf": num_high_conf,
        "ref_num_detections": ref_num_detections,
        "test_num_detections": test_num_detections,
        "class_match_rate": class_match_rate,
    }


def aggregate_results(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate comparison results across images."""
    return {
        "boxes_max_diff": max(d["boxes_max_diff"] for d in diffs),
        "boxes_mean_diff": float(np.mean([d["boxes_mean_diff"] for d in diffs])),
        "logits_max_diff": max(d["logits_max_diff"] for d in diffs),
        "logits_mean_diff": float(np.mean([d["logits_mean_diff"] for d in diffs])),
        "high_conf_boxes_max": max(d["high_conf_boxes_max"] for d in diffs),
        "high_conf_boxes_mean": float(np.mean([d["high_conf_boxes_mean"] for d in diffs])),
        "high_conf_logits_max": max(d["high_conf_logits_max"] for d in diffs),
        "high_conf_logits_mean": float(np.mean([d["high_conf_logits_mean"] for d in diffs])),
        "total_high_conf": sum(d["num_high_conf"] for d in diffs),
        "total_ref_detections": sum(d["ref_num_detections"] for d in diffs),
        "total_test_detections": sum(d["test_num_detections"] for d in diffs),
        "avg_class_match_rate": float(np.mean([d["class_match_rate"] for d in diffs])),
    }


def print_comparison(name: str, agg: dict[str, Any]) -> None:
    """Print comparison results."""
    print(f"  {name}:")
    print(f"    Boxes  - max: {agg['high_conf_boxes_max']:.6e}, mean: {agg['high_conf_boxes_mean']:.6e}")
    print(f"    Logits - max: {agg['high_conf_logits_max']:.6e}, mean: {agg['high_conf_logits_mean']:.6e}")
    print(f"    Detections: ref={agg['total_ref_detections']}, test={agg['total_test_detections']}")
    print(f"    Class match: {agg['avg_class_match_rate']:.2%}")

    if agg["high_conf_boxes_max"] < 1e-3 and agg["high_conf_logits_max"] < 1e-2:
        status = "PASS"
    elif agg["high_conf_boxes_max"] < 1e-1 and agg["high_conf_logits_max"] < 1.0:
        status = "OK"
    else:
        status = "WARNING"
    print(f"    Status: {status}")


def evaluate_all_backends(
    pt_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    trt_model: TensorRTInference,
    input_size: int,
    test_dir: Path,
    max_images: int = 50,
) -> dict[str, Any]:
    """Evaluate all backends on test dataset."""
    image_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    image_files = image_files[:max_images]

    onnx_vs_pt: list[dict[str, Any]] = []
    trt_vs_pt: list[dict[str, Any]] = []
    trt_vs_onnx: list[dict[str, Any]] = []

    for img_path in image_files:
        try:
            image = preprocess_image(img_path, input_size)

            # Run all backends
            pt_boxes, pt_logits = pytorch_inference(pt_model, image)
            onnx_boxes, onnx_logits = onnx_inference(onnx_session, image)
            trt_boxes, trt_logits = trt_model(image)

            # Compare pairs
            onnx_vs_pt.append(compare_outputs(pt_boxes, pt_logits, onnx_boxes, onnx_logits))
            trt_vs_pt.append(compare_outputs(pt_boxes, pt_logits, trt_boxes, trt_logits))
            trt_vs_onnx.append(compare_outputs(onnx_boxes, onnx_logits, trt_boxes, trt_logits))

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    return {
        "num_images": len(onnx_vs_pt),
        "onnx_vs_pytorch": aggregate_results(onnx_vs_pt) if onnx_vs_pt else {},
        "tensorrt_vs_pytorch": aggregate_results(trt_vs_pt) if trt_vs_pt else {},
        "tensorrt_vs_onnx": aggregate_results(trt_vs_onnx) if trt_vs_onnx else {},
    }


def main():
    print("=" * 60)
    print("TensorRT vs ONNX vs PyTorch Comparison")
    print("=" * 60)

    # Test license plate detection model
    print("\n[1/2] License Plate Detection Model")
    print("-" * 40)

    if LP_DETECTION_MODEL.exists() and LP_ONNX.exists() and LP_TRT.exists():
        print("Loading PyTorch model...")
        pt_model, input_size = load_pytorch_model(LP_DETECTION_MODEL)
        print("Loading ONNX model...")
        onnx_session = load_onnx_model(LP_ONNX)
        print("Loading TensorRT engine...")
        trt_model = TensorRTInference(LP_TRT)

        print(f"Testing on {LP_TEST_DIR.name}...")
        results = evaluate_all_backends(pt_model, onnx_session, trt_model, input_size, LP_TEST_DIR)

        print(f"\nResults ({results['num_images']} images, high-conf predictions):")
        print_comparison("ONNX vs PyTorch", results["onnx_vs_pytorch"])
        print_comparison("TensorRT vs PyTorch", results["tensorrt_vs_pytorch"])
        print_comparison("TensorRT vs ONNX", results["tensorrt_vs_onnx"])
    else:
        missing = [p for p in [LP_DETECTION_MODEL, LP_ONNX, LP_TRT] if not p.exists()]
        print(f"Skipping - missing: {[p.name for p in missing]}")

    # Test character detection model
    print("\n[2/2] Character Detection Model")
    print("-" * 40)

    if CHAR_DETECTION_MODEL.exists() and CHAR_ONNX.exists() and CHAR_TRT.exists():
        print("Loading PyTorch model...")
        pt_model, input_size = load_pytorch_model(CHAR_DETECTION_MODEL)
        print("Loading ONNX model...")
        onnx_session = load_onnx_model(CHAR_ONNX)
        print("Loading TensorRT engine...")
        trt_model = TensorRTInference(CHAR_TRT)

        print(f"Testing on {CHAR_TEST_DIR.name}...")
        results = evaluate_all_backends(pt_model, onnx_session, trt_model, input_size, CHAR_TEST_DIR)

        print(f"\nResults ({results['num_images']} images, high-conf predictions):")
        print_comparison("ONNX vs PyTorch", results["onnx_vs_pytorch"])
        print_comparison("TensorRT vs PyTorch", results["tensorrt_vs_pytorch"])
        print_comparison("TensorRT vs ONNX", results["tensorrt_vs_onnx"])
    else:
        missing = [p for p in [CHAR_DETECTION_MODEL, CHAR_ONNX, CHAR_TRT] if not p.exists()]
        print(f"Skipping - missing: {[p.name for p in missing]}")

    print("\n" + "=" * 60)
    print("Comparison complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
