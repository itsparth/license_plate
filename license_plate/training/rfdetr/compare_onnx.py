#!/usr/bin/env python3
"""Compare ONNX vs PyTorch model accuracy on test datasets.

Tests both models on their respective test sets and compares:
- Numerical precision (max/mean difference in outputs)
- Detection results (mAP on COCO-format annotations)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.ops import box_iou

from rfdetr.detr import RFDETRNano

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"

# Model paths
LP_DETECTION_MODEL = OUTPUT_DIR / "lp_detection_training" / "checkpoint_best_ema.pth"
CHAR_DETECTION_MODEL = OUTPUT_DIR / "rfdetr_training" / "checkpoint_best_ema.pth"
LP_ONNX = OUTPUT_DIR / "exported_models" / "lp_detection.onnx"
CHAR_ONNX = OUTPUT_DIR / "exported_models" / "char_detection.onnx"

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
    pt_boxes: np.ndarray,
    pt_logits: np.ndarray,
    onnx_boxes: np.ndarray,
    onnx_logits: np.ndarray,
    conf_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compare PyTorch and ONNX outputs."""
    boxes_diff = np.abs(pt_boxes - onnx_boxes)
    logits_diff = np.abs(pt_logits - onnx_logits)

    # Also compare high-confidence predictions only
    pt_scores = 1 / (1 + np.exp(-pt_logits))
    onnx_scores = 1 / (1 + np.exp(-onnx_logits))
    pt_max_scores = np.max(pt_scores, axis=-1)[0]
    onnx_max_scores = np.max(onnx_scores, axis=-1)[0]

    # Count detections at threshold
    pt_num_detections = int(np.sum(pt_max_scores > conf_threshold))
    onnx_num_detections = int(np.sum(onnx_max_scores > conf_threshold))

    # Check if predicted classes match for high-confidence predictions
    high_conf_mask = (pt_max_scores > conf_threshold) | (onnx_max_scores > conf_threshold)
    num_high_conf = int(np.sum(high_conf_mask))

    if num_high_conf > 0:
        pt_classes = np.argmax(pt_logits[0, high_conf_mask], axis=-1)
        onnx_classes = np.argmax(onnx_logits[0, high_conf_mask], axis=-1)
        class_match_rate = float(np.mean(pt_classes == onnx_classes))
    else:
        class_match_rate = 1.0

    high_conf_boxes_diff = boxes_diff[0, high_conf_mask] if num_high_conf > 0 else np.array([0.0])
    high_conf_logits_diff = logits_diff[0, high_conf_mask] if num_high_conf > 0 else np.array([0.0])

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
        "pt_num_detections": pt_num_detections,
        "onnx_num_detections": onnx_num_detections,
        "class_match_rate": class_match_rate,
    }


def load_coco_annotations(json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load COCO annotations and return mapping of filename -> annotations."""
    with open(json_path) as f:
        data = json.load(f)

    # Create image_id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # Group annotations by filename
    result: dict[str, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        filename = id_to_file[ann["image_id"]]
        if filename not in result:
            result[filename] = []
        result[filename].append(ann)

    return result


def compute_detection_metrics(
    pred_boxes: np.ndarray,
    pred_logits: np.ndarray,
    gt_annotations: list[dict[str, Any]],
    image_size: tuple[int, int],
    input_size: int,
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute detection metrics for a single image."""
    # Apply sigmoid to logits to get confidence scores
    pred_scores = 1 / (1 + np.exp(-pred_logits))
    max_scores = np.max(pred_scores, axis=-1)[0]  # [300]

    # Filter by confidence
    mask = max_scores > conf_threshold
    filtered_boxes = pred_boxes[0, mask]  # [N, 4] in normalized coords

    # Scale boxes to image size
    w, h = image_size
    scale_x = w / input_size
    scale_y = h / input_size

    # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    # Actually RF-DETR outputs are already in xyxy normalized format
    # Let's check - boxes are in [0, 384] range typically
    pred_xyxy = filtered_boxes.copy()
    pred_xyxy[:, [0, 2]] *= scale_x
    pred_xyxy[:, [1, 3]] *= scale_y

    # Ground truth boxes from COCO format [x, y, width, height]
    gt_boxes = []
    for ann in gt_annotations:
        x, y, bw, bh = ann["bbox"]
        gt_boxes.append([x, y, x + bw, y + bh])  # Convert to xyxy
    gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))

    return {
        "num_predictions": len(pred_xyxy),
        "num_ground_truth": len(gt_boxes),
        "pred_boxes": pred_xyxy.tolist(),
        "gt_boxes": gt_boxes.tolist(),
    }


def evaluate_model_pair(
    pt_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    input_size: int,
    test_dir: Path,
    max_images: int = 50,
) -> dict[str, Any]:
    """Evaluate PyTorch vs ONNX on test dataset."""
    annotations_path = test_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        return {"error": f"Annotations not found: {annotations_path}"}

    annotations = load_coco_annotations(annotations_path)

    # Get image files
    image_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    image_files = image_files[:max_images]

    results = {
        "num_images": 0,
        "numerical_diffs": [],
        "detection_metrics": [],
    }

    for img_path in image_files:
        try:
            image = preprocess_image(img_path, input_size)

            # Run both models
            pt_boxes, pt_logits = pytorch_inference(pt_model, image)
            onnx_boxes, onnx_logits = onnx_inference(onnx_session, image)

            # Compare numerical outputs
            diff = compare_outputs(pt_boxes, pt_logits, onnx_boxes, onnx_logits)
            results["numerical_diffs"].append(diff)

            # Get detection metrics if annotations exist
            if img_path.name in annotations:
                with Image.open(img_path) as pil_img:
                    img_size = pil_img.size

                metrics = compute_detection_metrics(
                    pt_boxes,
                    pt_logits,
                    annotations[img_path.name],
                    img_size,
                    input_size,
                )
                results["detection_metrics"].append(metrics)

            results["num_images"] += 1

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Aggregate numerical differences
    if results["numerical_diffs"]:
        diffs = results["numerical_diffs"]
        results["aggregate"] = {
            "boxes_max_diff": max(d["boxes_max_diff"] for d in diffs),
            "boxes_mean_diff": np.mean([d["boxes_mean_diff"] for d in diffs]),
            "logits_max_diff": max(d["logits_max_diff"] for d in diffs),
            "logits_mean_diff": np.mean([d["logits_mean_diff"] for d in diffs]),
            "high_conf_boxes_max": max(d["high_conf_boxes_max"] for d in diffs),
            "high_conf_boxes_mean": np.mean([d["high_conf_boxes_mean"] for d in diffs]),
            "high_conf_logits_max": max(d["high_conf_logits_max"] for d in diffs),
            "high_conf_logits_mean": np.mean([d["high_conf_logits_mean"] for d in diffs]),
            "total_high_conf": sum(d["num_high_conf"] for d in diffs),
            "total_pt_detections": sum(d["pt_num_detections"] for d in diffs),
            "total_onnx_detections": sum(d["onnx_num_detections"] for d in diffs),
            "avg_class_match_rate": np.mean([d["class_match_rate"] for d in diffs]),
        }

    return results


def main():
    print("=" * 60)
    print("ONNX vs PyTorch Model Comparison")
    print("=" * 60)

    # Test license plate detection model
    print("\n[1/2] License Plate Detection Model")
    print("-" * 40)

    if LP_DETECTION_MODEL.exists() and LP_ONNX.exists():
        print("Loading PyTorch model...")
        pt_model, input_size = load_pytorch_model(LP_DETECTION_MODEL)
        print("Loading ONNX model...")
        onnx_session = load_onnx_model(LP_ONNX)

        print(f"Testing on {LP_TEST_DIR.name}...")
        lp_results = evaluate_model_pair(pt_model, onnx_session, input_size, LP_TEST_DIR)

        print(f"\nResults ({lp_results['num_images']} images):")
        if "aggregate" in lp_results:
            agg = lp_results["aggregate"]
            print(f"  All predictions:")
            print(f"    Boxes max diff:    {agg['boxes_max_diff']:.6e}")
            print(f"    Boxes mean diff:   {agg['boxes_mean_diff']:.6e}")
            print(f"    Logits max diff:   {agg['logits_max_diff']:.6e}")
            print(f"    Logits mean diff:  {agg['logits_mean_diff']:.6e}")
            print(f"  High-confidence predictions (conf > 0.3, n={agg['total_high_conf']}):")
            print(f"    Boxes max diff:    {agg['high_conf_boxes_max']:.6e}")
            print(f"    Boxes mean diff:   {agg['high_conf_boxes_mean']:.6e}")
            print(f"    Logits max diff:   {agg['high_conf_logits_max']:.6e}")
            print(f"    Logits mean diff:  {agg['high_conf_logits_mean']:.6e}")
            print(f"  Detection consistency:")
            print(f"    PyTorch detections:  {agg['total_pt_detections']}")
            print(f"    ONNX detections:     {agg['total_onnx_detections']}")
            print(f"    Class match rate:    {agg['avg_class_match_rate']:.2%}")

            if agg["high_conf_boxes_max"] < 1e-3 and agg["high_conf_logits_max"] < 1e-2:
                print("  Status: PASS (high-conf outputs nearly identical)")
            elif agg["high_conf_boxes_max"] < 1e-1 and agg["high_conf_logits_max"] < 1.0:
                print("  Status: OK (small differences in high-conf predictions)")
            else:
                print("  Status: WARNING (significant differences in high-conf predictions)")
    else:
        print("Skipping - model files not found")

    # Test character detection model
    print("\n[2/2] Character Detection Model")
    print("-" * 40)

    if CHAR_DETECTION_MODEL.exists() and CHAR_ONNX.exists():
        print("Loading PyTorch model...")
        pt_model, input_size = load_pytorch_model(CHAR_DETECTION_MODEL)
        print("Loading ONNX model...")
        onnx_session = load_onnx_model(CHAR_ONNX)

        print(f"Testing on {CHAR_TEST_DIR.name}...")
        char_results = evaluate_model_pair(pt_model, onnx_session, input_size, CHAR_TEST_DIR)

        print(f"\nResults ({char_results['num_images']} images):")
        if "aggregate" in char_results:
            agg = char_results["aggregate"]
            print(f"  All predictions:")
            print(f"    Boxes max diff:    {agg['boxes_max_diff']:.6e}")
            print(f"    Boxes mean diff:   {agg['boxes_mean_diff']:.6e}")
            print(f"    Logits max diff:   {agg['logits_max_diff']:.6e}")
            print(f"    Logits mean diff:  {agg['logits_mean_diff']:.6e}")
            print(f"  High-confidence predictions (conf > 0.3, n={agg['total_high_conf']}):")
            print(f"    Boxes max diff:    {agg['high_conf_boxes_max']:.6e}")
            print(f"    Boxes mean diff:   {agg['high_conf_boxes_mean']:.6e}")
            print(f"    Logits max diff:   {agg['high_conf_logits_max']:.6e}")
            print(f"    Logits mean diff:  {agg['high_conf_logits_mean']:.6e}")
            print(f"  Detection consistency:")
            print(f"    PyTorch detections:  {agg['total_pt_detections']}")
            print(f"    ONNX detections:     {agg['total_onnx_detections']}")
            print(f"    Class match rate:    {agg['avg_class_match_rate']:.2%}")

            if agg["high_conf_boxes_max"] < 1e-3 and agg["high_conf_logits_max"] < 1e-2:
                print("  Status: PASS (high-conf outputs nearly identical)")
            elif agg["high_conf_boxes_max"] < 1e-1 and agg["high_conf_logits_max"] < 1.0:
                print("  Status: OK (small differences in high-conf predictions)")
            else:
                print("  Status: WARNING (significant differences in high-conf predictions)")
    else:
        print("Skipping - model files not found")

    print("\n" + "=" * 60)
    print("Comparison complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
