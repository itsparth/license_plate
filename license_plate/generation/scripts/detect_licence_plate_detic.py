import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import supervision as sv
import torch

VOCAB = "lvis"
CONFIDENCE_THRESHOLD = 0.3
HOME = os.path.expanduser("~")
DETIC_DIR = os.path.join(HOME, ".cache/detic")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_cfg(args):
    from centernet.config import add_centernet_config  # type: ignore
    from detic.config import add_detic_config  # type: ignore
    from detectron2.config import get_cfg

    detic_path = os.path.join(DETIC_DIR, "Detic")
    os.chdir(detic_path)
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu" if args.cpu else "cuda"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False
    cfg.freeze()
    return cfg


def load_detic_model():
    mp.set_start_method("spawn", force=True)
    from detectron2.utils.logger import setup_logger

    setup_logger(name="fvcore")

    args = argparse.Namespace()
    args.confidence_threshold = CONFIDENCE_THRESHOLD
    args.vocabulary = VOCAB
    args.opts = []
    args.config_file = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    args.cpu = False if torch.cuda.is_available() else True
    args.opts.append("MODEL.WEIGHTS")
    args.opts.append("models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    args.output = None
    args.webcam = None
    args.video_input = None
    args.pred_all_class = True

    detic_path = os.path.join(DETIC_DIR, "Detic")
    sys.path.insert(0, f"{detic_path}/third_party/CenterNet2/")
    sys.path.insert(0, detic_path)
    curr_dir = os.getcwd()

    cfg = setup_cfg(args)

    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("lvis_v1_val")
    classifier = f"{detic_path}/datasets/metadata/lvis_v1_clip_a+cname.npy"
    os.chdir(curr_dir)

    from detic.modeling.utils import reset_cls_test  # type: ignore

    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)

    return predictor, metadata


def check_dependencies():
    import importlib.util

    original_dir = os.getcwd()
    os.makedirs(DETIC_DIR, exist_ok=True)
    os.chdir(DETIC_DIR)

    if importlib.util.find_spec("detectron2") is None:
        print("📦 Installing detectron2...")
        subprocess.run(["uv", "pip", "install", "torch", "torchvision"], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "git+https://github.com/facebookresearch/detectron2.git",
                "--no-build-isolation",
            ],
            check=True,
        )

    if importlib.util.find_spec("clip") is None:
        print("📦 Installing CLIP...")
        subprocess.run(
            ["uv", "pip", "install", "git+https://github.com/openai/CLIP.git"],
            check=True,
        )

    detic_path = os.path.join(DETIC_DIR, "Detic")
    model_path = os.path.join(
        detic_path, "models", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    )

    # Check if Detic repo exists and has required files
    needs_install = not os.path.isdir(detic_path) or not os.path.isfile(
        os.path.join(
            detic_path,
            "configs",
            "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        )
    )

    if needs_install:
        print("📦 Cloning Detic repository...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/Detic.git",
                "--recurse-submodules",
            ],
            check=True,
        )
        os.chdir(detic_path)
        print("📦 Installing Detic requirements...")
        subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], check=True)

    # Check if model file exists and is complete (should be ~670MB)
    needs_download = (
        not os.path.isfile(model_path) or os.path.getsize(model_path) < 700_000_000
    )

    if needs_download:
        models_dir = os.path.join(detic_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Remove partial download if exists
        if os.path.isfile(model_path):
            print("⚠ Removing incomplete model file...")
            os.remove(model_path)

        print("⬇ Downloading model weights (670MB, this may take a while)...")
        model_url = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        subprocess.run(["wget", "-c", model_url, "-O", model_path], check=True)

        if os.path.getsize(model_path) < 700_000_000:
            raise RuntimeError(
                f"Model download incomplete. Expected ~670MB, got {os.path.getsize(model_path) / 1e6:.1f}MB"
            )

        print("✓ Model downloaded successfully")

    os.chdir(original_dir)


check_dependencies()


def detect(image_path: str) -> sv.Detections:
    import cv2
    from detectron2.data.detection_utils import read_image

    if not hasattr(detect, "predictor"):
        detect.predictor, detect.metadata = load_detic_model()

    # Read image and convert to grayscale, then back to 3-channel BGR
    img = read_image(image_path, format="BGR")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    predictions = detect.predictor(img_gray_3ch)

    pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    pred_classes = predictions["instances"].pred_classes.cpu().numpy()
    pred_scores = predictions["instances"].scores.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()

    if len(pred_classes) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=pred_boxes,
        mask=pred_masks,
        class_id=pred_classes,
        confidence=pred_scores,
    )


def detect_license_plate(
    image_path: str, confidence_threshold: float = 0.3
) -> sv.Detections:
    """Detect license plates in an image.

    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence threshold for detections (default: 0.3)

    Returns:
        sv.Detections containing only license plate detections
    """
    if not hasattr(detect, "predictor"):
        detect.predictor, detect.metadata = load_detic_model()

    # Get all detections
    detections = detect(image_path)

    # Find license_plate class ID in LVIS metadata
    class_names = detect.metadata.thing_classes
    license_plate_ids = [
        i
        for i, name in enumerate(class_names)
        if "license" in name.lower() and "plate" in name.lower()
    ]

    if len(detections) == 0 or len(license_plate_ids) == 0:
        return sv.Detections.empty()

    if detections.class_id is None or detections.confidence is None:
        return sv.Detections.empty()

    # Filter for license plates only
    mask = np.isin(detections.class_id, license_plate_ids) & (
        detections.confidence >= confidence_threshold
    )

    return cast(sv.Detections, detections[mask])


def detect_all_lvis(
    image_path: str, confidence_threshold: float = 0.3
) -> sv.Detections:
    """Detect all LVIS classes in an image.

    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence threshold for detections (default: 0.3)

    Returns:
        sv.Detections containing all detections
    """
    if not hasattr(detect, "predictor"):
        detect.predictor, detect.metadata = load_detic_model()

    # Get all detections
    detections = detect(image_path)

    if len(detections) == 0:
        return sv.Detections.empty()

    if detections.confidence is None:
        return sv.Detections.empty()

    # Filter by confidence
    mask = detections.confidence >= confidence_threshold

    return cast(sv.Detections, detections[mask])


def detect_license_plates_batch(
    image_paths: list[str],
    confidence_threshold: float = 0.3,
    batch_size: int = 2,
) -> list[sv.Detections]:
    """Fast batch detection of license plates. Optimized for bulk processing.

    ~10 FPS on RTX 4000 Ada. 10K images in ~16-17 minutes.

    Args:
        image_paths: List of paths to image files
        confidence_threshold: Minimum confidence threshold (default: 0.3)
        batch_size: Batch size for inference (default: 2, optimal for most GPUs)

    Returns:
        List of sv.Detections, one per input image (bbox only, no masks)
    """
    if not hasattr(detect, "predictor"):
        detect.predictor, detect.metadata = load_detic_model()

    predictor = detect.predictor
    metadata = detect.metadata
    model = predictor.model
    model.eval()

    # Find license plate class IDs
    class_names = metadata.thing_classes
    license_plate_ids = [
        i
        for i, name in enumerate(class_names)
        if "license" in name.lower() and "plate" in name.lower()
    ]

    all_results: list[sv.Detections] = []

    with torch.no_grad():
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            # Prepare batch inputs
            inputs = []
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    inputs.append(None)
                    continue
                height, width = img.shape[:2]
                img_tensor = predictor.aug.get_transform(img).apply_image(img)
                img_tensor = torch.as_tensor(
                    img_tensor.astype("float32").transpose(2, 0, 1)
                )
                inputs.append({"image": img_tensor, "height": height, "width": width})

            # Filter out failed loads
            valid_indices = [i for i, inp in enumerate(inputs) if inp is not None]
            valid_inputs = [inputs[i] for i in valid_indices]

            if not valid_inputs:
                all_results.extend([sv.Detections.empty()] * len(batch_paths))
                continue

            # Run batch inference
            outputs = model(valid_inputs)

            # Process results
            result_idx = 0
            for i in range(len(batch_paths)):
                if i not in valid_indices:
                    all_results.append(sv.Detections.empty())
                    continue

                instances = outputs[result_idx]["instances"]
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()

                # Filter for license plates
                mask = np.isin(classes, license_plate_ids) & (
                    scores >= confidence_threshold
                )

                if not mask.any():
                    all_results.append(sv.Detections.empty())
                else:
                    all_results.append(
                        sv.Detections(
                            xyxy=boxes[mask],
                            confidence=scores[mask],
                            class_id=classes[mask],
                        )
                    )
                result_idx += 1

    return all_results


def detect_license_plates_fast(
    image_paths: list[str],
    confidence_threshold: float = 0.3,
    batch_size: int = 2,
    show_progress: bool = True,
) -> list[np.ndarray]:
    """Fastest license plate detection - returns only bboxes as numpy arrays.

    ~10 FPS on RTX 4000 Ada. 10K images in ~16-17 minutes.

    Args:
        image_paths: List of paths to image files
        confidence_threshold: Minimum confidence threshold (default: 0.3)
        batch_size: Batch size for inference (default: 2)
        show_progress: Print progress every 100 images

    Returns:
        List of numpy arrays with shape (N, 4) for xyxy bboxes per image
    """
    if not hasattr(detect, "predictor"):
        detect.predictor, detect.metadata = load_detic_model()

    predictor = detect.predictor
    metadata = detect.metadata
    model = predictor.model
    model.eval()

    # Find license plate class IDs
    class_names = metadata.thing_classes
    license_plate_ids = set(
        i
        for i, name in enumerate(class_names)
        if "license" in name.lower() and "plate" in name.lower()
    )

    all_boxes: list[np.ndarray] = []
    total = len(image_paths)

    with torch.no_grad():
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_paths = image_paths[batch_start:batch_end]

            if show_progress and batch_start % 100 == 0:
                print(f"Processing {batch_start}/{total}...")

            # Prepare batch
            inputs = []
            valid_mask = []
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    valid_mask.append(False)
                    continue
                valid_mask.append(True)
                h, w = img.shape[:2]
                t = predictor.aug.get_transform(img).apply_image(img)
                inputs.append({
                    "image": torch.as_tensor(t.astype("float32").transpose(2, 0, 1)),
                    "height": h,
                    "width": w,
                })

            if not inputs:
                all_boxes.extend([np.zeros((0, 4))] * len(batch_paths))
                continue

            # Inference
            outputs = model(inputs)

            # Extract results
            out_idx = 0
            for is_valid in valid_mask:
                if not is_valid:
                    all_boxes.append(np.zeros((0, 4)))
                    continue

                inst = outputs[out_idx]["instances"]
                boxes = inst.pred_boxes.tensor.cpu().numpy()
                scores = inst.scores.cpu().numpy()
                classes = inst.pred_classes.cpu().numpy()

                if len(classes) == 0:
                    all_boxes.append(np.zeros((0, 4)))
                else:
                    class_mask = np.array([c in license_plate_ids for c in classes], dtype=bool)
                    score_mask = scores >= confidence_threshold
                    mask = class_mask & score_mask
                    all_boxes.append(boxes[mask] if mask.any() else np.zeros((0, 4)))
                out_idx += 1

    if show_progress:
        print(f"Done. Processed {total} images.")

    return all_boxes


if __name__ == "__main__":
    # Input and output directories
    vehicles_dir = Path(__file__).parent.parent / "assets/vehicles"
    output_dir = Path(__file__).parent.parent / "assets/vehicle_plates"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all image files
    image_files = list(vehicles_dir.glob("*.jpg")) + list(vehicles_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {vehicles_dir}")

    processed = 0
    skipped = 0

    for image_path in sorted(image_files):
        print(f"\nProcessing: {image_path.name}")

        try:
            # Detect license plates
            detections = detect_license_plate(str(image_path))

            if len(detections) == 0:
                print("  No license plates detected, skipping...")
                skipped += 1
                continue

            # Find largest license plate by area
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                detections.xyxy[:, 3] - detections.xyxy[:, 1]
            )
            largest_idx = np.argmax(areas)
            largest_bbox = detections.xyxy[largest_idx]
            largest_conf = (
                detections.confidence[largest_idx]
                if detections.confidence is not None
                else 1.0
            )

            print(
                f"  Found {len(detections)} plate(s), largest: {largest_conf:.2%} confidence"
            )

            # Load image for annotation and dimensions
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]

            # Create single detection object for the largest plate
            largest_detection = sv.Detections(
                xyxy=np.array([largest_bbox]),
                confidence=np.array([largest_conf]),
                class_id=np.array([0]),
            )

            # 1. Save Label (YOLO format)
            # class_id x_center y_center width height (normalized)
            x_min, y_min, x_max, y_max = largest_bbox

            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            label_path = output_dir / f"{image_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                )

            # 2. Save Annotated Image
            box_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.GREEN)

            annotated_image = box_annotator.annotate(
                scene=image.copy(), detections=largest_detection
            )

            output_image_path = output_dir / image_path.name
            cv2.imwrite(str(output_image_path), annotated_image)

            print(f"  Saved label to {label_path.name}")
            print(f"  Saved annotated image to {output_image_path.name}")

            processed += 1

        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            skipped += 1

    print("\nProcessing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {output_dir}")
