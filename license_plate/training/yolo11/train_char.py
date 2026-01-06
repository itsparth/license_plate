#!/usr/bin/env python3
"""Train YOLO11 for license plate character detection.

Uses synthetic data with 36 classes (A-Z, 0-9).
Optimized for cropped license plate images (256x256).
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import yaml
from ultralytics import YOLO  # type: ignore[attr-defined]

OUTPUT_DIR = Path("/home/openwebui/license_plate/output")
DATASET_DIR = OUTPUT_DIR / "training_data"
TRAINING_DIR = OUTPUT_DIR / "yolo11_char_training"

ModelSize = Literal["n", "s", "m", "l", "x"]

# 36 character classes
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def create_dataset_yaml(dataset_dir: Path, output_dir: Path) -> Path:
    """Create YOLO-format dataset.yaml from COCO annotations."""
    yaml_path = output_dir / "dataset.yaml"
    
    # Read class names from COCO annotations
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    with open(train_ann) as f:
        coco = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    names = [categories[i] for i in sorted(categories.keys())]
    
    config = {
        "path": str(dataset_dir.absolute()),
        "train": "train",
        "val": "valid",
        "test": "test",
        "names": {i: name for i, name in enumerate(names)},
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config: {yaml_path}")
    print(f"Classes ({len(names)}): {names}")
    return yaml_path


def find_latest_checkpoint(output_dir: Path, model_size: str) -> Path | None:
    """Find the latest checkpoint to resume from."""
    runs_dir = output_dir / f"yolo11{model_size}_char"
    weights_dir = runs_dir / "weights"
    
    if weights_dir.exists():
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            return last_pt
    return None


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 for character detection")
    parser.add_argument(
        "-d", "--dataset",
        type=Path,
        default=DATASET_DIR,
        help=f"Dataset directory (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=TRAINING_DIR,
        help=f"Output directory (default: {TRAINING_DIR})",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32, higher for smaller images)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="Image size (default: 256 for cropped plates)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: 0, 1, cpu, etc. (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience - epochs without improvement (default: 20)",
    )
    args = parser.parse_args()

    # Verify dataset
    for split in ["train", "valid"]:
        ann_file = args.dataset / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing: {ann_file}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Create dataset yaml
    yaml_path = create_dataset_yaml(args.dataset, args.output)

    # Model setup
    model_name = f"yolo11{args.model}.pt"
    project_name = f"yolo11{args.model}_char"
    
    print("=" * 60)
    print(f"Training YOLO11-{args.model.upper()} for Character Detection")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output / project_name}")
    print(f"Batch: {args.batch_size}")
    print(f"Image Size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Early Stopping: patience={args.patience}")
    print(f"Device: {args.device}")
    print()

    # Check for resume
    resume_from: bool | str = False
    if args.resume:
        checkpoint = find_latest_checkpoint(args.output, args.model)
        if checkpoint:
            resume_from = str(checkpoint)
            print(f"Resuming from: {resume_from}")
        else:
            print("No checkpoint found, starting fresh")
    
    # Load model
    if resume_from:
        model = YOLO(resume_from)
    else:
        model = YOLO(model_name)

    # Train with character-specific augmentations
    # Key differences from plate detection:
    # - fliplr=0.0: Text should NOT be horizontally flipped
    # - flipud=0.0: Text should NOT be vertically flipped  
    # - Lower perspective/shear: Characters need to stay readable
    # - Lower mosaic: Don't want to scramble character positions too much
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(args.output),
        name=project_name,
        exist_ok=True,
        resume=bool(resume_from),
        # Augmentation settings for character detection
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,  # Less rotation for characters
        translate=0.1,
        scale=0.3,  # Less scale variation
        shear=2.0,  # Less shear
        perspective=0.0001,  # Much less perspective
        flipud=0.0,  # Never flip vertically
        fliplr=0.0,  # Never flip horizontally (text would be mirrored!)
        mosaic=0.5,  # Less mosaic
        mixup=0.1,  # Less mixup
        copy_paste=0.1,
        erasing=0.3,
        # Save settings
        save=True,
        save_period=10,
        plots=True,
    )

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print final metrics
    best_weights = args.output / project_name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"Best weights: {best_weights}")
    
    return results


if __name__ == "__main__":
    main()
