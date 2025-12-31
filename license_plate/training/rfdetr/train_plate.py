#!/usr/bin/env python3
"""Train RF-DETR Nano for license plate detection.

Uses the combined dataset from Roboflow + Detic-labeled images.
Model: RFDETRNano (384x384) optimized for edge deployment.
"""

import argparse
import os
from pathlib import Path

from rfdetr.detr import RFDETRNano

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"
DATASET_DIR = OUTPUT_DIR / "lp_detection_combined"
TRAINING_DIR = OUTPUT_DIR / "lp_detection_training"
PRETRAINED_DIR = OUTPUT_DIR / "pretrained"


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint to resume from."""
    checkpoint = output_dir / "checkpoint.pth"
    if checkpoint.exists():
        return checkpoint
    return None


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR Nano for license plate detection")
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
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't auto-resume",
    )
    args = parser.parse_args()

    # Verify dataset
    for split in ["train", "valid"]:
        ann_file = args.dataset / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing: {ann_file}")

    # Auto-resume
    resume_path = args.resume
    if resume_path is None and not args.no_resume:
        resume_path = find_latest_checkpoint(args.output)
        if resume_path:
            print(f"Auto-resuming from: {resume_path}")

    print("=" * 60)
    print("Training RF-DETR Nano for License Plate Detection")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print()

    # Setup pretrained weights directory
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    
    original_cwd = os.getcwd()
    os.chdir(PRETRAINED_DIR)

    # Initialize model
    model = RFDETRNano()

    os.chdir(original_cwd)

    # Train
    model.train(
        dataset_dir=str(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(args.output),
        early_stopping=True,
        early_stopping_patience=15,
        tensorboard=True,
        checkpoint_interval=5,
        multi_scale=True,  # License plates vary in size
        num_workers=8,
        use_ema=True,
        resume=str(resume_path) if resume_path else None,
    )

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model: {args.output / 'checkpoint_best_total.pth'}")
    print(f"EMA model: {args.output / 'checkpoint_best_ema.pth'}")
    print(f"TensorBoard: tensorboard --logdir {args.output}")


if __name__ == "__main__":
    main()
