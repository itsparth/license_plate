#!/usr/bin/env python3
"""Train RF-DETR for license plate character detection.

Uses RFDETRNano (384x384) optimized for deepstream export.

Note: RF-DETR internally applies RandomHorizontalFlip during training.
For license plate text (which shouldn't be flipped), consider:
1. Training with flipped data anyway (model learns both orientations)
2. Forking RF-DETR to modify rfdetr/datasets/coco.py make_coco_transforms()
"""

import argparse
from pathlib import Path

from rfdetr.detr import RFDETRNano

DATASET_DIR = Path(__file__).parent.parent.parent.parent / "output" / "training_data"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "rfdetr_training"


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR Nano model (384x384)")
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help=f"Dataset directory with train/valid splits (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for checkpoints (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch = batch_size * grad_accum)",
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
    args = parser.parse_args()

    # Verify dataset structure
    for split in ["train", "valid"]:
        ann_file = args.dataset / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing: {ann_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")

    # RFDETRNano: 384x384, smallest model for edge deployment
    model = RFDETRNano()

    # Train with built-in augmentations (RandomResize, RandomSizeCrop, Normalize)
    # Note: multi_scale=False to keep consistent 384x384 resolution
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
        checkpoint_interval=10,
        multi_scale=False,  # Disable multi-scale for consistent 384x384
        num_workers=4,
        resume=str(args.resume) if args.resume else None,
    )

    print("Training complete!")
    print(f"Best model: {args.output / 'checkpoint_best_total.pth'}")


if __name__ == "__main__":
    main()
