#!/usr/bin/env python3
"""Train RF-DETR Medium for license plate character detection.

Uses RFDETRMedium at 384x384 resolution for:
- Stronger encoder/decoder architecture than Nano
- Same resolution for compatibility with small plate crops
- Max accuracy settings (EMA, multi-scale, etc.)
- PyTorch inference (not DeepStream optimized)
"""

import argparse
import os
from pathlib import Path

from rfdetr.detr import RFDETRMedium

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
DATASET_DIR = OUTPUT_DIR / "training_data"
TRAINING_DIR = OUTPUT_DIR / "rfdetr_medium_training"
PRETRAINED_DIR = OUTPUT_DIR / "pretrained"


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint to resume from."""
    checkpoint = output_dir / "checkpoint.pth"
    if checkpoint.exists():
        return checkpoint
    return None


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR Medium model (384x384)")
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
        default=TRAINING_DIR,
        help=f"Output directory for checkpoints (default: {TRAINING_DIR})",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs (default: 150 for better convergence)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16, Medium model needs more VRAM)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, effective batch = 32)",
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
        help="Resume from checkpoint (auto-detects if not specified)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't auto-resume",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=384,
        help="Training resolution (default: 384, must be divisible by 56)",
    )
    args = parser.parse_args()

    # Verify dataset structure
    for split in ["train", "valid"]:
        ann_file = args.dataset / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing: {ann_file}")

    # Validate resolution (must be divisible by patch_size=16)
    if args.resolution % 16 != 0:
        raise ValueError(f"Resolution must be divisible by 16, got {args.resolution}")

    # Auto-resume from latest checkpoint if available
    resume_path = args.resume
    if resume_path is None and not args.no_resume:
        resume_path = find_latest_checkpoint(args.output)
        if resume_path:
            print(f"Auto-resuming from: {resume_path}")

    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")

    # Ensure pretrained weights download to output/pretrained/
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(PRETRAINED_DIR)

    # RFDETRMedium with custom resolution
    # Medium has: 4 decoder layers, stronger architecture than Nano
    model = RFDETRMedium(resolution=args.resolution)

    # Restore working directory
    os.chdir(original_cwd)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Train with settings optimized for max accuracy
    model.train(
        dataset_dir=str(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=1.5e-4,  # Slightly higher LR for encoder fine-tuning
        output_dir=str(args.output),
        # Multi-scale training for better small object detection
        multi_scale=True,
        expanded_scales=True,  # More aggressive scale variation
        # EMA for smoother, more accurate final weights
        use_ema=True,
        ema_decay=0.9997,  # Higher decay for more stable EMA
        # Early stopping with patience
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_use_ema=True,  # Use EMA weights for early stopping metric
        # Longer warmup for stability
        warmup_epochs=3.0,
        # Logging
        tensorboard=True,
        checkpoint_interval=5,
        # More workers for faster data loading
        num_workers=8,
        # Resume if checkpoint exists
        resume=str(resume_path) if resume_path else None,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model (EMA): {args.output / 'checkpoint_best_ema.pth'}")
    print(f"Best model (Regular): {args.output / 'checkpoint_best_regular.pth'}")
    print(f"Best model (Total): {args.output / 'checkpoint_best_total.pth'}")
    print(f"TensorBoard: tensorboard --logdir {args.output}")


if __name__ == "__main__":
    main()
