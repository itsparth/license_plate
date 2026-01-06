"""Train YOLO11 for single-class character detection (localization only)."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def convert_to_single_class(src_dir: Path, dst_dir: Path) -> None:
    """Convert multi-class YOLO labels to single class (all chars -> class 0)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        src_split = src_dir / split
        dst_split = dst_dir / split
        
        if not src_split.exists():
            continue
            
        # Create output directories
        (dst_split / "images").mkdir(parents=True, exist_ok=True)
        (dst_split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Check if source has images/ subdir or flat structure
        if (src_split / "images").exists():
            img_src = src_split / "images"
            label_src = src_split / "labels"
        else:
            # Flat structure - images and labels in same folder
            img_src = src_split
            label_src = src_split
        
        # Copy images and convert labels
        img_count = 0
        for img_file in img_src.glob("*.jpg"):
            shutil.copy(img_file, dst_split / "images" / img_file.name)
            
            # Convert label - change all class IDs to 0
            label_file = label_src / f"{img_file.stem}.txt"
            if label_file.exists():
                new_lines = []
                for line in label_file.read_text().strip().split("\n"):
                    if line.strip():
                        parts = line.split()
                        # Replace class ID with 0, keep bbox coords
                        parts[0] = "0"
                        new_lines.append(" ".join(parts))
                
                (dst_split / "labels" / f"{img_file.stem}.txt").write_text("\n".join(new_lines))
            
            img_count += 1
        
        print(f"{split}: {img_count} images converted")


def create_yaml(data_dir: Path) -> Path:
    """Create dataset YAML for single-class detection."""
    yaml_path = data_dir / "data.yaml"
    yaml_content = f"""path: {data_dir}
train: train/images
val: valid/images
test: test/images

names:
  0: char
"""
    yaml_path.write_text(yaml_content)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11 single-class char detection")
    parser.add_argument("-d", "--data", type=str, default="output/training_data",
                        help="Source multi-class dataset directory")
    parser.add_argument("-o", "--output", type=str, default="output/char_detection_data",
                        help="Output single-class dataset directory")
    parser.add_argument("-m", "--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLO11 model size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=256, help="Image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--skip-convert", action="store_true", help="Skip dataset conversion")
    args = parser.parse_args()

    src_dir = Path(args.data)
    dst_dir = Path(args.output)
    
    # Convert dataset to single class
    if not args.skip_convert:
        print("Converting multi-class dataset to single class...")
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        convert_to_single_class(src_dir, dst_dir)
    
    # Create YAML
    yaml_path = create_yaml(dst_dir)
    print(f"Dataset YAML: {yaml_path}")
    
    # Load model
    model_name = f"yolo11{args.model}.pt"
    model = YOLO(model_name)
    print(f"Loaded {model_name}")
    
    # Train
    output_dir = Path("output/yolo11_char_detection_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        project=str(output_dir),
        name=f"yolo11{args.model}_char_detect",
        # Augmentation - no flipping for text
        fliplr=0.0,
        flipud=0.0,
        # Standard augmentations
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Training params
        optimizer="auto",
        lr0=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # Validation
        val=True,
        plots=True,
        save=True,
    )
    
    # Validate best model
    best_weights = output_dir / f"yolo11{args.model}_char_detect" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nValidating best model: {best_weights}")
        model = YOLO(str(best_weights))
        model.val(data=str(yaml_path), split="test")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
