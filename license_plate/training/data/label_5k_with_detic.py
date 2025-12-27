#!/usr/bin/env python3
"""Label 5k vehicle images with Detic for license plate detection.

Uses the fast batch detection to generate COCO format annotations.
Estimated time: ~8-10 minutes for 5k images on RTX 4000 Ada.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from license_plate.generation.scripts.detect_licence_plate_detic import (
    detect_license_plates_fast,
)

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
IMAGES_DIR = OUTPUT_DIR / "lp_dataset_5k"
LABELED_DIR = OUTPUT_DIR / "lp_detection_datasets" / "detic_labeled_5k"


def create_coco_dataset(
    image_paths: list[Path],
    all_boxes: list[np.ndarray],
    output_dir: Path,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> None:
    """Create COCO format dataset with train/valid/test splits."""
    
    # Filter to only images with detections
    valid_pairs = [
        (path, boxes) for path, boxes in zip(image_paths, all_boxes)
        if len(boxes) > 0
    ]
    
    print(f"Images with detections: {len(valid_pairs)} / {len(image_paths)}")
    
    # Shuffle for random splits
    np.random.seed(42)
    indices = np.random.permutation(len(valid_pairs))
    
    n_train = int(len(valid_pairs) * split_ratios[0])
    n_valid = int(len(valid_pairs) * split_ratios[1])
    
    splits = {
        "train": indices[:n_train],
        "valid": indices[n_train:n_train + n_valid],
        "test": indices[n_train + n_valid:],
    }
    
    # COCO category (single class: license_plate)
    categories = [{"id": 0, "name": "license_plate", "supercategory": "vehicle"}]
    
    for split_name, split_indices in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        images = []
        annotations = []
        ann_id = 0
        
        for img_id, idx in enumerate(split_indices):
            img_path, boxes = valid_pairs[idx]
            
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Copy image to split directory
            dest_name = f"{img_id:05d}{img_path.suffix}"
            dest_path = split_dir / dest_name
            cv2.imwrite(str(dest_path), img)
            
            # Add image entry
            images.append({
                "id": img_id,
                "file_name": dest_name,
                "width": w,
                "height": h,
            })
            
            # Add annotations for each detected plate
            for box in boxes:
                x1, y1, x2, y2 = box
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                })
                ann_id += 1
        
        # Save COCO annotations
        coco_data = {
            "info": {
                "description": f"License plate detection dataset - {split_name}",
                "date_created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        
        ann_file = split_dir / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco_data, f)
        
        print(f"  {split_name}: {len(images)} images, {len(annotations)} annotations")


def main():
    print("=" * 60)
    print("Labeling 5K Images with Detic for License Plate Detection")
    print("=" * 60)
    
    # Get all images
    image_paths = sorted(IMAGES_DIR.glob("*.webp")) + sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {IMAGES_DIR}")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Run Detic detection
    print()
    print("Running Detic detection (batch_size=2, ~10 FPS)...")
    start = time.perf_counter()
    
    all_boxes = detect_license_plates_fast(
        [str(p) for p in image_paths],
        confidence_threshold=0.3,
        batch_size=2,
        show_progress=True,
    )
    
    elapsed = time.perf_counter() - start
    fps = len(image_paths) / elapsed
    print(f"Detection complete: {elapsed:.1f}s ({fps:.1f} FPS)")
    
    # Stats
    n_with_plates = sum(1 for b in all_boxes if len(b) > 0)
    n_total_plates = sum(len(b) for b in all_boxes)
    print(f"Images with plates: {n_with_plates} / {len(image_paths)} ({100*n_with_plates/len(image_paths):.1f}%)")
    print(f"Total plates detected: {n_total_plates}")
    
    # Create COCO dataset
    print()
    print("Creating COCO dataset...")
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    create_coco_dataset(image_paths, all_boxes, LABELED_DIR)
    
    print()
    print(f"Output: {LABELED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
