#!/usr/bin/env python3
"""Combine all license plate datasets into unified COCO format for training.

Merges:
- 3 Roboflow datasets (pre-labeled)
- Detic-labeled 5k images

Output: output/lp_detection_combined/ with train/valid/test splits
"""

import json
import shutil
from pathlib import Path

import cv2

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
DATASETS_DIR = OUTPUT_DIR / "lp_detection_datasets"
COMBINED_DIR = OUTPUT_DIR / "lp_detection_combined"

# All source datasets
DATASETS = [
    DATASETS_DIR / "hari-ui7zw_license-plates-gycam_v1",
    DATASETS_DIR / "kashinath_evs-and-fuel-vehicles_v1",
    DATASETS_DIR / "indiannumberplatesdetection_bike-carnumberplate_v1",
    DATASETS_DIR / "detic_labeled_5k",
]


def normalize_category_to_license_plate(categories: list[dict]) -> dict[int, int]:
    """Map all category IDs to 0 (license_plate). Returns old_id -> new_id mapping."""
    mapping = {}
    for cat in categories:
        # Map all categories to 0 (we only care about license plates)
        mapping[cat["id"]] = 0
    return mapping


def merge_split(split: str, source_dirs: list[Path], output_dir: Path) -> dict:
    """Merge a single split from all source datasets."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    merged_images = []
    merged_annotations = []
    
    image_id = 0
    ann_id = 0
    file_counter = 0

    for ds_dir in source_dirs:
        ann_file = ds_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        with open(ann_file) as f:
            data = json.load(f)

        # Build category mapping (normalize to license_plate = 0)
        cat_mapping = normalize_category_to_license_plate(data.get("categories", []))

        # Build image ID mapping
        old_to_new_img_id = {}

        for img_info in data.get("images", []):
            old_img_id = img_info["id"]
            
            # Source image path
            src_path = ds_dir / split / img_info["file_name"]
            if not src_path.exists():
                continue

            # New filename with counter prefix to avoid collisions
            new_filename = f"{file_counter:06d}_{src_path.name}"
            dst_path = split_dir / new_filename
            
            # Copy image
            shutil.copy2(src_path, dst_path)

            # Add to merged
            old_to_new_img_id[old_img_id] = image_id
            merged_images.append({
                "id": image_id,
                "file_name": new_filename,
                "width": img_info["width"],
                "height": img_info["height"],
            })

            image_id += 1
            file_counter += 1

        # Merge annotations
        for ann in data.get("annotations", []):
            old_img_id = ann["image_id"]
            if old_img_id not in old_to_new_img_id:
                continue

            merged_annotations.append({
                "id": ann_id,
                "image_id": old_to_new_img_id[old_img_id],
                "category_id": 0,  # Always license_plate
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id += 1

    return {
        "images": merged_images,
        "annotations": merged_annotations,
    }


def main():
    print("=" * 60)
    print("Combining License Plate Detection Datasets")
    print("=" * 60)

    # Check all datasets exist
    for ds_dir in DATASETS:
        if not ds_dir.exists():
            print(f"[WARN] Missing: {ds_dir}")
        else:
            print(f"[OK] Found: {ds_dir.name}")

    print()

    # Clear output directory
    if COMBINED_DIR.exists():
        print("Clearing existing combined directory...")
        shutil.rmtree(COMBINED_DIR)
    COMBINED_DIR.mkdir(parents=True)

    # Single category for license plate detection
    categories = [{"id": 0, "name": "license_plate", "supercategory": "vehicle"}]

    # Merge each split
    for split in ["train", "valid", "test"]:
        print(f"\nMerging {split} split...")
        result = merge_split(split, DATASETS, COMBINED_DIR)

        # Save COCO annotations
        coco_data = {
            "info": {
                "description": f"Combined license plate detection dataset - {split}",
                "version": "1.0",
            },
            "licenses": [],
            "categories": categories,
            "images": result["images"],
            "annotations": result["annotations"],
        }

        ann_file = COMBINED_DIR / split / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco_data, f)

        print(f"  {split}: {len(result['images'])} images, {len(result['annotations'])} annotations")

    # Summary
    print()
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    total_images = 0
    total_anns = 0

    for split in ["train", "valid", "test"]:
        ann_file = COMBINED_DIR / split / "_annotations.coco.json"
        with open(ann_file) as f:
            data = json.load(f)
        n_img = len(data["images"])
        n_ann = len(data["annotations"])
        total_images += n_img
        total_anns += n_ann
        print(f"  {split}: {n_img} images, {n_ann} annotations")

    print(f"\nTotal: {total_images} images, {total_anns} annotations")
    print(f"Output: {COMBINED_DIR}")
    print()
    print("Ready for training with:")
    print(f"  uv run python license_plate/training/scripts/train_lp_detector.py")


if __name__ == "__main__":
    main()
