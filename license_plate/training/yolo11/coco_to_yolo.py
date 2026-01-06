#!/usr/bin/env python3
"""Convert COCO format annotations to YOLO format.

Reads _annotations.coco.json and creates .txt label files for each image.
"""

import argparse
import json
from pathlib import Path


def coco_to_yolo(dataset_dir: Path, splits: list[str] | None = None):
    """Convert COCO annotations to YOLO format for all splits."""
    if splits is None:
        splits = ["train", "valid", "test"]
    
    for split in splits:
        split_dir = dataset_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        
        if not ann_file.exists():
            print(f"Skipping {split}: no annotations found")
            continue
        
        with open(ann_file) as f:
            coco = json.load(f)
        
        # Build image info lookup
        images = {img["id"]: img for img in coco["images"]}
        
        # Group annotations by image
        annotations_by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        converted = 0
        for img_id, img_info in images.items():
            img_w = img_info["width"]
            img_h = img_info["height"]
            img_name = img_info["file_name"]
            
            # Create label file path
            label_name = Path(img_name).stem + ".txt"
            label_path = split_dir / label_name
            
            anns = annotations_by_image.get(img_id, [])
            if not anns:
                # Create empty label file
                label_path.write_text("")
                continue
            
            lines = []
            for ann in anns:
                # COCO bbox: [x, y, width, height] (absolute pixels, top-left corner)
                bbox = ann["bbox"]
                x, y, w, h = bbox
                
                # Convert to YOLO format: class_id x_center y_center width height (normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Our COCO category_id already starts at 0, use directly
                class_id = ann["category_id"]
                
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            label_path.write_text("\n".join(lines))
            converted += 1
        
        print(f"{split}: Converted {converted} images, {len(coco['annotations'])} annotations")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument(
        "-d", "--dataset",
        type=Path,
        default=Path("/home/openwebui/license_plate/output/training_data"),
        help="Dataset directory",
    )
    args = parser.parse_args()
    
    print(f"Converting: {args.dataset}")
    coco_to_yolo(args.dataset)
    print("Done!")


if __name__ == "__main__":
    main()
