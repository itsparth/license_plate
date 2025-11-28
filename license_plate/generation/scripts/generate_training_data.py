#!/usr/bin/env python3
"""Generate synthetic training data with license plates on vehicle images."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from license_plate.generation.generator import (
    AssetLoader,
    PlateGenerator,
    TemplateStyle,
    VehicleImageAsset,
    create_augmentation_pipeline,
    get_contrasting_color_with_alpha,
    get_templates_for_aspect_ratio,
    sample_plate_color,
)
from license_plate.generation.layout import render_tight_and_scale


class CharacterAnnotation(BaseModel):
    label: str
    x: int
    y: int
    width: int
    height: int


class SampleAnnotation(BaseModel):
    image_path: str
    plate_text: str
    characters: list[CharacterAnnotation]


OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "training_data"


def is_multi_line_plate(name: str) -> bool:
    """Check if the plate template is multi-line based on filename."""
    return "double" in name.lower()


def select_vehicle_for_template(
    loader: AssetLoader, aspect_ratio: float, is_multi_line: bool
) -> VehicleImageAsset | None:
    """Select a random vehicle image that matches the template requirements."""
    vehicles = loader.vehicles()
    if not vehicles:
        return None

    candidates = []
    for v in vehicles:
        name = v.image_path.stem.lower()
        plate_is_multi = is_multi_line_plate(name)
        plate_ar = v.bbox.w / v.bbox.h if v.bbox.h > 0 else 0

        if is_multi_line != plate_is_multi:
            continue

        if aspect_ratio * 0.5 <= plate_ar <= aspect_ratio * 2.0:
            candidates.append(v)

    if not candidates:
        candidates = [
            v
            for v in vehicles
            if is_multi_line_plate(v.image_path.stem) == is_multi_line
        ]

    return random.choice(candidates) if candidates else random.choice(vehicles)


def generate_sample(
    loader: AssetLoader,
    output_dir: Path,
    sample_id: int,
    augment,
) -> SampleAnnotation | None:
    """Generate a single training sample."""
    is_bharat = random.random() < 0.1
    plate = PlateGenerator.generate(is_bharat_series=is_bharat)

    fonts = list(loader.iter_fonts())
    if not fonts:
        print("No fonts available")
        return None
    font_path = random.choice(fonts)

    use_multi_line = random.random() < 0.4

    vehicle = select_vehicle_for_template(loader, 4.0, use_multi_line)
    if not vehicle:
        print("No matching vehicle found")
        return None

    plate_ar = vehicle.bbox.w / vehicle.bbox.h if vehicle.bbox.h > 0 else 4.0

    templates = get_templates_for_aspect_ratio(
        plate_ar,
        is_bharat=is_bharat,
        multi_line_only=use_multi_line,
        single_line_only=not use_multi_line,
    )
    if not templates:
        templates = get_templates_for_aspect_ratio(plate_ar, is_bharat=is_bharat)
    if not templates:
        print(f"No template for AR {plate_ar:.2f}")
        return None

    template = random.choice(templates)

    # Fixed font sizes - will be scaled to fit plate
    base_size = random.randint(40, 60)
    size_xlarge = int(base_size * 1.4)
    size_large = int(base_size * 1.2)
    size_small = int(base_size * 0.7)

    style = TemplateStyle(
        font_path=font_path,
        font_size=base_size,
        font_size_small=size_small,
        font_size_large=size_large,
        font_size_xlarge=size_xlarge,
        padding_h=random.randint(5, 15),
        padding_v=random.randint(5, 15),
        gap=random.randint(2, 8),
        row_gap=random.randint(2, 8),
        letter_spacing=random.randint(0, 3),
    )

    left, top, plate_w, plate_h = vehicle.pixel_bbox

    img = vehicle.image.convert("RGB")
    plate_bg_color = sample_plate_color(img, left, top, plate_w, plate_h)
    font_color = get_contrasting_color_with_alpha(plate_bg_color)
    style.color = font_color

    img_w, img_h = img.size

    # Margin around plate - sometimes tight, sometimes with more vehicle context
    # 10-100% of plate height (larger margins show more vehicle)
    margin_ratio = random.uniform(0.1, 1.0)
    crop_margin = int(plate_h * margin_ratio)
    crop_left = max(0, left - crop_margin)
    crop_top = max(0, top - crop_margin)
    crop_right = min(img_w, left + plate_w + crop_margin)
    crop_bottom = min(img_h, top + plate_h + crop_margin)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    plate_left = left - crop_left
    plate_top = top - crop_top

    # Render widget at natural size with base scale, then scale to fit plate
    widget = template(plate, style)
    plate_img, char_boxes = render_tight_and_scale(
        widget,
        target_width=plate_w,
        target_height=plate_h,
        base_scale=1.0,
        padding_ratio=0.05,  # 5% padding on each side
    )

    result = cropped.convert("RGBA")
    result.paste(plate_img, (plate_left, plate_top), plate_img)
    result = result.convert("RGB")

    final_chars: list[CharacterAnnotation] = []
    for box in char_boxes:
        if box.label.strip():
            final_chars.append(
                CharacterAnnotation(
                    label=box.label,
                    x=plate_left + box.x,
                    y=plate_top + box.y,
                    width=box.width,
                    height=box.height,
                )
            )

    result_np = np.array(result)

    bboxes = [[c.x, c.y, c.width, c.height] for c in final_chars]
    labels = [c.label for c in final_chars]

    try:
        augmented = augment(image=result_np, bboxes=bboxes, labels=labels)
        result_np = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["labels"]
    except Exception:
        aug_bboxes = bboxes
        aug_labels = labels

    final_chars = [
        CharacterAnnotation(
            label=label,
            x=int(bbox[0]),
            y=int(bbox[1]),
            width=int(bbox[2]),
            height=int(bbox[3]),
        )
        for bbox, label in zip(aug_bboxes, aug_labels)
    ]

    img_filename = f"{sample_id:06d}.jpg"
    img_path = output_dir / "images" / img_filename
    cv2.imwrite(str(img_path), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

    return SampleAnnotation(
        image_path=img_filename,
        plate_text=plate.formatted,
        characters=final_chars,
    )


def generate_dataset(
    num_samples: int,
    output_dir: Path,
    *,
    seed: int | None = None,
):
    """Generate training dataset."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    loader = AssetLoader()
    augment = create_augmentation_pipeline()

    annotations: list[dict] = []
    success_count = 0

    print(f"Generating {num_samples} samples...")
    print(f"Output: {output_dir}")
    print("-" * 50)

    for i in range(num_samples):
        sample = generate_sample(loader, output_dir, i, augment)
        if sample:
            annotations.append(sample.model_dump())
            success_count += 1
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} ({success_count} successful)")
        else:
            print(f"  Failed sample {i}")

    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print("-" * 50)
    print(f"Generated {success_count}/{num_samples} samples")
    print(f"Annotations: {annotations_path}")

    export_yolo_format(annotations, output_dir)


def export_yolo_format(annotations: list[dict], output_dir: Path):
    """Export annotations in YOLO format for training."""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    class_to_id = {c: i for i, c in enumerate(classes)}

    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(classes))

    for ann in annotations:
        img_name = ann["image_path"]
        label_name = Path(img_name).stem + ".txt"

        img_path = output_dir / "images" / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        lines = []
        for char in ann["characters"]:
            label = char["label"].upper()
            if label not in class_to_id:
                continue

            class_id = class_to_id[label]
            cx = (char["x"] + char["width"] / 2) / img_w
            cy = (char["y"] + char["height"] / 2) / img_h
            w = char["width"] / img_w
            h = char["height"] / img_h

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(labels_dir / label_name, "w") as f:
            f.write("\n".join(lines))

    print(f"YOLO labels: {labels_dir}")


def generate_debug_visualization(output_dir: Path, max_samples: int = 10):
    """Generate debug images with bounding boxes drawn."""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    annotations_path = output_dir / "annotations.json"
    with open(annotations_path) as f:
        annotations = json.load(f)

    for i, ann in enumerate(annotations[:max_samples]):
        img_path = output_dir / "images" / ann["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for char in ann["characters"]:
            x, y, w, h = char["x"], char["y"], char["width"], char["height"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                img,
                char["label"],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        debug_path = debug_dir / f"debug_{i:04d}.jpg"
        cv2.imwrite(str(debug_path), img)

    print(f"Debug images: {debug_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic license plate training data"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug visualization images with bounding boxes",
    )
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output,
        seed=args.seed,
    )

    if args.debug:
        generate_debug_visualization(args.output)


if __name__ == "__main__":
    main()
