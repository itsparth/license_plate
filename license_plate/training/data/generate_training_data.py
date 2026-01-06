#!/usr/bin/env python3
"""Generate synthetic training data with license plates on vehicle images."""

import argparse
import json
import os
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from license_plate.generation.generator import (
    AssetLoader,
    PlateGenerator,
    TemplateStyle,
    VehicleImageAsset,
    create_effects_pipeline,
    create_geometric_pipeline,
    get_contrasting_color_with_alpha,
    random_template,
    sample_plate_color,
    tight_crop_around_bboxes,
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
TRAIN_SPLIT = 0.8  # 80% train
VALID_SPLIT = 0.1  # 10% valid
# Remaining 10% goes to test
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Worker-local state (initialized once per process)
_worker_loader: AssetLoader | None = None
_worker_geometric_aug = None
_worker_effects_aug = None
_worker_output_size: int = 384
_worker_small_plate_sim: bool = True


def _init_worker(
    seed_base: int | None,
    worker_id: int,
    output_size: int = 384,
    small_plate_simulation: bool = True,
):
    """Initialize worker-local resources."""
    global _worker_loader, _worker_geometric_aug, _worker_effects_aug
    global _worker_output_size, _worker_small_plate_sim
    _worker_loader = AssetLoader()
    _worker_geometric_aug = create_geometric_pipeline()
    _worker_output_size = output_size
    _worker_small_plate_sim = small_plate_simulation
    _worker_effects_aug = create_effects_pipeline(
        output_size=output_size,
        small_plate_simulation=small_plate_simulation,
    )
    # Seed each worker differently for variety
    if seed_base is not None:
        worker_seed = seed_base + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)


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


def _generate_sample_task(args: tuple[int, Path]) -> tuple[int, SampleAnnotation | None]:
    """Worker task for multiprocessing. Returns (sample_id, annotation)."""
    sample_id, output_dir = args
    result = generate_sample(output_dir, sample_id)
    return sample_id, result


def generate_sample(
    output_dir: Path,
    sample_id: int,
) -> SampleAnnotation | None:
    """Generate a single training sample using worker-local resources."""
    loader = _worker_loader
    geometric_aug = _worker_geometric_aug
    effects_aug = _worker_effects_aug

    if loader is None:
        raise RuntimeError("Worker not initialized")

    is_bharat = random.random() < 0.1
    plate = PlateGenerator.generate(is_bharat_series=is_bharat)

    fonts = list(loader.iter_fonts())
    if not fonts:
        print("No fonts available")
        return None

    # Weight license plate fonts higher (70%) vs decorative fonts (30%)
    # Plate fonts: HSRP-like clean, bold fonts for realistic plates
    plate_font_names = {
        "hsrp.ttf",
        "license_plate.ttf",
        "license_plate_closed.ttf",
        "gl_nummernschild_eng.ttf",
        "gl_nummernschild_mtl.ttf",
        "road_numbers.ttf",
        "montserrat_bold.ttf",
        "ibm_plex_mono_bold.ttf",
        "anton.ttf",
    }
    plate_fonts = [f for f in fonts if f.split("/")[-1].lower() in plate_font_names]
    other_fonts = [f for f in fonts if f not in plate_fonts]

    if plate_fonts and other_fonts and random.random() < 0.7:
        font_path = random.choice(plate_fonts)
    elif plate_fonts:
        font_path = random.choice(plate_fonts)
    else:
        font_path = random.choice(fonts)

    use_multi_line = random.random() < 0.4

    vehicle = select_vehicle_for_template(loader, 4.0, use_multi_line)
    if not vehicle:
        print("No matching vehicle found")
        return None

    plate_ar = vehicle.bbox.w / vehicle.bbox.h if vehicle.bbox.h > 0 else 4.0

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

    # Optionally add a logo (30% chance)
    logo = None
    if random.random() < 0.3:
        logo = loader.random_logo()

    img_w, img_h = img.size

    # Moderate margin for geometric transform leeway (30%)
    # Tight crop to 5% buffer happens after augmentation
    margin_ratio = random.uniform(0.25, 0.35)
    crop_margin = int(plate_h * margin_ratio)
    crop_left = max(0, left - crop_margin)
    crop_top = max(0, top - crop_margin)
    crop_right = min(img_w, left + plate_w + crop_margin)
    crop_bottom = min(img_h, top + plate_h + crop_margin)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    plate_left = left - crop_left
    plate_top = top - crop_top

    # Render widget at natural size with base scale, then scale to fit plate
    widget = random_template(
        plate,
        style,
        aspect_ratio=plate_ar,
        multi_line_only=use_multi_line,
        single_line_only=not use_multi_line,
        logo=logo,
    )
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
        # Step 1: Apply geometric transforms (rotation, perspective, shear)
        geo_result = geometric_aug(image=result_np, bboxes=bboxes, labels=labels)
        result_np = geo_result["image"]
        bboxes = list(geo_result["bboxes"])
        labels = list(geo_result["labels"])

        # Step 2: Tight crop around bboxes (15% horizontal, 10% vertical buffer)
        if bboxes:
            result_np, bboxes, labels = tight_crop_around_bboxes(
                result_np, bboxes, labels
            )

        # Step 3: Apply visual effects and resize to 256x256
        eff_result = effects_aug(image=result_np, bboxes=bboxes, labels=labels)
        result_np = eff_result["image"]
        aug_bboxes = eff_result["bboxes"]
        aug_labels = eff_result["labels"]
    except Exception:
        aug_bboxes = bboxes
        aug_labels = labels

    # Validate result image
    if result_np is None or result_np.size == 0:
        return None

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
    img_path = output_dir / img_filename
    try:
        cv2.imwrite(str(img_path), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return None

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
    num_workers: int | None = None,
    output_size: int = 384,
    small_plate_simulation: bool = True,
):
    """Generate training dataset in COCO format with train/valid/test splits.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory for generated data
        seed: Random seed for reproducibility
        num_workers: Number of worker processes
        output_size: Output image size (square, default 384 for RF-DETR)
        small_plate_simulation: If True, includes random scaling to simulate small plates
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create split directories
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Determine split indices (80/10/10)
    num_train = int(num_samples * TRAIN_SPLIT)
    num_valid = int(num_samples * VALID_SPLIT)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = set(indices[:num_train])
    valid_indices = set(indices[num_train : num_train + num_valid])
    # Rest goes to test

    # Build task list: (sample_id, output_dir)
    tasks: list[tuple[int, Path]] = []
    for i in range(num_samples):
        if i in train_indices:
            tasks.append((i, train_dir))
        elif i in valid_indices:
            tasks.append((i, valid_dir))
        else:
            tasks.append((i, test_dir))

    # Determine worker count
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    num_workers = max(1, min(num_workers, cpu_count()))

    print(
        f"Generating {num_samples} samples (train: {num_train}, valid: {num_valid}, test: {num_samples - num_train - num_valid})..."
    )
    print(f"Output: {output_dir}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Small plate simulation: {small_plate_simulation}")
    print(f"Workers: {num_workers}")
    print("-" * 50)

    train_samples: list[SampleAnnotation] = []
    valid_samples: list[SampleAnnotation] = []
    test_samples: list[SampleAnnotation] = []

    # Use multiprocessing pool
    completed = 0
    failed = 0

    def init_worker_wrapper(
        seed_base: int | None,
        img_size: int,
        small_sim: bool,
    ):
        """Wrapper to initialize worker with unique seed and settings."""
        worker_id = os.getpid()
        _init_worker(seed_base, worker_id, img_size, small_sim)

    with Pool(
        num_workers,
        initializer=init_worker_wrapper,
        initargs=(seed, output_size, small_plate_simulation),
    ) as pool:
        for sample_id, result in pool.imap_unordered(_generate_sample_task, tasks, chunksize=32):
            if result:
                if sample_id in train_indices:
                    train_samples.append(result)
                elif sample_id in valid_indices:
                    valid_samples.append(result)
                else:
                    test_samples.append(result)
                completed += 1
            else:
                failed += 1

            total = completed + failed
            if total % 500 == 0:
                print(f"  Progress: {total}/{num_samples} ({completed} ok, {failed} failed)")

    # Export COCO format
    export_coco_format(train_samples, train_dir)
    export_coco_format(valid_samples, valid_dir)
    export_coco_format(test_samples, test_dir)

    print("-" * 50)
    print(
        f"Generated {len(train_samples)} train + {len(valid_samples)} valid + {len(test_samples)} test samples"
    )
    print(f"Dataset: {output_dir}")


def export_coco_format(samples: list[SampleAnnotation], output_dir: Path):
    """Export annotations in COCO format for RF-DETR training."""
    class_to_id = {c: i for i, c in enumerate(CLASSES)}

    coco = {
        "info": {
            "description": "License Plate Character Detection Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "character"}
            for i, name in enumerate(CLASSES)
        ],
    }

    annotation_id = 0

    for image_id, sample in enumerate(samples):
        img_path = output_dir / sample.image_path
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        coco["images"].append(
            {
                "id": image_id,
                "file_name": sample.image_path,
                "width": img_w,
                "height": img_h,
            }
        )

        for char in sample.characters:
            label = char.label.upper()
            if label not in class_to_id:
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[label],
                    "bbox": [char.x, char.y, char.width, char.height],
                    "area": char.width * char.height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(output_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"COCO annotations: {output_dir / '_annotations.coco.json'} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)"
    )


def generate_debug_visualization(output_dir: Path, max_samples: int = 10):
    """Generate debug images with bounding boxes drawn from COCO annotations."""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Use train split for debug visualization
    train_dir = output_dir / "train"
    annotations_path = train_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        print("No annotations found for debug visualization")
        return

    with open(annotations_path) as f:
        coco = json.load(f)

    # Build lookup for annotations by image_id
    img_to_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img_to_anns.setdefault(img_id, []).append(ann)

    id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for i, img_info in enumerate(coco["images"][:max_samples]):
        img_path = train_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for ann in img_to_anns.get(img_info["id"], []):
            x, y, w, h = [int(v) for v in ann["bbox"]]
            label = id_to_name.get(ann["category_id"], "?")
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                img,
                label,
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
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: {max(1, cpu_count() - 1)})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="Output image size (square, default: 384 for RF-DETR)",
    )
    parser.add_argument(
        "--no-small-plate-sim",
        action="store_true",
        help="Disable small plate simulation (random scaling)",
    )
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output,
        seed=args.seed,
        num_workers=args.workers,
        output_size=args.size,
        small_plate_simulation=not args.no_small_plate_sim,
    )

    if args.debug:
        generate_debug_visualization(args.output)


if __name__ == "__main__":
    main()
