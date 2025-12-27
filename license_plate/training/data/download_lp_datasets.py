#!/usr/bin/env python3
"""Download Roboflow license plate datasets and prepare for training."""

import json
import os
import shutil
from pathlib import Path

from roboflow import Roboflow

ROBOFLOW_KEY = "QX0zM93sXRSEGJbRX2st"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
DATASETS_DIR = OUTPUT_DIR / "lp_detection_datasets"

# License plate detection datasets from Roboflow
DATASETS = [
    # (workspace, project, version)
    ("hari-ui7zw", "license-plates-gycam", 1),
    ("kashinath", "evs-and-fuel-vehicles", 1),
    ("indiannumberplatesdetection", "bike-carnumberplate", 1),
]


def download_datasets() -> list[Path]:
    """Download all Roboflow datasets in COCO format."""
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    downloaded = []

    for workspace, project_name, version in DATASETS:
        dest = DATASETS_DIR / f"{workspace}_{project_name}_v{version}"

        # Check if already downloaded
        if dest.exists():
            has_train = (dest / "train" / "_annotations.coco.json").exists()
            has_valid = (dest / "valid" / "_annotations.coco.json").exists()
            if has_train or has_valid:
                print(f"[OK] Already exists: {dest.name}")
                downloaded.append(dest)
                continue

        dest.mkdir(parents=True, exist_ok=True)
        orig_cwd = os.getcwd()
        os.chdir(dest)

        try:
            print(f"[DL] Downloading: {workspace}/{project_name} v{version}")
            project = rf.workspace(workspace).project(project_name)
            ds = project.version(version).download("coco", location=str(dest), overwrite=True)
            print(f"     Saved to: {ds.location}")
            downloaded.append(dest)
        except Exception as e:
            print(f"[ERR] Failed: {project_name} - {e}")
        finally:
            os.chdir(orig_cwd)

    return downloaded


def get_dataset_stats(dataset_dir: Path) -> dict:
    """Get stats for a dataset."""
    stats = {"images": 0, "annotations": 0, "splits": []}

    for split in ["train", "valid", "test"]:
        ann_file = dataset_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        with open(ann_file) as f:
            data = json.load(f)

        n_images = len(data.get("images", []))
        n_anns = len(data.get("annotations", []))
        stats["images"] += n_images
        stats["annotations"] += n_anns
        stats["splits"].append(f"{split}: {n_images} imgs, {n_anns} anns")

    return stats


def main():
    print("=" * 60)
    print("Downloading Roboflow License Plate Datasets")
    print("=" * 60)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = download_datasets()

    print()
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    total_images = 0
    total_anns = 0

    for ds_dir in datasets:
        stats = get_dataset_stats(ds_dir)
        total_images += stats["images"]
        total_anns += stats["annotations"]
        print(f"\n{ds_dir.name}:")
        for split_info in stats["splits"]:
            print(f"  {split_info}")

    print()
    print(f"Total: {total_images} images, {total_anns} annotations")
    print(f"Output: {DATASETS_DIR}")


if __name__ == "__main__":
    main()
