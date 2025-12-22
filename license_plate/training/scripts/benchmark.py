#!/usr/bin/env python3
"""Benchmark license plate recognition on multiple Roboflow datasets."""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.genai as genai
import numpy as np
from PIL import Image
from pydantic import BaseModel
from roboflow import Roboflow

# Config
ROBOFLOW_KEY = "QX0zM93sXRSEGJbRX2st"


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate similarity as 1 - (edit_distance / max_len)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return 1 - dp[n] / max(m, n)


DATASETS = [
    ("pnmr", "indian_license_plate-sjlpn", 1),  # Original dataset
    ("yolomdata", "indian-license", 1),  # New multi-plate dataset
]
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"
CACHE_DIR = OUTPUT_DIR / "gemini_cache"
MODEL_PATH = OUTPUT_DIR / "rfdetr_training" / "checkpoint_best_ema.pth"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
MAX_PARALLEL = 20


class Sample(BaseModel):
    image_path: Path
    bbox: tuple[int, int, int, int]  # x, y, w, h
    cache_key: str


def download_datasets() -> list[Path]:
    """Download all datasets, return list of dataset directories."""
    dirs = []
    rf = Roboflow(api_key=ROBOFLOW_KEY)

    for workspace, project_name, version in DATASETS:
        dest = OUTPUT_DIR / "datasets" / f"{workspace}_{project_name}_v{version}"
        if dest.exists() and (dest / "train" / "_annotations.coco.json").exists():
            print(f"Dataset exists: {dest.name}")
            dirs.append(dest)
            continue

        dest.mkdir(parents=True, exist_ok=True)
        orig_cwd = os.getcwd()
        os.chdir(dest)

        try:
            project = rf.workspace(workspace).project(project_name)
            ds = project.version(version).download(
                "coco", location=str(dest), overwrite=True
            )
            print(f"Downloaded: {ds.location}")
            dirs.append(dest)
        except Exception as e:
            print(f"Failed to download {project_name}: {e}")
        finally:
            os.chdir(orig_cwd)

    return dirs


def load_samples(dataset_dirs: list[Path]) -> list[Sample]:
    """Load samples, keeping only the largest plate per image."""
    # Group annotations by image
    image_anns: dict[str, list[tuple[Path, dict]]] = {}

    for ds_dir in dataset_dirs:
        for split in ["train", "valid", "test"]:
            ann_file = ds_dir / split / "_annotations.coco.json"
            if not ann_file.exists():
                continue

            with open(ann_file) as f:
                data = json.load(f)

            img_map = {img["id"]: img for img in data["images"]}
            for ann in data["annotations"]:
                img = img_map.get(ann["image_id"])
                if not img:
                    continue

                image_path = ds_dir / split / img["file_name"]
                key = str(image_path)

                if key not in image_anns:
                    image_anns[key] = []
                image_anns[key].append((image_path, ann))

    # Pick largest plate per image
    samples = []
    for key, anns in image_anns.items():
        # Find largest by area
        best = max(anns, key=lambda x: x[1]["bbox"][2] * x[1]["bbox"][3])
        image_path, ann = best
        x, y, w, h = [int(v) for v in ann["bbox"]]

        if w < 20 or h < 10:
            continue

        cache_key = f"{image_path.parent.parent.name}_{image_path.stem}"
        samples.append(
            Sample(image_path=image_path, bbox=(x, y, w, h), cache_key=cache_key)
        )

    return samples


# Gemini OCR with caching
def get_cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def read_cache(key: str) -> str | None:
    p = get_cache_path(key)
    if p.exists():
        return json.load(open(p)).get("text")
    return None


def write_cache(key: str, text: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"text": text}, open(get_cache_path(key), "w"))


def ocr_single(args: tuple[str, Image.Image, str]) -> tuple[str, str | None]:
    """OCR a single plate."""
    cache_key, image, api_key = args

    if cached := read_cache(cache_key):
        return cache_key, cached

    client = genai.Client(api_key=api_key)
    prompt = """Read the text from this Indian license plate image.
Return ONLY the license plate text in uppercase, no spaces, no punctuation.
If you cannot read the plate clearly, return "UNREADABLE".
Example outputs: MH12AB1234, KA01CD5678, DL3CAB1234"""

    for retry in range(5):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash", contents=[prompt, image]
            )
            text = re.sub(r"[^A-Z0-9]", "", (resp.text or "").upper()) or "UNREADABLE"
            write_cache(cache_key, text)
            return cache_key, text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep((2**retry) * 2)
            else:
                return cache_key, None
    return cache_key, None


def ocr_all(samples: list[Sample], api_key: str) -> dict[str, str]:
    """OCR all samples in parallel."""
    tasks = []
    cached = 0

    for s in samples:
        if read_cache(s.cache_key):
            cached += 1
            continue
        img = Image.open(s.image_path).convert("RGB")
        x, y, w, h = s.bbox
        crop = img.crop((x, y, x + w, y + h))
        tasks.append((s.cache_key, crop, api_key))

    print(f"Cached: {cached}, To OCR: {len(tasks)}")

    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
            futures = {ex.submit(ocr_single, t): t[0] for t in tasks}
            done = 0
            for f in as_completed(futures):
                done += 1
                if done % 20 == 0:
                    print(f"  OCR progress: {done}/{len(tasks)}")

    return {s.cache_key: read_cache(s.cache_key) or "" for s in samples}


# Model inference
def deduplicate(class_ids, bboxes, confidences, iou_thresh=0.5):
    """NMS-like deduplication for RF-DETR outputs."""
    if len(bboxes) == 0:
        return class_ids, bboxes, confidences

    order = confidences.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        rem = order[1:]
        x1 = np.maximum(bboxes[i, 0], bboxes[rem, 0])
        y1 = np.maximum(bboxes[i, 1], bboxes[rem, 1])
        x2 = np.minimum(bboxes[i, 2], bboxes[rem, 2])
        y2 = np.minimum(bboxes[i, 3], bboxes[rem, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        areas_rem = (bboxes[rem, 2] - bboxes[rem, 0]) * (
            bboxes[rem, 3] - bboxes[rem, 1]
        )
        iou = inter / (area_i + areas_rem - inter + 1e-6)
        order = rem[iou < iou_thresh]

    keep = np.array(keep)
    return class_ids[keep], bboxes[keep], confidences[keep]


def parse_plate(class_ids, bboxes, confidences, conf_thresh=0.3) -> str:
    """Parse detections into plate string."""
    if len(class_ids) == 0:
        return ""

    mask = confidences >= conf_thresh
    if not mask.any():
        return ""

    class_ids, bboxes, confidences = class_ids[mask], bboxes[mask], confidences[mask]
    class_ids, bboxes, confidences = deduplicate(class_ids, bboxes, confidences)

    if len(class_ids) == 0:
        return ""

    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    heights = bboxes[:, 3] - bboxes[:, 1]
    avg_h = heights.mean()

    x_order = np.argsort(cx)
    cy_sorted = cy[x_order]

    # Check for multi-line
    is_multi = False
    if len(cy_sorted) > 1:
        y_diffs = np.abs(np.diff(cy_sorted))
        is_multi = y_diffs.max() > avg_h * 1.2

    if is_multi:
        split_idx = np.argmax(y_diffs) + 1
        line1, line2 = x_order[:split_idx], x_order[split_idx:]

        if cy[line1].mean() < cy[line2].mean():
            top, bot = line1, line2
        else:
            top, bot = line2, line1

        top = top[np.argsort(cx[top])]
        bot = bot[np.argsort(cx[bot])]
        ordered = list(top) + list(bot)
    else:
        ordered = x_order.tolist()

    chars = [CLASSES[class_ids[i]] for i in ordered]
    text = "".join(chars)

    # Replace O with 0 except in state code
    if len(text) > 2:
        text = text[:2] + text[2:].replace("O", "0")

    return text


def preprocess(img: Image.Image) -> Image.Image:
    """Convert to grayscale and resize to 256x256 with padding."""
    gray = img.convert("L").convert("RGB")
    w, h = gray.size
    scale = 256 / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = gray.resize((nw, nh), Image.Resampling.LANCZOS)

    padded = Image.new("RGB", (256, 256), (0, 0, 0))
    padded.paste(resized, ((256 - nw) // 2, (256 - nh) // 2))
    return padded


def run_model(samples: list[Sample], ocr_results: dict[str, str]) -> list[dict]:
    """Run model inference on all samples."""
    from rfdetr.detr import RFDETRNano

    print(f"\nLoading model from {MODEL_PATH}")
    model = RFDETRNano(pretrain_weights=str(MODEL_PATH))

    results = []
    for s in samples:
        gt = ocr_results.get(s.cache_key, "")
        if not gt or gt == "UNREADABLE":
            continue

        img = Image.open(s.image_path).convert("RGB")
        x, y, w, h = s.bbox
        crop = img.crop((x, y, x + w, y + h))
        processed = preprocess(crop)

        dets = model.predict(processed, threshold=0.2)

        if len(dets) == 0:
            pred = ""
        else:
            det = dets[0] if isinstance(dets, list) else dets
            if det.xyxy is None or len(det.xyxy) == 0:
                pred = ""
            else:
                pred = parse_plate(
                    det.class_id.astype(np.int32),  # type: ignore
                    det.xyxy.astype(np.float32),
                    det.confidence.astype(np.float32),  # type: ignore
                )

        # Normalize for comparison
        gt_n = re.sub(r"[^A-Z0-9]", "", gt.upper()).replace("O", "0").replace("I", "1")
        pred_n = (
            re.sub(r"[^A-Z0-9]", "", pred.upper()).replace("O", "0").replace("I", "1")
        )

        exact = gt_n == pred_n
        sim = levenshtein_similarity(gt_n, pred_n)

        results.append(
            {
                "image": s.image_path.name,
                "path": str(s.image_path),
                "gt": gt,
                "pred": pred,
                "exact": exact,
                "sim": sim,
            }
        )

        mark = "OK" if exact else "X"
        print(f"  [{mark}] GT: {gt:15} | Pred: {pred:15} | Sim: {sim:.2f}")

    return results


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable")

    print("=== Downloading datasets ===")
    dataset_dirs = download_datasets()

    print("\n=== Loading samples ===")
    samples = load_samples(dataset_dirs)
    print(f"Total samples (largest plate per image): {len(samples)}")

    print("\n=== Phase 1: Gemini OCR ===")
    ocr_results = ocr_all(samples, api_key)

    print("\n=== Phase 2: Model Inference ===")
    results = run_model(samples, ocr_results)

    # Summary
    exact = sum(1 for r in results if r["exact"])
    avg_sim = sum(r["sim"] for r in results) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(
        f"RESULTS: {exact}/{len(results)} exact ({100*exact/len(results):.1f}%), avg sim: {avg_sim:.2f}"
    )

    # Save results
    out_file = OUTPUT_DIR / "benchmark_results.json"
    json.dump(
        {"total": len(results), "exact": exact, "avg_sim": avg_sim, "results": results},
        open(out_file, "w"),
        indent=2,
    )
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
