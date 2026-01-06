#!/usr/bin/env python3
"""Export YOLO11 models to ONNX for DeepStream/Savant and create deployment packages.

Creates zip packages compatible with Savant framework's remote model loading.
Uses DeepStream-Yolo export script for proper ONNX format.

Usage:
    uv run python -m license_plate.training.yolo11.export_onnx
"""

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"
EXPORT_DIR = OUTPUT_DIR / "yolo11_export"
PACKAGES_DIR = OUTPUT_DIR / "model_packages"

# Character class labels (36 classes: A-Z, 0-9)
CHAR_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Model configurations
MODELS = {
    "lp_detection": {
        "checkpoint": OUTPUT_DIR / "yolo11_training" / "yolo11n_lp" / "weights" / "best.pt",
        "labels": ["license_plate"],
        "imgsz": 640,
    },
    "char_detection": {
        "checkpoint": OUTPUT_DIR / "yolo11_char_training" / "yolo11n_char" / "weights" / "best.pt",
        "labels": CHAR_LABELS,
        "imgsz": 256,
    },
}


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run command and return result."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def setup_export_environment() -> Path:
    """Set up isolated environment for ONNX export with PyTorch 2.1."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    ultralytics_dir = EXPORT_DIR / "ultralytics"
    deepstream_dir = EXPORT_DIR / "DeepStream-Yolo"
    venv_dir = EXPORT_DIR / "venv"

    # Clone ultralytics if needed
    if not ultralytics_dir.exists():
        print("Cloning ultralytics repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ultralytics/ultralytics.git"],
            cwd=EXPORT_DIR,
            check=True,
        )

    # Clone DeepStream-Yolo if needed
    if not deepstream_dir.exists():
        print("Cloning DeepStream-Yolo repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/marcoslucianops/DeepStream-Yolo.git"],
            cwd=EXPORT_DIR,
            check=True,
        )

    # Copy export script to ultralytics dir
    export_script = deepstream_dir / "utils" / "export_yolo11.py"
    dest_script = ultralytics_dir / "export_yolo11.py"
    if export_script.exists() and not dest_script.exists():
        print("Copying export_yolo11.py to ultralytics...")
        shutil.copy(export_script, dest_script)

    # Create venv with PyTorch 2.1 (legacy ONNX exporter)
    if not venv_dir.exists():
        print("Creating venv with Python 3.11 and PyTorch 2.1...")
        subprocess.run(["uv", "venv", str(venv_dir), "--python", "3.11"], check=True)

        python_bin = venv_dir / "bin" / "python"
        uv_pip = ["uv", "pip", "install", "--python", str(python_bin)]

        # PyTorch 2.1 has legacy ONNX exporter that works with DeepStream-Yolo
        subprocess.run([*uv_pip, "torch==2.1.0", "torchvision==0.16.0"], check=True)
        subprocess.run([*uv_pip, "-e", str(ultralytics_dir)], check=True)
        subprocess.run([*uv_pip, "onnx", "onnxslim", "onnxruntime"], check=True)

    return ultralytics_dir


def export_yolo11_onnx(ultralytics_dir: Path, checkpoint: Path, imgsz: int = 640) -> Path | None:
    """Export YOLO11 checkpoint to ONNX using DeepStream-Yolo script."""
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return None

    venv_python = EXPORT_DIR / "venv" / "bin" / "python"

    print(f"\nExporting {checkpoint.name} to ONNX...")
    result = subprocess.run(
        [
            str(venv_python),
            "export_yolo11.py",
            "-w", str(checkpoint),
            "-s", str(imgsz),
            "--dynamic",
            "--simplify",
        ],
        cwd=ultralytics_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Export failed:\n{result.stderr}")
        return None

    print(result.stdout)

    # Find generated ONNX file
    onnx_path = checkpoint.with_suffix(".onnx")
    if onnx_path.exists():
        return onnx_path

    # Check in ultralytics dir (export script might put it there)
    alt_onnx = ultralytics_dir / checkpoint.with_suffix(".onnx").name
    if alt_onnx.exists():
        return alt_onnx

    print("ONNX file not found after export")
    return None


def create_savant_package(
    name: str,
    onnx_path: Path,
    labels: list[str],
) -> tuple[Path, str]:
    """Create Savant-compatible zip package with ONNX and labels."""
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    pkg_dir = PACKAGES_DIR / f"{name}_temp"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy ONNX with package name
        onnx_dest = pkg_dir / f"{name}.onnx"
        shutil.copy(onnx_path, onnx_dest)
        print(f"Copied ONNX: {onnx_dest.name} ({onnx_dest.stat().st_size / 1024 / 1024:.1f} MB)")

        # Create labels.txt
        labels_file = pkg_dir / "labels.txt"
        labels_file.write_text("\n".join(labels) + "\n")
        print(f"Created labels.txt with {len(labels)} classes")

        # Create zip
        zip_path = PACKAGES_DIR / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(pkg_dir.iterdir()):
                zf.write(file, file.name)
                print(f"  Added: {file.name}")

        # Calculate MD5
        md5_hash = hashlib.md5(zip_path.read_bytes()).hexdigest()
        md5_path = PACKAGES_DIR / f"{name}.md5"
        md5_path.write_text(f"{md5_hash} {name}.zip\n")

        print(f"\nPackage created: {zip_path}")
        print(f"MD5: {md5_hash}")

        return zip_path, md5_hash

    finally:
        # Cleanup temp dir
        shutil.rmtree(pkg_dir, ignore_errors=True)


def main() -> None:
    print("=" * 60)
    print("YOLO11 ONNX Export for DeepStream/Savant")
    print("=" * 60)
    print(f"Export directory: {EXPORT_DIR}")
    print(f"Packages directory: {PACKAGES_DIR}")
    print()

    # Setup environment
    ultralytics_dir = setup_export_environment()

    # Process each model
    packages = []
    for name, config in MODELS.items():
        print(f"\n{'=' * 40}")
        print(f"Processing: {name}")
        print("=" * 40)

        checkpoint = config["checkpoint"]
        if not checkpoint.exists():
            print(f"Checkpoint not found: {checkpoint}")
            continue

        # Export to ONNX
        onnx_path = export_yolo11_onnx(
            ultralytics_dir,
            checkpoint,
            config.get("imgsz", 640),
        )

        if not onnx_path:
            print(f"Failed to export {name}")
            continue

        # Move ONNX to export directory
        final_onnx = EXPORT_DIR / f"{name}.onnx"
        if onnx_path != final_onnx:
            shutil.move(onnx_path, final_onnx)
            onnx_path = final_onnx

        # Create package
        zip_path, md5 = create_savant_package(
            name=name,
            onnx_path=onnx_path,
            labels=config["labels"],
        )
        packages.append((name, zip_path, md5))

    # Cleanup labels.txt from ultralytics dir if generated
    labels_cleanup = ultralytics_dir / "labels.txt"
    if labels_cleanup.exists():
        labels_cleanup.unlink()

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nONNX models: {EXPORT_DIR}")
    print(f"Packages: {PACKAGES_DIR}")

    for name, zip_path, md5 in packages:
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"\n{name}:")
        print(f"  ZIP: {zip_path.name} ({size_mb:.1f} MB)")
        print(f"  MD5: {md5}")
        print(f"  Usage in Savant pipeline:")
        print(f"    remote:")
        print(f"      url: s3://your-bucket/models/{name}/{name}.zip")
        print(f"      checksum_url: s3://your-bucket/models/{name}/{name}.md5")
        print(f"      parameters:")
        print(f"        endpoint: https://your-s3-endpoint")


if __name__ == "__main__":
    main()
