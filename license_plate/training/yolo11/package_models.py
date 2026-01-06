#!/usr/bin/env python3
"""Package YOLO11 models for Savant framework deployment.

Creates zip packages with ONNX models and labels for remote model loading.

Usage:
    uv run python -m license_plate.training.yolo11.package_models

Output format matches Savant's remote model loading:
    remote:
        url: s3://bucket/models/lp_detection/lp_detection.zip
        checksum_url: s3://bucket/models/lp_detection/lp_detection.md5
        parameters:
            endpoint: https://your-endpoint.com
"""

import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from pydantic import BaseModel

OUTPUT_DIR = Path("/home/openwebui/license_plate/output")
PACKAGES_DIR = OUTPUT_DIR / "model_packages"
EXPORT_DIR = OUTPUT_DIR / "onnx_export"
EXPORTED_MODELS_DIR = OUTPUT_DIR / "exported_models"


class ModelConfig(BaseModel):
    """Model packaging configuration."""
    name: str
    checkpoint: Path | None = None  # PyTorch checkpoint for export
    onnx_path: Path | None = None   # Pre-exported ONNX (preferred)
    labels: list[str]
    imgsz: int = 640


# DeepStream/Savant nvinfer config template
NVINFER_CONFIG = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={name}.onnx
model-engine-file=model_b1_gpu0_fp16.engine
labelfile-path=labels.txt
batch-size=1
network-mode=2
num-detected-classes={num_classes}
interval=0
network-type=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/opt/savant/lib/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
"""


def export_to_onnx(checkpoint: Path, imgsz: int, output_path: Path) -> bool:
    """Export PyTorch checkpoint to ONNX using standard ultralytics export.
    
    This produces Savant-compatible format: [batch, num_classes+4, anchors]
    """
    try:
        from ultralytics import YOLO
        
        model = YOLO(str(checkpoint))
        model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            simplify=True,
            opset=12,
        )
        
        # YOLO exports to same dir as checkpoint
        onnx_src = checkpoint.with_suffix(".onnx")
        if onnx_src.exists():
            shutil.copy(str(onnx_src), str(output_path))
            return True
        return False
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def create_package(config: ModelConfig) -> tuple[Path, str] | None:
    """Create Savant-compatible zip package."""
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    onnx_path = EXPORT_DIR / f"{config.name}.onnx"
    
    # Use pre-exported ONNX if available
    if config.onnx_path and config.onnx_path.exists():
        print(f"Using pre-exported ONNX: {config.onnx_path}")
        shutil.copy(config.onnx_path, onnx_path)
    elif not onnx_path.exists():
        # Export from checkpoint
        if not config.checkpoint or not config.checkpoint.exists():
            print(f"No ONNX or checkpoint found for {config.name}")
            return None
        print(f"Exporting {config.name} to ONNX...")
        if not export_to_onnx(config.checkpoint, config.imgsz, onnx_path):
            print(f"Failed to export {config.name}")
            return None
    else:
        print(f"Using existing ONNX: {onnx_path}")
    
    # Create package in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir)
        
        # Copy ONNX
        onnx_dest = pkg_dir / f"{config.name}.onnx"
        shutil.copy(onnx_path, onnx_dest)
        size_mb = onnx_dest.stat().st_size / 1024 / 1024
        print(f"  ONNX: {size_mb:.1f} MB")
        
        # Create zip (Savant format: just ONNX file)
        zip_path = PACKAGES_DIR / f"{config.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(onnx_dest, onnx_dest.name)
        
        # Calculate MD5
        md5_hash = hashlib.md5(zip_path.read_bytes()).hexdigest()
        md5_path = PACKAGES_DIR / f"{config.name}.md5"
        md5_path.write_text(f"{md5_hash}  {config.name}.zip\n")
        
        # Also save labels.txt separately (for DeepStream config)
        labels_file = PACKAGES_DIR / f"{config.name}_labels.txt"
        labels_file.write_text("\n".join(config.labels) + "\n")
        
        print(f"  ZIP: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  MD5: {md5_hash}")
        print(f"  Labels: {labels_file.name} ({len(config.labels)} classes)")
        
        return zip_path, md5_hash


def main() -> None:
    print("=" * 60)
    print("Packaging YOLO11 Models for Savant")
    print("=" * 60)
    
    # Model configurations - use standard ultralytics export for Savant compatibility
    # LP detection was trained in /home/openwebui/output (old export_onnx.py location)
    LP_TRAINING_DIR = Path("/home/openwebui/output")
    
    models = [
        ModelConfig(
            name="lp_detection",
            checkpoint=LP_TRAINING_DIR / "yolo11_training" / "yolo11n_lp" / "weights" / "best.pt",
            labels=["license_plate"],
            imgsz=640,
        ),
        ModelConfig(
            name="char_detection",
            checkpoint=OUTPUT_DIR / "yolo11_char_detection_training" / "yolo11n_char_detect5" / "weights" / "best.pt",
            labels=["char"],  # Single class - detection only, no classification
            imgsz=256,
        ),
    ]
    
    results = []
    for model in models:
        print(f"\n{'=' * 40}")
        print(f"Packaging: {model.name}")
        print("=" * 40)
        
        result = create_package(model)
        if result:
            results.append((model.name, *result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Packaging Complete!")
    print("=" * 60)
    print(f"\nOutput: {PACKAGES_DIR}")
    
    for name, zip_path, md5 in results:
        print(f"\n{name}:")
        print(f"  {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  {name}.md5: {md5}")
        print()
        print(f"  Savant config:")
        print(f"    remote:")
        print(f"      url: s3://savant-data/models/{name}/{name}.zip")
        print(f"      checksum_url: s3://savant-data/models/{name}/{name}.md5")
        print(f"      parameters:")
        print(f"        endpoint: https://eu-central-1.linodeobjects.com")


if __name__ == "__main__":
    main()
