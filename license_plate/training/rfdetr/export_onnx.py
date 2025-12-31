#!/usr/bin/env python3
"""Export RF-DETR models to ONNX for DeepStream and create deployment packages.

Sets up isolated environment with ultralytics repo as per DeepStream-Yolo instructions:
https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/RFDETR.md
"""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "output"
ONNX_DIR = OUTPUT_DIR / "onnx_export"
LP_MODEL = OUTPUT_DIR / "lp_detection_training" / "checkpoint_best_ema.pth"
CHAR_MODEL = OUTPUT_DIR / "rfdetr_training" / "checkpoint_best_ema.pth"
PACKAGES_DIR = OUTPUT_DIR / "model_packages"

# DeepStream config template for RF-DETR
CONFIG_TEMPLATE = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
# 0=RGB, 1=BGR, 2=GRAY
model-color-format=0
onnx-file={onnx_file}
model-engine-file=model_b1_gpu0_{precision}.engine
#int8-calib-file=calib.table
labelfile-path=labels.txt
batch-size=1
# Integer 0: FP32 1: INT8 2: FP16
network-mode={network_mode}
num-detected-classes={num_classes}
interval=0
#gie-unique-id=1
#process-mode=1
network-type=0
cluster-mode=4
maintain-aspect-ratio=0
#force-implicit-batch-dim=1
#workspace-size=2000
parse-bbox-func-name=NvDsInferParseYolo
#parse-bbox-func-name=NvDsInferParseYoloCuda
custom-lib-path=/opt/savant/lib/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
"""


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def setup_environment() -> Path:
    """Clone repos and create venv, return path to ultralytics dir."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    ultralytics_dir = ONNX_DIR / "ultralytics"
    deepstream_dir = ONNX_DIR / "DeepStream-Yolo"
    venv_dir = ONNX_DIR / "venv"

    # Clone ultralytics
    if not ultralytics_dir.exists():
        print("Cloning ultralytics repo...")
        run(["git", "clone", "--depth", "1", "https://github.com/ultralytics/ultralytics.git"], cwd=ONNX_DIR)

    # Clone DeepStream-Yolo
    if not deepstream_dir.exists():
        print("Cloning DeepStream-Yolo repo...")
        run(["git", "clone", "--depth", "1", "https://github.com/marcoslucianops/DeepStream-Yolo.git"], cwd=ONNX_DIR)

    # Copy export script
    export_script = deepstream_dir / "utils" / "export_rfdetr.py"
    dest_script = ultralytics_dir / "export_rfdetr.py"
    if not dest_script.exists():
        print("Copying export_rfdetr.py to ultralytics...")
        shutil.copy(export_script, dest_script)

    # Create venv with uv
    if not venv_dir.exists():
        print("Creating venv...")
        run(["uv", "venv", str(venv_dir), "--python", "3.12"])

        # Install dependencies
        # Pin torch<2.5 for legacy ONNX export compatibility (DeepStream-Yolo script)
        print("Installing dependencies...")
        uv_pip = ["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python")]
        run([*uv_pip, "torch>=2.4,<2.5", "torchvision"])
        run([*uv_pip, "-e", str(ultralytics_dir)])
        run([*uv_pip, "onnx", "onnxslim", "onnxruntime", "rfdetr"])

    return ultralytics_dir


def export_model(ultralytics_dir: Path, model_path: Path, output_name: str) -> Path | None:
    """Export a single model to ONNX."""
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    print(f"\nExporting {model_path.name}...")
    venv_python = ONNX_DIR / "venv" / "bin" / "python"

    run(
        [
            str(venv_python),
            "export_rfdetr.py",
            "-m", "rfdetr-nano",
            "-w", str(model_path),
            "-s", "384",
            "--dynamic",
            "--simplify",
        ],
        cwd=ultralytics_dir,
    )

    # Find and move the generated ONNX file
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        dest = ONNX_DIR / output_name
        shutil.move(onnx_path, dest)
        print(f"Exported: {dest}")
        return dest

    return None


def create_package(onnx_path: Path, name: str, labels: list[str], num_classes: int, network_mode: int) -> Path:
    """Create a deployment zip package with ONNX, config, and labels."""
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create temp directory for package contents
    pkg_dir = PACKAGES_DIR / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONNX file
    onnx_name = f"{name}.onnx"
    shutil.copy(onnx_path, pkg_dir / onnx_name)

    # Create config file with model-specific name
    precision = {0: "fp32", 1: "int8", 2: "fp16"}[network_mode]
    config_content = CONFIG_TEMPLATE.format(
        onnx_file=onnx_name,
        num_classes=num_classes,
        network_mode=network_mode,
        precision=precision,
    )
    (pkg_dir / f"{name}.txt").write_text(config_content)

    # Create labels file
    labels_content = "\n".join(labels)
    (pkg_dir / "labels.txt").write_text(labels_content)

    # Create zip
    zip_path = PACKAGES_DIR / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in pkg_dir.iterdir():
            zf.write(file, file.name)

    # Create MD5 checksum
    md5_hash = hashlib.md5(zip_path.read_bytes()).hexdigest()
    md5_path = PACKAGES_DIR / f"{name}.md5"
    md5_path.write_text(f"{md5_hash}  {name}.zip\n")

    # Cleanup temp directory
    shutil.rmtree(pkg_dir)

    print(f"Created package: {zip_path}")
    print(f"  MD5: {md5_hash}")
    return zip_path


def main() -> None:
    print(f"ONNX export directory: {ONNX_DIR}")

    ultralytics_dir = setup_environment()

    # Define models with their labels
    models = [
        {
            "checkpoint": LP_MODEL,
            "onnx_name": "lp_detection_deepstream.onnx",
            "package_name": "lp_detection",
            "labels": ["numberPlate"],
            "num_classes": 1,
            "network_mode": 2,  # FP16
        },
        {
            "checkpoint": CHAR_MODEL,
            "onnx_name": "char_detection_deepstream.onnx",
            "package_name": "char_detection",
            "labels": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "num_classes": 36,
            "network_mode": 0,  # FP32
        },
    ]

    exported = []
    for model in models:
        onnx_path = ONNX_DIR / model["onnx_name"]

        # Export if not already done
        if not onnx_path.exists():
            result = export_model(ultralytics_dir, model["checkpoint"], model["onnx_name"])
            if result:
                onnx_path = result
            else:
                continue
        else:
            print(f"ONNX already exists: {onnx_path}")

        # Create deployment package
        pkg_path = create_package(
            onnx_path,
            model["package_name"],
            model["labels"],
            model["num_classes"],
            model["network_mode"],
        )
        exported.append(pkg_path)

    # Move labels.txt if generated by export script
    labels_file = ultralytics_dir / "labels.txt"
    if labels_file.exists():
        labels_file.unlink()

    print(f"\nExport complete!")
    print(f"ONNX models: {ONNX_DIR}")
    print(f"Packages: {PACKAGES_DIR}")
    for p in exported:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
