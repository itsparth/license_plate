#!/usr/bin/env python3
"""Test RF-DETR models with DeepStream.

Sets up DeepStream-Yolo and runs inference for accuracy testing.
Requires: docker with nvcr.io/nvidia/deepstream:7.0-gc-triton-devel
"""

import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEEPSTREAM_YOLO_DIR = OUTPUT_DIR / "onnx_export" / "DeepStream-Yolo"
PACKAGES_DIR = OUTPUT_DIR / "model_packages"
TEST_DIR = OUTPUT_DIR / "deepstream_test"

DOCKER_IMAGE = "nvcr.io/nvidia/deepstream:7.0-gc-triton-devel"
CUDA_VER = "12.2"  # DeepStream 7.0


def setup_test_dir() -> None:
    """Create test directory with models and configs."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy DeepStream-Yolo source for compiling the lib
    ds_yolo_dest = TEST_DIR / "DeepStream-Yolo"
    if not ds_yolo_dest.exists():
        print(f"Copying DeepStream-Yolo to {ds_yolo_dest}")
        shutil.copytree(DEEPSTREAM_YOLO_DIR, ds_yolo_dest)

    # Extract model packages
    import zipfile
    for pkg in ["lp_detection.zip", "char_detection.zip"]:
        pkg_path = PACKAGES_DIR / pkg
        if pkg_path.exists():
            print(f"Extracting {pkg}")
            with zipfile.ZipFile(pkg_path, 'r') as zf:
                zf.extractall(ds_yolo_dest)

    # Create setup script to run inside container
    setup_script = TEST_DIR / "setup_inside_container.sh"
    setup_script.write_text(f"""#!/bin/bash
set -e

cd /workspace/DeepStream-Yolo

# Install build tools
apt-get update && apt-get install -y build-essential

# Compile the custom lib
export CUDA_VER={CUDA_VER}
make -C nvdsinfer_custom_impl_Yolo clean
make -C nvdsinfer_custom_impl_Yolo

echo "Library compiled: nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
ls -la nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
""")
    setup_script.chmod(0o755)

    # Create test script for LP detection
    test_lp_script = TEST_DIR / "test_lp_detection.sh"
    test_lp_script.write_text("""#!/bin/bash
cd /workspace/DeepStream-Yolo

# Update config to use local lib path
sed -i 's|custom-lib-path=.*|custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so|' lp_detection.txt

# Create a simple test pipeline config
cat > test_lp_app_config.txt << 'EOF'
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1

[tiled-display]
enable=0

[source0]
enable=1
type=3
uri=file:///workspace/test_video.mp4
num-sources=1
gpu-id=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=-1
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=lp_detection.txt

[sink0]
enable=1
type=1
sync=0
gpu-id=0
nvbuf-memory-type=0
EOF

echo "Run: deepstream-app -c test_lp_app_config.txt"
echo "Make sure to provide a test video at /workspace/test_video.mp4"
""")
    test_lp_script.chmod(0o755)

    print(f"\nTest directory created: {TEST_DIR}")
    print("\nTo test with DeepStream Docker:")
    print(f"  docker run --gpus all -it --rm \\")
    print(f"    -v {TEST_DIR}:/workspace \\")
    print(f"    {DOCKER_IMAGE} bash")
    print("\nInside container:")
    print("  1. ./setup_inside_container.sh  # Compile the lib")
    print("  2. Copy a test video to /workspace/test_video.mp4")
    print("  3. ./test_lp_detection.sh")
    print("  4. deepstream-app -c test_lp_app_config.txt")


def main() -> None:
    setup_test_dir()


if __name__ == "__main__":
    main()
