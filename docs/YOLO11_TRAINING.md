# YOLO11 License Plate Detection Training Guide

Complete workflow for training, exporting, and deploying YOLO11 models for license plate detection with DeepStream/Savant.

## Prerequisites

```bash
# Clone the repository
git clone <repo-url>
cd license_plate

# Install dependencies with uv
uv sync
```

## Dataset Preparation

### Directory Structure

```
output/lp_detection_combined/
├── train/
│   ├── image1.jpg
│   ├── image1.txt          # YOLO format labels
│   └── _annotations.coco.json
├── valid/
│   └── ...
└── test/
    └── ...
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].

### Check Dataset Size

```bash
for split in train valid test; do
  count=$(find output/lp_detection_combined/$split -name "*.jpg" | wc -l)
  echo "$split: $count images"
done
```

## Training

### Basic Training

```bash
uv run python -m license_plate.training.yolo11.train_plate \
  -d output/lp_detection_combined \
  -m n \
  -e 100 \
  --patience 20
```

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `-d, --dataset` | Dataset directory | `output/lp_detection_combined` |
| `-m, --model` | Model size: n/s/m/l/x | `n` (nano) |
| `-e, --epochs` | Training epochs | 100 |
| `-b, --batch-size` | Batch size | 16 |
| `--imgsz` | Input image size | 640 |
| `--patience` | Early stopping patience | 20 |
| `--resume` | Resume from checkpoint | False |
| `--device` | GPU device | 0 |

### Augmentations Applied

The training script uses aggressive augmentations for small datasets:

| Augmentation | Value | Description |
|--------------|-------|-------------|
| `hsv_h` | 0.015 | Hue shift ±1.5% |
| `hsv_s` | 0.7 | Saturation ±70% |
| `hsv_v` | 0.4 | Brightness ±40% |
| `degrees` | 10.0 | Rotation ±10° |
| `scale` | 0.5 | Scale ±50% |
| `shear` | 3.0 | Shear distortion |
| `perspective` | 0.0003 | Perspective warp |
| `mosaic` | 1.0 | Mosaic (100%) |
| `mixup` | 0.2 | MixUp (20%) |
| `copy_paste` | 0.2 | Copy-paste augmentation |
| `erasing` | 0.4 | Random erasing |

### Output

Training outputs are saved to:
```
output/yolo11_training/yolo11n_lp/
├── weights/
│   ├── best.pt      # Best model
│   └── last.pt      # Last checkpoint
├── results.csv
└── plots/
```

## Evaluation

### Validate on Test Set

```bash
uv run python -c "
from ultralytics import YOLO

model = YOLO('output/yolo11_training/yolo11n_lp/weights/best.pt')
results = model.val(
    data='output/yolo11_training/dataset.yaml',
    split='test',
    imgsz=640,
    conf=0.25,
    iou=0.5,
)
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
"
```

### Expected Results (1.9K images, nano model)

| Metric | Value |
|--------|-------|
| mAP50 | ~0.815 |
| mAP50-95 | ~0.642 |
| Precision | ~0.831 |
| Recall | ~0.730 |

## ONNX Export for DeepStream

### Automated Export

```bash
uv run python license_plate/training/yolo11/export_onnx.py
```

This script:
1. Sets up isolated venv with PyTorch 2.1 (required for legacy ONNX exporter)
2. Clones ultralytics + DeepStream-Yolo repos
3. Exports ONNX using DeepStream-Yolo's `export_yolo11.py`
4. Creates Savant-compatible zip package with ONNX + labels.txt

### Manual Export (Alternative)

```bash
# Create export environment
cd output/yolo11_export
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git

# Copy export script
cp DeepStream-Yolo/utils/export_yolo11.py ultralytics/

# Create venv with PyTorch 2.1 (legacy ONNX exporter)
uv venv venv --python 3.11
source venv/bin/activate
uv pip install torch==2.1.0 torchvision==0.16.0
uv pip install -e ultralytics
uv pip install onnx onnxslim onnxruntime

# Export
cd ultralytics
python export_yolo11.py \
  -w /path/to/best.pt \
  -s 640 \
  --dynamic \
  --simplify
```

### Output

```
output/model_packages/
├── lp_detection.zip      # ONNX + labels.txt
└── lp_detection.md5      # Checksum
```

Package contents:
```
lp_detection.zip
├── lp_detection.onnx    # DeepStream-compatible ONNX
└── labels.txt           # Class labels (one per line)
```

## DeepStream Testing

### Run Test in Docker Container

```bash
cd output/deepstream_test
bash run_yolo11_test.sh
```

This script:
1. Starts DeepStream 7.0 container
2. Compiles `nvdsinfer_custom_impl_Yolo`
3. Builds TensorRT FP16 engine
4. Runs inference on sample video
5. Outputs KITTI format detections

### Manual DeepStream Setup

```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  -v $(pwd)/DeepStream-Yolo:/mnt/DeepStream-Yolo \
  -w /mnt/DeepStream-Yolo \
  nvcr.io/nvidia/deepstream:7.0-gc-triton-devel

# Inside container
apt-get update && apt-get install -y build-essential
export CUDA_VER=12.2
make -C nvdsinfer_custom_impl_Yolo

# Run DeepStream
deepstream-app -c deepstream_lp_yolo11_test.txt
```

### DeepStream Config Files

**config_infer_lp_yolo11.txt:**
```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=lp_detection.onnx
model-engine-file=lp_detection_b1_gpu0_fp16.engine
labelfile-path=labels.txt
batch-size=1
network-mode=2
num-detected-classes=1
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

### TensorRT FP16 Accuracy

| Metric | PyTorch FP32 | TensorRT FP16 | Diff |
|--------|-------------|---------------|------|
| mAP50 | 0.815 | 0.816 | +0.1% |
| Precision | 0.831 | 0.817 | -1.4% |
| Recall | 0.730 | 0.718 | -1.2% |
| Inference | 4.6ms | 1.1ms | **4x faster** |

## Deployment Package Upload

### Upload to SeaweedFS Filer

```bash
cd output/model_packages

# Upload files
curl -X POST "https://filer.kryptonait.com/models/lp_detection/" \
  -F "file=@lp_detection.zip"

curl -X POST "https://filer.kryptonait.com/models/lp_detection/" \
  -F "file=@lp_detection.md5"
```

### Verify Upload

```bash
curl -s "https://filer.kryptonait.com/models/lp_detection/lp_detection.md5"
```

## Savant Pipeline Integration

### Model Configuration

```yaml
model:
  remote:
    url: https://filer.kryptonait.com/models/lp_detection/lp_detection.zip
    checksum_url: https://filer.kryptonait.com/models/lp_detection/lp_detection.md5
```

### Full Pipeline Example

```yaml
name: lp_detection_pipeline

parameters:
  batch_size: 1

pipeline:
  elements:
    - element: nvinfer@detector
      name: lp_detector
      model:
        remote:
          url: https://filer.kryptonait.com/models/lp_detection/lp_detection.zip
          checksum_url: https://filer.kryptonait.com/models/lp_detection/lp_detection.md5
        format: onnx
        config_file: lp_detection.txt
        precision: fp16
        input:
          shape: [3, 640, 640]
          maintain_aspect_ratio: true
        output:
          num_detected_classes: 1
          layer_names: [output]
```

## Quick Reference

### Full Training → Deployment Pipeline

```bash
# 1. Train model
uv run python -m license_plate.training.yolo11.train_plate \
  -d output/lp_detection_combined -m n -e 100 --patience 20

# 2. Export to ONNX + create package
uv run python license_plate/training/yolo11/export_onnx.py

# 3. Upload to storage
cd output/model_packages
curl -X POST "https://filer.kryptonait.com/models/lp_detection/" \
  -F "file=@lp_detection.zip"
curl -X POST "https://filer.kryptonait.com/models/lp_detection/" \
  -F "file=@lp_detection.md5"

# 4. Test in DeepStream (optional)
cd output/deepstream_test
bash run_yolo11_test.sh
```

### File Locations

| File | Location |
|------|----------|
| Training script | `license_plate/training/yolo11/train_plate.py` |
| Export script | `license_plate/training/yolo11/export_onnx.py` |
| Trained weights | `output/yolo11_training/yolo11n_lp/weights/best.pt` |
| ONNX model | `output/yolo11_export/lp_detection.onnx` |
| Deployment package | `output/model_packages/lp_detection.zip` |
| DeepStream configs | `output/deepstream_test/DeepStream-Yolo/` |

## Troubleshooting

### ONNX Export Fails with PyTorch 2.9+

The new PyTorch ONNX exporter is incompatible with DeepStream-Yolo. Use PyTorch 2.1:

```bash
uv pip install torch==2.1.0 torchvision==0.16.0
```

### DeepStream TensorRT Build Slow

First-time engine build takes 5-10 minutes. Subsequent runs use cached engine.

### Low mAP Score

- Increase training data (aim for 5K+ images)
- Use larger model (s/m instead of n)
- Increase epochs
- Check label quality

### SeaweedFS S3 Access Denied

Use the filer endpoint instead of S3 endpoint for uploads:
- ✗ `https://s3.kryptonait.com/models/...`
- ✓ `https://filer.kryptonait.com/models/...`
