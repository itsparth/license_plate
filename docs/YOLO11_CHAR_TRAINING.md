# YOLO11 Character Detection Training Guide

Training workflow for the character detection model (36 classes: A-Z, 0-9).

## Prerequisites

```bash
# Ensure dependencies are installed
uv sync
```

## Dataset

The training data is in `output/training_data/` with:
- **Train**: ~18K images (256x256 cropped license plates)
- **Valid**: ~4K images
- **Test**: ~4K images
- **Classes**: 36 (A-Z, 0-9)

## Training

### Basic Training

```bash
uv run python -m license_plate.training.yolo11.train_char \
  -d output/training_data \
  -m n \
  -e 100 \
  --patience 20
```

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `-d, --dataset` | Dataset directory | `output/training_data` |
| `-m, --model` | Model size: n/s/m/l/x | `n` (nano) |
| `-e, --epochs` | Training epochs | 100 |
| `-b, --batch-size` | Batch size | 32 |
| `--imgsz` | Input image size | 256 |
| `--patience` | Early stopping patience | 20 |
| `--resume` | Resume from checkpoint | False |

### Augmentation Strategy

Character detection uses more conservative augmentations than plate detection:

| Setting | Value | Reason |
|---------|-------|--------|
| `fliplr` | 0.0 | Text shouldn't be mirrored |
| `flipud` | 0.0 | Text shouldn't be flipped |
| `degrees` | 5.0 | Less rotation (text is usually upright) |
| `mosaic` | 0.5 | Less aggressive to preserve readability |
| `mixup` | 0.1 | Minimal to avoid confusion |

### Output

```
output/yolo11_char_training/yolo11n_char/
├── weights/
│   ├── best.pt      # Best model
│   └── last.pt      # Last checkpoint
├── results.csv
└── plots/
```

## ONNX Export

After training, export both models:

```bash
uv run python -m license_plate.training.yolo11.export_onnx
```

This exports:
- `lp_detection.zip` - License plate detection (640x640)
- `char_detection.zip` - Character detection (256x256)

Each package contains:
- `<name>.onnx` - DeepStream-compatible ONNX model
- `labels.txt` - Class labels

## DeepStream Testing

### Character Detection Test

```bash
cd output/deepstream_test
bash run_char_detection_test.sh
```

This:
1. Starts DeepStream 7.0 container
2. Compiles `nvdsinfer_custom_impl_Yolo`
3. Builds TensorRT FP16 engine
4. Runs inference on validation images
5. Outputs KITTI format detections

### DeepStream Config

The character detection nvinfer config uses:

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=char_detection.onnx
labelfile-path=char_detection_labels.txt
batch-size=1
network-mode=2
num-detected-classes=36
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

## Expected Results

For YOLO11n (nano) model with ~18K training images:

| Metric | Expected |
|--------|----------|
| mAP50 | ~0.85-0.90 |
| mAP50-95 | ~0.65-0.75 |
| Precision | ~0.85 |
| Recall | ~0.80 |

## Full Pipeline

```bash
# 1. Train character detection
uv run python -m license_plate.training.yolo11.train_char \
  -d output/training_data -m n -e 100 --patience 20

# 2. Export to ONNX
uv run python -m license_plate.training.yolo11.export_onnx

# 3. Copy ONNX to DeepStream-Yolo directory
cp output/yolo11_export/char_detection.onnx output/deepstream_test/DeepStream-Yolo/
# Create labels file
python3 -c "print('\n'.join('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))" > \
  output/deepstream_test/DeepStream-Yolo/char_detection_labels.txt

# 4. Test in DeepStream
cd output/deepstream_test
bash run_char_detection_test.sh
```

## File Locations

| File | Location |
|------|----------|
| Training script | `license_plate/training/yolo11/train_char.py` |
| Export script | `license_plate/training/yolo11/export_onnx.py` |
| Trained weights | `output/yolo11_char_training/yolo11n_char/weights/best.pt` |
| ONNX model | `output/yolo11_export/char_detection.onnx` |
| Package | `output/model_packages/char_detection.zip` |
| DeepStream test | `output/deepstream_test/run_char_detection_test.sh` |
