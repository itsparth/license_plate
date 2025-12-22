# Indian License Plate Recognition

End-to-end license plate character detection using RF-DETR trained on synthetic data.

## Project Structure

```
license_plate/
├── generation/          # Synthetic data generation
│   ├── assets/          # Fonts, vehicle images, plate templates
│   ├── generator/       # Core synthesis + augmentation
│   ├── layout/          # Widget-tree layout system
│   └── scripts/         # Font download, preview generation
├── training/
│   └── scripts/
│       ├── generate_training_data.py  # Generate synthetic dataset
│       ├── train_rfdetr.py            # Train RF-DETR model
│       └── benchmark.py               # Validate on real datasets
└── inference/           # Production inference package
```

## Quick Start

```bash
# Install dependencies
uv sync

# Generate synthetic training data (12,000 samples)
uv run python -m license_plate.training.scripts.generate_training_data -n 12000

# Train RF-DETR Nano model
uv run python -m license_plate.training.scripts.train_rfdetr -e 30

# Benchmark on real datasets
GEMINI_API_KEY=your_key uv run python license_plate/training/scripts/benchmark.py
```

## Training Data Generation

Generate synthetic license plates with character-level bounding boxes:

```bash
uv run python -m license_plate.training.scripts.generate_training_data \
    -n 12000 \
    -o output/training_data \
    --seed 42
```

**Features:**
- Indian license plate formats (single/double line, Bharat series)
- 17 fonts with 70% weight on realistic plate fonts
- Geometric augmentations: rotation, perspective, shear
- Visual effects: blur, noise, compression, weather
- Plate wear: scratches, dirt, fading
- Output: 256x256 grayscale, COCO format annotations

**Output structure:**
```
output/training_data/
├── train/
│   ├── 000000.jpg
│   └── _annotations.coco.json
├── valid/
└── test/
```

## Model Training

Train RF-DETR Nano (384x384 input, optimized for edge deployment):

```bash
uv run python -m license_plate.training.scripts.train_rfdetr \
    -d output/training_data \
    -o output/rfdetr_training \
    -e 30 \
    -b 4
```

**Output:**
- `checkpoint_best_ema.pth` - Best model weights
- Training logs with mAP metrics

## Benchmark

Validate model accuracy on real-world datasets from Roboflow:

```bash
GEMINI_API_KEY=your_api_key uv run python license_plate/training/scripts/benchmark.py
```

**What it does:**
1. Downloads multiple Roboflow datasets (auto-cached)
2. Extracts largest license plate per image
3. Uses Gemini 2.0 Flash for ground truth OCR (cached)
4. Runs RF-DETR model inference
5. Calculates accuracy metrics

**Datasets included:**
- `pnmr/indian_license_plate-sjlpn` (38 images)
- `yolomdata/indian-license` (1100+ images)

**Output:** `output/benchmark_results.json`
```json
{
  "total": 1077,
  "exact": 104,
  "avg_sim": 0.55,
  "results": [...]
}
```

**Metrics:**
- **Exact match**: Plate text matches exactly (after O→0, I→1 normalization)
- **Similarity**: Levenshtein-based similarity (0-1)

## Current Results

| Metric | Value |
|--------|-------|
| Exact matches | 104/1077 (9.7%) |
| Avg similarity | 0.55 |

## Adding New Datasets

Edit `DATASETS` in `benchmark.py`:

```python
DATASETS = [
    ("workspace", "project_name", version),
    # Add more Roboflow datasets here
]
```

## Requirements

- Python 3.13+
- CUDA-capable GPU (for training)
- Roboflow API key (included)
- Gemini API key (for benchmark OCR)
