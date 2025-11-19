# License Plate Recognition System

## Project Architecture

Three-phase pipeline for Indian license plate recognition:

1. **Generation** (`license_plate/generation/`) - Synthetic data generation with precise character bounding boxes
2. **Training** (`license_plate/training/`) - Two models: plate detection + character detection (UltraLytics/Roboflow)
3. **Inference** (`license_plate/inference/`) - Pip-installable package with auto-downloading models (DeepFace-style)

## Core Principles

- **Minimalism**: Concise, readable code. No verbose docstrings or unnecessary abstractions
- **Type Safety**: Use Pydantic models everywhere for configuration and data validation
- **Python 3.13+**: Use modern syntax (`list[T]`, `dict[K, V]`, `Literal`, `@property`)
- **Error-Free**: Ensure code is free of linting errors (Pylance/MyPy). Handle imports and types correctly. Use `# type: ignore` sparingly and only when necessary.
- **Clean Output**: Do not use emojis in code output or comments. Keep logs professional and simple.

## Dependency Management with uv

```bash
# Add dependencies
uv add <package>              # Add to pyproject.toml + install
uv add --dev <package>        # Add dev dependency

# Run commands
uv run python script.py       # Run with project venv
uv run pytest                 # Run tests

# Sync environment
uv sync                       # Install all dependencies from lockfile
```

No manual venv activation needed - `uv run` handles everything.

## Data Generation Details

### Font System

- **6 categories**: `hsrp`, `display`, `italic`, `condensed`, `challenging`, `extreme`
- **HSRP fonts**: Clean, bold, sans-serif (legal Indian plates)
- **Challenging fonts**: Character ambiguity (O/0, I/1, B/8, M/H)
- **Extreme fonts**: Heavily stylized (stencil, pixel, futuristic) - maximum OCR difficulty
- **Download fonts**: `uv run python license_plate/generation/scripts/download_fonts.py`
- **Generate previews**: `uv run python license_plate/generation/scripts/generate_font_previews.py`
- See `download_fonts.py` for implementation pattern

### Asset Structure

```
license_plate/generation/assets/
  fonts/         # 25 TTF fonts across 6 categories
  vehicles/      # Front-view vehicle images with plate bboxes
  plates/        # License plate templates with character placement bboxes
```

### Layout Templates

- Widget-tree approach (Flutter-inspired) for flexible character arrangement
- Define rows, columns, alignment, gaps, padding programmatically
- Include screw assets for realism

### Augmentation Pipeline

- Use Albumentations for vehicle/plate/text colorization
- Apply HSV adjustments (training on grayscale but varied inputs)
- Filters: blur, rotation, skew, motion blur
- **Critical**: Maintain exact character coordinate tracking through all transformations

## Code Patterns

### Pydantic Models

```python
from pydantic import BaseModel
from typing import Literal

Category = Literal["type1", "type2", "type3"]

class Config(BaseModel):
    name: str
    url: str
    category: Category

    @property
    def derived_field(self) -> str:
        return f"{self.name}.ext"
```

### HTTP with httpx

```python
import httpx

with httpx.Client(timeout=30, follow_redirects=True) as client:
    response = client.get(url)
    response.raise_for_status()
    file.write_bytes(response.content)
```

### Image Processing with Pillow

```python
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype(str(font_path), size)
img = Image.new("RGB", (width, height), bg_color)
draw = ImageDraw.Draw(img)
draw.text((x, y), text, font=font, fill=color)
img.save(output_path)
```

### Script Structure

- Executable scripts in `license_plate/*/scripts/`
- Use `Path(__file__).parent` for relative asset paths
- Print concise progress with emojis: `✓`, `✗`, `⬇`, `📦`
- See `download_fonts.py` and `generate_font_previews.py` for reference patterns

## Training Workflow

- **Plate Detection Model**: Detect overall license plate bbox on vehicle
- **Character Detection Model**: Detect individual character bboxes within plate
- Export datasets to Roboflow, train with UltraLytics
- Track model versions in `license_plate/training/models/`

## Inference Goals

- Simple API: `import license_plate; result = license_plate.detect(image_path)`
- Auto-download cached models on first use
- Return structured output: character bboxes + parsed string
- Minimize dependencies for pip package

## File Organization

```
license_plate/
  generation/
    scripts/        # Executable generation utilities
    assets/         # Fonts, vehicles, plates, screws
    augument/       # Augmentation pipeline code
    generator/      # Core synthesis logic
    layout/         # Widget-tree layout system
    templates/      # Plate format templates
  training/
    datasets/       # Generated/exported datasets
    models/         # Trained weights + configs
  inference/        # Standalone pip package
  evaluations/      # Model comparison charts
```
