#!/usr/bin/env python3
"""Generate preview images for all templates with a sample plate."""

from pathlib import Path

from PIL import Image, ImageDraw

from license_plate.generation.generator import (
    ALL_TEMPLATES,
    PlateGenerator,
    TemplateStyle,
)
from license_plate.generation.generator.asset_loader import AssetLoader
from license_plate.generation.layout import render_bounding_boxes, render_layout, u

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "template_previews"


def render_template_preview(
    template,
    plate,
    style: TemplateStyle,
    *,
    show_boxes: bool = True,
    height: int = 300,
) -> Image.Image:
    """Render a single template with optional bounding boxes."""
    # Use midpoint of aspect ratio range for optimal fit
    aspect_ratio = (template.min_aspect_ratio + template.max_aspect_ratio) / 2
    width = int(height * aspect_ratio)

    widget = template(plate, style)
    img, boxes = render_layout(widget, width, height)

    # Add white background
    bg = Image.new("RGB", img.size, "white")
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)

    if show_boxes:
        bg = render_bounding_boxes(bg, boxes, color="red", width=2, copy_image=False)

    return bg


def generate_all_previews(
    *,
    plate=None,
    font_path: str | None = None,
    show_boxes: bool = True,
):
    """Generate preview images for all templates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get font
    if font_path is None:
        loader = AssetLoader()
        fonts = list(loader.iter_fonts())
        if not fonts:
            print("No fonts found in assets, using system font")
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        else:
            font_path = fonts[1]

    # Generate plate if not provided
    if plate is None:
        plate = PlateGenerator.generate(separators=("", "", ""))

    style = TemplateStyle(
        font_path=font_path,
        font_size=u("30vh"),
        font_size_small=u("20vh"),
        font_size_large=u("40vh"),
        font_size_xlarge=u("50vh"),
        padding_h=u("3vw"),
        padding_v=u("5vh"),
        gap=u("4vh"),
        row_gap=u("3vh"),
        letter_spacing=u("1vh"),  # 2x the default (0.5vh)
    )

    print(f"Plate: {plate.formatted}")
    print(f"Font: {Path(font_path).name}")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 50)

    for template in ALL_TEMPLATES:
        img = render_template_preview(template, plate, style, show_boxes=show_boxes)

        # Add template info as header
        info_height = 30
        final = Image.new("RGB", (img.width, img.height + info_height), "#f0f0f0")
        draw = ImageDraw.Draw(final)

        # Template info text
        line_type = "multi-line" if template.is_multi_line else "single-line"
        bharat = " [bharat]" if template.is_bharat_only else ""
        info = f"{template.name} | AR: {template.min_aspect_ratio}-{template.max_aspect_ratio} | {line_type}{bharat}"
        draw.text((5, 5), info, fill="black")

        # Paste rendered plate
        final.paste(img, (0, info_height))

        # Save
        filename = f"{template.name}.png"
        final.save(OUTPUT_DIR / filename)
        print(f"  {template.name}")

    print("-" * 50)
    print(f"Generated {len(ALL_TEMPLATES)} previews")

    # Create a combined grid image
    create_grid_preview(OUTPUT_DIR)


def create_grid_preview(output_dir: Path):
    """Create a grid of all template previews."""
    images = []
    for template in ALL_TEMPLATES:
        img_path = output_dir / f"{template.name}.png"
        if img_path.exists():
            images.append(Image.open(img_path))

    if not images:
        return

    # Calculate grid dimensions
    cols = 3
    rows = (len(images) + cols - 1) // cols

    # Get max dimensions
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    # Create grid
    padding = 10
    grid_w = cols * max_w + (cols + 1) * padding
    grid_h = rows * max_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), "#ffffff")

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + padding)
        grid.paste(img, (x, y))

    grid.save(output_dir / "_all_templates.png")
    print("  Grid: _all_templates.png")


if __name__ == "__main__":
    generate_all_previews()
