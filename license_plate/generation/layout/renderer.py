from __future__ import annotations

from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base import BoundingBox, Box, Constraints, RenderContext, Widget


def render_layout(
    widget: Widget,
    width: int,
    height: int,
    *,
    scale: float = 1.0,
) -> Tuple[Image.Image, List[BoundingBox]]:
    """Render widget at specified size with given scale factor."""
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    constraints = Constraints(
        min_width=width,
        max_width=width,
        min_height=height,
        max_height=height,
    )
    widget.layout(constraints, scale)
    render_ctx = RenderContext(image=image, draw=draw, scale=scale)
    boxes = widget.render(Box(x=0, y=0, width=width, height=height), render_ctx)
    return image, boxes


def render_tight(
    widget: Widget,
    *,
    scale: float = 1.0,
) -> Tuple[Image.Image, List[BoundingBox]]:
    """Render widget at its natural (tight) size with no extra padding."""
    # Measure natural size
    unconstrained = Constraints(
        min_width=0, max_width=10000, min_height=0, max_height=10000
    )
    measured = widget.layout(unconstrained, scale)
    natural_w = max(1, measured.width)
    natural_h = max(1, measured.height)

    # Render at natural size
    image = Image.new("RGBA", (natural_w, natural_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    render_ctx = RenderContext(image=image, draw=draw, scale=scale)
    boxes = widget.render(Box(x=0, y=0, width=natural_w, height=natural_h), render_ctx)

    return image, boxes


def render_tight_and_scale(
    widget: Widget,
    target_width: int,
    target_height: int,
    *,
    base_scale: float = 1.0,
    padding_ratio: float = 0.05,
) -> Tuple[Image.Image, List[BoundingBox]]:
    """
    Render widget at its natural size, then scale and center to fit target bounds.

    1. Measure natural content size with unconstrained layout
    2. Render at natural size
    3. Scale down to fit target bounds (with padding)
    4. Center on transparent target-sized image
    5. Scale bounding boxes accordingly
    """
    # Measure natural size with base scale
    unconstrained = Constraints(
        min_width=0, max_width=10000, min_height=0, max_height=10000
    )
    measured = widget.layout(unconstrained, base_scale)
    natural_w = max(1, measured.width)
    natural_h = max(1, measured.height)

    # Render at natural size
    natural_img = Image.new("RGBA", (natural_w, natural_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(natural_img)
    render_ctx = RenderContext(image=natural_img, draw=draw, scale=base_scale)
    boxes = widget.render(Box(x=0, y=0, width=natural_w, height=natural_h), render_ctx)

    # Calculate scale to fit within target bounds (with padding)
    usable_w = target_width * (1 - 2 * padding_ratio)
    usable_h = target_height * (1 - 2 * padding_ratio)
    fit_scale = min(usable_w / natural_w, usable_h / natural_h)

    # Scale image
    scaled_w = max(1, int(natural_w * fit_scale))
    scaled_h = max(1, int(natural_h * fit_scale))
    scaled_img = natural_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    # Center on target
    target_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    offset_x = (target_width - scaled_w) // 2
    offset_y = (target_height - scaled_h) // 2
    target_img.paste(scaled_img, (offset_x, offset_y), scaled_img)

    # Scale and offset bounding boxes
    scaled_boxes = [
        BoundingBox(
            label=box.label,
            x=int(box.x * fit_scale) + offset_x,
            y=int(box.y * fit_scale) + offset_y,
            width=max(1, int(box.width * fit_scale)),
            height=max(1, int(box.height * fit_scale)),
        )
        for box in boxes
    ]

    return target_img, scaled_boxes


def render_bounding_boxes(
    image: Image.Image,
    boxes: Iterable[BoundingBox],
    *,
    color: str = "red",
    width: int = 2,
    draw_labels: bool = False,
    label_fill: str = "white",
    copy_image: bool = False,
) -> Image.Image:
    target = image.copy() if copy_image else image
    draw = ImageDraw.Draw(target)
    font = ImageFont.load_default() if draw_labels else None

    for box in boxes:
        x1 = box.x
        y1 = box.y
        x2 = box.x + box.width
        y2 = box.y + box.height
        for offset in range(width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )
        if draw_labels and box.label:
            draw.text((x1 + 2, y1 + 2), box.label, fill=label_fill, font=font)

    return target
