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
    min_render_scale: float = 2.0,
) -> Tuple[Image.Image, List[BoundingBox]]:
    """
    Render widget larger than target, then scale DOWN to fit (never up).

    1. Measure natural content size
    2. Calculate render scale to ensure rendered size >= target
    3. Render at larger size for crisp downscaling
    4. Scale down to fit target bounds (with padding)
    5. Center on transparent target-sized image
    """
    unconstrained = Constraints(
        min_width=0, max_width=10000, min_height=0, max_height=10000
    )

    # First measure at base scale to get natural proportions
    measured = widget.layout(unconstrained, base_scale)
    natural_w = max(1, measured.width)
    natural_h = max(1, measured.height)

    # Calculate usable target area
    usable_w = target_width * (1 - 2 * padding_ratio)
    usable_h = target_height * (1 - 2 * padding_ratio)

    # Calculate scale needed so rendered size >= target (for downscaling only)
    # We want: natural_w * render_scale >= usable_w and natural_h * render_scale >= usable_h
    scale_for_width = usable_w / natural_w if natural_w > 0 else 1.0
    scale_for_height = usable_h / natural_h if natural_h > 0 else 1.0
    # Use the larger scale to ensure both dimensions are covered
    needed_scale = max(scale_for_width, scale_for_height)
    # Ensure we render at least min_render_scale times larger for quality
    render_scale = max(base_scale * needed_scale * min_render_scale, base_scale)

    # Re-measure and render at the computed scale
    measured = widget.layout(unconstrained, render_scale)
    render_w = max(1, measured.width)
    render_h = max(1, measured.height)

    render_img = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(render_img)
    render_ctx = RenderContext(image=render_img, draw=draw, scale=render_scale)
    boxes = widget.render(Box(x=0, y=0, width=render_w, height=render_h), render_ctx)

    # Calculate scale to fit within target bounds (should be <= 1.0 now)
    fit_scale = min(usable_w / render_w, usable_h / render_h)

    # Scale image down
    scaled_w = max(1, int(render_w * fit_scale))
    scaled_h = max(1, int(render_h * fit_scale))
    scaled_img = render_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

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
