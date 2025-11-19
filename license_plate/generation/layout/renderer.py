from __future__ import annotations

from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base import BoundingBox, Box, Constraints, RenderContext, Widget


def render_layout(
    widget: Widget, width: int, height: int
) -> Tuple[Image.Image, List[BoundingBox]]:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    constraints = Constraints(
        min_width=width,
        max_width=width,
        min_height=height,
        max_height=height,
    )
    widget.layout(constraints, width, height)
    render_ctx = RenderContext(
        image=image, draw=draw, root_width=width, root_height=height
    )
    boxes = widget.render(Box(x=0, y=0, width=width, height=height), render_ctx)
    return image, boxes


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
