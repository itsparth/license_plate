from __future__ import annotations

from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from .base import BoundingBox, Box, Constraints, RenderContext, Widget
from .units import UnitField


class Text(Widget):
    content: str
    font_path: str
    font_size: UnitField
    color: str = "black"

    def _font(self, root_width: int, root_height: int) -> ImageFont.FreeTypeFont:
        size_px = int(
            self.font_size.resolve(
                parent=root_height, root_width=root_width, root_height=root_height
            )
        )
        return ImageFont.truetype(self.font_path, size_px)

    def layout(
        self, constraints: Constraints, root_width: int, root_height: int
    ) -> Box:
        font = self._font(root_width, root_height)
        ascent, descent = font.getmetrics()
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), self.content, font=font)

        text_width = bbox[2] - bbox[0]
        # Use tight glyph bounding box height (no extra padding)
        text_height = bbox[3] - bbox[1]

        width = int(min(constraints.max_width, text_width))
        height = int(min(constraints.max_height, text_height))
        return Box(x=0, y=0, width=width, height=height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        font = self._font(ctx.root_width, ctx.root_height)

        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        full_bbox = draw.textbbox((0, 0), self.content, font=font)
        offset_x = -full_bbox[0]
        offset_y = -full_bbox[1]

        # Draw text shifted so glyph region top-left is at box origin
        draw_x = box.x + offset_x
        draw_y = box.y + offset_y
        ctx.draw.text((draw_x, draw_y), self.content, font=font, fill=self.color)

        boxes: List[BoundingBox] = []
        cursor_x = 0

        for ch in self.content:
            # Measure character at current advance position to capture tight vertical bounds
            left, top, right, bottom = draw.textbbox((cursor_x, 0), ch, font=font)
            width = int(right - left)
            height = int(bottom - top)
            # Global coordinates incorporate drawing offset and per-char local bbox
            global_x = draw_x + left
            global_y = draw_y + top
            boxes.append(
                BoundingBox(
                    label=ch,
                    x=int(global_x),
                    y=int(global_y),
                    width=width,
                    height=height,
                )
            )
            cursor_x += width

        return boxes


class ImageWidget(Widget):
    path: str
    width: Optional[UnitField] = None
    height: Optional[UnitField] = None

    def _load(self) -> Image.Image:
        return Image.open(self.path).convert("RGBA")

    def layout(
        self, constraints: Constraints, root_width: int, root_height: int
    ) -> Box:
        img = self._load()
        w, h = img.size
        if self.width is not None:
            w = int(
                self.width.resolve(
                    parent=constraints.max_width,
                    root_width=root_width,
                    root_height=root_height,
                )
            )
        if self.height is not None:
            h = int(
                self.height.resolve(
                    parent=constraints.max_height,
                    root_width=root_width,
                    root_height=root_height,
                )
            )
        w = min(constraints.max_width, w)
        h = min(constraints.max_height, h)
        return Box(x=0, y=0, width=w, height=h)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        img = self._load().resize((box.width, box.height))
        ctx.image.paste(img, (box.x, box.y), img)
        return []
