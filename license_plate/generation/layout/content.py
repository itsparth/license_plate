from __future__ import annotations

from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from .base import BoundingBox, Box, Constraints, RenderContext, Widget
from .units import UnitField, px


class Text(Widget):
    content: str
    font_path: str
    font_size: UnitField
    color: str = "black"
    letter_spacing: UnitField = px(0)  # Extra spacing between characters

    def _font(self, root_width: int, root_height: int) -> ImageFont.FreeTypeFont:
        size_px = int(
            self.font_size.resolve(
                parent=root_height, root_width=root_width, root_height=root_height
            )
        )
        return ImageFont.truetype(self.font_path, size_px)

    def _get_letter_spacing(self, root_width: int, root_height: int) -> int:
        return int(
            self.letter_spacing.resolve(
                parent=root_height, root_width=root_width, root_height=root_height
            )
        )

    def layout(
        self, constraints: Constraints, root_width: int, root_height: int
    ) -> Box:
        font = self._font(root_width, root_height)
        spacing = self._get_letter_spacing(root_width, root_height)
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        # Calculate total width with letter spacing
        total_width = 0
        max_height = 0
        for i, ch in enumerate(self.content):
            bbox = draw.textbbox((0, 0), ch, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            total_width += char_width
            if i < len(self.content) - 1:
                total_width += spacing
            max_height = max(max_height, char_height)

        width = int(min(constraints.max_width, total_width))
        height = int(min(constraints.max_height, max_height))
        return Box(x=0, y=0, width=width, height=height)

    def render(self, box: Box, ctx: RenderContext) -> List[BoundingBox]:
        font = self._font(ctx.root_width, ctx.root_height)
        spacing = self._get_letter_spacing(ctx.root_width, ctx.root_height)

        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        # Find baseline offset from first character
        if self.content:
            first_bbox = draw.textbbox((0, 0), self.content[0], font=font)
            offset_y = -first_bbox[1]
        else:
            offset_y = 0

        boxes: List[BoundingBox] = []
        cursor_x = box.x

        for i, ch in enumerate(self.content):
            # Get character bounds
            char_bbox = draw.textbbox((0, 0), ch, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            char_offset_x = -char_bbox[0]
            # char_offset_y = -char_bbox[1]

            # Draw character
            draw_x = cursor_x + char_offset_x
            draw_y = box.y + offset_y
            ctx.draw.text((draw_x, draw_y), ch, font=font, fill=self.color)

            # Record bounding box
            boxes.append(
                BoundingBox(
                    label=ch,
                    x=int(cursor_x),
                    y=int(box.y + offset_y + char_bbox[1]),
                    width=int(char_width),
                    height=int(char_height),
                )
            )

            # Advance cursor
            cursor_x += char_width
            if i < len(self.content) - 1:
                cursor_x += spacing

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
