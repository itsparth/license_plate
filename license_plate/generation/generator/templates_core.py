from __future__ import annotations

import random
from typing import Callable, Literal, TypeAlias

from pydantic import BaseModel

from ..layout import Column, ImageWidget, Padding, Row, Text, Widget
from .asset_loader import LogoAsset

CrossAlign = Literal["start", "center", "end"]
LogoPosition = Literal["left", "right", "top", "bottom"]


class TemplateStyle(BaseModel):
    font_path: str
    font_size: int = 50
    font_size_small: int = 35
    font_size_large: int = 60
    font_size_xlarge: int = 70
    color: str = "black"
    padding_h: int = 10
    padding_v: int = 10
    gap: int = 5
    row_gap: int = 5
    letter_spacing: int = 2

    def text(self, content: str, size: int | None = None) -> Text:
        return Text(
            content=content,
            font_path=self.font_path,
            font_size=size or self.font_size,
            color=self.color,
            letter_spacing=self.letter_spacing,
        )

    def text_small(self, content: str) -> Text:
        return self.text(content, self.font_size_small)

    def text_large(self, content: str) -> Text:
        return self.text(content, self.font_size_large)

    def text_xlarge(self, content: str) -> Text:
        return self.text(content, self.font_size_xlarge)

    def padding(self, child: Widget) -> Padding:
        return Padding(
            left=self.padding_h,
            right=self.padding_h,
            top=self.padding_v,
            bottom=self.padding_v,
            child=child,
        )

    def row(self, children: list[Widget], cross_align: CrossAlign = "center") -> Row:
        return Row(children=children, gap=self.gap, cross_axis_alignment=cross_align)

    def column(
        self, children: list[Widget], cross_align: CrossAlign = "center"
    ) -> Column:
        return Column(
            children=children, gap=self.row_gap, cross_axis_alignment=cross_align
        )


# Import here to avoid circular import
from .plate_generator import IndianLicensePlate  # noqa: E402

TemplateFunc: TypeAlias = Callable[[IndianLicensePlate, TemplateStyle], Widget]


class Template(BaseModel):
    """Template descriptor with render function and aspect ratio bounds"""

    name: str
    func: TemplateFunc  # type: ignore[valid-type]
    min_aspect_ratio: float  # width / height (minimum)
    max_aspect_ratio: float  # width / height (maximum)
    is_multi_line: bool = False
    is_bharat_only: bool = False

    def __call__(self, plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
        return self.func(plate, style)


def random_cross_align() -> CrossAlign:
    return random.choice(["start", "center", "end"])


def get_logo_position(logo: LogoAsset, squarish_threshold: float = 2.0) -> LogoPosition:
    """Squarish logos (AR < threshold) go left/right, wide logos go top/bottom"""
    if logo.aspect_ratio < squarish_threshold:
        return random.choice(["left", "right"])
    return random.choice(["top", "bottom"])


def wrap_with_logo(
    plate_widget: Widget,
    logo: LogoAsset,
    style: TemplateStyle,
) -> Widget:
    """Wrap a plate template with a logo based on its aspect ratio

    Logo maintains aspect ratio and takes max 20% of space.
    """
    position = get_logo_position(logo)

    # Calculate logo dimensions maintaining aspect ratio, max 20% of space
    if position in ("left", "right"):
        # Horizontal: base height on font size, calculate width from AR
        # Max width ~20% of plate (assume plate ~5x font_size wide)
        max_logo_w = int(style.font_size * 1.0)  # 20% of ~5*font_size
        logo_h = style.font_size
        logo_w = int(logo_h * logo.aspect_ratio)
        if logo_w > max_logo_w:
            logo_w = max_logo_w
            logo_h = (
                int(logo_w / logo.aspect_ratio) if logo.aspect_ratio > 0 else logo_w
            )
    else:
        # Vertical: base on expected plate width, max 20% height
        # Max height ~20% of plate height (assume plate ~2*font_size tall)
        max_logo_h = int(style.font_size * 0.4)  # 20% of ~2*font_size
        logo_w = int(style.font_size * 2)
        logo_h = int(logo_w / logo.aspect_ratio) if logo.aspect_ratio > 0 else logo_w
        if logo_h > max_logo_h:
            logo_h = max_logo_h
            logo_w = int(logo_h * logo.aspect_ratio)

    logo_img = ImageWidget(path=str(logo.path), width=logo_w, height=logo_h)

    if position == "left":
        inner = Row(
            children=[logo_img, plate_widget],
            gap=style.gap,
            cross_axis_alignment="center",
        )
    elif position == "right":
        inner = Row(
            children=[plate_widget, logo_img],
            gap=style.gap,
            cross_axis_alignment="center",
        )
    elif position == "top":
        inner = Column(
            children=[logo_img, plate_widget],
            gap=style.row_gap,
            cross_axis_alignment="center",
        )
    else:  # bottom
        inner = Column(
            children=[plate_widget, logo_img],
            gap=style.row_gap,
            cross_axis_alignment="center",
        )

    return style.padding(inner)


__all__ = [
    "CrossAlign",
    "LogoPosition",
    "Template",
    "TemplateFunc",
    "TemplateStyle",
    "get_logo_position",
    "random_cross_align",
    "wrap_with_logo",
]
