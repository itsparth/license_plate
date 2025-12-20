from __future__ import annotations

import random

from ..layout import Column, Container, Padding, Row, Widget
from .asset_loader import LogoAsset
from .plate_generator import IndianLicensePlate
from .templates_core import (
    CrossAlign,
    LogoPosition,
    Template,
    TemplateFunc,
    TemplateStyle,
    random_cross_align,
    wrap_with_logo,
)


# -----------------------------------------------------------------------------
# Single-line templates
# -----------------------------------------------------------------------------


def _simple(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """Single line with all characters"""
    return style.padding(style.text(plate.formatted))


TEMPLATE_SIMPLE = Template(
    name="simple", func=_simple, min_aspect_ratio=4.0, max_aspect_ratio=8.0
)


def _large_digits(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State+district+letters in normal size, digits in large font"""
    s1, s2, s3 = plate.separators
    prefix = f"{plate.state_code}{s1}{plate.district_formatted}{s2}{plate.letters}{s3}"
    return style.padding(
        Row(
            children=[style.text(prefix), style.text_large(plate.digits)],
            gap=style.gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_LARGE_DIGITS = Template(
    name="large_digits", func=_large_digits, min_aspect_ratio=4.5, max_aspect_ratio=8.0
)


def _multi_size_digits(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State+district+letters normal, first digit(s) small, rest large"""
    s1, s2, s3 = plate.separators
    prefix = f"{plate.state_code}{s1}{plate.district_formatted}{s2}{plate.letters}{s3}"
    split = random.randint(1, 2)
    first_digits = plate.digits[:split]
    rest_digits = plate.digits[split:]
    return style.padding(
        Row(
            children=[
                style.text(prefix),
                style.text_small(first_digits),
                style.text_large(rest_digits),
            ],
            gap=style.gap,
            cross_axis_alignment="end",
        )
    )


TEMPLATE_MULTI_SIZE_DIGITS = Template(
    name="multi_size_digits",
    func=_multi_size_digits,
    min_aspect_ratio=4.5,
    max_aspect_ratio=8.0,
)


def _state_district_stacked(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State and district stacked vertically, letters+digits to the right"""
    s3 = plate.separators[2]
    return style.padding(
        Row(
            children=[
                Column(
                    children=[
                        style.text_small(plate.state_code),
                        style.text_small(plate.district_formatted),
                    ],
                    gap=style.row_gap,
                    cross_axis_alignment="center",
                ),
                style.text_large(f"{plate.letters}{s3}{plate.digits}"),
            ],
            gap=style.gap,
            cross_axis_alignment="center",
        )
    )


TEMPLATE_STATE_DISTRICT_STACKED = Template(
    name="state_district_stacked",
    func=_state_district_stacked,
    min_aspect_ratio=4.0,
    max_aspect_ratio=7.0,
)


def _state_district_letters_stacked(
    plate: IndianLicensePlate, style: TemplateStyle
) -> Widget:
    """State+district and letters stacked, digits large to the right"""
    s1 = plate.separators[0]
    return style.padding(
        Row(
            children=[
                Column(
                    children=[
                        style.text_small(
                            f"{plate.state_code}{s1}{plate.district_formatted}"
                        ),
                        style.text_small(plate.letters),
                    ],
                    gap=style.row_gap,
                    cross_axis_alignment="center",
                ),
                style.text_large(plate.digits),
            ],
            gap=style.gap,
            cross_axis_alignment="center",
        )
    )


TEMPLATE_STATE_DISTRICT_LETTERS_STACKED = Template(
    name="state_district_letters_stacked",
    func=_state_district_letters_stacked,
    min_aspect_ratio=3.5,
    max_aspect_ratio=6.0,
)


# -----------------------------------------------------------------------------
# Multi-line templates
# -----------------------------------------------------------------------------


def _digits_second_line(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State+district+letters on first line, digits large on second line"""
    s1, s2, _ = plate.separators
    first_line = f"{plate.state_code}{s1}{plate.district_formatted}{s2}{plate.letters}"
    digit_style = random.choice([style.text_large, style.text_xlarge])
    return style.padding(
        Column(
            children=[style.text(first_line), digit_style(plate.digits)],
            gap=style.row_gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_DIGITS_SECOND_LINE = Template(
    name="digits_second_line",
    func=_digits_second_line,
    min_aspect_ratio=1.0,
    max_aspect_ratio=4.0,
    is_multi_line=True,
)


def _letters_digits_second_line(
    plate: IndianLicensePlate, style: TemplateStyle
) -> Widget:
    """State+district on first line, letters+digits on second line"""
    s1, _, s3 = plate.separators
    first_line = f"{plate.state_code}{s1}{plate.district_formatted}"
    second_line = f"{plate.letters}{s3}{plate.digits}"
    return style.padding(
        Column(
            children=[style.text(first_line), style.text(second_line)],
            gap=style.row_gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_LETTERS_DIGITS_SECOND_LINE = Template(
    name="letters_digits_second_line",
    func=_letters_digits_second_line,
    min_aspect_ratio=1.0,
    max_aspect_ratio=4.0,
    is_multi_line=True,
)


def _equal_lines(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """Split plate roughly in half across two lines"""
    chars = plate.formatted
    mid = len(chars) // 2
    first_half = chars[:mid]
    second_half = chars[mid:]
    return style.padding(
        Column(
            children=[style.text(first_half), style.text(second_half)],
            gap=style.row_gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_EQUAL_LINES = Template(
    name="equal_lines",
    func=_equal_lines,
    min_aspect_ratio=1.0,
    max_aspect_ratio=4.0,
    is_multi_line=True,
)


def _triple_lines(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State+district, letters, and digits on three separate lines"""
    s1 = plate.separators[0]
    return style.padding(
        Column(
            children=[
                style.text(f"{plate.state_code}{s1}{plate.district_formatted}"),
                style.text(plate.letters),
                style.text(plate.digits),
            ],
            gap=style.row_gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_TRIPLE_LINES = Template(
    name="triple_lines",
    func=_triple_lines,
    min_aspect_ratio=0.5,
    max_aspect_ratio=1.5,
    is_multi_line=True,
)


def _compact_two_line(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """State on first line, rest on second line"""
    s1, s2, s3 = plate.separators
    second = f"{plate.district_formatted}{s2}{plate.letters}{s3}{plate.digits}"
    return style.padding(
        Container(
            child=Column(
                children=[style.text_small(plate.state_code), style.text(second)],
                gap=style.row_gap,
                cross_axis_alignment="center",
            )
        )
    )


TEMPLATE_COMPACT_TWO_LINE = Template(
    name="compact_two_line",
    func=_compact_two_line,
    min_aspect_ratio=1.0,
    max_aspect_ratio=4.0,
    is_multi_line=True,
)


def _large_letters_center(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """Three lines: state+district small, letters large center, digits normal"""
    s1 = plate.separators[0]
    return style.padding(
        Column(
            children=[
                style.text_small(f"{plate.state_code}{s1}{plate.district_formatted}"),
                style.text_xlarge(plate.letters),
                style.text(plate.digits),
            ],
            gap=style.row_gap,
            cross_axis_alignment="center",
        )
    )


TEMPLATE_LARGE_LETTERS_CENTER = Template(
    name="large_letters_center",
    func=_large_letters_center,
    min_aspect_ratio=0.5,
    max_aspect_ratio=1.5,
    is_multi_line=True,
)


# -----------------------------------------------------------------------------
# Bharat series templates
# -----------------------------------------------------------------------------


def _bharat_simple(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """Bharat series: YYBH####XX format in single line"""
    return style.padding(style.text(plate.formatted))


TEMPLATE_BHARAT_SIMPLE = Template(
    name="bharat_simple",
    func=_bharat_simple,
    min_aspect_ratio=4.0,
    max_aspect_ratio=8.0,
    is_bharat_only=True,
)


def _bharat_two_line(plate: IndianLicensePlate, style: TemplateStyle) -> Widget:
    """Bharat series: YY BH on first line, digits and letters on second"""
    s1, _, s3 = plate.separators
    first = f"{plate.district_formatted}{s1}{plate.state_code}"
    second = f"{plate.digits}{s3}{plate.letters}"
    return style.padding(
        Column(
            children=[style.text(first), style.text_large(second)],
            gap=style.row_gap,
            cross_axis_alignment=random_cross_align(),
        )
    )


TEMPLATE_BHARAT_TWO_LINE = Template(
    name="bharat_two_line",
    func=_bharat_two_line,
    min_aspect_ratio=1.0,
    max_aspect_ratio=4.0,
    is_multi_line=True,
    is_bharat_only=True,
)


# -----------------------------------------------------------------------------
# Template registry
# -----------------------------------------------------------------------------

SINGLE_LINE_TEMPLATES: list[Template] = [
    TEMPLATE_SIMPLE,
    TEMPLATE_LARGE_DIGITS,
    TEMPLATE_MULTI_SIZE_DIGITS,
    TEMPLATE_STATE_DISTRICT_STACKED,
    TEMPLATE_STATE_DISTRICT_LETTERS_STACKED,
]

MULTI_LINE_TEMPLATES: list[Template] = [
    TEMPLATE_DIGITS_SECOND_LINE,
    TEMPLATE_LETTERS_DIGITS_SECOND_LINE,
    TEMPLATE_EQUAL_LINES,
    TEMPLATE_TRIPLE_LINES,
    TEMPLATE_COMPACT_TWO_LINE,
    TEMPLATE_LARGE_LETTERS_CENTER,
]

BHARAT_TEMPLATES: list[Template] = [
    TEMPLATE_BHARAT_SIMPLE,
    TEMPLATE_BHARAT_TWO_LINE,
]

ALL_TEMPLATES: list[Template] = (
    SINGLE_LINE_TEMPLATES + MULTI_LINE_TEMPLATES + BHARAT_TEMPLATES
)


def get_templates_for_aspect_ratio(
    aspect_ratio: float,
    *,
    is_bharat: bool = False,
    single_line_only: bool = False,
    multi_line_only: bool = False,
) -> list[Template]:
    """Get templates that fit within the given aspect ratio (min <= AR <= max)"""
    if is_bharat:
        candidates = BHARAT_TEMPLATES
    elif single_line_only:
        candidates = SINGLE_LINE_TEMPLATES
    elif multi_line_only:
        candidates = MULTI_LINE_TEMPLATES
    else:
        candidates = SINGLE_LINE_TEMPLATES + MULTI_LINE_TEMPLATES

    return [
        t
        for t in candidates
        if t.min_aspect_ratio <= aspect_ratio <= t.max_aspect_ratio
    ]


def random_template(
    plate: IndianLicensePlate,
    style: TemplateStyle,
    *,
    aspect_ratio: float | None = None,
    single_line_only: bool = False,
    multi_line_only: bool = False,
    logo: LogoAsset | None = None,
) -> Widget:
    """Select and apply a random template based on plate type and aspect ratio

    Args:
        logo: Optional LogoAsset to add. Squarish logos (AR < 2) go left/right,
              wide logos (AR >= 2) go top/bottom.
    """
    if aspect_ratio is not None:
        templates = get_templates_for_aspect_ratio(
            aspect_ratio,
            is_bharat=plate.is_bharat_series,
            single_line_only=single_line_only,
            multi_line_only=multi_line_only,
        )
        if not templates:
            # Fallback to template with lowest aspect ratio requirement
            if plate.is_bharat_series:
                templates = [min(BHARAT_TEMPLATES, key=lambda t: t.min_aspect_ratio)]
            elif multi_line_only:
                templates = [
                    min(MULTI_LINE_TEMPLATES, key=lambda t: t.min_aspect_ratio)
                ]
            else:
                templates = [min(ALL_TEMPLATES, key=lambda t: t.min_aspect_ratio)]
    else:
        if plate.is_bharat_series:
            templates = BHARAT_TEMPLATES
        elif single_line_only:
            templates = SINGLE_LINE_TEMPLATES
        elif multi_line_only:
            templates = MULTI_LINE_TEMPLATES
        else:
            templates = SINGLE_LINE_TEMPLATES + MULTI_LINE_TEMPLATES

    template = random.choice(templates)
    widget = template(plate, style)

    if logo is not None:
        # Remove existing padding to avoid double padding
        if isinstance(widget, Padding):
            inner = widget.child
        else:
            inner = widget
        widget = wrap_with_logo(inner, logo, style)

    return widget


__all__ = [
    # Re-export from templates_core
    "CrossAlign",
    "LogoPosition",
    "Template",
    "TemplateFunc",
    "TemplateStyle",
    # Template instances
    "TEMPLATE_SIMPLE",
    "TEMPLATE_LARGE_DIGITS",
    "TEMPLATE_MULTI_SIZE_DIGITS",
    "TEMPLATE_STATE_DISTRICT_STACKED",
    "TEMPLATE_STATE_DISTRICT_LETTERS_STACKED",
    "TEMPLATE_DIGITS_SECOND_LINE",
    "TEMPLATE_LETTERS_DIGITS_SECOND_LINE",
    "TEMPLATE_EQUAL_LINES",
    "TEMPLATE_TRIPLE_LINES",
    "TEMPLATE_COMPACT_TWO_LINE",
    "TEMPLATE_LARGE_LETTERS_CENTER",
    "TEMPLATE_BHARAT_SIMPLE",
    "TEMPLATE_BHARAT_TWO_LINE",
    # Template lists
    "SINGLE_LINE_TEMPLATES",
    "MULTI_LINE_TEMPLATES",
    "BHARAT_TEMPLATES",
    "ALL_TEMPLATES",
    # Functions
    "get_templates_for_aspect_ratio",
    "random_template",
]
