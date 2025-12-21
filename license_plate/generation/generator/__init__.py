from __future__ import annotations

from .asset_loader import (
    AssetLoader,
    LogoAsset,
    VehicleImageAsset,
    VehiclePlateBBox,
)
from .augment import (
    create_augmentation_pipeline,
    create_effects_pipeline,
    create_geometric_pipeline,
    tight_crop_around_bboxes,
)
from .color import get_contrasting_color_with_alpha, sample_plate_color
from .plate_generator import (
    IndianLicensePlate,
    PlateGenerator,
    PLATE_DIGITS,
    PLATE_LETTERS,
    STATE_CODES,
)
from .templates import (
    Template,
    TemplateStyle,
    TemplateFunc,
    ALL_TEMPLATES,
    SINGLE_LINE_TEMPLATES,
    MULTI_LINE_TEMPLATES,
    BHARAT_TEMPLATES,
    TEMPLATE_SIMPLE,
    TEMPLATE_LARGE_DIGITS,
    TEMPLATE_MULTI_SIZE_DIGITS,
    TEMPLATE_STATE_DISTRICT_STACKED,
    TEMPLATE_STATE_DISTRICT_LETTERS_STACKED,
    TEMPLATE_DIGITS_SECOND_LINE,
    TEMPLATE_LETTERS_DIGITS_SECOND_LINE,
    TEMPLATE_EQUAL_LINES,
    TEMPLATE_TRIPLE_LINES,
    TEMPLATE_COMPACT_TWO_LINE,
    TEMPLATE_LARGE_LETTERS_CENTER,
    TEMPLATE_BHARAT_SIMPLE,
    TEMPLATE_BHARAT_TWO_LINE,
    get_templates_for_aspect_ratio,
    random_template,
)

__all__ = [
    "AssetLoader",
    "IndianLicensePlate",
    "LogoAsset",
    "PLATE_DIGITS",
    "PLATE_LETTERS",
    "PlateGenerator",
    "STATE_CODES",
    "VehicleImageAsset",
    "VehiclePlateBBox",
    # Augmentation
    "create_augmentation_pipeline",
    "create_effects_pipeline",
    "create_geometric_pipeline",
    "tight_crop_around_bboxes",
    # Color
    "get_contrasting_color_with_alpha",
    "sample_plate_color",
    # Templates
    "Template",
    "TemplateStyle",
    "TemplateFunc",
    "ALL_TEMPLATES",
    "SINGLE_LINE_TEMPLATES",
    "MULTI_LINE_TEMPLATES",
    "BHARAT_TEMPLATES",
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
    "get_templates_for_aspect_ratio",
    "random_template",
]
