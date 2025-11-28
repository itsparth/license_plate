"""Color utilities for plate generation."""

import random

import numpy as np
from PIL import Image


def sample_plate_color(
    img: Image.Image, left: int, top: int, width: int, height: int
) -> tuple[int, int, int]:
    """Sample the dominant color from the plate region."""
    plate_region = img.crop((left, top, left + width, top + height))
    small = plate_region.resize((10, 10))
    arr = np.array(small)
    avg_r = int(arr[:, :, 0].mean())
    avg_g = int(arr[:, :, 1].mean())
    avg_b = int(arr[:, :, 2].mean())
    return (avg_r, avg_g, avg_b)


def get_contrasting_color_with_alpha(bg_color: tuple[int, int, int]) -> str:
    """Generate a contrasting font color with transparency based on background."""
    r, g, b = bg_color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    if luminance > 0.5:
        # Light background -> dark text
        base_r = random.randint(0, 40)
        base_g = random.randint(0, 40)
        base_b = random.randint(0, 40)
    else:
        # Dark background -> light text
        base_r = random.randint(200, 255)
        base_g = random.randint(200, 255)
        base_b = random.randint(200, 255)

    # Add color variation for realism
    if random.random() < 0.3:
        # Warm tint (yellowish)
        base_r = min(255, base_r + random.randint(10, 30))
        base_g = min(255, base_g + random.randint(5, 20))
    elif random.random() < 0.2:
        # Cool tint (bluish)
        base_b = min(255, base_b + random.randint(10, 25))

    # Transparency for blending (alpha 180-230)
    alpha = random.randint(180, 230)

    return f"#{base_r:02x}{base_g:02x}{base_b:02x}{alpha:02x}"


__all__ = ["sample_plate_color", "get_contrasting_color_with_alpha"]
