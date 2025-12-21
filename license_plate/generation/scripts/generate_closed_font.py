"""
Generate a font with closed gaps using morphological dilation.
Takes license_plate.ttf and applies dilation to fill small gaps.
Uses 2x canvas space to allow expansion without clipping.
"""

from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"
INPUT_PATH = FONTS_DIR / "license_plate.ttf"
OUTPUT_PATH = FONTS_DIR / "license_plate_closed.ttf"

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RENDER_SIZE = 1200
PADDING = 100


def get_glyph_commands(font: TTFont, char: str) -> list[tuple]:
    """Extract drawing commands from a glyph."""
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    code = ord(char)
    if code not in cmap:
        return []

    glyph_name = cmap[code]
    if glyph_name not in glyph_set:
        return []

    pen = RecordingPen()
    glyph_set[glyph_name].draw(pen)
    return pen.value


def get_bounds(commands: list[tuple]) -> tuple[float, float, float, float]:
    """Get bounding box from commands."""
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for cmd, args in commands:
        if cmd in ("moveTo", "lineTo"):
            x, y = args[0]
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)

    return min_x, min_y, max_x, max_y


def commands_to_image(commands: list[tuple]) -> tuple[np.ndarray, dict]:
    """Render glyph commands to binary image with 2x space for expansion."""
    if not commands:
        return np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8), {}

    min_x, min_y, max_x, max_y = get_bounds(commands)
    if min_x == float("inf"):
        return np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8), {}

    width = max_x - min_x
    height = max_y - min_y
    if width == 0 or height == 0:
        return np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8), {}

    # Use 1x canvas size
    canvas_width = width
    canvas_height = height

    # Scale to fit in render size with padding
    scale = (RENDER_SIZE - 2 * PADDING) / max(canvas_width, canvas_height)

    # Center the glyph in the canvas
    offset_x = 0.0
    offset_y = 0.0

    transform = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": width,
        "height": height,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "canvas_height": canvas_height,
    }

    img = np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8)
    contours = []
    current: list[tuple[int, int]] = []

    for cmd, args in commands:
        if cmd == "moveTo":
            if current:
                contours.append(np.array(current, dtype=np.int32))
            # Shift by offset to center, then scale
            fx = (args[0][0] - min_x + offset_x) * scale + PADDING
            fy = (canvas_height - (args[0][1] - min_y + offset_y)) * scale + PADDING
            current = [(int(fx), int(fy))]
        elif cmd == "lineTo":
            fx = (args[0][0] - min_x + offset_x) * scale + PADDING
            fy = (canvas_height - (args[0][1] - min_y + offset_y)) * scale + PADDING
            current.append((int(fx), int(fy)))
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                fx = (x - min_x + offset_x) * scale + PADDING
                fy = (canvas_height - (y - min_y + offset_y)) * scale + PADDING
                current.append((int(fx), int(fy)))
        elif cmd in ("closePath", "endPath"):
            if current:
                contours.append(np.array(current, dtype=np.int32))
            current = []

    if current:
        contours.append(np.array(current, dtype=np.int32))

    if contours:
        cv2.fillPoly(img, contours, color=255)  # type: ignore[call-overload]

    return img, transform


def apply_closing(img: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Apply morphological dilation to fill V-shaped gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    result = img.copy()
    for _ in range(iterations):
        result = cv2.dilate(result, kernel)

    # Smooth edges
    blurred = cv2.GaussianBlur(result, (3, 3), 0.5)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed


def image_to_commands(img: np.ndarray, transform: dict) -> list[tuple]:
    """Convert binary image back to glyph commands."""
    if not transform:
        return []

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    commands = []

    min_x = transform["min_x"]
    min_y = transform["min_y"]
    scale = transform["scale"]
    offset_x = transform["offset_x"]
    offset_y = transform["offset_y"]
    canvas_height = transform["canvas_height"]

    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        # Use very small epsilon to preserve curves
        epsilon = 0.3
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) < 3:
            continue

        is_hole = hierarchy[i][3] != -1
        points = contour.reshape(-1, 2)

        # Convert pixel coords back to font coords
        font_points = []
        for px, py in points:
            # Reverse the transform
            fx = (px - PADDING) / scale - offset_x + min_x
            fy = min_y - offset_y + canvas_height - (py - PADDING) / scale
            font_points.append((fx, fy))

        # Check winding direction
        area = 0.0
        n = len(font_points)
        for j in range(n):
            x1, y1 = font_points[j]
            x2, y2 = font_points[(j + 1) % n]
            area += (x2 - x1) * (y2 + y1)
        area /= 2

        # Outer contours = counter-clockwise (negative area)
        # Holes = clockwise (positive area)
        if is_hole:
            if area < 0:
                font_points = font_points[::-1]
        else:
            if area > 0:
                font_points = font_points[::-1]

        commands.append(("moveTo", (font_points[0],)))
        for pt in font_points[1:]:
            commands.append(("lineTo", (pt,)))
        commands.append(("closePath", ()))

    return commands


# Only these characters get gap-filling, others preserve original curves
FILL_CHARS = {"M", "N", "W", "V", "A", "K", "X", "Y", "4"}


def process_glyph(
    font: TTFont, char: str, kernel_size: int, iterations: int
) -> list[tuple]:
    """Process a single glyph with morphological closing."""
    commands = get_glyph_commands(font, char)
    if not commands:
        return []

    # Skip characters not in FILL_CHARS - preserve original
    if char not in FILL_CHARS:
        return commands

    img, transform = commands_to_image(commands)
    if img.sum() == 0:
        return commands

    closed = apply_closing(img, kernel_size, iterations)
    new_commands = image_to_commands(closed, transform)

    return new_commands if new_commands else commands


def get_original_metrics(
    font: TTFont,
) -> tuple[dict[str, int], int, int, int, int, int]:
    """Get original widths and vertical metrics from source font."""
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]  # type: ignore[index]
    widths = {}

    for char in CHARS:
        code = ord(char)
        if code in cmap:
            glyph_name = cmap[code]
            widths[char] = hmtx[glyph_name][0]  # type: ignore[index]

    orig_hhea = font["hhea"]  # type: ignore[index]
    orig_os2 = font["OS/2"]  # type: ignore[index]

    return (
        widths,
        orig_hhea.ascent,  # type: ignore[attr-defined]
        orig_hhea.descent,  # type: ignore[attr-defined]
        orig_os2.sTypoAscender,  # type: ignore[attr-defined]
        orig_os2.sTypoDescender,  # type: ignore[attr-defined]
        orig_os2.sTypoLineGap,  # type: ignore[attr-defined]
    )


def create_closed_font(kernel_size: int = 15, iterations: int = 3):
    """Create font with closed gaps."""
    print(f"Loading: {INPUT_PATH}")
    font = TTFont(INPUT_PATH)

    units_per_em = font["head"].unitsPerEm  # type: ignore[union-attr]
    print(f"Units per em: {units_per_em}")

    # Get metrics before processing
    original_widths, ascender, descender, typo_asc, typo_desc, typo_gap = (
        get_original_metrics(font)
    )

    fb = FontBuilder(unitsPerEm=units_per_em, isTTF=True)
    fb.setupGlyphOrder([".notdef", "space"] + list(CHARS))

    cmap = {ord(c): c for c in CHARS}
    cmap[ord(" ")] = "space"
    fb.setupCharacterMap(cmap)

    pen_glyphs = {
        ".notdef": TTGlyphPen(None).glyph(),
        "space": TTGlyphPen(None).glyph(),
    }
    char_widths: dict[str, int] = {".notdef": 500, "space": 300}

    print(f"\nProcessing glyphs (kernel={kernel_size}, iterations={iterations}):")

    for char in CHARS:
        commands = process_glyph(font, char, kernel_size, iterations)

        if not commands:
            print(f"  {char}: FAILED")
            continue

        pen = TTGlyphPen(None)
        for cmd, args in commands:
            if cmd == "moveTo":
                pen.moveTo(args[0])
            elif cmd == "lineTo":
                pen.lineTo(args[0])
            elif cmd == "qCurveTo":
                pen.qCurveTo(*args)
            elif cmd == "curveTo":
                pen.curveTo(*args)
            elif cmd == "closePath":
                pen.closePath()
            elif cmd == "endPath":
                pen.endPath()

        pen_glyphs[char] = pen.glyph()
        char_widths[char] = original_widths.get(char, 600)
        print(f"  {char}: OK (width={char_widths[char]})")

    font.close()

    # Build font
    fb.setupGlyf(pen_glyphs)
    fb.setupHorizontalMetrics(
        {name: (char_widths.get(name, 600), 0) for name in fb.font.getGlyphOrder()}
    )
    fb.setupHorizontalHeader(ascent=ascender, descent=descender)
    fb.setupHead(unitsPerEm=units_per_em)
    fb.setupMaxp()
    fb.setupOS2(
        sTypoAscender=typo_asc,
        sTypoDescender=typo_desc,
        sTypoLineGap=typo_gap,
    )
    fb.setupPost()
    fb.setupNameTable({"familyName": "License Plate Closed", "styleName": "Regular"})

    fb.save(str(OUTPUT_PATH))
    print(f"\nFont saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate font with closed gaps")
    parser.add_argument(
        "--kernel", type=int, default=25, help="Closing kernel size (default: 25)"
    )
    parser.add_argument(
        "--iterations", type=int, default=4, help="Closing iterations (default: 4)"
    )
    args = parser.parse_args()

    create_closed_font(kernel_size=args.kernel, iterations=args.iterations)
