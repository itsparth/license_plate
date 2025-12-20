"""
Generate HSRP font: extract from reference.jpg, use DIN+morphology for missing chars.
"""

from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw, ImageFont


ASSETS_DIR = Path(__file__).parent.parent / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
HSRP_DIR = FONTS_DIR / "hsrp_font"
REFERENCE_PATH = HSRP_DIR / "reference.jpg"
DIN_FONT_PATH = HSRP_DIR / "DINEngschriftStd.otf"
OUTPUT_PATH = FONTS_DIR / "hsrp.ttf"
PREVIEW_PATH = HSRP_DIR / "preview.png"

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RENDER_SIZE = 600


def extract_char_bboxes(img: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    """Auto-detect character bounding boxes from reference image."""
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 20:
            bboxes.append((x, y, w, h))

    # Group by rows
    rows_dict: dict[int, list] = {}
    for bbox in bboxes:
        row_key = bbox[1] // 50
        if row_key not in rows_dict:
            rows_dict[row_key] = []
        rows_dict[row_key].append(bbox)

    rows = [sorted(rows_dict[k], key=lambda b: b[0]) for k in sorted(rows_dict.keys())]

    char_bboxes = {}

    # Row 0: A-K (excluding I)
    if len(rows) > 0:
        row0_chars = "ABCDEFGHJK"
        if len(rows[0]) != len(row0_chars):
            print(
                f"Warning: Row 0 has {len(rows[0])} blobs, expected {len(row0_chars)}"
            )
        for i, char in enumerate(row0_chars):
            if i < len(rows[0]):
                char_bboxes[char] = rows[0][i]

    # Row 1: R-P mixed
    if len(rows) > 1:
        row1_chars = "RSUWLMNP"
        if len(rows[1]) != len(row1_chars):
            print(
                f"Warning: Row 1 has {len(rows[1])} blobs, expected {len(row1_chars)}"
            )
        for i, char in enumerate(row1_chars):
            if i < len(rows[1]):
                char_bboxes[char] = rows[1][i]

    # Row 2: 0-9
    if len(rows) > 2:
        row2_chars = "0123456789"
        if len(rows[2]) != len(row2_chars):
            print(
                f"Warning: Row 2 has {len(rows[2])} blobs, expected {len(row2_chars)}"
            )
        for i, char in enumerate(row2_chars):
            if i < len(rows[2]):
                char_bboxes[char] = rows[2][i]

    return char_bboxes


def extract_and_smooth_char(
    img: np.ndarray, bbox: tuple[int, int, int, int], target_height: int = 700
) -> list[tuple]:
    """Extract character from reference, upscale, smooth, and convert to commands."""
    x, y, w, h = bbox
    pad = 3
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    char_img = img[y1:y2, x1:x2]

    # Upscale for better quality
    scale_factor = target_height / h
    new_size = (
        int(char_img.shape[1] * scale_factor),
        int(char_img.shape[0] * scale_factor),
    )
    upscaled = cv2.resize(char_img, new_size, interpolation=cv2.INTER_CUBIC)

    # Threshold
    _, binary = cv2.threshold(upscaled, 140, 255, cv2.THRESH_BINARY_INV)

    # Smooth edges with blur + re-threshold
    blurred = cv2.GaussianBlur(binary, (5, 5), 1)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    commands = []

    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        # Simplify contour
        epsilon = 1.5
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) < 3:
            continue

        is_hole = hierarchy[i][3] != -1
        points = contour.reshape(-1, 2)

        # Convert to font coordinates (y-flip)
        img_h = smoothed.shape[0]
        font_points = [(float(px), float(img_h - py)) for px, py in points]

        # Check winding
        area = 0
        n = len(font_points)
        for j in range(n):
            x1, y1 = font_points[j]
            x2, y2 = font_points[(j + 1) % n]
            area += (x2 - x1) * (y2 + y1)
        area /= 2

        # Outer = negative area, holes = positive
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

    # Normalize to min_y = 0 and scale to target_height
    min_y = float("inf")
    max_y = float("-inf")
    for cmd, args in commands:
        if cmd in ("moveTo", "lineTo"):
            y = args[0][1]
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    if min_y != float("inf") and max_y > min_y:
        current_height = max_y - min_y
        scale = target_height / current_height

        normalized = []
        for cmd, args in commands:
            if cmd in ("moveTo", "lineTo"):
                x, y = args[0]
                new_y = (y - min_y) * scale
                normalized.append((cmd, ((x, new_y),)))
            else:
                normalized.append((cmd, args))
        return normalized

    return commands


def get_din_commands_with_fill(
    din_font: TTFont,
    char: str,
    target_height: int = 700,
    gap_px: int = 35,
    iterations: int = 10,
) -> list[tuple]:
    """Get DIN glyph with morphological gap filling."""
    glyph_set = din_font.getGlyphSet()
    cmap = din_font.getBestCmap()

    code = ord(char)
    if code not in cmap:
        return []

    glyph_name = cmap[code]
    if glyph_name not in glyph_set:
        return []

    pen = RecordingPen()
    glyph_set[glyph_name].draw(pen)
    commands = pen.value

    # Calculate raw bounds
    raw_min_x, raw_min_y = float("inf"), float("inf")
    raw_max_x, raw_max_y = float("-inf"), float("-inf")

    for cmd, args in commands:
        if cmd in ("moveTo", "lineTo"):
            x, y = args[0]
            raw_min_x, raw_min_y = min(raw_min_x, x), min(raw_min_y, y)
            raw_max_x, raw_max_y = max(raw_max_x, x), max(raw_max_y, y)
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                raw_min_x, raw_min_y = min(raw_min_x, x), min(raw_min_y, y)
                raw_max_x, raw_max_y = max(raw_max_x, x), max(raw_max_y, y)

    if raw_min_x == float("inf"):
        return []

    raw_height = raw_max_y - raw_min_y
    if raw_height == 0:
        return []

    scale = target_height / raw_height

    # Scale and shift commands
    scaled = []
    for cmd, args in commands:
        if cmd in ("moveTo", "lineTo"):
            x, y = args[0]
            scaled.append((cmd, (((x - raw_min_x) * scale, (y - raw_min_y) * scale),)))
        elif cmd in ("qCurveTo", "curveTo"):
            scaled.append(
                (
                    cmd,
                    tuple(
                        ((x - raw_min_x) * scale, (y - raw_min_y) * scale)
                        for x, y in args
                    ),
                )
            )
        else:
            scaled.append((cmd, args))

    # Render to image
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for cmd, args in scaled:
        if cmd in ("moveTo", "lineTo"):
            x, y = args[0]
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)

    if min_x == float("inf"):
        return scaled

    padding = 20
    size = RENDER_SIZE
    width, height = max_x - min_x, max_y - min_y
    render_scale = (size - 2 * padding) / max(width, height)

    img = np.zeros((size, size), dtype=np.uint8)

    # Draw contours
    contours = []
    current = []

    for cmd, args in scaled:
        if cmd == "moveTo":
            if current:
                contours.append(np.array(current, dtype=np.int32))
            px = int((args[0][0] - min_x) * render_scale + padding)
            py = int((max_y - args[0][1]) * render_scale + padding)
            current = [(px, py)]
        elif cmd == "lineTo":
            px = int((args[0][0] - min_x) * render_scale + padding)
            py = int((max_y - args[0][1]) * render_scale + padding)
            current.append((px, py))
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                px = int((x - min_x) * render_scale + padding)
                py = int((max_y - y) * render_scale + padding)
                current.append((px, py))
        elif cmd == "closePath":
            if current:
                contours.append(np.array(current, dtype=np.int32))
            current = []

    if current:
        contours.append(np.array(current, dtype=np.int32))

    cv2.fillPoly(img, contours, color=(255,))

    # Apply morphological closing per component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px, gap_px))

    num_labels, labels = cv2.connectedComponents(img)
    result = np.zeros_like(img)

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        closed = component.copy()
        for _ in range(iterations):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        result = cv2.bitwise_or(result, closed)

    # Smooth
    blurred = cv2.GaussianBlur(result, (5, 5), 1)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Convert back to commands
    contours, hierarchy = cv2.findContours(
        smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return scaled

    hierarchy = hierarchy[0]
    new_commands = []

    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        epsilon = 1.5
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) < 3:
            continue

        is_hole = hierarchy[i][3] != -1
        points = contour.reshape(-1, 2)

        # Convert back to font coordinates
        font_points = []
        for px, py in points:
            fx = (px - padding) / render_scale + min_x
            fy = max_y - (py - padding) / render_scale
            font_points.append((fx, fy))

        # Check winding
        area = 0
        n = len(font_points)
        for j in range(n):
            x1, y1 = font_points[j]
            x2, y2 = font_points[(j + 1) % n]
            area += (x2 - x1) * (y2 + y1)
        area /= 2

        if is_hole:
            if area < 0:
                font_points = font_points[::-1]
        else:
            if area > 0:
                font_points = font_points[::-1]

        new_commands.append(("moveTo", (font_points[0],)))
        for pt in font_points[1:]:
            new_commands.append(("lineTo", (pt,)))
        new_commands.append(("closePath", ()))

    return new_commands if new_commands else scaled


def measure_width(commands: list[tuple]) -> int:
    """Measure width of glyph from commands."""
    min_x, max_x = float("inf"), float("-inf")
    for cmd, args in commands:
        if cmd in ("moveTo", "lineTo"):
            x = args[0][0]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
        elif cmd in ("qCurveTo", "curveTo"):
            for x, y in args:
                min_x = min(min_x, x)
                max_x = max(max_x, x)

    if min_x == float("inf"):
        return 600

    # Add padding for advance width
    return int(max_x + 50)


def create_hybrid_font(
    gap_px: int = 35, iterations: int = 10, target_units: int = 1000
):
    """Create font using reference chars where available, DIN+fill for others."""
    # Load reference
    ref_img = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Reference not found: {REFERENCE_PATH}")

    char_bboxes = extract_char_bboxes(ref_img)
    print(f"Reference chars: {sorted(char_bboxes.keys())}")

    # Load DIN font
    din_font = TTFont(DIN_FONT_PATH)
    din_scale = target_units / din_font["head"].unitsPerEm  # type: ignore
    print(f"DIN font loaded, scale: {din_scale:.3f}")

    # Setup font builder
    fb = FontBuilder(unitsPerEm=target_units, isTTF=True)
    fb.setupGlyphOrder([".notdef", "space"] + list(CHARS))

    cmap = {ord(c): c for c in CHARS}
    cmap[ord(" ")] = "space"
    fb.setupCharacterMap(cmap)

    pen_glyphs = {
        ".notdef": TTGlyphPen(None).glyph(),
        "space": TTGlyphPen(None).glyph(),
    }
    char_widths = {".notdef": 500, "space": 300}

    for char in CHARS:
        if char in char_bboxes:
            # Extract from reference
            commands = extract_and_smooth_char(
                ref_img, char_bboxes[char], target_height=700
            )
            source = "ref"
        else:
            commands = []
            source = None

        if not commands:
            # Fallback to DIN with morphological filling
            commands = get_din_commands_with_fill(
                din_font,
                char,
                target_height=700,
                gap_px=gap_px,
                iterations=iterations,
            )
            source = "DIN"

        if not commands:
            print(f"  {char}: FAILED")
            continue

        # Build glyph
        pen = TTGlyphPen(None)
        for cmd, args in commands:
            if cmd == "moveTo":
                pen.moveTo(args[0])
            elif cmd == "lineTo":
                pen.lineTo(args[0])
            elif cmd == "closePath":
                pen.closePath()

        pen_glyphs[char] = pen.glyph()

        # Width from commands
        char_widths[char] = measure_width(commands)

        print(f"  {char}: {source} (width={char_widths[char]})")

    din_font.close()

    # Build font
    fb.setupGlyf(pen_glyphs)

    ascender, descender = 800, -200
    fb.setupHorizontalMetrics(
        {name: (char_widths.get(name, 600), 0) for name in fb.font.getGlyphOrder()}
    )
    fb.setupHorizontalHeader(ascent=ascender, descent=descender)
    fb.setupHead(unitsPerEm=target_units)
    fb.setupMaxp()
    fb.setupOS2(sTypoAscender=ascender, sTypoDescender=descender)
    fb.setupPost()
    fb.setupNameTable({"familyName": "HSRP", "styleName": "Regular"})

    fb.save(str(OUTPUT_PATH))
    print(f"\nFont saved: {OUTPUT_PATH}")

    missing_from_ref = [c for c in CHARS if c not in char_bboxes]
    print(f"\nCharacters using DIN fallback: {', '.join(sorted(missing_from_ref))}")


def generate_preview():
    """Generate preview image."""
    font_size = 72
    font = ImageFont.truetype(str(OUTPUT_PATH), font_size)

    lines = [
        "ABCDEFGHIJKLM",
        "NOPQRSTUVWXYZ",
        "0123456789",
    ]

    padding = 40
    line_height = font_size + 20
    img_width = 800
    img_height = padding * 2 + len(lines) * line_height

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        if line:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (img_width - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), line, font=font, fill="black")
        y += line_height

    img.save(PREVIEW_PATH)
    print(f"Preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HSRP hybrid font")
    parser.add_argument("--preview", action="store_true", help="Generate preview image")
    args = parser.parse_args()

    create_hybrid_font(gap_px=35, iterations=10)

    if args.preview:
        generate_preview()
