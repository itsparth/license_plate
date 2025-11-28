#!/usr/bin/env python3
"""Generate preview images for all downloaded fonts."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT_SIZE = 48
CHAR_SPACING = 10
PADDING = 20
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)


def generate_preview(font_path: Path, output_path: Path) -> None:
    """Generate a preview image for a single font."""
    try:
        font = ImageFont.truetype(str(font_path), FONT_SIZE)
    except Exception as e:
        print(f"✗ {font_path.stem}: {e}")
        return

    # Calculate dimensions
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    char_widths = [
        dummy_draw.textbbox((0, 0), char, font=font)[2] for char in CHARACTERS
    ]
    max_char_width = max(char_widths)
    char_height = dummy_draw.textbbox((0, 0), "A", font=font)[3]

    # 10 chars per row (A-J, K-T, U-Z0-9, 0-9 on separate rows)
    rows = [CHARACTERS[:10], CHARACTERS[10:20], CHARACTERS[20:30], CHARACTERS[30:]]

    img_width = int(10 * (max_char_width + CHAR_SPACING) + 2 * PADDING)
    img_height = int(len(rows) * (char_height + CHAR_SPACING) + 2 * PADDING)

    # Create image
    img = Image.new("RGB", (img_width, img_height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw characters
    y = PADDING
    for row in rows:
        x = PADDING
        for char in row:
            draw.text((x, y), char, font=font, fill=TEXT_COLOR)
            x += max_char_width + CHAR_SPACING
        y += char_height + CHAR_SPACING

    # Save
    img.save(output_path)
    print(f"✓ {font_path.stem}")


def main() -> None:
    """Generate previews for all fonts."""
    fonts_dir = Path(__file__).parent.parent / "assets" / "fonts"
    previews_dir = fonts_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    font_files = sorted(fonts_dir.glob("*.ttf"))

    print(f"📁 {previews_dir}\n📦 {len(font_files)} fonts\n")

    for font_path in font_files:
        output_path = previews_dir / f"{font_path.stem}.jpg"
        generate_preview(font_path, output_path)

    total = len(list(previews_dir.glob("*.jpg")))
    print(f"\n✅ {total}/{len(font_files)} previews generated")


if __name__ == "__main__":
    main()
