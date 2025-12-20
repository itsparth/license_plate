#!/usr/bin/env python3
"""Download fonts for Indian license plate generation."""

import io
import zipfile
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel


FontCategory = Literal[
    "hsrp", "display", "italic", "condensed", "challenging", "extreme"
]


class FontConfig(BaseModel):
    """Font configuration."""

    name: str
    url: str
    category: FontCategory
    zip_path: str | None = None  # Path inside zip file to extract (for zip downloads)

    @property
    def filename(self) -> str:
        return f"{self.name}.ttf"

    @property
    def is_zip(self) -> bool:
        return self.zip_path is not None


class DownloadStats(BaseModel):
    """Download statistics."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def success_rate(self) -> float:
        return (self.success / self.total * 100) if self.total else 0.0


# Font collection
FONTS: list[FontConfig] = [
    # HSRP-style: clean, bold, sans-serif - resembles Indian license plate fonts (8 fonts)
    FontConfig(
        name="montserrat_bold",
        url="https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
        category="hsrp",
    ),
    FontConfig(
        name="poppins_bold",
        url="https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
        category="hsrp",
    ),
    FontConfig(
        name="lato_bold",
        url="https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf",
        category="hsrp",
    ),
    # License plate style fonts (monospace/technical fonts work well for HSRP)
    FontConfig(
        name="ibm_plex_mono_bold",
        url="https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Bold.ttf",
        category="hsrp",
    ),
    FontConfig(
        name="road_numbers",
        url="https://github.com/google/fonts/raw/main/ofl/b612mono/B612Mono-Bold.ttf",
        category="hsrp",
    ),
    # GL-Nummernschild - German license plate font (very similar to Indian HSRP)
    FontConfig(
        name="gl_nummernschild_eng",
        url="https://fontesk.com/download/128024/?tmstv=1702529571",
        category="hsrp",
        zip_path="fonts/ttf/GL-Nummernschild-Eng.ttf",
    ),
    FontConfig(
        name="gl_nummernschild_mtl",
        url="https://fontesk.com/download/128024/?tmstv=1702529571",
        category="hsrp",
        zip_path="fonts/ttf/GL-Nummernschild-Mtl.ttf",
    ),
    # Display: bold, fancy (3 fonts)
    FontConfig(
        name="anton",
        url="https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        category="display",
    ),
    FontConfig(
        name="righteous",
        url="https://github.com/google/fonts/raw/main/ofl/righteous/Righteous-Regular.ttf",
        category="display",
    ),
    FontConfig(
        name="alfa_slab",
        url="https://github.com/google/fonts/raw/main/ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
        category="display",
    ),
    # Italic: challenging angles (3 fonts)
    FontConfig(
        name="montserrat_italic",
        url="https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-BoldItalic.ttf",
        category="italic",
    ),
    FontConfig(
        name="poppins_italic",
        url="https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-BoldItalic.ttf",
        category="italic",
    ),
    FontConfig(
        name="lato_italic",
        url="https://github.com/google/fonts/raw/main/ofl/lato/Lato-BoldItalic.ttf",
        category="italic",
    ),
    # Condensed: M/H/N ambiguity (1 font)
    FontConfig(
        name="fjalla_one",
        url="https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf",
        category="condensed",
    ),
    # Challenging: O/0, I/1, B/8 confusion (3 fonts)
    FontConfig(
        name="staatliches",
        url="https://github.com/google/fonts/raw/main/ofl/staatliches/Staatliches-Regular.ttf",
        category="challenging",
    ),
    FontConfig(
        name="antonio_bold",
        url="https://github.com/google/fonts/raw/main/ofl/antonio/Antonio%5Bwght%5D.ttf",
        category="challenging",
    ),
    FontConfig(
        name="exo_2_bold",
        url="https://github.com/google/fonts/raw/main/ofl/exo2/Exo2%5Bwght%5D.ttf",
        category="challenging",
    ),
    # Extreme: Heavily stylized fonts (stencil, pixel, futuristic) - maximum OCR difficulty (4 fonts)
    FontConfig(
        name="black_ops_one",
        url="https://github.com/google/fonts/raw/main/ofl/blackopsone/BlackOpsOne-Regular.ttf",
        category="extreme",
    ),
    FontConfig(
        name="rubik_mono_one",
        url="https://github.com/google/fonts/raw/main/ofl/rubikmonoone/RubikMonoOne-Regular.ttf",
        category="extreme",
    ),
    FontConfig(
        name="bungee",
        url="https://github.com/google/fonts/raw/main/ofl/bungee/Bungee-Regular.ttf",
        category="extreme",
    ),
]


def download_font(font: FontConfig, dest_dir: Path, stats: DownloadStats) -> None:
    """Download a single font file."""
    dest_file = dest_dir / font.filename

    if dest_file.exists():
        print(f"✓ {font.name} already exists")
        stats.skipped += 1
        return

    try:
        print(f"⬇ {font.name}...", end=" ", flush=True)
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(font.url)
            response.raise_for_status()

            if font.is_zip and font.zip_path:
                # Extract specific file from zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    with zf.open(font.zip_path) as src:
                        dest_file.write_bytes(src.read())
            else:
                dest_file.write_bytes(response.content)
        print("✓")
        stats.success += 1
    except Exception as e:
        print(f"✗ {e}")
        stats.failed += 1


def main() -> None:
    """Download all configured fonts."""
    fonts_dir = Path(__file__).parent.parent / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 {fonts_dir}\n📦 {len(FONTS)} fonts\n")

    stats = DownloadStats(total=len(FONTS))

    for font in FONTS:
        download_font(font, fonts_dir, stats)

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"✓ Downloaded: {stats.success} | ⊙ Existed: {stats.skipped} | ✗ Failed: {stats.failed}"
    )
    print(f"📊 Success rate: {stats.success_rate:.1f}%")
    print(f"{'=' * 60}")

    # Categories
    print("\n📋 Categories:")
    categories: dict[FontCategory, list[str]] = {}
    for font in FONTS:
        categories.setdefault(font.category, []).append(font.name)

    for cat, names in sorted(categories.items()):
        print(f"\n  {cat.upper()} ({len(names)}):")
        for name in sorted(names):
            status = "✓" if (fonts_dir / f"{name}.ttf").exists() else "✗"
            print(f"    {status} {name}")

    total_available = sum(1 for font in FONTS if (fonts_dir / font.filename).exists())
    print(f"\n✅ {total_available}/{len(FONTS)} fonts available in {fonts_dir}")


if __name__ == "__main__":
    main()
