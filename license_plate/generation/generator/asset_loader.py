from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional
import random

from pydantic import BaseModel, PrivateAttr
from PIL import Image

ASSETS_ROOT = Path(__file__).parent.parent / "assets"
FONTS_DIR = ASSETS_ROOT / "fonts"
LOGOS_DIR = ASSETS_ROOT / "logos"
VEHICLE_PLATES_DIR = ASSETS_ROOT / "vehicle_plates"
VEHICLES_DIR = ASSETS_ROOT / "vehicles"


class LogoAsset(BaseModel):
    path: Path
    width: int
    height: int
    _image: Optional[Image.Image] = PrivateAttr(default=None)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def is_long(self, *, min_aspect: float = 3.0) -> bool:
        return self.aspect_ratio >= min_aspect

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            self._image = Image.open(self.path).copy()
        return self._image


class VehiclePlateBBox(BaseModel):
    class_id: int
    cx: float  # normalized center x (0-1)
    cy: float  # normalized center y (0-1)
    w: float  # normalized width (0-1)
    h: float  # normalized height (0-1)

    def to_pixels(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        px_w = int(round(self.w * image_width))
        px_h = int(round(self.h * image_height))
        px_cx = int(round(self.cx * image_width))
        px_cy = int(round(self.cy * image_height))
        left = px_cx - px_w // 2
        top = px_cy - px_h // 2
        return left, top, px_w, px_h


class VehicleImageAsset(BaseModel):
    image_path: Path
    annotation_path: Path
    bbox: VehiclePlateBBox
    width: int
    height: int
    _image: Optional[Image.Image] = PrivateAttr(default=None)

    @property
    def pixel_bbox(self) -> tuple[int, int, int, int]:
        return self.bbox.to_pixels(self.width, self.height)

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            self._image = Image.open(self.image_path).copy()
        return self._image


class AssetLoader(BaseModel):
    assets_root: Path = ASSETS_ROOT
    _logo_cache: Optional[list[LogoAsset]] = None
    _vehicle_cache: Optional[list[VehicleImageAsset]] = None

    @property
    def fonts_dir(self) -> Path:
        return self.assets_root / "fonts"

    @property
    def logos_dir(self) -> Path:
        return self.assets_root / "logos"

    @property
    def vehicle_plates_dir(self) -> Path:
        return self.assets_root / "vehicle_plates"

    @property
    def vehicles_dir(self) -> Path:
        return self.assets_root / "vehicles"

    # Fonts -----------------------------------------------------------------
    def iter_fonts(self) -> Iterator[str]:
        if not self.fonts_dir.exists():
            return iter(())
        for p in sorted(self.fonts_dir.iterdir()):
            if p.is_dir():
                # skip previews subdir and any others
                continue
            if p.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            yield str(p)

    # Logos -----------------------------------------------------------------
    def _load_logos(self) -> list[LogoAsset]:
        items: list[LogoAsset] = []
        if self.logos_dir.exists():
            for p in sorted(self.logos_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                try:
                    im = Image.open(p).copy()
                    w, h = im.size
                except Exception:
                    continue
                asset = LogoAsset(path=p, width=w, height=h)
                asset._image = im
                items.append(asset)
        return items

    def logos(
        self, *, min_aspect: Optional[float] = None, max_aspect: Optional[float] = None
    ) -> list[LogoAsset]:
        if self._logo_cache is None:
            self._logo_cache = self._load_logos()
        items = list(self._logo_cache)
        if min_aspect is not None:
            items = [logo for logo in items if logo.aspect_ratio >= min_aspect]
        if max_aspect is not None:
            items = [logo for logo in items if logo.aspect_ratio <= max_aspect]
        return items

    def random_logo(
        self, *, min_aspect: Optional[float] = None, max_aspect: Optional[float] = None
    ) -> Optional[LogoAsset]:
        candidates = self.logos(min_aspect=min_aspect, max_aspect=max_aspect)
        return random.choice(candidates) if candidates else None

    # Vehicles ---------------------------------------------------------------
    def _parse_plate_bbox(self, txt_path: Path) -> Optional[VehiclePlateBBox]:
        try:
            raw = txt_path.read_text().strip()
        except FileNotFoundError:
            return None
        if not raw:
            return None
        # Use first non-empty line
        line = next((ln for ln in raw.splitlines() if ln.strip()), "")
        parts = line.split()
        if len(parts) not in (4, 5):
            return None
        if len(parts) == 5:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        else:
            class_id = 0
            cx, cy, w, h = map(float, parts)
        return VehiclePlateBBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h)

    def _load_vehicles(self) -> list[VehicleImageAsset]:
        items: list[VehicleImageAsset] = []
        if not self.vehicles_dir.exists():
            return items
        for img_path in sorted(self.vehicles_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            base = img_path.stem
            annotation_path = self.vehicle_plates_dir / f"{base}.txt"
            bbox = self._parse_plate_bbox(annotation_path)
            if bbox is None:
                continue
            try:
                im = Image.open(img_path).copy()
                w, h = im.size
            except Exception:
                continue
            asset = VehicleImageAsset(
                image_path=img_path,
                annotation_path=annotation_path,
                bbox=bbox,
                width=w,
                height=h,
            )
            asset._image = im
            items.append(asset)
        return items

    def vehicles(
        self, *, min_aspect: Optional[float] = None
    ) -> list[VehicleImageAsset]:
        if self._vehicle_cache is None:
            self._vehicle_cache = self._load_vehicles()
        items = list(self._vehicle_cache)
        if min_aspect is not None:
            items = [
                v for v in items if v.bbox.h and (v.bbox.w / v.bbox.h >= min_aspect)
            ]
        return items

    def random_vehicle(
        self, *, min_aspect: Optional[float] = None
    ) -> Optional[VehicleImageAsset]:
        candidates = self.vehicles(min_aspect=min_aspect)
        return random.choice(candidates) if candidates else None


__all__ = [
    "AssetLoader",
    "LogoAsset",
    "VehicleImageAsset",
    "VehiclePlateBBox",
]
