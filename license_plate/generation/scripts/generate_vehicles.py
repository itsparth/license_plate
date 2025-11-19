"""Generate vehicle images with empty license plates for Indian vehicles."""

import os
from pathlib import Path
from typing import Literal

import google.genai as genai
from pydantic import BaseModel


VehicleType = Literal[
    "car", "bus", "truck", "auto", "bike", "tempo", "minivan", "mini_truck"
]
ViewType = Literal["front", "back"]
PlatePosition = Literal["center", "left", "right"]  # For back plates on trucks/buses
PlateStyle = Literal[
    "white_single",  # White plate, single line (old private)
    "white_double",  # White plate, double line (HSRP private)
    "yellow_single",  # Yellow plate, single line (old commercial)
    "yellow_double",  # Yellow plate, double line (HSRP commercial)
    "black_yellow",  # Black with yellow text (transport)
    "green",  # Green plate (electric/CNG)
]


class VehicleConfig(BaseModel):
    """Vehicle generation configuration."""

    vehicle_type: VehicleType
    view: ViewType
    plate_style: PlateStyle
    plate_position: PlatePosition = "center"  # Where plate is mounted
    realistic_bg: bool = True  # Use realistic background vs studio

    @property
    def filename(self) -> str:
        pos_suffix = (
            f"_{self.plate_position}" if self.plate_position != "center" else ""
        )
        bg_suffix = "_studio" if not self.realistic_bg else ""
        return f"{self.vehicle_type}_{self.view}_{self.plate_style}{pos_suffix}{bg_suffix}.jpg"

    @property
    def prompt(self) -> str:
        """Generate detailed prompt for vehicle image."""
        view_desc = "front view" if self.view == "front" else "rear view"
        plate_desc = self._get_plate_description()

        # Background setting
        if self.realistic_bg:
            bg_desc = "realistic Indian street scene, parked on road, natural outdoor lighting, daytime"
        else:
            bg_desc = "clean neutral studio background, professional lighting"

        # Framing - tight crop focusing on license plate area
        framing = "close-up framing focused on license plate area, vehicle centered, plate takes significant portion of frame"

        # Plate position for rear plates on trucks/buses
        position_desc = ""
        if self.view == "back" and self.vehicle_type in ["truck", "bus"]:
            if self.plate_position == "left":
                position_desc = "license plate mounted on left side of rear, "
            elif self.plate_position == "right":
                position_desc = "license plate mounted on right side of rear, "
            else:
                position_desc = "license plate centered on rear, "

        base = (
            f"Professional photograph of an Indian {self.vehicle_type}, "
            f"absolute straight-on {view_desc}, {framing}, perfectly centered vehicle, "
            f"{bg_desc}, "
            f"{position_desc}"
            f"{plate_desc}, "
            f"realistic, blank empty plate with absolutely no text, graphic or numbers, "
            f"sharp focus, clear plate visibility"
        )

        return base

    def _get_plate_description(self) -> str:
        """Get plate style description based on Indian standards."""
        styles = {
            "white_single": "empty white rectangular license plate with black border, single-line format",
            "white_double": "empty white rectangular license plate with black border, double-line format",
            "yellow_single": "empty yellow rectangular license plate with black border, single-line format",
            "yellow_double": "empty yellow rectangular license plate with black border, double-line format",
            "black_yellow": "empty black rectangular license plate with yellow border, transport vehicle format",
            "green": "empty green rectangular license plate with white border, electric vehicle format",
        }
        return styles[self.plate_style]


class GenerationStats(BaseModel):
    """Generation statistics."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def success_rate(self) -> float:
        return (self.success / self.total * 100) if self.total else 0.0


# Comprehensive configurations (~50 vehicles covering all variations)
VEHICLE_CONFIGS: list[VehicleConfig] = [
    # ========== CARS (16 variations) ==========
    # Front views - all plate styles
    VehicleConfig(vehicle_type="car", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="white_double"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="black_yellow"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="green"),
    # Back views - common styles
    VehicleConfig(vehicle_type="car", view="back", plate_style="white_single"),
    VehicleConfig(vehicle_type="car", view="back", plate_style="white_double"),
    VehicleConfig(vehicle_type="car", view="back", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="car", view="back", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="car", view="back", plate_style="black_yellow"),
    VehicleConfig(vehicle_type="car", view="back", plate_style="green"),
    # Additional front variations (different car models implied by regeneration)
    VehicleConfig(vehicle_type="car", view="front", plate_style="white_double"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="car", view="front", plate_style="green"),
    # ========== BIKES (12 variations) ==========
    # Front views
    VehicleConfig(vehicle_type="bike", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="bike", view="front", plate_style="white_double"),
    VehicleConfig(vehicle_type="bike", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="bike", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="bike", view="front", plate_style="green"),
    VehicleConfig(vehicle_type="bike", view="front", plate_style="black_yellow"),
    # Back views
    VehicleConfig(vehicle_type="bike", view="back", plate_style="white_single"),
    VehicleConfig(vehicle_type="bike", view="back", plate_style="white_double"),
    VehicleConfig(vehicle_type="bike", view="back", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="bike", view="back", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="bike", view="back", plate_style="green"),
    VehicleConfig(vehicle_type="bike", view="back", plate_style="black_yellow"),
    # ========== AUTOS (10 variations) ==========
    # Front views
    VehicleConfig(vehicle_type="auto", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="auto", view="front", plate_style="white_double"),
    VehicleConfig(vehicle_type="auto", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="auto", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="auto", view="front", plate_style="black_yellow"),
    # Back views
    VehicleConfig(vehicle_type="auto", view="back", plate_style="white_single"),
    VehicleConfig(vehicle_type="auto", view="back", plate_style="white_double"),
    VehicleConfig(vehicle_type="auto", view="back", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="auto", view="back", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="auto", view="back", plate_style="black_yellow"),
    # ========== TRUCKS (12 variations with side plates) ==========
    # Front views
    VehicleConfig(vehicle_type="truck", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="truck", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="truck", view="front", plate_style="black_yellow"),
    VehicleConfig(vehicle_type="truck", view="front", plate_style="white_single"),
    # Back views - center mounted
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="yellow_single",
        plate_position="center",
    ),
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="yellow_double",
        plate_position="center",
    ),
    # Back views - left side mounted (common for trucks)
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="yellow_single",
        plate_position="left",
    ),
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="black_yellow",
        plate_position="left",
    ),
    # Back views - right side mounted
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="yellow_double",
        plate_position="right",
    ),
    VehicleConfig(
        vehicle_type="truck",
        view="back",
        plate_style="white_single",
        plate_position="right",
    ),
    # ========== BUSES (12 variations with side plates) ==========
    # Front views
    VehicleConfig(vehicle_type="bus", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="bus", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="bus", view="front", plate_style="black_yellow"),
    VehicleConfig(vehicle_type="bus", view="front", plate_style="white_single"),
    # Back views - center mounted
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="yellow_single",
        plate_position="center",
    ),
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="yellow_double",
        plate_position="center",
    ),
    # Back views - left side mounted (common for buses)
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="yellow_single",
        plate_position="left",
    ),
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="black_yellow",
        plate_position="left",
    ),
    # Back views - right side mounted
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="yellow_double",
        plate_position="right",
    ),
    VehicleConfig(
        vehicle_type="bus",
        view="back",
        plate_style="white_single",
        plate_position="right",
    ),
    # ========== TEMPOS (8 variations) ==========
    # Front views
    VehicleConfig(vehicle_type="tempo", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="tempo", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="tempo", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="tempo", view="front", plate_style="black_yellow"),
    # Back views
    VehicleConfig(vehicle_type="tempo", view="back", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="tempo", view="back", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="tempo", view="back", plate_style="white_single"),
    VehicleConfig(vehicle_type="tempo", view="back", plate_style="black_yellow"),
    # ========== MINIVANS (8 variations) ==========
    # Front views
    VehicleConfig(vehicle_type="minivan", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="minivan", view="front", plate_style="white_double"),
    VehicleConfig(vehicle_type="minivan", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="minivan", view="front", plate_style="yellow_double"),
    # Back views
    VehicleConfig(vehicle_type="minivan", view="back", plate_style="white_single"),
    VehicleConfig(vehicle_type="minivan", view="back", plate_style="white_double"),
    VehicleConfig(vehicle_type="minivan", view="back", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="minivan", view="back", plate_style="yellow_double"),
    # ========== MINI TRUCKS (8 variations with side plates) ==========
    # Front views
    VehicleConfig(vehicle_type="mini_truck", view="front", plate_style="yellow_single"),
    VehicleConfig(vehicle_type="mini_truck", view="front", plate_style="yellow_double"),
    VehicleConfig(vehicle_type="mini_truck", view="front", plate_style="white_single"),
    VehicleConfig(vehicle_type="mini_truck", view="front", plate_style="black_yellow"),
    # Back views - varied positions
    VehicleConfig(
        vehicle_type="mini_truck",
        view="back",
        plate_style="yellow_single",
        plate_position="center",
    ),
    VehicleConfig(
        vehicle_type="mini_truck",
        view="back",
        plate_style="yellow_double",
        plate_position="left",
    ),
    VehicleConfig(
        vehicle_type="mini_truck",
        view="back",
        plate_style="white_single",
        plate_position="right",
    ),
    VehicleConfig(
        vehicle_type="mini_truck",
        view="back",
        plate_style="black_yellow",
        plate_position="center",
    ),
]

# Set to True to regenerate existing images
REGENERATE = False


def generate_vehicle(
    config: VehicleConfig, dest_dir: Path, client: genai.Client, stats: GenerationStats
) -> None:
    """Generate a single vehicle image."""
    dest_file = dest_dir / config.filename

    if dest_file.exists() and not REGENERATE:
        print(f"✓ {config.filename} already exists")
        stats.skipped += 1
        return

    try:
        print(f"⬇ {config.filename}...", end=" ", flush=True)

        # Try Gemini image generation for better prompt adherence
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=config.prompt,
        )

        if response.parts:
            for part in response.parts:
                if part.inline_data and part.inline_data.data:
                    from PIL import Image
                    from io import BytesIO

                    Image.open(BytesIO(part.inline_data.data)).save(dest_file)
                    print("✓")
                    stats.success += 1
                    return

        print("✗ No image returned")
        stats.failed += 1

    except Exception as e:
        print(f"✗ {e}")
        stats.failed += 1


def main() -> None:
    """Generate initial test vehicle images."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("CRITICAL ERROR: 'GEMINI_API_KEY' not set.")

    client = genai.Client(api_key=api_key)

    vehicles_dir = Path(__file__).parent.parent / "assets" / "vehicles"
    vehicles_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 {vehicles_dir}\n📦 {len(VEHICLE_CONFIGS)} vehicles\n")

    stats = GenerationStats(total=len(VEHICLE_CONFIGS))

    for config in VEHICLE_CONFIGS:
        generate_vehicle(config, vehicles_dir, client, stats)

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"✓ Generated: {stats.success} | ⊙ Existed: {stats.skipped} | ✗ Failed: {stats.failed}"
    )
    print(f"📊 Success rate: {stats.success_rate:.1f}%")
    print(f"{'=' * 60}")

    # Categories
    print("\n📋 Vehicle Types:")
    vehicle_types: dict[VehicleType, list[str]] = {}
    for config in VEHICLE_CONFIGS:
        vehicle_types.setdefault(config.vehicle_type, []).append(config.filename)

    for vtype, files in sorted(vehicle_types.items()):
        print(f"\n  {vtype.upper()} ({len(files)}):")
        for filename in sorted(files):
            status = "✓" if (vehicles_dir / filename).exists() else "✗"
            print(f"    {status} {filename}")

    total_available = sum(
        1 for config in VEHICLE_CONFIGS if (vehicles_dir / config.filename).exists()
    )
    print(f"\n✅ {total_available}/{len(VEHICLE_CONFIGS)} vehicles available")


if __name__ == "__main__":
    main()
