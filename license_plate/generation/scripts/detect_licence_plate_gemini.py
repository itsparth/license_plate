"""Detect license plates using Gemini vision model."""

import os
import time
from pathlib import Path

import cv2
import google.genai as genai
import numpy as np
import supervision as sv
from PIL import Image
from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized 0-1)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_xyxy(self, img_width: int, img_height: int) -> list[float]:
        """Convert normalized coordinates to pixel coordinates."""
        return [
            self.x_min * img_width,
            self.y_min * img_height,
            self.x_max * img_width,
            self.y_max * img_height,
        ]


class LicensePlateDetection(BaseModel):
    """License plate detection result."""

    confidence: float
    bbox: BoundingBox
    description: str = ""


def detect_license_plate_gemini(
    image_path: str | Path, api_key: str | None = None
) -> sv.Detections:
    """Detect empty license plates using Gemini vision model.

    Args:
        image_path: Path to the image file
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)

    Returns:
        sv.Detections containing license plate detections
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Prompt for license plate detection
    prompt = """Analyze this image and detect all empty license plates on vehicles.

For each empty license plate found, provide:
1. Bounding box coordinates (normalized 0-1): x_min, y_min, x_max, y_max
2. Confidence score (0-1)
3. Brief description (e.g., "white rectangular plate on front bumper")

Return results in JSON format:
{
  "plates": [
    {
      "bbox": {"x_min": 0.1, "y_min": 0.5, "x_max": 0.3, "y_max": 0.6},
      "confidence": 0.95,
      "description": "white rectangular license plate"
    }
  ]
}

Focus on:
- Empty/blank license plates (no text/numbers visible)
- Rectangular plate shapes with borders
- Typical mounting positions (front/rear bumpers)
- Indian vehicle license plate formats (white, yellow, black, green backgrounds)

Return empty array if no plates detected."""

    try:
        # Generate content with image
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, image],
        )

        # Parse response
        import json
        import re

        text = response.text
        if not text:
            return sv.Detections.empty()

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result = json.loads(text)

        plates = result.get("plates", [])

        if not plates:
            return sv.Detections.empty()

        # Convert to supervision format
        detections = []
        confidences = []

        for plate in plates:
            bbox = BoundingBox(**plate["bbox"])
            xyxy = bbox.to_xyxy(img_width, img_height)
            detections.append(xyxy)
            confidences.append(plate.get("confidence", 0.9))

        return sv.Detections(
            xyxy=np.array(detections, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(
                len(detections), dtype=int
            ),  # All class 0 (license plate)
        )

    except Exception as e:
        print(f"Error detecting with Gemini: {e}")
        return sv.Detections.empty()


if __name__ == "__main__":
    # Input and output directories
    vehicles_dir = Path(__file__).parent.parent / "assets/vehicles"
    output_dir = Path(__file__).parent.parent / "assets/vehicle_plates"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all image files
    image_files = list(vehicles_dir.glob("*.jpg")) + list(vehicles_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {vehicles_dir}")

    processed = 0
    skipped = 0

    for image_path in sorted(image_files):
        print(f"\nProcessing: {image_path.name}")

        detections = None
        retries = 0
        max_retries = 5

        while retries < max_retries:
            try:
                # Detect license plates
                detections = detect_license_plate_gemini(str(image_path))
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (2**retries) * 5  # Exponential backoff
                    print(f"  Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"  Error: {e}")
                    break

        if detections is None or len(detections) == 0:
            print("  No license plates detected, skipping...")
            skipped += 1
            continue

        # Find largest license plate by area
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
            detections.xyxy[:, 3] - detections.xyxy[:, 1]
        )
        largest_idx = np.argmax(areas)
        largest_bbox = detections.xyxy[largest_idx]
        largest_conf = (
            detections.confidence[largest_idx]
            if detections.confidence is not None
            else 1.0
        )

        print(
            f"  Found {len(detections)} plate(s), largest: {largest_conf:.2%} confidence"
        )

        # Load image for annotation and dimensions
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        # Create single detection object for the largest plate
        largest_detection = sv.Detections(
            xyxy=np.array([largest_bbox]),
            confidence=np.array([largest_conf]),
            class_id=np.array([0]),
        )

        # 1. Save Label (YOLO format)
        # class_id x_center y_center width height (normalized)
        x_min, y_min, x_max, y_max = largest_bbox

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height

        label_path = output_dir / f"{image_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write(
                f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
            )

        # 2. Save Annotated Image
        box_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.GREEN)

        annotated_image = box_annotator.annotate(
            scene=image.copy(), detections=largest_detection
        )

        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), annotated_image)

        print(f"  Saved label to {label_path.name}")
        print(f"  Saved annotated image to {output_image_path.name}")

        processed += 1

        # Small delay to be nice to the API
        time.sleep(2)

    print("\nProcessing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {output_dir}")
