"""Augmentation pipeline for training data generation."""

import albumentations as A


def create_geometric_pipeline() -> A.Compose:
    """Geometric transforms applied first with extra margin.

    Returns transformed image/bboxes, then caller crops tightly.
    """
    return A.Compose(
        [
            # Rotation - slight tilts from mounting angles
            A.Rotate(
                limit=15,
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=0,  # Black fill
                p=0.5,
            ),
            # Perspective - CCTV/dashcam viewing angles
            A.Perspective(
                scale=(0.02, 0.08),
                fit_output=True,
                fill=0,  # Black fill
                p=0.5,
            ),
            # Affine - shear and slight scale variations
            A.Affine(
                scale=(0.9, 1.1),
                shear=(-10, 10),
                rotate=0,  # Already handled by Rotate
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=0,  # Black fill
                p=0.4,
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height]
            label_fields=["labels"],
            min_visibility=0.7,
        ),
    )


def create_effects_pipeline(*, output_size: int = 256) -> A.Compose:
    """Visual effects applied after tight crop."""
    return A.Compose(
        [
            # Weather and environmental effects
            A.OneOf(
                [
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_limit=(1, 2),
                        shadow_dimension=5,
                        p=1.0,
                    ),
                    A.RandomFog(fog_coef_range=(0.05, 0.2), alpha_coef=0.1, p=1.0),
                    A.RandomSunFlare(src_radius=40, p=1.0),
                ],
                p=0.3,
            ),
            # Color augmentations (simulate different lighting)
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.08,
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25, contrast_limit=0.25, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=25,
                        val_shift_limit=25,
                        p=1.0,
                    ),
                    A.RandomToneCurve(scale=0.15, p=1.0),
                ],
                p=0.65,
            ),
            # Gamma adjustment
            A.RandomGamma(gamma_limit=(70, 130), p=0.25),
            # Noise effects (realistic camera sensor noise)
            A.OneOf(
                [
                    A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.08, 0.3), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.07), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ],
                p=0.85,
            ),
            # Blur effects (always apply some blur for realism)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(
                        blur_limit=(5, 15),
                        angle_range=(0, 360),
                        direction_range=(-1.0, 1.0),
                        p=1.0,
                    ),
                    A.Defocus(radius=(1, 3), p=1.0),
                ],
                p=1.0,
            ),
            # Compression artifacts (JPEG from dashcams/CCTV)
            A.ImageCompression(quality_range=(40, 85), p=0.7),
            # Downscale (simulate distance)
            A.Downscale(scale_range=(0.5, 0.85), p=0.3),
            # Convert to grayscale (training target)
            A.ToGray(p=1.0),
            # Resize longest side, then pad to square with black padding
            A.LongestMaxSize(max_size=output_size),
            A.PadIfNeeded(
                min_height=output_size,
                min_width=output_size,
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=0,  # Black padding
                position="center",  # Symmetric padding
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height]
            label_fields=["labels"],
            min_visibility=0.7,
        ),
    )


def tight_crop_around_bboxes(
    image,
    bboxes: list,
    labels: list,
    buffer_ratio_x: float = 0.15,
    buffer_ratio_y: float = 0.10,
) -> tuple:
    """Crop tightly around bboxes with buffer margins (15% horizontal, 10% vertical)."""
    if not bboxes:
        return image, bboxes, labels

    h, w = image.shape[:2]

    # Find bounding box of all character bboxes
    min_x = min(b[0] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_x = max(b[0] + b[2] for b in bboxes)
    max_y = max(b[1] + b[3] for b in bboxes)

    # Calculate buffer based on plate size
    plate_w = max_x - min_x
    plate_h = max_y - min_y
    buffer_x = int(plate_w * buffer_ratio_x)
    buffer_y = int(plate_h * buffer_ratio_y)

    # Crop region with buffer
    crop_x1 = max(0, int(min_x) - buffer_x)
    crop_y1 = max(0, int(min_y) - buffer_y)
    crop_x2 = min(w, int(max_x) + buffer_x)
    crop_y2 = min(h, int(max_y) + buffer_y)

    # Crop image
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Adjust bboxes to new coordinates
    new_bboxes = [[b[0] - crop_x1, b[1] - crop_y1, b[2], b[3]] for b in bboxes]

    return cropped, new_bboxes, labels


# Legacy function for backward compatibility
def create_augmentation_pipeline(*, output_size: int = 256) -> A.Compose:
    """Full pipeline - use create_geometric_pipeline + tight_crop + create_effects_pipeline instead."""
    return create_effects_pipeline(output_size=output_size)


__all__ = [
    "create_geometric_pipeline",
    "create_effects_pipeline",
    "tight_crop_around_bboxes",
    "create_augmentation_pipeline",
]
