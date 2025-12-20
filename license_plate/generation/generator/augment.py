"""Augmentation pipeline for training data generation."""

import albumentations as A


def create_augmentation_pipeline(*, output_size: int = 192) -> A.Compose:
    """Create albumentations pipeline for realistic augmentation.

    Designed to simulate real-world conditions for license plate recognition:
    - Various lighting and weather conditions
    - Camera motion and focus blur
    - Different capture distances and angles
    - Noise and compression artifacts
    """
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
                p=0.45,
            ),
            # Blur effects
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.Defocus(radius=(1, 3), p=1.0),
                ],
                p=0.4,
            ),
            # Compression artifacts (JPEG from dashcams/CCTV)
            A.ImageCompression(quality_range=(55, 95), p=0.35),
            # Simulate different capture distances
            A.Downscale(scale_range=(0.5, 0.85), p=0.25),
            # Tight crop around plate (simulates detector output)
            A.BBoxSafeRandomCrop(erosion_rate=0.02, p=0.45),
            # Convert to grayscale (training target)
            A.ToGray(p=1.0),
            # Geometric transforms (no rotation - done during training)
            A.Affine(
                scale=(0.92, 1.08),
                shear=(-8, 8),
                p=0.3,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.35),
            # Final resize
            A.LongestMaxSize(max_size=output_size),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height]
            label_fields=["labels"],
            min_visibility=0.7,  # Keep chars that are at least 70% visible
        ),
    )


__all__ = ["create_augmentation_pipeline"]
