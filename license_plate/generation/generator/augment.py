"""Augmentation pipeline for training data generation."""

import albumentations as A


def create_augmentation_pipeline(*, output_size: int = 192) -> A.Compose:
    """Create albumentations pipeline for realistic augmentation."""
    return A.Compose(
        [
            # Lighting effects
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3,
            ),
            A.RandomSunFlare(
                src_radius=50,
                p=0.1,
            ),
            # Color augmentations
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0,
                    ),
                    A.RandomToneCurve(scale=0.1, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                ],
                p=0.7,
            ),
            # Noise effects (light noise only)
            A.OneOf(
                [
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
                ],
                p=0.25,
            ),
            # Blur effects (very subtle - plates should remain readable)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.MotionBlur(blur_limit=(3, 3), p=1.0),
                ],
                p=0.15,
            ),
            # Compression artifacts (mild)
            A.ImageCompression(quality_range=(70, 95), p=0.2),
            # Tight crop around plate (simulates detector output)
            # erosion_rate=0.05 allows up to 5% of edge chars to be cropped
            A.BBoxSafeRandomCrop(erosion_rate=0.05, p=0.8),
            # Convert to grayscale (training target)
            A.ToGray(p=1.0),
            # Geometric transforms
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.Rotate(limit=5, border_mode=0, p=0.3),
            # Final resize
            A.LongestMaxSize(max_size=output_size),
        ],
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height]
            label_fields=["labels"],
            min_visibility=0.9,  # Keep chars that are at least 90% visible
        ),
    )


__all__ = ["create_augmentation_pipeline"]
