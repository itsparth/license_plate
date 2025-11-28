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
            # Noise effects (realistic camera sensor noise)
            A.OneOf(
                [
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ],
                p=0.5,
            ),
            # Blur effects (realistic camera motion/focus blur)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.Defocus(radius=(1, 3), p=1.0),
                ],
                p=0.4,
            ),
            # Additional motion blur (simulates camera/vehicle movement)
            A.MotionBlur(blur_limit=(3, 7), p=0.25),
            # Compression artifacts (mild)
            A.ImageCompression(quality_range=(70, 95), p=0.2),
            # Tight crop around plate (simulates detector output)
            # erosion_rate=0.01 allows very minimal edge cropping (rare)
            A.BBoxSafeRandomCrop(erosion_rate=0.01, p=0.6),
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
            min_visibility=0.6,  # Keep chars that are at least 60% visible
        ),
    )


__all__ = ["create_augmentation_pipeline"]
