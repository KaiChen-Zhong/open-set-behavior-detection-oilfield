"""
Data augmentation pipeline for safety violation detection.

Uses albumentations for efficient image and bounding-box augmentation.
Designed for oil depot imagery with domain-specific augmentations.
"""

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class ViolationAugmentation:
    """
    Augmentation pipeline tailored for oil depot safety violation detection.

    Applies appropriate augmentations that preserve the semantic meaning
    of safety violations while improving model generalization.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = "train",
        mosaic_prob: float = 0.5,
    ):
        self.img_size = img_size
        self.mode = mode
        if mode == "train":
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_val_transforms()

    def _build_train_transforms(self) -> A.Compose:
        """Strong augmentation for training."""
        return A.Compose(
            [
                # Geometric transforms
                A.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                # Color transforms (domain-specific for outdoor/industrial)
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3,
                            contrast_limit=0.3,
                            p=1.0,
                        ),
                        A.ColorJitter(
                            brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.1,
                            p=1.0,
                        ),
                    ],
                    p=0.7,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.4,
                ),
                # Weather / environment simulation for outdoor scenes
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_range=(0.1, 0.3), p=1.0),
                        A.RandomSunFlare(p=1.0),
                        A.RandomShadow(p=1.0),
                    ],
                    p=0.2,
                ),
                # Blur / noise
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.1), p=1.0),
                    ],
                    p=0.3,
                ),
                # Safety equipment occlusion simulation
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    fill=0,
                    p=0.3,
                ),
                # Normalization (ImageNet stats)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.3,
                min_area=16,
            ),
        )

    def _build_val_transforms(self) -> A.Compose:
        """Minimal augmentation for validation/testing."""
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(self.img_size)),
                A.PadIfNeeded(
                    min_height=self.img_size[0],
                    min_width=self.img_size[1],
                    border_mode=0,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List,
        class_labels: List,
    ) -> Dict:
        """Apply transforms to image and bounding boxes."""
        return self.transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )


def get_train_transforms(img_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Get training augmentation pipeline."""
    aug = ViolationAugmentation(img_size=img_size, mode="train")
    return aug.transform


def get_val_transforms(img_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Get validation/test transform pipeline."""
    aug = ViolationAugmentation(img_size=img_size, mode="val")
    return aug.transform


class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into one, improving small object detection.
    Adapted from YOLOv5/v8 mosaic implementation.
    """

    def __init__(
        self,
        img_size: int = 640,
        fill_value: int = 114,
    ):
        self.img_size = img_size
        self.fill_value = fill_value

    def __call__(
        self,
        images: List[np.ndarray],
        all_boxes: List[np.ndarray],
        all_labels: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create mosaic from 4 images.

        Args:
            images: List of 4 images (HxWxC numpy arrays).
            all_boxes: List of 4 box arrays in xyxy format.
            all_labels: List of 4 label arrays.

        Returns:
            mosaic_img: Combined mosaic image.
            mosaic_boxes: Combined bounding boxes.
            mosaic_labels: Combined labels.
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        s = self.img_size
        # Random center point
        cx = int(np.random.uniform(s * 0.25, s * 0.75))
        cy = int(np.random.uniform(s * 0.25, s * 0.75))

        mosaic_img = np.full((2 * s, 2 * s, 3), self.fill_value, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []

        # Placement positions for each of 4 images
        positions = [
            (0, 0, cx, cy),       # Top-left
            (cx, 0, 2 * s, cy),   # Top-right
            (0, cy, cx, 2 * s),   # Bottom-left
            (cx, cy, 2 * s, 2 * s),  # Bottom-right
        ]

        for i, (img, boxes, labels) in enumerate(zip(images, all_boxes, all_labels)):
            x1, y1, x2, y2 = positions[i]
            h, w = y2 - y1, x2 - x1
            # Resize image to fit its quadrant
            img_resized = np.array(
                __import__("PIL").Image.fromarray(img).resize((w, h))
            )
            mosaic_img[y1:y2, x1:x2] = img_resized

            if len(boxes) > 0:
                orig_h, orig_w = img.shape[:2]
                # Scale boxes
                scale_x = w / orig_w
                scale_y = h / orig_h
                scaled_boxes = boxes.copy().astype(np.float32)
                scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]] * scale_x + x1
                scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]] * scale_y + y1
                mosaic_boxes.append(scaled_boxes)
                mosaic_labels.append(labels)

        # Crop to img_size x img_size and clip boxes
        mosaic_img = mosaic_img[s // 2: s // 2 + s, s // 2: s // 2 + s]
        offset = s // 2
        if mosaic_boxes:
            all_b = np.concatenate(mosaic_boxes, axis=0)
            all_l = np.concatenate(mosaic_labels, axis=0)
            all_b[:, [0, 2]] -= offset
            all_b[:, [1, 3]] -= offset
            all_b = np.clip(all_b, 0, s)
            # Filter degenerate boxes
            valid = (all_b[:, 2] - all_b[:, 0] > 1) & (all_b[:, 3] - all_b[:, 1] > 1)
            all_b = all_b[valid]
            all_l = all_l[valid]
        else:
            all_b = np.zeros((0, 4), dtype=np.float32)
            all_l = np.zeros(0, dtype=np.int64)

        return mosaic_img, all_b, all_l
