"""
COCO-format dataset class for safety violation detection in oil depot operations.

Supports:
- Standard COCO annotation format
- Open-set class handling (known vs unknown)
- Multiple violation types: no_helmet, phone_usage, gas_cylinder_viol, crane_outrigger
- Spatial metadata for distance-based violation reasoning
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# Violation category definitions
VIOLATION_CATEGORIES = {
    "no_helmet": {
        "id": 0,
        "supercategory": "ppe_violation",
        "description": "Worker not wearing safety helmet",
    },
    "phone_usage": {
        "id": 1,
        "supercategory": "behavioral_violation",
        "description": "Worker using mobile phone in restricted area",
    },
    "gas_cylinder_viol": {
        "id": 2,
        "supercategory": "equipment_violation",
        "description": "Gas cylinder placement violation",
    },
    "crane_outrigger": {
        "id": 3,
        "supercategory": "equipment_violation",
        "description": "Crane outrigger not properly deployed",
    },
    "unknown_violation": {
        "id": 4,
        "supercategory": "unknown",
        "description": "Unknown / open-set safety violation",
    },
}

CATEGORY_ID_TO_NAME = {v["id"]: k for k, v in VIOLATION_CATEGORIES.items()}
CATEGORY_NAME_TO_ID = {k: v["id"] for k, v in VIOLATION_CATEGORIES.items()}


class COCOViolationDataset(Dataset):
    """
    Dataset class for loading COCO-format safety violation annotations.

    Args:
        ann_file: Path to COCO-format annotation JSON file.
        img_dir: Directory containing images.
        transforms: Optional transform pipeline (albumentations or torchvision).
        num_known_classes: Number of known (closed-set) classes.
        use_unknown_class: Whether to include unknown class in training.
        filter_empty: Whether to filter out images with no annotations.
        max_samples: Optional limit on number of samples (for debugging).
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        transforms: Optional[Callable] = None,
        num_known_classes: int = 4,
        use_unknown_class: bool = False,
        filter_empty: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.num_known_classes = num_known_classes
        self.use_unknown_class = use_unknown_class

        self.data = self._load_annotations(ann_file)
        self.ids = list(self.data["images"].keys())

        if filter_empty:
            self.ids = [
                img_id for img_id in self.ids if img_id in self.data["annotations"]
            ]

        if max_samples is not None:
            self.ids = self.ids[:max_samples]

    def _load_annotations(self, ann_file: str) -> Dict[str, Any]:
        """Load and index COCO-format annotations."""
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Index images by id
        images = {img["id"]: img for img in coco_data["images"]}

        # Build category mapping (COCO cat_id -> local class id)
        categories = {cat["id"]: cat for cat in coco_data.get("categories", [])}
        cat_id_map = self._build_category_map(categories)

        # Index annotations by image_id
        annotations: Dict[int, List[Dict]] = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "cat_id_map": cat_id_map,
        }

    def _build_category_map(
        self, categories: Dict[int, Dict]
    ) -> Dict[int, int]:
        """Map COCO category IDs to local class IDs."""
        cat_id_map = {}
        for coco_id, cat in categories.items():
            name = cat["name"]
            if name in CATEGORY_NAME_TO_ID:
                cat_id_map[coco_id] = CATEGORY_NAME_TO_ID[name]
            else:
                # Map to unknown class
                cat_id_map[coco_id] = CATEGORY_NAME_TO_ID["unknown_violation"]
        return cat_id_map

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.ids[idx]
        img_info = self.data["images"][img_id]
        anns = self.data["annotations"].get(img_id, [])

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Parse annotations
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy format
            coco_cat_id = ann["category_id"]
            label = self.data["cat_id_map"].get(
                coco_cat_id, CATEGORY_NAME_TO_ID["unknown_violation"]
            )
            # Remap unknown labels during training if not using unknown class
            if not self.use_unknown_class and label >= self.num_known_classes:
                label = self.num_known_classes - 1  # Assign to last known class
            labels.append(label)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
        areas = np.array(areas, dtype=np.float32) if areas else np.zeros(0, dtype=np.float32)

        # Apply transforms (albumentations-compatible)
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                class_labels=labels.tolist(),
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed["class_labels"], dtype=np.int64) if transformed["class_labels"] else np.zeros(0, dtype=np.int64)

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.uint8),
            "orig_size": torch.tensor(
                [img_info["height"], img_info["width"]], dtype=torch.long
            ),
        }

        return {"image": image, "target": target, "img_id": img_id}

    def get_category_info(self) -> Dict[str, Any]:
        """Return category metadata."""
        return {
            "categories": VIOLATION_CATEGORIES,
            "num_known_classes": self.num_known_classes,
            "cat_id_map": self.data["cat_id_map"],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        class_counts: Dict[int, int] = {}
        total_annotations = 0
        for img_id in self.ids:
            for ann in self.data["annotations"].get(img_id, []):
                coco_id = ann["category_id"]
                label = self.data["cat_id_map"].get(coco_id, self.num_known_classes)
                class_counts[label] = class_counts.get(label, 0) + 1
                total_annotations += 1

        return {
            "num_images": len(self.ids),
            "num_annotations": total_annotations,
            "class_distribution": {
                CATEGORY_ID_TO_NAME.get(k, f"class_{k}"): v
                for k, v in class_counts.items()
            },
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for variable-size detection targets."""
    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]
    img_ids = [item["img_id"] for item in batch]
    return {"images": images, "targets": targets, "img_ids": img_ids}


def build_dataloader(
    ann_file: str,
    img_dir: str,
    transforms: Optional[Callable],
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    num_known_classes: int = 4,
    use_unknown_class: bool = False,
    pin_memory: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for the violation detection dataset."""
    dataset = COCOViolationDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=transforms,
        num_known_classes=num_known_classes,
        use_unknown_class=use_unknown_class,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
