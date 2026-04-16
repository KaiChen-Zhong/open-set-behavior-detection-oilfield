"""
Dataset analysis and visualization scripts for oilfield safety violation data.

Provides tools for:
- Class distribution analysis
- Bounding box statistics
- Image quality assessment
- Visualization of annotations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .dataset import CATEGORY_ID_TO_NAME, VIOLATION_CATEGORIES


# Color palette for violation categories
VIOLATION_COLORS = {
    "no_helmet": "#FF4444",         # Red
    "phone_usage": "#FF8C00",       # Orange
    "gas_cylinder_viol": "#FFD700", # Yellow
    "crane_outrigger": "#00CED1",   # Teal
    "unknown_violation": "#9370DB", # Purple
}


class DatasetAnalyzer:
    """
    Analyzes COCO-format annotation datasets for the oilfield safety detection task.

    Args:
        ann_file: Path to COCO-format annotation JSON file.
        img_dir: Optional directory for images (needed for image-level analysis).
    """

    def __init__(self, ann_file: str, img_dir: Optional[str] = None):
        self.ann_file = ann_file
        self.img_dir = Path(img_dir) if img_dir else None
        self.data = self._load(ann_file)

    def _load(self, ann_file: str) -> Dict:
        with open(ann_file, "r") as f:
            return json.load(f)

    def get_summary(self) -> Dict:
        """Return a summary of the dataset."""
        images = self.data.get("images", [])
        annotations = self.data.get("annotations", [])
        categories = self.data.get("categories", [])

        class_counts: Dict[str, int] = {}
        box_areas = []
        aspect_ratios = []

        for ann in annotations:
            cat_id = ann["category_id"]
            # Find category name
            cat_name = next(
                (c["name"] for c in categories if c["id"] == cat_id),
                "unknown",
            )
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
            x, y, w, h = ann["bbox"]
            box_areas.append(w * h)
            if h > 0:
                aspect_ratios.append(w / h)

        return {
            "num_images": len(images),
            "num_annotations": len(annotations),
            "num_categories": len(categories),
            "avg_annotations_per_image": (
                len(annotations) / len(images) if images else 0
            ),
            "class_distribution": class_counts,
            "box_stats": {
                "mean_area": float(np.mean(box_areas)) if box_areas else 0,
                "median_area": float(np.median(box_areas)) if box_areas else 0,
                "std_area": float(np.std(box_areas)) if box_areas else 0,
                "mean_aspect_ratio": (
                    float(np.mean(aspect_ratios)) if aspect_ratios else 0
                ),
            },
        }

    def plot_class_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot bar chart of class distribution."""
        summary = self.get_summary()
        class_dist = summary["class_distribution"]

        # Sort by violation type order
        ordered_names = list(VIOLATION_CATEGORIES.keys())
        counts = [class_dist.get(name, 0) for name in ordered_names]
        colors = [
            VIOLATION_COLORS.get(name, "#888888") for name in ordered_names
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(ordered_names, counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Violation Category", fontsize=12)
        ax.set_ylabel("Number of Annotations", fontsize=12)
        ax.set_title(
            "Class Distribution in Oilfield Safety Violation Dataset",
            fontsize=14,
        )
        ax.tick_params(axis="x", rotation=30)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def plot_box_size_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot distribution of bounding box sizes."""
        annotations = self.data.get("annotations", [])
        categories = self.data.get("categories", [])
        cat_map = {c["id"]: c["name"] for c in categories}

        # Collect areas per class
        class_areas: Dict[str, List[float]] = {}
        for ann in annotations:
            _, _, w, h = ann["bbox"]
            area = w * h
            cat_name = cat_map.get(ann["category_id"], "unknown")
            class_areas.setdefault(cat_name, []).append(area)

        fig, axes = plt.subplots(
            1, len(class_areas), figsize=(4 * len(class_areas), 4)
        )
        if len(class_areas) == 1:
            axes = [axes]

        for ax, (name, areas) in zip(axes, class_areas.items()):
            color = VIOLATION_COLORS.get(name, "#888888")
            ax.hist(areas, bins=30, color=color, edgecolor="black", linewidth=0.3)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Box Area (px²)")
            ax.set_ylabel("Count")

        plt.suptitle("Bounding Box Size Distribution by Class", fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def visualize_sample(
        self,
        img_id: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """Visualize annotations on a sample image."""
        if self.img_dir is None:
            raise ValueError("img_dir must be provided to visualize samples.")

        images = {img["id"]: img for img in self.data.get("images", [])}
        categories = {c["id"]: c["name"] for c in self.data.get("categories", [])}

        if img_id not in images:
            print(f"Image ID {img_id} not found.")
            return None

        img_info = images[img_id]
        img_path = self.img_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"Image file not found: {img_path}")
            return None

        image = np.array(Image.open(img_path).convert("RGB"))
        anns = [
            a for a in self.data.get("annotations", []) if a["image_id"] == img_id
        ]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        legend_patches = {}
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_name = categories.get(ann["category_id"], "unknown")
            color = VIOLATION_COLORS.get(cat_name, "#888888")
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x, y - 4, cat_name.replace("_", " "),
                fontsize=9, color="white",
                bbox=dict(facecolor=color, alpha=0.8, pad=1),
            )
            if cat_name not in legend_patches:
                legend_patches[cat_name] = mpatches.Patch(
                    color=color, label=cat_name.replace("_", " ")
                )

        ax.legend(handles=list(legend_patches.values()), loc="upper right")
        ax.set_title(f"Image: {img_info['file_name']} | Annotations: {len(anns)}")
        ax.axis("off")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def print_summary(self) -> None:
        """Print formatted dataset summary to stdout."""
        summary = self.get_summary()
        print("=" * 60)
        print("  OILFIELD SAFETY VIOLATION DATASET SUMMARY")
        print("=" * 60)
        print(f"  Total Images       : {summary['num_images']:,}")
        print(f"  Total Annotations  : {summary['num_annotations']:,}")
        print(f"  Number of Classes  : {summary['num_categories']}")
        print(
            f"  Avg Ann/Image      : {summary['avg_annotations_per_image']:.2f}"
        )
        print("-" * 60)
        print("  Class Distribution:")
        for cls, cnt in sorted(
            summary["class_distribution"].items(), key=lambda x: -x[1]
        ):
            bar = "█" * min(int(cnt / max(1, summary["num_annotations"]) * 40), 40)
            print(f"    {cls:<25} {cnt:>6}  {bar}")
        print("-" * 60)
        bs = summary["box_stats"]
        print(f"  Box Mean Area      : {bs['mean_area']:.1f} px²")
        print(f"  Box Median Area    : {bs['median_area']:.1f} px²")
        print(f"  Box Std Area       : {bs['std_area']:.1f} px²")
        print(f"  Mean Aspect Ratio  : {bs['mean_aspect_ratio']:.3f}")
        print("=" * 60)
