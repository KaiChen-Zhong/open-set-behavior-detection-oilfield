#!/usr/bin/env python3
"""
Demo video generation script.

Creates a synthetic demonstration video with annotated safety violations
for project showcase purposes (useful when real demo footage is unavailable).

Usage:
    python scripts/generate_demo_video.py \\
        --checkpoint checkpoints/best_model.pth \\
        --input-images data/demo_images/ \\
        --output demo/oilfield_safety_demo.mp4 \\
        --fps 15
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo video from images")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional model checkpoint for running actual inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_config.yaml",
    )
    parser.add_argument(
        "--input-images",
        type=str,
        default=None,
        help="Directory of images to create video from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo/oilfield_safety_demo.mp4",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def create_synthetic_frame(
    width: int = 1280,
    height: int = 720,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Create a synthetic oil depot scene with annotated violations.

    This is used when no real footage is available, to generate a
    visually clear demonstration of the detection system capabilities.
    """
    # Background: industrial gray-brown
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background (sky + ground)
    for y in range(height):
        sky_color = np.array([200, 180, 160]) * (1 - y / height)
        ground_color = np.array([100, 90, 80]) * (y / height)
        frame[y] = (sky_color + ground_color).clip(0, 255).astype(np.uint8)

    # Draw industrial scene elements
    # Storage tanks
    for i, (cx, cy, r) in enumerate([
        (200, 400, 80), (400, 380, 100), (900, 420, 90)
    ]):
        cv2.ellipse(frame, (cx, cy), (r, r // 3), 0, 180, 360, (120, 120, 120), -1)
        cv2.rectangle(frame, (cx - r, cy), (cx + r, cy + 120), (100, 100, 100), -1)
        cv2.ellipse(frame, (cx, cy + 120), (r, r // 3), 0, 0, 180, (80, 80, 80), -1)

    # Ground
    cv2.rectangle(frame, (0, 550), (width, height), (70, 65, 60), -1)

    # Safety violation annotations
    violations = [
        # no_helmet
        {
            "box": [650 + int(10 * np.sin(frame_idx * 0.05)), 250, 720, 380],
            "label": "NO HELMET",
            "color": (0, 0, 255),
        },
        # phone usage
        {
            "box": [350, 300, 430, 440],
            "label": "PHONE USAGE",
            "color": (0, 140, 255),
        },
        # gas cylinder violation
        {
            "box": [800, 460, 880, 540],
            "label": "GAS CYLINDER VIOL.",
            "color": (0, 215, 255),
        },
    ]

    for v in violations:
        x1, y1, x2, y2 = v["box"]
        color = v["color"]
        # Pulsing effect
        thickness = 2 + (frame_idx // 10) % 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # Label
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(v["label"]) * 9, y1), color, cv2.FILLED)
        cv2.putText(
            frame, v["label"], (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Spatial violation line (distance warning)
    if frame_idx > 30:
        cx1 = 685 + int(10 * np.sin(frame_idx * 0.05))
        cy1 = 315
        cx2 = 840
        cy2 = 500
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2, cv2.LINE_AA)
        mid_x = (cx1 + cx2) // 2
        mid_y = (cy1 + cy2) // 2
        cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(
            frame, "UNSAFE PROXIMITY",
            (mid_x + 12, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
        )

    # Header bar
    cv2.rectangle(frame, (0, 0), (width, 40), (30, 30, 30), cv2.FILLED)
    cv2.putText(
        frame,
        "OILFIELD SAFETY MONITORING SYSTEM | Open-Set Multi-Task Detection",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1,
    )

    # Alert banner
    if len(violations) > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 40), (width, height), (0, 0, 180), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(
            frame,
            f"⚠ SAFETY ALERT: {len(violations)} VIOLATION(S) DETECTED",
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    # Frame counter
    cv2.putText(
        frame, f"Frame: {frame_idx:04d}",
        (width - 150, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
    )

    return frame


def generate_from_images(
    image_dir: str,
    output_path: str,
    model_runner: Optional[callable],
    fps: int = 15,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Generate demo video from real images with model inference."""
    image_dir = Path(image_dir)
    image_files = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height))

        if model_runner is not None:
            frame = model_runner(frame)

        writer.write(frame)
        print(f"Processed {i+1}/{len(image_files)}: {img_path.name}")

    writer.release()
    print(f"Demo video saved to: {output_path}")


def main() -> None:
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output, fourcc, args.fps, (args.width, args.height)
    )

    if args.input_images and Path(args.input_images).exists():
        # Real images with optional model inference
        model_runner = None
        if args.checkpoint and Path(args.checkpoint).exists():
            try:
                from src.models import build_model
                import yaml

                with open(args.config, "r") as f:
                    cfg = yaml.safe_load(f)
                data_cfg = cfg.get("data", {})
                device = torch.device(args.device)
                model = build_model({
                    "backbone": cfg.get("model", {}).get("backbone", {"type": "CSPDarknet", "variant": "n"}),
                    "fpn_out_channels": 256,
                    "num_known_classes": data_cfg.get("num_known_classes", 4),
                    "num_classes": data_cfg.get("num_classes", 5),
                })
                ckpt = torch.load(args.checkpoint, map_location=device)
                if "model_state_dict" in ckpt:
                    ckpt = ckpt["model_state_dict"]
                model.load_state_dict(ckpt, strict=False)
                model.eval().to(device)

                from src.utils import Visualizer
                from src.utils.visualization import CLASS_NAMES
                visualizer = Visualizer(class_names=CLASS_NAMES)

                @torch.no_grad()
                def _runner(frame):
                    inp = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    inp = inp.unsqueeze(0).to(device)
                    outputs = model(inp)
                    pred = {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64), "scores": np.zeros(0)}
                    return visualizer.visualize(frame, pred)

                model_runner = _runner
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

        generate_from_images(
            args.input_images, args.output, model_runner,
            fps=args.fps, width=args.width, height=args.height,
        )
    else:
        # Synthetic demo
        print(f"Generating synthetic demo video: {args.output}")
        num_frames = args.fps * 10  # 10 seconds
        for frame_idx in range(num_frames):
            frame = create_synthetic_frame(args.width, args.height, frame_idx)
            writer.write(frame)

        writer.release()
        print(f"Synthetic demo video saved to: {args.output} ({num_frames} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
