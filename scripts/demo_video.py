#!/usr/bin/env python3
"""
Real-time video inference demo for open-set safety violation detection.

Runs the trained model on a video file or webcam feed,
displaying detections, spatial relationship violations, and a live alert overlay.

Usage:
    # On a video file:
    python scripts/demo_video.py \\
        --checkpoint checkpoints/best_model.pth \\
        --source data/demo_video.mp4 \\
        --output output/demo_result.mp4

    # On webcam:
    python scripts/demo_video.py \\
        --checkpoint checkpoints/best_model.pth \\
        --source 0

    # With spatial reasoning visualization:
    python scripts/demo_video.py \\
        --checkpoint checkpoints/best_model.pth \\
        --source data/demo_video.mp4 \\
        --show-spatial --show-heatmap
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import yaml

from src.models import build_model
from src.utils import Visualizer
from src.utils.visualization import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time video inference demo")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_config.yaml",
        help="Model config path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: path to video file or webcam index (0, 1, ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--show-spatial", action="store_true")
    parser.add_argument("--show-heatmap", action="store_true")
    parser.add_argument("--show-fps", action="store_true", default=True)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (for headless environments)",
    )
    return parser.parse_args()


def preprocess(
    frame: np.ndarray, img_size: int, device: torch.device
) -> torch.Tensor:
    """Preprocess a video frame for inference."""
    # Resize
    h, w = frame.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Pad to square
    pad_h = img_size - new_h
    pad_w = img_size - new_w
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=114)

    # Normalize and convert to tensor
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
             torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.unsqueeze(0).to(device)


def postprocess(
    outputs: Dict,
    orig_shape: tuple,
    img_size: int,
    conf_threshold: float = 0.25,
) -> Dict:
    """Decode model outputs to detection results."""
    h, w = orig_shape[:2]
    scale = img_size / max(h, w)

    if "det_cls_scores" not in outputs or not outputs["det_cls_scores"]:
        return {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64), "scores": np.zeros(0)}

    # Use first FPN level for simplified demo
    cls_map = outputs["det_cls_scores"][0][0]  # (C, H, W)
    C, fH, fW = cls_map.shape

    scores_flat, pred_labels = cls_map.reshape(C, -1).permute(1, 0).sigmoid().max(dim=-1)
    keep = scores_flat > conf_threshold

    if keep.sum() == 0:
        return {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64), "scores": np.zeros(0)}

    # Grid positions -> boxes (simplified)
    ys = torch.arange(fH, device=cls_map.device).float()
    xs = torch.arange(fW, device=cls_map.device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_flat = torch.stack([
        grid_x.reshape(-1), grid_y.reshape(-1),
        grid_x.reshape(-1) + 1, grid_y.reshape(-1) + 1,
    ], dim=-1)  # (H*W, 4)

    # Scale to original image size
    scale_tensor = torch.tensor([w / fW, h / fH, w / fW, h / fH], device=cls_map.device)
    boxes = (grid_flat[keep] * scale_tensor).cpu().numpy()
    scores = scores_flat[keep].cpu().numpy()
    labels = pred_labels[keep].cpu().numpy()

    return {"boxes": boxes, "scores": scores, "labels": labels}


def add_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """Add FPS counter to frame."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (frame.shape[1] - 120, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return frame


def add_alert_banner(frame: np.ndarray, num_violations: int) -> np.ndarray:
    """Add alert banner if violations detected."""
    if num_violations > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 45), (frame.shape[1], frame.shape[0]), (0, 0, 200), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(
            frame,
            f"⚠ SAFETY VIOLATION DETECTED ({num_violations} instance{'s' if num_violations > 1 else ''})",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2,
        )
    return frame


@torch.no_grad()
def run_demo(args: argparse.Namespace) -> None:
    """Run the video inference demo."""
    # Load config and model
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("_base_", None)

    device = torch.device(args.device)
    data_cfg = cfg.get("data", {})

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

    visualizer = Visualizer(
        class_names=CLASS_NAMES,
        conf_threshold=args.conf_threshold,
        show_unknown=True,
    )

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (frame_w, frame_h))

    # Inference loop
    frame_count = 0
    fps_tracker = []
    print(f"Running inference on: {args.source}")
    print(f"Press 'q' to quit, 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # Preprocess
        inp = preprocess(frame, args.img_size, device)

        # Inference
        outputs = model(inp)

        # Postprocess
        prediction = postprocess(outputs, frame.shape, args.img_size, args.conf_threshold)

        # Visualize
        vis_frame = visualizer.visualize(
            frame,
            prediction,
            show_heatmap=args.show_heatmap,
            show_spatial=args.show_spatial,
        )

        # FPS overlay
        elapsed = time.time() - t0
        fps_tracker.append(1.0 / (elapsed + 1e-6))
        if len(fps_tracker) > 30:
            fps_tracker.pop(0)
        avg_fps = sum(fps_tracker) / len(fps_tracker)

        if args.show_fps:
            vis_frame = add_fps_overlay(vis_frame, avg_fps)

        num_violations = len(prediction["boxes"])
        vis_frame = add_alert_banner(vis_frame, num_violations)

        # Write output
        if writer is not None:
            writer.write(vis_frame)

        # Display
        if not args.no_display:
            cv2.imshow("Oilfield Safety Monitor", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                screenshot_path = f"screenshot_{frame_count:04d}.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"Screenshot saved: {screenshot_path}")

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames | FPS: {avg_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_count} frames.")
    if args.output:
        print(f"Output saved to: {args.output}")


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
