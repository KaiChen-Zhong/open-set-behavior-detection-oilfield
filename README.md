# Open-Set Multi-Task Detection for Oilfield Safety Violations

An **Open-Set Multi-Task Deep Learning System** for intelligent identification of safety violations and environmental hazards in oilfield maintenance operations. This project implements incremental/continual learning to detect both **known and unknown** violation categories in unconstrained real-world scenarios.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Industrial maintenance sites, particularly oil depots, present complex safety monitoring challenges:
- Workers may not wear required PPE (helmets, vests)
- Improper equipment placement (gas cylinders, crane outriggers)
- Behavioral violations (phone usage in restricted zones)
- **Unknown violation types** that were not seen during training

This system addresses these challenges using:
- **Open-Set Detection**: Detects both known violations and flags unknown ones
- **Multi-Task Learning**: Simultaneous detection, classification, and spatial reasoning
- **Continual Learning**: Incrementally learns new violation types without forgetting old ones

---

## Project Structure

```
open-set-behavior-detection-oilfield/
├── configs/
│   ├── base_config.yaml                  # Base configuration
│   ├── detection/
│   │   ├── yolov8_baseline.yaml          # YOLOv8 baseline (Ultralytics)
│   │   └── open_set_detector.yaml        # Custom open-set detector
│   ├── multitask/
│   │   └── multitask_config.yaml         # Multi-task learning config
│   └── continual/
│       └── continual_config.yaml         # Continual/incremental learning
├── src/
│   ├── data/
│   │   ├── dataset.py                    # COCO-format dataset class
│   │   ├── augmentation.py               # Albumentations pipeline
│   │   └── analysis.py                   # Dataset analysis & visualization
│   ├── models/
│   │   ├── backbone.py                   # CSPDarknet / Swin Transformer
│   │   ├── detection_head.py             # Open-set detection head
│   │   ├── spatial_reasoning.py          # Spatial relationship reasoning
│   │   ├── multitask_model.py            # Multi-task fusion model
│   │   └── continual_learning.py         # Replay buffer & EWC
│   ├── training/
│   │   ├── trainer.py                    # Training loop with W&B
│   │   ├── losses.py                     # Focal, GIoU, SupCon losses
│   │   └── optimizer.py                  # Optimizer & scheduler builders
│   ├── evaluation/
│   │   ├── metrics.py                    # mAP, WI, A-OSE metrics
│   │   ├── evaluator.py                  # Main evaluator
│   │   └── ablation.py                   # Ablation study framework
│   └── utils/
│       ├── visualization.py              # Detection visualization
│       └── logger.py                     # Logging utilities
├── scripts/
│   ├── train.py                          # Main training script
│   ├── evaluate.py                       # Evaluation script
│   ├── compare_baselines.py              # Baseline comparison
│   ├── demo_video.py                     # Real-time video inference
│   └── generate_demo_video.py            # Demo video generation
├── tests/
│   └── test_basic.py                     # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input Images
     │
     ▼
┌─────────────────────────────────────────────┐
│          Shared Backbone                     │
│    (Swin Transformer / CSPDarknet)           │
│    Multi-scale feature maps: P3, P4, P5     │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│     Feature Pyramid Network (FPN/PANet)      │
│     Fuses multi-scale features               │
└─────────────────────────────────────────────┘
     │
     ├──────────────────┬────────────────────┐
     ▼                  ▼                    ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│ Open-Set │    │ Violation    │    │ Spatial          │
│ Detection│    │ Classification│    │ Relationship     │
│ Head     │    │ Head         │    │ Reasoning Head   │
│          │    │              │    │                  │
│ Known +  │    │ Image-level  │    │ Graph Attention  │
│ Unknown  │    │ class scores │    │ + Distance Model │
│ classes  │    │              │    │                  │
└──────────┘    └──────────────┘    └─────────────────┘
     │                  │                    │
     └──────────────────┴────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Cross-Task Attention│
              │  Fusion Module       │
              └─────────────────────┘
                         │
                         ▼
              Final Predictions:
              - Bounding boxes with class labels
              - Confidence scores (known vs. unknown)
              - Spatial violation pairs
```

### Key Components

#### 1. Open-Set Detection Head
- **Unknown-Aware Classifier**: Adjusts logits to detect known and unknown violations
- **Prototype Memory Bank**: Stores per-class feature prototypes updated via EMA
- **Objectness Branch**: Separates object presence from class identity

#### 2. Spatial Relationship Reasoning
- **Relation Embedding**: Encodes geometric relationships between bounding boxes
- **Graph Attention Network**: Propagates spatial context between objects
- **Distance-Based Violation Detection**: Identifies unsafe proximity between hazards

#### 3. Continual Learning
- **Replay Buffer**: Reservoir/herding-based exemplar memory
- **Elastic Weight Consolidation (EWC)**: Prevents forgetting via Fisher regularization
- **Learning Without Forgetting (LwF)**: Knowledge distillation from previous task models

---

## Supported Violation Types

| Category | Description | Detection Approach |
|---|---|---|
| `no_helmet` | Worker not wearing safety helmet | Object detection |
| `phone_usage` | Mobile phone use in restricted area | Behavioral detection |
| `gas_cylinder_viol` | Gas cylinder placement violation | Object + spatial |
| `crane_outrigger` | Crane outrigger not properly deployed | Equipment detection |
| `unknown_violation` | Open-set / unseen violation type | Unknown-aware scoring |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/KaiChen-Zhong/open-set-behavior-detection-oilfield.git
cd open-set-behavior-detection-oilfield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (if CUDA is available):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Dataset Preparation

The system uses **COCO-format annotations**. Prepare your dataset as follows:

### Directory Structure

```
data/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── annotations/
    ├── train.json       # COCO-format training annotations
    ├── val.json         # COCO-format validation annotations
    └── test.json        # COCO-format test annotations
```

### COCO Annotation Format

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg", "height": 720, "width": 1280}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1200.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "no_helmet", "supercategory": "ppe_violation"},
    {"id": 2, "name": "phone_usage", "supercategory": "behavioral_violation"},
    {"id": 3, "name": "gas_cylinder_viol", "supercategory": "equipment_violation"},
    {"id": 4, "name": "crane_outrigger", "supercategory": "equipment_violation"}
  ]
}
```

### Dataset Analysis

```bash
# Analyze your dataset
python -c "
from src.data.analysis import DatasetAnalyzer
analyzer = DatasetAnalyzer('data/annotations/train.json', 'data/images/train')
analyzer.print_summary()
analyzer.plot_class_distribution(save_path='analysis/class_dist.png', show=False)
analyzer.plot_box_size_distribution(save_path='analysis/box_sizes.png', show=False)
"
```

---

## Training

### 1. YOLOv8 Baseline (Ultralytics API)

First, create a dataset YAML for Ultralytics:

```yaml
# data/oilfield.yaml
path: data/
train: images/train
val: images/val
nc: 4
names: [no_helmet, phone_usage, gas_cylinder_viol, crane_outrigger]
```

Then train:

```bash
python scripts/train.py \
    --config configs/detection/yolov8_baseline.yaml \
    --use-ultralytics \
    --device cuda
```

### 2. Custom Open-Set Multi-Task Detector

```bash
# Standard training
python scripts/train.py \
    --config configs/multitask/multitask_config.yaml \
    --device cuda \
    --use-wandb

# Resume from checkpoint
python scripts/train.py \
    --config configs/multitask/multitask_config.yaml \
    --resume checkpoints/checkpoint_epoch_0050.pth

# Debug mode (small subset)
python scripts/train.py \
    --config configs/multitask/multitask_config.yaml \
    --debug
```

### 3. Continual / Incremental Learning

```bash
# Incremental learning with replay buffer
python scripts/train.py \
    --config configs/continual/continual_config.yaml \
    --device cuda
```

### Training Configuration

Edit `configs/base_config.yaml` to customize:

```yaml
training:
  epochs: 100
  batch_size: 16
  device: "cuda"
  amp: true  # Automatic Mixed Precision

optimizer:
  name: "AdamW"
  lr: 1.0e-4
  weight_decay: 1.0e-4

wandb:
  enabled: true
  project: "oilfield-safety-detection"
```

---

## Evaluation

### Standard Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/multitask/multitask_config.yaml \
    --split test \
    --output results/test_metrics.json
```

### Baseline Comparison

```bash
python scripts/compare_baselines.py \
    --baseline-checkpoint checkpoints/yolov8_baseline.pth \
    --proposed-checkpoint checkpoints/best_model.pth \
    --config configs/multitask/multitask_config.yaml \
    --output results/comparison.json
```

### Ablation Study

```python
from src.models.multitask_model import build_model
from src.evaluation import AblationStudy

# Define evaluation function
def eval_fn(model):
    evaluator = Evaluator(model, test_loader, ...)
    return evaluator.run()

# Run ablation
study = AblationStudy(
    base_model_builder=build_model,
    base_cfg=base_config,
    eval_fn=eval_fn,
    output_dir="ablation_results/",
)
study.run_all()  # Tests all standard ablations
```

---

## Inference and Demo

### Real-Time Video Inference

```bash
# On a video file
python scripts/demo_video.py \
    --checkpoint checkpoints/best_model.pth \
    --source data/demo_video.mp4 \
    --output output/demo_result.mp4 \
    --show-spatial \
    --show-fps

# On webcam
python scripts/demo_video.py \
    --checkpoint checkpoints/best_model.pth \
    --source 0
```

### Generate Demo Video

```bash
# Generate synthetic demo (no real footage needed)
python scripts/generate_demo_video.py \
    --output demo/oilfield_safety_demo.mp4 \
    --fps 15

# Generate from real images with model inference
python scripts/generate_demo_video.py \
    --checkpoint checkpoints/best_model.pth \
    --input-images data/demo_images/ \
    --output demo/oilfield_demo_real.mp4
```

### Programmatic Inference

```python
import torch
from src.models import build_model
from src.utils import Visualizer

# Load model
model = build_model({
    "backbone": {"type": "SwinTransformer", "variant": "swin_tiny"},
    "num_known_classes": 4,
    "num_classes": 5,
})
model.load_state_dict(torch.load("checkpoints/best_model.pth")["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    outputs = model(images)

# Visualize
visualizer = Visualizer(conf_threshold=0.25)
annotated = visualizer.visualize(image_np, prediction)
```

---

## Evaluation Metrics

### Standard Detection Metrics

| Metric | Description |
|---|---|
| **mAP@50** | Mean AP at IoU=0.5 |
| **mAP@75** | Mean AP at IoU=0.75 |
| **Precision** | TP / (TP + FP) at fixed threshold |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of precision and recall |

### Open-Set Metrics

| Metric | Description | Lower is Better |
|---|---|---|
| **WI** (Wilderness Impact) | Precision drop from unknown objects | ✓ |
| **A-OSE** (Absolute Open-Set Error) | Count of unknowns classified as known | ✓ |

### Expected Results (Reference)

| Model | mAP@50 | mAP@75 | WI↓ | A-OSE↓ |
|---|---|---|---|---|
| YOLOv8n (baseline) | ~65% | ~42% | N/A | N/A |
| Open-Set Detector | ~68% | ~45% | ~0.05 | ~120 |
| Multi-Task + Spatial | ~71% | ~48% | ~0.03 | ~85 |

*Results depend on dataset size and quality. Train on your own data for accurate numbers.*

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_basic.py::TestMultiTaskDetector -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## W&B Experiment Tracking

Enable Weights & Biases tracking:

```bash
# Login to W&B
wandb login

# Set W&B project in config
# configs/base_config.yaml:
# wandb:
#   enabled: true
#   project: "oilfield-safety-detection"
#   entity: "your-username"

# Train with tracking
python scripts/train.py --config configs/multitask/multitask_config.yaml --use-wandb
```

---

## Extending the System

### Adding New Violation Categories

1. Add category to `src/data/dataset.py`:
   ```python
   VIOLATION_CATEGORIES["new_violation"] = {
       "id": 5, "supercategory": "...", "description": "..."
   }
   ```

2. Update `num_classes` in your config YAML

3. Run continual learning to add without forgetting:
   ```bash
   python scripts/train.py \
       --config configs/continual/continual_config.yaml \
       --resume checkpoints/best_model.pth
   ```

### Using Grounding DINO for Open-Vocabulary Detection

Set `grounding_dino.enabled: true` in `configs/detection/open_set_detector.yaml`:

```yaml
grounding_dino:
  enabled: true
  model_id: "IDEA-Research/grounding-dino-tiny"
  text_queries:
    - "worker not wearing helmet"
    - "gas cylinder placed improperly"
    - "unknown safety violation"
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{oilfield-safety-detection-2024,
  title = {Open-Set Multi-Task Detection for Oilfield Safety Violations},
  author = {KaiChen-Zhong},
  year = {2024},
  url = {https://github.com/KaiChen-Zhong/open-set-behavior-detection-oilfield},
}
```

### Related Work

- [YOLOv8](https://github.com/ultralytics/ultralytics) (Jocher et al., 2023)
- [OpenDet](https://github.com/csuhan/opendet2) (Han et al., CVPR 2022)
- [Towards Open World Object Detection](https://github.com/JosephKJ/OWOD) (Joseph et al., CVPR 2021)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Liu et al., 2023)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Liu et al., ICCV 2021)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
