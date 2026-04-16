"""
Open-Set Safety Violation Detection for Oil Depot Maintenance.
"""

from src.data import COCOViolationDataset, ViolationAugmentation
from src.models import MultiTaskDetector
from src.training import Trainer
from src.evaluation import Evaluator

__version__ = "1.0.0"
__all__ = [
    "COCOViolationDataset",
    "ViolationAugmentation",
    "MultiTaskDetector",
    "Trainer",
    "Evaluator",
]
