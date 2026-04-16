"""
Logging utilities for experiment tracking and debugging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "oilfield_safety",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level.
        format_str: Optional custom format string.

    Returns:
        Configured logger.
    """
    if format_str is None:
        format_str = (
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """
    Simple metric logger for tracking training and evaluation metrics.

    Stores a history of metrics and provides aggregation utilities.
    """

    def __init__(self):
        self._history: dict = {}

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(
                {"value": value, "step": step}
            )

    def get(self, key: str) -> list:
        """Get history for a specific metric."""
        return self._history.get(key, [])

    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value for a metric."""
        history = self.get(key)
        return history[-1]["value"] if history else None

    def summary(self) -> dict:
        """Return summary statistics for all tracked metrics."""
        import statistics
        summary = {}
        for key, values in self._history.items():
            vals = [v["value"] for v in values if isinstance(v["value"], (int, float))]
            if vals:
                summary[key] = {
                    "last": vals[-1],
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.mean(vals),
                }
        return summary

    def clear(self) -> None:
        """Clear all logged metrics."""
        self._history.clear()
