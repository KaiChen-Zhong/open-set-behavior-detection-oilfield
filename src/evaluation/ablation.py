"""
Ablation study framework for systematically evaluating model components.

Allows running controlled experiments to measure the contribution of:
- Different backbone architectures
- Open-set detection head variants
- Spatial reasoning module
- Multi-task learning vs. single-task
- Continual learning strategies
"""

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class AblationConfig:
    """Configuration for a single ablation experiment."""

    def __init__(
        self,
        name: str,
        description: str,
        model_modifications: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.model_modifications = model_modifications

    def __repr__(self) -> str:
        return f"AblationConfig(name={self.name})"


# Predefined ablation configurations
STANDARD_ABLATIONS: List[Dict] = [
    {
        "name": "no_spatial_reasoning",
        "description": "Disable spatial reasoning module",
        "model_modifications": {
            "enable_spatial_reasoning": False,
        },
    },
    {
        "name": "no_classification_head",
        "description": "Disable global classification task",
        "model_modifications": {
            "enable_classification": False,
        },
    },
    {
        "name": "single_task_detection_only",
        "description": "Detection only (no multi-task)",
        "model_modifications": {
            "enable_spatial_reasoning": False,
            "enable_classification": False,
        },
    },
    {
        "name": "csp_backbone",
        "description": "Use CSPDarknet instead of Swin Transformer",
        "model_modifications": {
            "backbone_cfg": {"type": "CSPDarknet", "variant": "n"},
        },
    },
    {
        "name": "no_prototypes",
        "description": "Disable prototype memory in detection head",
        "model_modifications": {
            "detection_head_cfg": {"use_prototypes": False},
        },
    },
]


class AblationStudy:
    """
    Framework for running systematic ablation studies.

    Args:
        base_model_builder: Callable that builds a base model from config.
        base_cfg: Base model configuration dict.
        eval_fn: Callable that evaluates a model and returns metrics dict.
        output_dir: Directory to save ablation results.
    """

    def __init__(
        self,
        base_model_builder: Callable[[Dict], nn.Module],
        base_cfg: Dict[str, Any],
        eval_fn: Callable[[nn.Module], Dict[str, float]],
        output_dir: str = "ablation_results",
    ):
        self.base_model_builder = base_model_builder
        self.base_cfg = base_cfg
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def run_experiment(
        self,
        ablation_cfg: Dict[str, Any],
        pretrained_weights: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run a single ablation experiment.

        Args:
            ablation_cfg: Ablation configuration dict.
            pretrained_weights: Optional path to pretrained weights.

        Returns:
            Metrics dictionary.
        """
        name = ablation_cfg["name"]
        description = ablation_cfg.get("description", "")
        modifications = ablation_cfg.get("model_modifications", {})

        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"  {description}")
        print(f"{'='*60}")

        # Build modified config
        modified_cfg = copy.deepcopy(self.base_cfg)
        for key, value in modifications.items():
            if isinstance(value, dict) and key in modified_cfg:
                modified_cfg[key].update(value)
            else:
                modified_cfg[key] = value

        # Build model
        try:
            model = self.base_model_builder(modified_cfg)
        except Exception as e:
            print(f"  ERROR building model: {e}")
            self.results[name] = {"error": str(e)}
            return {}

        # Load pretrained weights (shared backbone if available)
        if pretrained_weights is not None:
            try:
                state_dict = torch.load(pretrained_weights, map_location="cpu")
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                # Load compatible weights only
                model_state = model.state_dict()
                compatible = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                model.load_state_dict(compatible, strict=False)
                print(f"  Loaded {len(compatible)}/{len(model_state)} weights")
            except Exception as e:
                print(f"  Warning: Could not load weights: {e}")

        # Run evaluation
        try:
            metrics = self.eval_fn(model)
        except Exception as e:
            print(f"  ERROR during evaluation: {e}")
            metrics = {"error": str(e)}

        self.results[name] = {
            "description": description,
            "modifications": modifications,
            "metrics": metrics,
        }

        # Save results
        self._save_results()

        return metrics

    def run_all(
        self,
        ablation_configs: Optional[List[Dict]] = None,
        pretrained_weights: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run all ablation experiments.

        Args:
            ablation_configs: List of ablation configs. Defaults to STANDARD_ABLATIONS.
            pretrained_weights: Optional shared pretrained weights path.

        Returns:
            Dictionary mapping experiment names to results.
        """
        if ablation_configs is None:
            ablation_configs = STANDARD_ABLATIONS

        # Run baseline first
        print("\n" + "="*60)
        print("  Running BASELINE (full model)")
        print("="*60)
        baseline_metrics = self.eval_fn(self.base_model_builder(self.base_cfg))
        self.results["baseline"] = {
            "description": "Full model (all components enabled)",
            "modifications": {},
            "metrics": baseline_metrics,
        }

        # Run each ablation
        for cfg in ablation_configs:
            self.run_experiment(cfg, pretrained_weights)

        # Print summary
        self.print_summary()
        return self.results

    def print_summary(self) -> None:
        """Print a summary table of ablation results."""
        print("\n" + "="*80)
        print("  ABLATION STUDY SUMMARY")
        print("="*80)
        header = f"{'Experiment':<35} {'mAP@50':>8} {'F1':>8} {'WI':>8} {'A-OSE':>8}"
        print(header)
        print("-"*80)

        for name, result in self.results.items():
            if "error" in result:
                print(f"  {name:<33} ERROR: {result['error']}")
                continue
            m = result.get("metrics", {})
            mAP = m.get("mAP", float("nan"))
            f1 = m.get("f1", float("nan"))
            wi = m.get("WI", float("nan"))
            aose = m.get("A-OSE", "N/A")
            print(
                f"  {name:<33} {mAP:>8.4f} {f1:>8.4f} "
                f"{wi if isinstance(wi, str) else f'{wi:>8.4f}'} {aose:>8}"
            )

        print("="*80)

    def _save_results(self) -> None:
        """Save results to JSON file."""
        output_path = self.output_dir / "ablation_results.json"

        # Convert non-serializable values
        def make_serializable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            if isinstance(obj, float) and (obj != obj):  # NaN check
                return "NaN"
            return obj

        with open(output_path, "w") as f:
            json.dump(make_serializable(self.results), f, indent=2)

    def compare_with_baseline(
        self, experiment_name: str
    ) -> Dict[str, float]:
        """
        Compare an ablation result with the baseline.

        Returns:
            Dict of metric deltas (positive = improvement over baseline).
        """
        if "baseline" not in self.results:
            raise ValueError("Baseline not found. Run baseline first.")
        if experiment_name not in self.results:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        baseline_m = self.results["baseline"]["metrics"]
        exp_m = self.results[experiment_name]["metrics"]

        deltas = {}
        for key in set(baseline_m) | set(exp_m):
            if key in baseline_m and key in exp_m:
                try:
                    deltas[key] = float(exp_m[key]) - float(baseline_m[key])
                except (TypeError, ValueError):
                    pass

        return deltas
