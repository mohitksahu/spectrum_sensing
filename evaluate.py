#!/usr/bin/env python3
"""Evaluation script for SpectraSense model.

Loads best checkpoint and evaluates on validation or test set with
comprehensive metrics and visualization generation.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --split test
    python evaluate.py --checkpoint checkpoints/best.pt --split val --save-figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.spectrasense import SpectraSense, build_spectrasense
from src.datasets.spectrum_dataset import SpectrumDataset
from src.evaluation.evaluator import SpectraSenseEvaluator
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_analysis import plot_roc_curve, plot_precision_recall_curve
from src.evaluation.snr_analysis import (
    plot_snr_error_distribution,
    plot_per_snr_accuracy,
    plot_training_curves,
)
from src.datasets.preprocessing import MODULATION_NAMES
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpectraSense model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml",
                        help="Model configuration file")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Evaluation batch size")
    parser.add_argument("--save-figures", action="store_true", default=True,
                        help="Save evaluation figures")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory for reports")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for evaluation")
    args = parser.parse_args()
    
    set_seed(42)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Load model config
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    
    # Build and load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = build_spectrasense(model_config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"  Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Load data
    data_path = Path(args.data_dir) / f"{args.split}.pt"
    print(f"\nLoading {args.split} data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    dataset = SpectrumDataset(
        psds=data["psds"].numpy() if isinstance(data["psds"], torch.Tensor) else data["psds"],
        pu_labels=data["pu_labels"].numpy() if isinstance(data["pu_labels"], torch.Tensor) else data["pu_labels"],
        mod_labels=data["mod_labels"].numpy() if isinstance(data["mod_labels"], torch.Tensor) else data["mod_labels"],
        snrs=data["snrs"].numpy() if isinstance(data["snrs"], torch.Tensor) else data["snrs"],
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"  Samples: {len(dataset):,}")
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = SpectraSenseEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=str(metrics_dir),
    )
    
    results = evaluator.evaluate()

    # Print per-SNR PU performance table
    per_snr = results.get("per_snr", {})
    per_snr_bins = []
    for key, metrics in per_snr.items():
        if key.endswith("dB") and not key.startswith("low_"):
            try:
                snr_value = int(key.replace("dB", ""))
            except ValueError:
                continue
            per_snr_bins.append((snr_value, key, metrics))
    per_snr_bins.sort(key=lambda item: item[0])

    if per_snr_bins:
        print("\nPer-SNR PU Performance (TPR/TNR/Balanced Acc)")
        header = f"{'SNR':>6} {'n_active':>9} {'n_inactive':>11} {'TPR':>8} {'TNR':>8} {'BalAcc':>8}"
        print(header)
        print("-" * len(header))

        def _fmt_rate(value: float) -> str:
            if np.isnan(value):
                return "  nan  "
            return f"{value * 100:6.2f}%"

        for _, key, metrics in per_snr_bins:
            tpr = metrics.get("pu_tpr", float("nan"))
            tnr = metrics.get("pu_tnr", float("nan"))
            bal = metrics.get("pu_balanced_accuracy", float("nan"))
            print(
                f"{key:>6}"
                f" {metrics.get('n_active', 0):>9}"
                f" {metrics.get('n_inactive', 0):>11}"
                f" {_fmt_rate(tpr):>8}"
                f" {_fmt_rate(tnr):>8}"
                f" {_fmt_rate(bal):>8}"
            )
    
    # Generate figures
    if args.save_figures:
        print("\nGenerating evaluation figures...")
        
        # 1. PU Confusion Matrix
        plot_confusion_matrix(
            cm=np.array(results["confusion_matrices"]["pu"]),
            class_names=["Idle", "Active"],
            title="PU Detection Confusion Matrix",
            save_path=str(figures_dir / "pu_confusion_matrix"),
        )
        
        # 2. Modulation Confusion Matrix
        mod_names = [MODULATION_NAMES[i] for i in range(5)]
        plot_confusion_matrix(
            cm=np.array(results["confusion_matrices"]["mod"]),
            class_names=mod_names,
            title="Modulation Classification Confusion Matrix",
            save_path=str(figures_dir / "mod_confusion_matrix"),
        )
        
        # 3. ROC Curve
        plot_roc_curve(
            targets=np.array(results["roc_data"]["pu_targets"]),
            probs=np.array(results["roc_data"]["pu_probs"]),
            title="ROC Curve - PU Detection",
            save_path=str(figures_dir / "roc_curve"),
        )
        
        # 4. Precision-Recall Curve
        plot_precision_recall_curve(
            targets=np.array(results["roc_data"]["pu_targets"]),
            probs=np.array(results["roc_data"]["pu_probs"]),
            title="Precision-Recall Curve - PU Detection",
            save_path=str(figures_dir / "pr_curve"),
        )
        
        # 5. SNR Error Distribution
        plot_snr_error_distribution(
            targets=np.array(results["snr_data"]["targets"]),
            predictions=np.array(results["snr_data"]["predictions"]),
            title="SNR Estimation Analysis",
            save_path=str(figures_dir / "snr_error"),
        )
        
        # 6. Per-SNR Balanced Accuracy
        plot_per_snr_accuracy(
            per_snr_metrics=results["per_snr"],
            title="Per-SNR Balanced Accuracy",
            save_path=str(figures_dir / "per_snr_accuracy"),
        )
        
        # 7. Training curves (if available)
        training_results_path = Path("checkpoints") / "training_results.pt"
        if training_results_path.exists():
            training_results = torch.load(training_results_path, weights_only=False)
            if "history" in training_results and training_results["history"]:
                plot_training_curves(
                    history=training_results["history"],
                    title="Training History",
                    save_path=str(figures_dir / "training_curves"),
                )
        
        print(f"  Figures saved to {figures_dir}")
    
    # Print comparison with targets
    print(f"""
{'='*60}
COMPARISON WITH TARGET PERFORMANCE
{'='*60}

{'Metric':<30} {'Achieved':<15} {'Target':<15} {'Status':<10}
{'-'*70}
PU Detection Accuracy         {results['overall']['pu_accuracy']*100:>6.2f}%       ≥ 95.00%      {'✓' if results['overall']['pu_accuracy'] >= 0.95 else '✗'}
Modulation Accuracy            {results['overall']['mod_accuracy']*100:>6.2f}%       ≥ 82.00%      {'✓' if results['overall']['mod_accuracy'] >= 0.82 else '✗'}
SNR MAE                        {results['overall']['snr_mae']:>6.4f} dB    ≤ 0.3000 dB   {'✓' if results['overall']['snr_mae'] <= 0.30 else '✗'}
Parameters                     {results['model_params']:>8,}        ~93,000       {'✓' if results['model_params'] < 120000 else '✗'}
""")
    
    # Low-SNR analysis
    if "low_snr_<8dB" in results["per_snr"]:
        low_snr = results["per_snr"]["low_snr_<8dB"]
        low_bal = low_snr.get("pu_balanced_accuracy", float("nan"))
        print(f"Low-SNR PU Balanced Acc (<8dB) {low_bal*100:>6.2f}%       ≥ 93.00%      {'✓' if low_bal >= 0.93 else '✗'}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()