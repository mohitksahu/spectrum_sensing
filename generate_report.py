#!/usr/bin/env python3
"""Generate comprehensive HTML evaluation report.

Creates a publication-quality experiment report with all metrics,
figures, and analysis.

Usage:
    python generate_report.py --results reports/metrics/test_results.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from jinja2 import Environment


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpectraSense Experiment Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        h3 { color: #7f8c8d; }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }
        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th { background: #3498db; color: white; font-weight: 600; }
        tr:hover { background: #f5f6fa; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .figure-container {
            text-align: center;
            margin: 20px 0;
        }
        .figure-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin: 20px 0;
        }
        .summary-box h2 { color: white; border-color: rgba(255,255,255,0.3); }
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🔬 SpectraSense Experiment Report</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    <p><strong>Model:</strong> Hybrid Decoupled-CNN + Transformer | <strong>Parameters:</strong> {{ total_params | comma }}</p>
    
    <div class="summary-box">
        <h2>Key Results</h2>
        <div class="metric-grid">
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value" style="color: white;">{{ "%.2f" | format(pu_accuracy * 100) }}%</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">PU Detection Accuracy</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value" style="color: white;">{{ "%.2f" | format(mod_accuracy * 100) }}%</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Modulation Accuracy</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value" style="color: white;">{{ "%.4f" | format(snr_mae) }} dB</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">SNR MAE</div>
            </div>
        </div>
    </div>
    
    <h2>📊 Performance Comparison with Targets</h2>
    <table>
        <tr><th>Metric</th><th>Achieved</th><th>Target</th><th>Baseline (Spectrum-SLM)</th><th>Status</th></tr>
        <tr>
            <td>PU Detection Accuracy</td>
            <td>{{ "%.2f" | format(pu_accuracy * 100) }}%</td>
            <td>≥ 95.00%</td>
            <td>93.32%</td>
            <td class="{{ 'pass' if pu_accuracy >= 0.95 else 'fail' }}">{{ '✓ PASS' if pu_accuracy >= 0.95 else '✗ FAIL' }}</td>
        </tr>
        <tr>
            <td>Modulation Accuracy (5-class)</td>
            <td>{{ "%.2f" | format(mod_accuracy * 100) }}%</td>
            <td>≥ 82.00%</td>
            <td>75.13%</td>
            <td class="{{ 'pass' if mod_accuracy >= 0.82 else 'fail' }}">{{ '✓ PASS' if mod_accuracy >= 0.82 else '✗ FAIL' }}</td>
        </tr>
        <tr>
            <td>SNR MAE</td>
            <td>{{ "%.4f" | format(snr_mae) }} dB</td>
            <td>≤ 0.30 dB</td>
            <td>0.371 dB</td>
            <td class="{{ 'pass' if snr_mae <= 0.30 else 'fail' }}">{{ '✓ PASS' if snr_mae <= 0.30 else '✗ FAIL' }}</td>
        </tr>
        <tr>
            <td>Parameters</td>
            <td>{{ total_params | comma }}</td>
            <td>~93K</td>
            <td>940K</td>
            <td class="{{ 'pass' if total_params < 120000 else 'fail' }}">{{ '✓ PASS' if total_params < 120000 else '✗ FAIL' }}</td>
        </tr>
    </table>
    
    <h2>📈 Detailed Metrics</h2>
    
    <h3>PU Detection</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{{ "%.2f" | format(pu_accuracy * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{{ "%.2f" | format(pu_precision * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{{ "%.2f" | format(pu_recall * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{{ "%.2f" | format(pu_f1 * 100) }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value">{{ "%.4f" | format(pu_roc_auc) }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">PR AUC</div>
            <div class="metric-value">{{ "%.4f" | format(pu_pr_auc) }}</div>
        </div>
    </div>
    
    <h3>Modulation Classification</h3>
    <table>
        <tr><th>Class</th><th>Accuracy</th></tr>
        {% for cls_name, cls_acc in mod_per_class.items() %}
        <tr><td>{{ cls_name }}</td><td>{{ "%.2f" | format(cls_acc * 100) }}%</td></tr>
        {% endfor %}
    </table>
    
    <h3>SNR Estimation</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{{ "%.4f" | format(snr_mae) }} dB</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{{ "%.4f" | format(snr_rmse) }} dB</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{{ "%.4f" | format(snr_r2) }}</div>
        </div>
    </div>
    
    <h2>📉 Per-SNR Analysis</h2>
    <table>
        <tr><th>SNR Bin</th><th>PU Accuracy</th><th>Mod Accuracy</th><th>SNR MAE</th><th>Samples</th></tr>
        {% for snr_bin, metrics in per_snr.items() %}
        {% if 'low_snr' not in snr_bin %}
        <tr>
            <td>{{ snr_bin }}</td>
            <td>{{ "%.2f" | format(metrics.pu_accuracy * 100) }}%</td>
            <td>{{ "%.2f" | format(metrics.mod_accuracy * 100) }}%</td>
            <td>{{ "%.3f" | format(metrics.snr_mae) }} dB</td>
            <td>{{ metrics.n_samples }}</td>
        </tr>
        {% endif %}
        {% endfor %}
    </table>
    
    <h2>🖼️ Figures</h2>
    
    {% if figures_exist %}
    <div class="figure-container">
        <h3>Training Curves</h3>
        <img src="../figures/training_curves.png" alt="Training Curves">
    </div>
    <div class="figure-container">
        <h3>PU Detection Confusion Matrix</h3>
        <img src="../figures/pu_confusion_matrix.png" alt="PU Confusion Matrix">
    </div>
    <div class="figure-container">
        <h3>Modulation Confusion Matrix</h3>
        <img src="../figures/mod_confusion_matrix.png" alt="Modulation Confusion Matrix">
    </div>
    <div class="figure-container">
        <h3>ROC Curve</h3>
        <img src="../figures/roc_curve.png" alt="ROC Curve">
    </div>
    <div class="figure-container">
        <h3>Precision-Recall Curve</h3>
        <img src="../figures/pr_curve.png" alt="PR Curve">
    </div>
    <div class="figure-container">
        <h3>SNR Estimation Analysis</h3>
        <img src="../figures/snr_error.png" alt="SNR Error">
    </div>
    <div class="figure-container">
        <h3>Per-SNR Performance</h3>
        <img src="../figures/per_snr_accuracy.png" alt="Per-SNR Accuracy">
    </div>
    {% endif %}
    
    <h2>🏗️ Model Architecture</h2>
    <table>
        <tr><th>Component</th><th>Parameters</th><th>Percentage</th></tr>
        {% for component, count in param_breakdown.items() %}
        {% if component != 'total' %}
        <tr>
            <td>{{ component }}</td>
            <td>{{ count | comma }}</td>
            <td>{{ "%.1f" | format(count / total_params * 100) }}%</td>
        </tr>
        {% endif %}
        {% endfor %}
        <tr style="font-weight: bold;">
            <td>Total</td>
            <td>{{ total_params | comma }}</td>
            <td>100.0%</td>
        </tr>
    </table>
    
    <footer>
        <p>SpectraSense — Hybrid Decoupled-CNN + Transformer for Single-Band Spectrum Sensing</p>
        <p>Report generated automatically by the SpectraSense evaluation framework</p>
    </footer>
</body>
</html>
"""


def comma_filter(value):
    """Jinja2 filter for comma-separated numbers."""
    return f"{int(value):,}"


def main():
    parser = argparse.ArgumentParser(description="Generate SpectraSense HTML report")
    parser.add_argument("--results", type=str, default="reports/metrics/test_results.json",
                        help="Path to test results JSON")
    parser.add_argument("--output", type=str, default="reports/html/experiment_report.html",
                        help="Output HTML report path")
    args = parser.parse_args()
    
    # Load results
    with open(args.results, "r") as f:
        results = json.load(f)
    
    overall = results["overall"]
    per_snr = results.get("per_snr", {})
    param_breakdown = results.get("param_breakdown", {})
    total_params = results.get("model_params", 93000)
    
    # Check if figures exist
    figures_dir = Path("reports/figures")
    figures_exist = figures_dir.exists() and any(figures_dir.glob("*.png"))
    
    # Prepare per-class modulation metrics
    from src.datasets.preprocessing import MODULATION_NAMES
    mod_per_class = {}
    for i in range(5):
        key = f"mod_acc_class_{i}"
        mod_per_class[MODULATION_NAMES.get(i, f"Class {i}")] = overall.get(key, 0)
    
    # Prepare per_snr with dot-access compatibility
    class DotDict(dict):
        __getattr__ = dict.get
    
    per_snr_dotted = {k: DotDict(v) for k, v in per_snr.items()}
    
    # Render template
    env = Environment(autoescape=True)
    env.filters["comma"] = comma_filter
    template = env.from_string(REPORT_TEMPLATE)
    
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_params=total_params,
        pu_accuracy=overall.get("pu_accuracy", 0),
        pu_precision=overall.get("pu_precision", 0),
        pu_recall=overall.get("pu_recall", 0),
        pu_f1=overall.get("pu_f1", 0),
        pu_roc_auc=overall.get("pu_roc_auc", 0),
        pu_pr_auc=overall.get("pu_pr_auc", 0),
        mod_accuracy=overall.get("mod_accuracy", 0),
        mod_per_class=mod_per_class,
        snr_mae=overall.get("snr_mae", 0),
        snr_rmse=overall.get("snr_rmse", 0),
        snr_r2=overall.get("snr_r2", 0),
        per_snr=per_snr_dotted,
        param_breakdown=param_breakdown,
        figures_exist=figures_exist,
    )
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()