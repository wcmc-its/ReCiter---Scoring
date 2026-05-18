#!/usr/bin/env python3
"""
calibration.py - Calibration metrics and visualization for ReCiter scoring models.

This module provides tools to measure and visualize probability calibration:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Extreme score verification (for G8: scores must be true probabilities)

A well-calibrated model outputs probabilities that match observed frequencies:
- If the model outputs 0.90, ~90% of those predictions should be correct
- If the model outputs 0.999, ~99.9% should be correct (enables auto-accept)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual accuracy.
    Lower is better; a perfectly calibrated model has ECE = 0.

    Formula:
        ECE = sum(|bucket_size / n| * |accuracy_in_bucket - avg_confidence_in_bucket|)

    Target: ECE < 0.02 (2%) for well-calibrated model.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted probabilities [0, 1]
        n_bins: Number of bins for bucketing predictions

    Returns:
        ECE value in [0, 1], lower is better
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])

        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)

    return ece / len(y_true)


def compute_mce(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the worst-case calibration error across all bins.
    Useful for identifying if there are specific probability ranges that are poorly calibrated.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted probabilities [0, 1]
        n_bins: Number of bins for bucketing predictions

    Returns:
        MCE value in [0, 1], lower is better
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])

        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred[mask].mean()
            max_error = max(max_error, abs(bin_accuracy - bin_confidence))

    return max_error


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data for reliability diagram.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Tuple of (mean_predicted_prob, actual_frequency, bin_counts)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mean_predicted = []
    actual_freq = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])

        if mask.sum() > 0:
            mean_predicted.append(y_pred[mask].mean())
            actual_freq.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            mean_predicted.append(np.nan)
            actual_freq.append(np.nan)
            bin_counts.append(0)

    return np.array(mean_predicted), np.array(actual_freq), np.array(bin_counts)


# =============================================================================
# EXTREME SCORE VERIFICATION (G8 Compliance)
# =============================================================================

def verify_extreme_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float] = [0.90, 0.95, 0.99, 0.999, 0.10, 0.05, 0.01, 0.001]
) -> Dict[str, Dict]:
    """
    Verify calibration at extreme probability thresholds.

    This is critical for G8 compliance: "0.999 should mean <0.1% error rate"

    For each threshold:
    - High thresholds (>0.5): Check accuracy of positive predictions
    - Low thresholds (<0.5): Check accuracy of negative predictions

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        thresholds: List of thresholds to verify

    Returns:
        Dictionary with verification results for each threshold
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}

    for thr in sorted(thresholds, reverse=True):
        if thr > 0.5:
            # High confidence positive predictions
            mask = y_pred >= thr
            key = f">=_{thr}"
            if mask.sum() > 0:
                actual_accuracy = y_true[mask].mean()
                expected_accuracy = thr
                gap = actual_accuracy - expected_accuracy
                results[key] = {
                    "count": int(mask.sum()),
                    "expected_accuracy": float(expected_accuracy),
                    "actual_accuracy": float(actual_accuracy),
                    "gap": float(gap),
                    "calibrated": abs(gap) < 0.05  # Within 5% is acceptable
                }
            else:
                results[key] = {
                    "count": 0,
                    "expected_accuracy": float(thr),
                    "actual_accuracy": None,
                    "gap": None,
                    "calibrated": None
                }
        else:
            # High confidence negative predictions
            mask = y_pred <= thr
            key = f"<=_{thr}"
            if mask.sum() > 0:
                actual_rejection_rate = 1 - y_true[mask].mean()
                expected_rejection_rate = 1 - thr
                gap = actual_rejection_rate - expected_rejection_rate
                results[key] = {
                    "count": int(mask.sum()),
                    "expected_rejection_rate": float(expected_rejection_rate),
                    "actual_rejection_rate": float(actual_rejection_rate),
                    "gap": float(gap),
                    "calibrated": abs(gap) < 0.05
                }
            else:
                results[key] = {
                    "count": 0,
                    "expected_rejection_rate": float(1 - thr),
                    "actual_rejection_rate": None,
                    "gap": None,
                    "calibrated": None
                }

    return results


def format_extreme_calibration_report(verification: Dict[str, Dict]) -> str:
    """Format extreme calibration verification as a readable report."""
    lines = ["Extreme Score Calibration Verification", "=" * 50]

    lines.append("\nHigh Confidence Positives (should accept):")
    lines.append("-" * 50)
    for key, data in verification.items():
        if key.startswith(">=_"):
            thr = key.replace(">=_", "")
            if data["count"] > 0:
                status = "OK" if data["calibrated"] else "FAIL"
                lines.append(
                    f"  Score >= {thr}: n={data['count']:,}, "
                    f"expected={data['expected_accuracy']:.1%}, "
                    f"actual={data['actual_accuracy']:.1%}, "
                    f"gap={data['gap']:+.1%} [{status}]"
                )
            else:
                lines.append(f"  Score >= {thr}: No samples")

    lines.append("\nHigh Confidence Negatives (should reject):")
    lines.append("-" * 50)
    for key, data in verification.items():
        if key.startswith("<=_"):
            thr = key.replace("<=_", "")
            if data["count"] > 0:
                status = "OK" if data["calibrated"] else "FAIL"
                lines.append(
                    f"  Score <= {thr}: n={data['count']:,}, "
                    f"expected_rej={data['expected_rejection_rate']:.1%}, "
                    f"actual_rej={data['actual_rejection_rate']:.1%}, "
                    f"gap={data['gap']:+.1%} [{status}]"
                )
            else:
                lines.append(f"  Score <= {thr}: No samples")

    return "\n".join(lines)


# =============================================================================
# RELIABILITY DIAGRAMS (VISUALIZATION)
# =============================================================================

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
    show_histogram: bool = True
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    A well-calibrated model should have points close to the diagonal line.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)
        show_histogram: Whether to show prediction histogram

    Returns:
        matplotlib Figure object
    """
    mean_pred, actual_freq, bin_counts = compute_calibration_curve(y_true, y_pred, n_bins)

    if show_histogram:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(8, 8))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Only plot non-NaN points
    valid = ~np.isnan(mean_pred) & ~np.isnan(actual_freq)
    ax1.plot(mean_pred[valid], actual_freq[valid], 'o-', color='blue', label='Model')

    # Error bars based on count (larger counts = more confident)
    for i in range(len(mean_pred)):
        if valid[i] and bin_counts[i] > 10:
            # Standard error of proportion
            se = np.sqrt(actual_freq[i] * (1 - actual_freq[i]) / bin_counts[i])
            ax1.errorbar(mean_pred[i], actual_freq[i], yerr=1.96*se,
                        color='blue', alpha=0.3, capsize=3)

    # Compute and show ECE
    ece = compute_ece(y_true, y_pred, n_bins)
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    if show_histogram:
        ax2.hist(y_pred, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predictions', fontsize=12)
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")

    return fig


def plot_reliability_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Model Calibration Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reliability diagrams for multiple models side-by-side.

    Args:
        y_true: Ground truth binary labels
        predictions: Dict mapping model name to predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        mean_pred, actual_freq, _ = compute_calibration_curve(y_true, y_pred, n_bins)
        ece = compute_ece(y_true, y_pred, n_bins)

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        valid = ~np.isnan(mean_pred) & ~np.isnan(actual_freq)
        ax.plot(mean_pred[valid], actual_freq[valid], 'o-',
                color=colors[idx], label=f'{model_name}')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{model_name}\nECE = {ece:.4f}')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison diagram saved to {save_path}")

    return fig


# =============================================================================
# CALIBRATION SUMMARY
# =============================================================================

def calibration_summary(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict:
    """
    Generate comprehensive calibration summary.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        model_name: Name for reporting

    Returns:
        Dictionary with all calibration metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    summary = {
        "model": model_name,
        "n_samples": len(y_true),
        "ece": float(compute_ece(y_true, y_pred)),
        "mce": float(compute_mce(y_true, y_pred)),
        "extreme_verification": verify_extreme_calibration(y_true, y_pred),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
        "pred_min": float(y_pred.min()),
        "pred_max": float(y_pred.max()),
        "actual_positive_rate": float(y_true.mean())
    }

    # G8 compliance check
    extreme = summary["extreme_verification"]
    g8_pass = True
    if ">=_0.99" in extreme and extreme[">=_0.99"]["count"] > 0:
        if not extreme[">=_0.99"]["calibrated"]:
            g8_pass = False
    if ">=_0.999" in extreme and extreme[">=_0.999"]["count"] > 0:
        if not extreme[">=_0.999"]["calibrated"]:
            g8_pass = False

    summary["g8_compliant"] = g8_pass

    return summary


def print_calibration_summary(summary: Dict) -> None:
    """Print calibration summary in readable format."""
    print(f"\n{'='*60}")
    print(f"Calibration Summary: {summary['model']}")
    print(f"{'='*60}")
    print(f"Samples: {summary['n_samples']:,}")
    print(f"ECE: {summary['ece']:.4f} (target < 0.02)")
    print(f"MCE: {summary['mce']:.4f}")
    print(f"Prediction range: [{summary['pred_min']:.3f}, {summary['pred_max']:.3f}]")
    print(f"Prediction mean: {summary['pred_mean']:.3f}")
    print(f"Actual positive rate: {summary['actual_positive_rate']:.3f}")
    print(f"G8 Compliant: {'YES' if summary['g8_compliant'] else 'NO'}")
    print(f"\n{format_extreme_calibration_report(summary['extreme_verification'])}")
