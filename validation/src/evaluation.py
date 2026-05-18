#!/usr/bin/env python3
"""
evaluation.py - Comprehensive evaluation metrics for ReCiter scoring models.

This module provides:
- Discrimination metrics (AUC-ROC, AUC-PR, F1, etc.)
- Calibration metrics (via calibration.py)
- Threshold selection
- Model comparison utilities
- Per-segment analysis (by uncertainRejectionRisk buckets)

Used by all training and comparison scripts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
    brier_score_loss, roc_curve
)

from calibration import (
    compute_ece, compute_mce, verify_extreme_calibration,
    calibration_summary, plot_reliability_diagram
)


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

def pick_threshold_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Select threshold that maximizes F1 score.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    thr = np.append(thr, 1.0)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])


def pick_threshold_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_precision: float = 0.99
) -> float:
    """
    Select threshold that achieves target precision.

    Useful for identifying "auto-accept" candidates where high precision is required.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        target_precision: Minimum precision to achieve

    Returns:
        Threshold value, or 1.0 if target not achievable
    """
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    thr = np.append(thr, 1.0)

    # Find lowest threshold that achieves target precision
    for i in range(len(prec)):
        if prec[i] >= target_precision:
            return float(thr[i])

    return 1.0  # Target not achievable


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
    tag: str = "eval"
) -> Dict:
    """
    Comprehensive evaluation of model predictions.

    Combines discrimination and calibration metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold (auto-selected if None)
        tag: Label for this evaluation

    Returns:
        Dictionary with all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if threshold is None:
        threshold = pick_threshold_f1(y_true, y_pred)

    y_hat = (y_pred >= threshold).astype(int)

    # Discrimination metrics
    metrics = {
        "tag": tag,
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()),

        # Classification metrics
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),

        # Ranking metrics
        "auc_roc": float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "auc_pr": float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,

        # Probability quality
        "brier": float(brier_score_loss(y_true, y_pred)),

        # Calibration metrics
        "ece": float(compute_ece(y_true, y_pred)),
        "mce": float(compute_mce(y_true, y_pred)),

        # Confusion matrix
        "confusion_matrix": confusion_matrix(y_true, y_hat).tolist(),

        # Extreme calibration (G8 compliance)
        "extreme_calibration": verify_extreme_calibration(y_true, y_pred)
    }

    return metrics


def print_evaluation(metrics: Dict) -> None:
    """Print evaluation metrics in readable format."""
    print(f"\n[{metrics['tag']}] n={metrics['support']:,}")
    print(f"  Threshold: {metrics['threshold']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    if metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f} (target < 0.02)")
    print(f"  MCE:       {metrics['mce']:.4f}")


# =============================================================================
# CURVE PLOTTING
# =============================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color='blue', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{title}\nAP = {ap:.4f}', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        y_true: Ground truth binary labels
        predictions: Dict mapping model name to predictions
        threshold: Shared threshold for all models (auto-selected if None)

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for model_name, y_pred in predictions.items():
        if threshold is None:
            thr = pick_threshold_f1(y_true, y_pred)
        else:
            thr = threshold

        metrics = evaluate_predictions(y_true, y_pred, thr, model_name)
        results.append({
            "Model": model_name,
            "AUC-ROC": metrics["auc_roc"],
            "AUC-PR": metrics["auc_pr"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Brier": metrics["brier"],
            "ECE": metrics["ece"],
            "MCE": metrics["mce"],
            "Threshold": metrics["threshold"]
        })

    df = pd.DataFrame(results)

    # Highlight best values
    print("\nModel Comparison:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\nBest by metric:")
    for col in ["AUC-ROC", "AUC-PR", "Precision", "Recall", "F1"]:
        best_idx = df[col].idxmax()
        print(f"  {col}: {df.loc[best_idx, 'Model']} ({df.loc[best_idx, col]:.4f})")
    for col in ["Brier", "ECE", "MCE"]:
        best_idx = df[col].idxmin()
        print(f"  {col} (lower=better): {df.loc[best_idx, 'Model']} ({df.loc[best_idx, col]:.4f})")

    return df


def plot_comparison_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_dir: Optional[str] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot PR and ROC curves for multiple models.

    Args:
        y_true: Ground truth binary labels
        predictions: Dict mapping model name to predictions
        save_dir: Directory to save figures (optional)

    Returns:
        Tuple of (PR figure, ROC figure)
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

    # PR Curve
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        ax_pr.plot(rec, prec, color=colors[idx], linewidth=2,
                   label=f'{model_name} (AP={ap:.4f})')

    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Comparison', fontsize=14)
    ax_pr.legend(loc='lower left')
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1])
    ax_pr.grid(True, alpha=0.3)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax_roc.plot(fpr, tpr, color=colors[idx], linewidth=2,
                    label=f'{model_name} (AUC={auc:.4f})')

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Comparison', fontsize=14)
    ax_roc.legend(loc='lower right')
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.grid(True, alpha=0.3)

    if save_dir:
        fig_pr.savefig(f"{save_dir}/pr_comparison.png", dpi=150, bbox_inches='tight')
        fig_roc.savefig(f"{save_dir}/roc_comparison.png", dpi=150, bbox_inches='tight')

    return fig_pr, fig_roc


# =============================================================================
# PER-SEGMENT ANALYSIS
# =============================================================================

def analyze_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_values: np.ndarray,
    segment_name: str = "uncertainRejectionRisk",
    n_buckets: int = 5
) -> pd.DataFrame:
    """
    Analyze model performance by segment (e.g., uncertainRejectionRisk buckets).

    This helps identify if the model performs differently for different subgroups.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        segment_values: Values to segment by
        segment_name: Name of the segment variable
        n_buckets: Number of buckets to create

    Returns:
        DataFrame with per-segment metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    segment_values = np.asarray(segment_values)

    # Create buckets
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bucket_edges = np.percentile(segment_values, percentiles)

    results = []
    for i in range(n_buckets):
        if i < n_buckets - 1:
            mask = (segment_values >= bucket_edges[i]) & (segment_values < bucket_edges[i + 1])
            bucket_label = f"[{bucket_edges[i]:.2f}, {bucket_edges[i+1]:.2f})"
        else:
            mask = (segment_values >= bucket_edges[i]) & (segment_values <= bucket_edges[i + 1])
            bucket_label = f"[{bucket_edges[i]:.2f}, {bucket_edges[i+1]:.2f}]"

        if mask.sum() > 0:
            y_t = y_true[mask]
            y_p = y_pred[mask]

            thr = pick_threshold_f1(y_t, y_p) if len(np.unique(y_t)) > 1 else 0.5
            y_hat = (y_p >= thr).astype(int)

            results.append({
                "Bucket": bucket_label,
                "Count": int(mask.sum()),
                "Positive_Rate": float(y_t.mean()),
                "Precision": float(precision_score(y_t, y_hat, zero_division=0)),
                "Recall": float(recall_score(y_t, y_hat, zero_division=0)),
                "F1": float(f1_score(y_t, y_hat, zero_division=0)),
                "AUC-ROC": float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else None,
                "ECE": float(compute_ece(y_t, y_p))
            })

    df = pd.DataFrame(results)
    print(f"\nPerformance by {segment_name}:")
    print("=" * 100)
    print(df.to_string(index=False))

    return df


# =============================================================================
# AUTO-ACCEPT ANALYSIS
# =============================================================================

def analyze_auto_accept_candidates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_thresholds: List[float] = [0.95, 0.99, 0.999]
) -> pd.DataFrame:
    """
    Analyze potential auto-accept candidates at various confidence thresholds.

    This supports G8: identifying articles that could be auto-accepted with
    known error rates.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        confidence_thresholds: Thresholds to analyze

    Returns:
        DataFrame with auto-accept analysis
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = []
    for thr in confidence_thresholds:
        mask = y_pred >= thr

        if mask.sum() > 0:
            actual_accuracy = y_true[mask].mean()
            error_rate = 1 - actual_accuracy
            expected_error = 1 - thr
            calibration_gap = error_rate - expected_error

            results.append({
                "Threshold": thr,
                "Count": int(mask.sum()),
                "Pct_of_Total": float(mask.sum() / len(y_true) * 100),
                "Actual_Accuracy": float(actual_accuracy),
                "Expected_Accuracy": float(thr),
                "Actual_Error_Rate": float(error_rate),
                "Expected_Error_Rate": float(expected_error),
                "Calibration_Gap": float(calibration_gap),
                "Safe_to_Auto_Accept": abs(calibration_gap) < 0.01  # Within 1%
            })
        else:
            results.append({
                "Threshold": thr,
                "Count": 0,
                "Pct_of_Total": 0.0,
                "Actual_Accuracy": None,
                "Expected_Accuracy": float(thr),
                "Actual_Error_Rate": None,
                "Expected_Error_Rate": float(1 - thr),
                "Calibration_Gap": None,
                "Safe_to_Auto_Accept": None
            })

    df = pd.DataFrame(results)

    print("\nAuto-Accept Analysis:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\nInterpretation:")
    print("  - 'Safe_to_Auto_Accept' = True means actual error rate is within 1% of expected")
    print("  - High 'Pct_of_Total' with 'Safe_to_Auto_Accept' = True enables significant automation")

    return df
