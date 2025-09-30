"""Performance monitoring utilities for machine learning models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


@dataclass
class PerformanceMetrics:
    """Container for common classification metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    class_balance: Dict[str, float] = field(default_factory=dict)
    mean_prediction: Optional[float] = None


@dataclass
class ThresholdAlert:
    """Alerts raised when a metric falls below a threshold."""

    metric: str
    value: float
    threshold: float


def evaluate_classification_performance(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    y_proba: Optional[Iterable[float]] = None,
    positive_label: int = 1,
) -> PerformanceMetrics:
    """Compute core classification metrics."""

    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    metrics = PerformanceMetrics()

    if y_true.size == 0:
        return metrics

    metrics.accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0
    )
    metrics.precision = float(precision)
    metrics.recall = float(recall)
    metrics.f1 = float(f1)

    if y_proba is not None:
        y_proba = np.asarray(list(y_proba))
        if y_proba.size == y_true.size:
            metrics.mean_prediction = float(y_proba.mean())
            try:
                metrics.roc_auc = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics.roc_auc = None
    else:
        metrics.mean_prediction = float(np.mean(y_pred))

    unique, counts = np.unique(y_true, return_counts=True)
    total = counts.sum()
    metrics.class_balance = {str(label): count / total for label, count in zip(unique, counts)}

    return metrics


def check_thresholds(
    metrics: PerformanceMetrics, thresholds: Dict[str, float]
) -> Dict[str, ThresholdAlert]:
    """Return alerts for metrics that violate the provided thresholds."""

    alerts: Dict[str, ThresholdAlert] = {}
    for metric_name, threshold in thresholds.items():
        value = getattr(metrics, metric_name, None)
        if value is not None and value < threshold:
            alerts[metric_name] = ThresholdAlert(
                metric=metric_name, value=value, threshold=threshold
            )
    return alerts


__all__ = [
    "PerformanceMetrics",
    "ThresholdAlert",
    "check_thresholds",
    "evaluate_classification_performance",
]
