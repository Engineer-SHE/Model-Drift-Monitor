from __future__ import annotations

import numpy as np

from model_drift_monitor.performance import (
    PerformanceMetrics,
    check_thresholds,
    evaluate_classification_performance,
)


def test_evaluate_classification_performance() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = evaluate_classification_performance(y_true, y_pred, y_proba=y_proba)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.accuracy == 0.75
    assert metrics.mean_prediction is not None
    assert metrics.roc_auc is not None


def test_check_thresholds_flags_alerts() -> None:
    metrics = PerformanceMetrics(accuracy=0.6, precision=0.5, recall=0.4)
    thresholds = {"accuracy": 0.7, "recall": 0.5}

    alerts = check_thresholds(metrics, thresholds)
    assert set(alerts.keys()) == {"accuracy", "recall"}
