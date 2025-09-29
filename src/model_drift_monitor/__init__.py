"""Model Drift Monitor package."""
from .baseline import BaselineStats, CategoricalBaseline, NumericBaseline, compute_baseline_statistics
from .drift import ColumnDriftResult, DriftReport, detect_data_drift, population_stability_index
from .performance import (
    PerformanceMetrics,
    ThresholdAlert,
    check_thresholds,
    evaluate_classification_performance,
)
from .reporting import build_evidently_report, save_report_html
from .synthetic import generate_drift_datasets
from .visualization import plot_drift_heatmap, plot_performance_metrics, plot_psi_trend

__all__ = [
    "BaselineStats",
    "CategoricalBaseline",
    "NumericBaseline",
    "compute_baseline_statistics",
    "ColumnDriftResult",
    "DriftReport",
    "detect_data_drift",
    "population_stability_index",
    "PerformanceMetrics",
    "ThresholdAlert",
    "check_thresholds",
    "evaluate_classification_performance",
    "build_evidently_report",
    "save_report_html",
    "generate_drift_datasets",
    "plot_drift_heatmap",
    "plot_performance_metrics",
    "plot_psi_trend",
]
