from __future__ import annotations

import numpy as np
import pandas as pd

from model_drift_monitor.baseline import compute_baseline_statistics
from model_drift_monitor.drift import detect_data_drift


def test_detect_data_drift_numeric_shift() -> None:
    rng = np.random.default_rng(42)
    baseline = pd.DataFrame({"feature": rng.normal(0, 1, size=1000)})
    current = pd.DataFrame({"feature": rng.normal(2, 1, size=1000)})

    baseline_stats = compute_baseline_statistics(baseline)
    report = detect_data_drift(current, baseline_stats, reference_df=baseline)

    result = report.column_results["feature"]
    assert result.psi is not None and result.psi > 0.2
    assert result.drift_detected


def test_detect_data_drift_categorical_shift() -> None:
    baseline = pd.DataFrame({"feature": ["a"] * 80 + ["b"] * 20})
    current = pd.DataFrame({"feature": ["a"] * 30 + ["b"] * 70})

    baseline_stats = compute_baseline_statistics(baseline)
    report = detect_data_drift(current, baseline_stats)

    result = report.column_results["feature"]
    assert result.psi is not None and result.psi > 0.2
    assert result.drift_detected
