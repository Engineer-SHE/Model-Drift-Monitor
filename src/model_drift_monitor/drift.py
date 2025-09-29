"""Drift detection utilities including PSI, KS, and Chi-square tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .baseline import BaselineStats, CategoricalBaseline, NumericBaseline


@dataclass
class ColumnDriftResult:
    """Stores drift metrics for a single column."""

    column: str
    column_type: str
    psi: Optional[float] = None
    psi_threshold: Optional[float] = None
    ks_pvalue: Optional[float] = None
    ks_threshold: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    chi2_threshold: Optional[float] = None
    drift_detected: bool = False
    unexpected_categories: List[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """Aggregated results for data drift detection."""

    column_results: Dict[str, ColumnDriftResult] = field(default_factory=dict)

    @property
    def drifted_columns(self) -> List[str]:
        """List the names of columns flagged as drifted."""

        return [
            column
            for column, result in self.column_results.items()
            if result.drift_detected
        ]


def population_stability_index(
    expected_probabilities: Sequence[float],
    actual_probabilities: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the population stability index (PSI)."""

    expected = np.asarray(expected_probabilities, dtype=float)
    actual = np.asarray(actual_probabilities, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError("Expected and actual probability vectors must match in shape")

    expected = np.where(expected <= 0, epsilon, expected)
    actual = np.where(actual <= 0, epsilon, actual)
    psi = np.sum((expected - actual) * np.log(expected / actual))
    return float(psi)


def _numeric_probabilities(series: pd.Series, baseline: NumericBaseline) -> np.ndarray:
    """Compute the observed numeric bucket probabilities for ``series``."""

    cleaned = series.dropna()
    if cleaned.empty:
        return np.zeros(len(baseline.probabilities))
    hist, _ = np.histogram(cleaned.values, bins=np.asarray(baseline.bins))
    total = hist.sum()
    return hist / total if total else np.zeros_like(hist, dtype=float)


def _categorical_probabilities(
    series: pd.Series, baseline: CategoricalBaseline
) -> tuple[np.ndarray, List[str]]:
    """Compute categorical probabilities aligned with the baseline categories."""

    cleaned = series.dropna().astype(str)
    total = cleaned.shape[0]
    categories = list(baseline.probabilities.keys())
    counts = dict.fromkeys(categories, 0)
    others = 0
    for value, count in cleaned.value_counts().items():
        if value in counts:
            counts[value] = count
        else:
            others += count
    if "__other__" in counts:
        counts["__other__"] += others
    elif others > 0:
        categories.append("__other__")
        counts["__other__"] = others
    probabilities = np.array(
        [counts.get(category, 0) / total if total else 0 for category in categories],
        dtype=float,
    )
    return probabilities, categories


def detect_data_drift(
    current_df: pd.DataFrame,
    baseline: BaselineStats,
    *,
    reference_df: Optional[pd.DataFrame] = None,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
    chi2_pvalue_threshold: float = 0.05,
) -> DriftReport:
    """Run data drift detection on the current dataset."""

    report = DriftReport()

    for column, numeric_baseline in baseline.numeric.items():
        current_series = current_df[column]
        psi_value = population_stability_index(
            numeric_baseline.probabilities,
            _numeric_probabilities(current_series, numeric_baseline),
        )
        ks_pvalue = None
        if reference_df is not None and column in reference_df:
            baseline_series = reference_df[column].dropna()
            if not baseline_series.empty and not current_series.dropna().empty:
                ks_stat, ks_pvalue = stats.ks_2samp(
                    baseline_series.values, current_series.dropna().values
                )
        drift_detected = False
        if psi_value >= psi_threshold:
            drift_detected = True
        if ks_pvalue is not None and ks_pvalue < ks_pvalue_threshold:
            drift_detected = True
        report.column_results[column] = ColumnDriftResult(
            column=column,
            column_type="numeric",
            psi=psi_value,
            psi_threshold=psi_threshold,
            ks_pvalue=ks_pvalue,
            ks_threshold=ks_pvalue_threshold,
            drift_detected=drift_detected,
        )

    for column, categorical_baseline in baseline.categorical.items():
        current_series = current_df[column]
        actual_probs, categories = _categorical_probabilities(
            current_series, categorical_baseline
        )
        expected_probs = np.array(
            [categorical_baseline.probabilities.get(category, 0) for category in categories]
        )
        psi_value = population_stability_index(expected_probs, actual_probs)

        current_counts = current_series.dropna().astype(str).value_counts()
        expected_counts = np.array(
            [
                categorical_baseline.probabilities.get(category, 0)
                * max(current_series.shape[0], 1)
                for category in categories
            ]
        )
        observed_counts = np.array([current_counts.get(category, 0) for category in categories])

        chi2_pvalue = None
        if expected_counts.sum() > 0:
            _, chi2_pvalue = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        unexpected_categories = [
            category for category in current_counts.index if category not in categories
        ]

        drift_detected = False
        if psi_value >= psi_threshold:
            drift_detected = True
        if chi2_pvalue is not None and chi2_pvalue < chi2_pvalue_threshold:
            drift_detected = True
        report.column_results[column] = ColumnDriftResult(
            column=column,
            column_type="categorical",
            psi=psi_value,
            psi_threshold=psi_threshold,
            chi2_pvalue=chi2_pvalue,
            chi2_threshold=chi2_pvalue_threshold,
            unexpected_categories=unexpected_categories,
            drift_detected=drift_detected,
        )

    return report


__all__ = [
    "ColumnDriftResult",
    "DriftReport",
    "detect_data_drift",
    "population_stability_index",
]
