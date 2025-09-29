"""Integration with Evidently AI for HTML reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset
    from evidently.report import Report
except ImportError:  # pragma: no cover - optional dependency
    ColumnMapping = object  # type: ignore[assignment]
    Report = object  # type: ignore[assignment]
    _EVIDENTLY_AVAILABLE = False
else:  # pragma: no cover - simply tracks availability
    _EVIDENTLY_AVAILABLE = True


def _ensure_evidently() -> None:
    if not _EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently is required for reporting. Install with `pip install evidently`."
        )


def build_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    column_mapping: Optional[ColumnMapping] = None,
) -> Report:
    """Create an Evidently report for data and concept drift."""

    _ensure_evidently()
    metrics = [DataDriftPreset()]
    if target_column is not None and prediction_column is not None:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    return report


def save_report_html(report: Report, path: Path | str) -> None:
    """Persist an Evidently report to an HTML file."""

    _ensure_evidently()
    Path(path).write_text(report.as_html())


__all__ = [
    "build_evidently_report",
    "save_report_html",
]
