"""Visualization helpers for the model drift monitoring system."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .drift import DriftReport


def plot_psi_trend(trend: pd.DataFrame, *, title: str = "PSI Trend") -> go.Figure:
    """Plot PSI values over time."""

    if "timestamp" not in trend.columns or "psi" not in trend.columns:
        raise ValueError("The trend dataframe must contain 'timestamp' and 'psi' columns")

    fig = px.line(trend, x="timestamp", y="psi", color="feature", markers=True, title=title)
    fig.update_layout(yaxis_title="PSI", xaxis_title="Timestamp", hovermode="x unified")
    return fig


def plot_performance_metrics(
    metrics: pd.DataFrame, *, title: str = "Performance Metrics"
) -> go.Figure:
    """Plot rolling model performance metrics."""

    if "timestamp" not in metrics.columns:
        raise ValueError("Metrics dataframe must include a 'timestamp' column")

    value_columns = [column for column in metrics.columns if column != "timestamp"]
    fig = px.line(metrics, x="timestamp", y=value_columns, title=title)
    fig.update_layout(yaxis_title="Metric Value", xaxis_title="Timestamp", hovermode="x unified")
    return fig


def plot_drift_heatmap(report: DriftReport, *, title: str = "Drift Heatmap") -> go.Figure:
    """Visualize drift status for each feature as a heatmap."""

    if not report.column_results:
        raise ValueError("Drift report is empty")

    columns = list(report.column_results.keys())
    status = [1 if report.column_results[column].drift_detected else 0 for column in columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=[status],
            x=columns,
            y=["drift"],
            colorscale=[[0, "#1f77b4"], [1, "#d62728"]],
            showscale=False,
            text=["Drift" if value else "Stable" for value in status],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=title, xaxis_title="Feature", yaxis=dict(showticklabels=False))
    return fig


__all__ = [
    "plot_drift_heatmap",
    "plot_performance_metrics",
    "plot_psi_trend",
]
