"""Streamlit dashboard for the model drift monitor."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from model_drift_monitor import (
    compute_baseline_statistics,
    detect_data_drift,
    evaluate_classification_performance,
    plot_drift_heatmap,
    plot_performance_metrics,
    plot_psi_trend,
)
from model_drift_monitor.synthetic import generate_drift_datasets


st.set_page_config(page_title="Model Drift Monitor", layout="wide")
st.title("📊 Model Drift Monitor Dashboard")

st.sidebar.header("Data Options")
flip_rate = st.sidebar.slider("Label flip rate", 0.0, 0.5, 0.2, 0.05)
drift_strength = st.sidebar.slider("Drift strength", 0.0, 1.5, 0.8, 0.1)

baseline_df, current_df = generate_drift_datasets(flip_rate=flip_rate, drift_strength=drift_strength)
feature_columns = [column for column in baseline_df.columns if column != "target"]

baseline_stats = compute_baseline_statistics(baseline_df[feature_columns])
report = detect_data_drift(
    current_df[feature_columns], baseline_stats, reference_df=baseline_df[feature_columns]
)

st.subheader("Drift Summary")
st.write(pd.DataFrame(
    {
        "feature": list(report.column_results.keys()),
        "psi": [result.psi for result in report.column_results.values()],
        "ks_pvalue": [result.ks_pvalue for result in report.column_results.values()],
        "chi2_pvalue": [result.chi2_pvalue for result in report.column_results.values()],
        "drift_detected": [result.drift_detected for result in report.column_results.values()],
    }
))

heatmap = plot_drift_heatmap(report)
st.plotly_chart(heatmap, use_container_width=True)

psi_trend = pd.DataFrame(
    {
        "timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=len(feature_columns)),
        "psi": [result.psi for result in report.column_results.values()],
        "feature": feature_columns,
    }
)
st.plotly_chart(plot_psi_trend(psi_trend), use_container_width=True)

st.subheader("Performance Monitoring")
st.caption("Upload model predictions to evaluate performance metrics.")

uploaded = st.file_uploader("Upload CSV with columns: y_true, y_pred, y_proba", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    metrics = evaluate_classification_performance(
        df["y_true"], df["y_pred"], y_proba=df.get("y_proba")
    )
    metrics_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp.utcnow(),
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "roc_auc": metrics.roc_auc,
        }
    ])
    st.plotly_chart(plot_performance_metrics(metrics_df), use_container_width=True)
    st.json(metrics.__dict__)
else:
    st.info("Upload a CSV file to evaluate performance metrics.")
