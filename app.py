"""Command-line interface for running a sample drift monitoring workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_drift_monitor import (
    compute_baseline_statistics,
    detect_data_drift,
    evaluate_classification_performance,
    plot_drift_heatmap,
)
from model_drift_monitor.synthetic import generate_drift_datasets


def run_demo(output_dir: Path) -> None:
    baseline_df, current_df = generate_drift_datasets()
    feature_columns = [column for column in baseline_df.columns if column != "target"]

    X_train, X_test, y_train, y_test = train_test_split(
        baseline_df[feature_columns], baseline_df["target"], test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(current_df[feature_columns])
    y_proba = model.predict_proba(current_df[feature_columns])[:, 1]

    baseline_stats = compute_baseline_statistics(baseline_df[feature_columns])
    report = detect_data_drift(
        current_df[feature_columns],
        baseline_stats,
        reference_df=baseline_df[feature_columns],
    )
    metrics = evaluate_classification_performance(current_df["target"], y_pred, y_proba=y_proba)

    output_dir.mkdir(parents=True, exist_ok=True)
    drift_heatmap = plot_drift_heatmap(report)
    drift_heatmap.write_html(output_dir / "drift_heatmap.html")

    drift_summary = pd.DataFrame(
        {
            "feature": list(report.column_results.keys()),
            "column_type": [result.column_type for result in report.column_results.values()],
            "psi": [result.psi for result in report.column_results.values()],
            "ks_pvalue": [result.ks_pvalue for result in report.column_results.values()],
            "chi2_pvalue": [result.chi2_pvalue for result in report.column_results.values()],
            "drift_detected": [result.drift_detected for result in report.column_results.values()],
        }
    )
    drift_summary.to_csv(output_dir / "drift_summary.csv", index=False)

    metrics_df = pd.DataFrame(
        {
            "accuracy": [metrics.accuracy],
            "precision": [metrics.precision],
            "recall": [metrics.recall],
            "roc_auc": [metrics.roc_auc],
        }
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    baseline_stats.to_json(output_dir / "baseline.json")

    print("Drifted columns:", ", ".join(report.drifted_columns) or "None")
    print("Metrics saved to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model drift monitor demo")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Directory where reports will be saved.",
    )
    args = parser.parse_args()
    run_demo(args.output)


if __name__ == "__main__":
    main()
