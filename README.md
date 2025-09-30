# Model Drift Monitor

Model Drift Monitor is a reference implementation of a **model monitoring system** that detects when a machine learning model is no longer performing as expected due to **data drift** or **concept drift**. The project combines statistical tests, performance tracking, and interactive visualisations to support decision making.

## Features

- **Baseline Calculation** – Compute histograms, means, and standard deviations for each feature in the training data and persist the statistics for future comparisons.
- **Data Drift Detection** – Monitor incoming feature distributions using:
  - Population Stability Index (PSI)
  - Kolmogorov–Smirnov (KS) test
  - Chi-square test for categorical variables
- **Concept & Performance Monitoring** – Track rolling performance metrics (accuracy, precision, recall, ROC AUC) and flag degradations when thresholds are breached.
- **Visualisation & Alerts** – Generate Plotly-based drift heatmaps, PSI trends, and performance metric charts. Alerts can be raised when metrics drop below configured thresholds.
- **Evidently Integration** – Optionally build interactive HTML reports using [Evidently AI](https://www.evidentlyai.com/).
- **Streamlit Dashboard** – Launch an interactive dashboard (`streamlit run streamlit_app.py`) to explore drift and upload predictions for monitoring.

## Getting Started

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project depends on `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, `streamlit`, and `evidently` (optional).

### Quick Demo

Run the sample workflow which trains a logistic regression model on synthetic data, generates drifted observations, and exports reports and metrics:

```bash
python app.py --output artifacts
```

Generated artefacts include:

- `baseline.json` – persisted baseline statistics
- `drift_summary.csv` – per-feature drift results
- `drift_heatmap.html` – interactive Plotly visualisation
- `metrics.csv` – classification metrics on the drifted dataset

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Use the sidebar controls to simulate different levels of label flips and feature drift. Upload a CSV file containing `y_true`, `y_pred`, and optional `y_proba` columns to evaluate performance metrics in real time.

## Project Structure

```
Model-Drift-Monitor/
├── app.py                  # CLI demo for the monitoring workflow
├── streamlit_app.py        # Streamlit dashboard
├── src/model_drift_monitor/
│   ├── __init__.py
│   ├── baseline.py         # Baseline statistics utilities
│   ├── drift.py            # PSI, KS, Chi-square drift detection
│   ├── performance.py      # Performance metrics and alerts
│   ├── reporting.py        # Evidently AI integration
│   ├── synthetic.py        # Synthetic dataset generation
│   └── visualization.py    # Plotly visualisations
└── tests/                  # Pytest-based unit tests
```

## Testing

```bash
pytest
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
