"""Synthetic dataset utilities for demos and tests."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_drift_datasets(
    *,
    n_samples: int = 5000,
    n_features: int = 5,
    drift_strength: float = 0.8,
    flip_rate: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate baseline and drifted datasets for demonstrations."""

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.6, 0.4],
        random_state=random_state,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    baseline_df = pd.DataFrame(X, columns=feature_names)
    baseline_df["target"] = y

    rng = np.random.default_rng(random_state + 1)
    drift_X = X + rng.normal(0, drift_strength, size=X.shape)
    drift_y = np.where(rng.random(len(y)) < flip_rate, 1 - y, y)
    current_df = pd.DataFrame(drift_X, columns=feature_names)
    current_df["target"] = drift_y

    return baseline_df, current_df


__all__ = ["generate_drift_datasets"]
