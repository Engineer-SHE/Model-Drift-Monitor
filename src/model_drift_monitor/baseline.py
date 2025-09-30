"""Utilities for computing and persisting baseline statistics for drift monitoring."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd


@dataclass
class NumericBaseline:
    """Summary statistics for a numeric feature."""

    bins: List[float]
    probabilities: List[float]
    mean: float
    std: float
    count: int

    @classmethod
    def from_series(cls, series: pd.Series, bins: int = 20) -> "NumericBaseline":
        """Create a numeric baseline from the provided series."""

        cleaned = series.dropna()
        count = int(cleaned.shape[0])
        if count == 0:
            # Edge case: empty column – generate a degenerate baseline
            return cls(
                bins=[0.0, 1.0],
                probabilities=[1.0],
                mean=float("nan"),
                std=float("nan"),
                count=0,
            )
        hist, bin_edges = np.histogram(cleaned.values, bins=bins)
        total = hist.sum()
        if total == 0:
            probabilities = [1.0 / len(hist)] * len(hist)
        else:
            probabilities = (hist / total).tolist()
        return cls(
            bins=bin_edges.tolist(),
            probabilities=probabilities,
            mean=float(cleaned.mean()),
            std=float(cleaned.std(ddof=0)),
            count=count,
        )


@dataclass
class CategoricalBaseline:
    """Summary statistics for a categorical feature."""

    probabilities: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        min_frequency: float = 0.01,
    ) -> "CategoricalBaseline":
        """Create a categorical baseline summarising the provided series."""

        cleaned = series.dropna().astype(str)
        count = int(cleaned.shape[0])
        if count == 0:
            return cls(probabilities={}, count=0)
        value_counts = cleaned.value_counts(normalize=True)
        if min_frequency > 0:
            major = value_counts[value_counts >= min_frequency]
            other = value_counts[value_counts < min_frequency].sum()
            probabilities: MutableMapping[str, float] = major.to_dict()
            if other > 0:
                probabilities["__other__"] = float(other)
        else:
            probabilities = value_counts.to_dict()
        return cls(probabilities=dict(probabilities), count=count)


@dataclass
class BaselineStats:
    """Container for all baseline statistics in a dataset."""

    numeric: Dict[str, NumericBaseline] = field(default_factory=dict)
    categorical: Dict[str, CategoricalBaseline] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Represent the baseline statistics as serialisable dictionaries."""

        return {
            "numeric": {key: asdict(value) for key, value in self.numeric.items()},
            "categorical": {key: asdict(value) for key, value in self.categorical.items()},
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: Path | str) -> None:
        """Serialise the baseline statistics to a JSON file."""

        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BaselineStats":
        """Instantiate ``BaselineStats`` from a dictionary payload."""

        numeric = {
            key: NumericBaseline(**value)
            for key, value in payload.get("numeric", {}).items()
        }
        categorical = {
            key: CategoricalBaseline(**value)
            for key, value in payload.get("categorical", {}).items()
        }
        metadata = dict(payload.get("metadata", {}))
        return cls(numeric=numeric, categorical=categorical, metadata=metadata)

    @classmethod
    def from_json(cls, path: Path | str) -> "BaselineStats":
        """Load baseline statistics from a JSON file."""

        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)


def compute_baseline_statistics(
    df: pd.DataFrame,
    *,
    numeric_columns: Optional[Iterable[str]] = None,
    categorical_columns: Optional[Iterable[str]] = None,
    histogram_bins: int = 20,
    min_category_frequency: float = 0.01,
    metadata: Optional[Dict[str, str]] = None,
) -> BaselineStats:
    """Compute baseline statistics for all features in a dataframe."""

    metadata = metadata or {}

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = [
            column
            for column in df.columns
            if column not in numeric_columns
        ]

    stats = BaselineStats(metadata=dict(metadata))

    for column in numeric_columns:
        stats.numeric[column] = NumericBaseline.from_series(
            df[column], bins=histogram_bins
        )

    for column in categorical_columns:
        stats.categorical[column] = CategoricalBaseline.from_series(
            df[column], min_frequency=min_category_frequency
        )

    return stats


__all__ = [
    "BaselineStats",
    "CategoricalBaseline",
    "NumericBaseline",
    "compute_baseline_statistics",
]
