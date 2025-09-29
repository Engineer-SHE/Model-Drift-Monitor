from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_drift_monitor.baseline import BaselineStats, compute_baseline_statistics


def test_compute_baseline_statistics(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5],
            "cat": ["a", "b", "a", "b", "c"],
        }
    )

    stats = compute_baseline_statistics(df)
    assert "num" in stats.numeric
    assert "cat" in stats.categorical
    assert stats.numeric["num"].mean == df["num"].mean()

    output = tmp_path / "baseline.json"
    stats.to_json(output)
    loaded = BaselineStats.from_json(output)
    assert loaded.numeric["num"].count == stats.numeric["num"].count
    assert loaded.categorical["cat"].probabilities == stats.categorical["cat"].probabilities
