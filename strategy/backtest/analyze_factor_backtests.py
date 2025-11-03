"""
Aggregate backtest performance metrics for a specific factor.

This script scans all `batch_summary.json` files within
`strategy/backtest/factor_report/<factor_name>/**/base_backtest/` and
consolidates metrics across multiple runs. For every parameter combo
(`rebalance_hours`, `top_n`, `bottom_n`), it reports statistics for:

- total_return
- sharpe_ratio
- max_drawdown
- win_rate

Each metric includes mean, median, 25th percentile (q1), 75th percentile (q3),
and variance across all matching backtests. The resulting summary is stored as
`factor_backtest_summary.json` in the factor's report directory.

Usage:
    python strategy/backtest/analyze_factor_backtests.py
    (edit the CONFIG_FACTOR_NAME value in the main() block)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FACTOR_REPORT_DIR = BASE_DIR / "factor_report"
SUMMARY_FILENAME = "base_backtest_summary.json"


@dataclass
class MetricSummary:
    mean: float
    median: float
    q1: float
    q3: float
    variance: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "variance": self.variance,
        }


def _compute_summary(values: pd.Series) -> MetricSummary:
    """Compute descriptive statistics from a pandas Series."""
    cleaned = values.dropna()
    return MetricSummary(
        mean=float(cleaned.mean()) if not cleaned.empty else float("nan"),
        median=float(cleaned.median()) if not cleaned.empty else float("nan"),
        q1=float(cleaned.quantile(0.25)) if not cleaned.empty else float("nan"),
        q3=float(cleaned.quantile(0.75)) if not cleaned.empty else float("nan"),
        variance=float(cleaned.var(ddof=1)) if len(cleaned) > 1 else 0.0,
    )


def _load_batch_summary(path: Path) -> List[Dict]:
    """Read a batch_summary.json file and return its list of backtests."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("backtests", [])


def gather_backtest_records(factor_name: str) -> pd.DataFrame:
    """Collect all backtest entries for the factor into a DataFrame."""
    factor_dir = FACTOR_REPORT_DIR / factor_name
    if not factor_dir.exists():
        raise FileNotFoundError(f"Factor report directory not found: {factor_dir}")

    summary_files = sorted(factor_dir.glob("**/base_backtest/batch_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No batch_summary.json files found under {factor_dir}")

    records: List[Dict] = []
    for file_path in summary_files:
        backtests = _load_batch_summary(file_path)
        for entry in backtests:
            cleaned = {
                "rebalance_hours": entry.get("rebalance_hours"),
                "top_n": entry.get("top_n"),
                "bottom_n": entry.get("bottom_n"),
                "total_return": entry.get("total_return"),
                "sharpe_ratio": entry.get("sharpe_ratio"),
                "max_drawdown": entry.get("max_drawdown"),
                "win_rate": entry.get("win_rate"),
                "source_file": str(file_path),
            }
            records.append(cleaned)

    if not records:
        raise ValueError(f"No backtest entries discovered for {factor_name}")

    df = pd.DataFrame.from_records(records)
    df.sort_values(
        by=["rebalance_hours", "top_n", "bottom_n", "source_file"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def summarise_metrics(df: pd.DataFrame) -> Dict:
    """Aggregate metrics for each parameter combination."""
    metric_fields = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    summary: Dict[str, Dict] = {}

    grouped = df.groupby(["rebalance_hours", "top_n", "bottom_n"], dropna=False)

    for (rebalance_hours, top_n, bottom_n), group in grouped:
        combo_key = f"{rebalance_hours}h_top{top_n}_bottom{bottom_n}"
        combo_result = {"count": int(len(group)), "metrics": {}}

        for field in metric_fields:
            summary_stats = _compute_summary(group[field])
            combo_result["metrics"][field] = summary_stats.as_dict()

        summary[combo_key] = combo_result

    return summary


def save_summary(factor_name: str, summary_payload: Dict) -> Path:
    """Write the summary JSON into the factor's report directory."""
    factor_dir = FACTOR_REPORT_DIR / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)

    output_path = factor_dir / SUMMARY_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return output_path


def analyse_factor(factor_name: str) -> Path:
    """End-to-end processing for a factor."""
    df = gather_backtest_records(factor_name)
    summary = summarise_metrics(df)
    payload = {
        "factor": factor_name,
        "total_records": int(len(df)),
        "combinations": summary,
    }
    return save_summary(factor_name, payload)


FACTOR_PREFIX = "cnn_10_03_72h_v4"
FACTOR_DATA_DIR = Path(__file__).resolve().parent / "factor_data"

def discover_factors(prefix: str = FACTOR_PREFIX) -> List[str]:
    """Return factor directory names that match the desired prefix."""
    if not FACTOR_DATA_DIR.exists():
        raise FileNotFoundError(f"Factor data directory not found: {FACTOR_DATA_DIR}")

    factors = [
        path.name
        for path in FACTOR_DATA_DIR.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    factors.sort()
    return factors


def main() -> None:

    factors = discover_factors()

    for CONFIG_FACTOR_NAME in factors:

        print(f"📊 Analysing backtest results for factor: {CONFIG_FACTOR_NAME}")
        output_path = analyse_factor(CONFIG_FACTOR_NAME)
        print(f"✅ Summary saved to {output_path}")


if __name__ == "__main__":
    main()

