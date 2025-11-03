"""
Summarise adjusted backtest outcomes for a given factor.

This utility scans `factor_report/<factor_name>/adjust_backtest/**/batch_summary.json`,
groups the results by strategy (i.e., leverage/stop configurations), and
computes descriptive statistics across experimental runs.

Metrics summarised per strategy:
    - total_return
    - sharpe_ratio
    - win_rate
    - max_drawdown

For each metric we report mean, median, variance, and the 25th/75th percentiles.
Results are written to `factor_report/<factor_name>/adjust_backtest/adjust_summary.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FACTOR_REPORT_DIR = BASE_DIR / "factor_report"
ADJUST_FOLDER = "adjust_backtest"
SUMMARY_FILENAME = "adjust_summary.json"


def find_batch_summaries(factor_name: str) -> List[Path]:
    """Return all batch_summary.json files under the factor's adjust_backtest tree."""
    factor_dir = FACTOR_REPORT_DIR / factor_name
    if not factor_dir.exists():
        raise FileNotFoundError(f"Factor report directory not found: {factor_dir}")

    adjust_dir = factor_dir / ADJUST_FOLDER
    if not adjust_dir.exists():
        raise FileNotFoundError(f"Adjust backtest directory not found: {adjust_dir}")

    summary_files = sorted(adjust_dir.glob("**/batch_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No batch_summary.json files found in {adjust_dir}")

    return summary_files


def load_backtests(summary_path: Path) -> List[Dict]:
    """Load backtest records from a single batch summary file."""
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("backtests", [])


def gather_records(factor_name: str) -> pd.DataFrame:
    """Assemble all backtest entries for a factor into a DataFrame."""
    summary_files = find_batch_summaries(factor_name)
    records: List[Dict] = []

    adjust_dir = FACTOR_REPORT_DIR / factor_name / ADJUST_FOLDER

    for path in summary_files:
        relative_parts = path.relative_to(adjust_dir).parts
        if len(relative_parts) < 2:
            # Expect at least date_range/strategy_dir/batch_summary.json
            continue

        date_range = relative_parts[0]
        strategy_dir = relative_parts[1]

        backtests = load_backtests(path)
        for entry in backtests:
            records.append(
                {
                    "date_range": date_range,
                    "strategy_dir": strategy_dir,
                    "combo": entry.get("output_dir", "").split("/")[-1],
                    "total_return": entry.get("total_return"),
                    "sharpe_ratio": entry.get("sharpe_ratio"),
                    "win_rate": entry.get("win_rate"),
                    "max_drawdown": entry.get("max_drawdown"),
                    "source_file": str(path),
                }
            )

    if not records:
        raise ValueError(f"No backtest entries collected for factor {factor_name}")

    df = pd.DataFrame.from_records(records)
    df.sort_values(
        by=["strategy_dir", "date_range", "combo", "source_file"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def metric_summary(series: pd.Series) -> Dict[str, float]:
    """Compute descriptive statistics for a metric series."""
    cleaned = series.dropna()
    if cleaned.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "variance": np.nan,
        }

    q1 = float(cleaned.quantile(0.25))
    q3 = float(cleaned.quantile(0.75))

    return {
        "mean": float(cleaned.mean()),
        "median": float(cleaned.median()),
        "q1": q1,
        "q3": q3,
        "variance": float(cleaned.var(ddof=1)) if len(cleaned) > 1 else 0.0,
    }


def summarise_adjust_backtests(df: pd.DataFrame) -> Dict:
    """Group backtest results by strategy and compute overall metric summaries."""
    metrics = ["total_return", "sharpe_ratio", "win_rate", "max_drawdown"]
    summary: Dict[str, Dict] = {}

    for strategy, group in df.groupby("strategy_dir"):
        overall = {"count": int(len(group)), "metrics": {}}
        for metric in metrics:
            overall["metrics"][metric] = metric_summary(group[metric])
        summary[strategy] = {"overall": overall}

    return summary


def save_summary(factor_name: str, summary_payload: Dict) -> Path:
    """Persist the summary JSON beneath the factor's adjust_backtest directory."""
    output_dir = FACTOR_REPORT_DIR / factor_name / ADJUST_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / SUMMARY_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    return output_path


def analyse_factor_adjust(factor_name: str) -> Path:
    """End-to-end analysis for a factor's adjusted backtests."""
    df = gather_records(factor_name)
    summary = summarise_adjust_backtests(df)
    payload = {
        "factor": factor_name,
        "total_records": int(len(df)),
        "strategies": summary,
    }
    return save_summary(factor_name, payload)


def main() -> None:
    CONFIG_FACTOR_NAME = "cnn_10_03_72h_v4_wma_10h_0.4"

    print(f"📊 Analysing adjusted backtests for factor: {CONFIG_FACTOR_NAME}")
    output_path = analyse_factor_adjust(CONFIG_FACTOR_NAME)
    print(f"✅ Summary saved to {output_path}")


if __name__ == "__main__":
    main()
