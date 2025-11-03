"""
Batch backtest runner.

This script discovers factors saved under `factor_data/` whose directory names
begin with `cnn_10_03_72h_v4`, then runs the existing `batch_backtest` routine
from `base_backtest.py` across a user-supplied list of time ranges.

Usage (edit the CONFIG block at the bottom):
    python strategy/backtest/batch_base_backtest.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from base_backtest import batch_backtest


FACTOR_PREFIX = "cnn_10_03_72h_v4"
FACTOR_DATA_DIR = Path(__file__).resolve().parent / "factor_data"


@dataclass
class BacktestJob:
    factor_name: str
    start_time: str
    end_time: str


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


def expand_jobs(
    factors: Sequence[str],
    time_ranges: Iterable[Sequence[str]],
) -> List[BacktestJob]:
    """Create BacktestJob objects for every factor/time-range combination."""
    jobs: List[BacktestJob] = []
    for factor in factors:
        for time_range in time_ranges:
            if len(time_range) != 2:
                raise ValueError(f"Time range must be [start, end]; got {time_range}")
            start, end = time_range
            jobs.append(BacktestJob(factor_name=factor, start_time=start, end_time=end))
    return jobs


def run_batch_backtests(
    time_ranges: Iterable[Sequence[str]],
    rebalance_hours_list: List[int],
    top_bottom_n_list: List[Sequence[int]],
) -> List[Tuple[str, str, str]]:
    """
    Execute batch_backtest for all discovered factors and requested time ranges.

    Returns a list of tuples (factor_name, start_time, end_time) for completed jobs.
    """
    factors = discover_factors()
    if not factors:
        raise FileNotFoundError(f"No factors found with prefix '{FACTOR_PREFIX}'.")

    jobs = expand_jobs(factors, time_ranges)
    completed: List[Tuple[str, str, str]] = []

    print(f"🚀 Running batch backtests for {len(jobs)} combinations...")
    for job in jobs:
        print(
            f"\n=== Factor: {job.factor_name} | Period: {job.start_time} to {job.end_time} ==="
        )
        batch_backtest(
            factor_name=job.factor_name,
            start_time=job.start_time,
            end_time=job.end_time,
            rebalance_hours_list=rebalance_hours_list,
            top_bottom_n_list=[list(pair) for pair in top_bottom_n_list],
            output_base_dir="./factor_report",
            factor_base_dir="./factor_data",
        )
        completed.append((job.factor_name, job.start_time, job.end_time))
    return completed


def main() -> None:
    """
    Update CONFIG_* values below to schedule batch backtests.
    Each entry in CONFIG_TIME_RANGES should be [start_time, end_time].
    """

    CONFIG_TIME_RANGES: List[List[str]] = [
        ["2025-04-02 15:00:00", "2025-08-31 23:00:00"],
    ]
    CONFIG_REBALANCE_HOURS: List[int] = [16]
    CONFIG_TOP_BOTTOM_PAIRS: List[List[int]] = [[i, i] for i in [1,2,3]]
    run_batch_backtests(
        time_ranges=CONFIG_TIME_RANGES,
        rebalance_hours_list=CONFIG_REBALANCE_HOURS,
        top_bottom_n_list=CONFIG_TOP_BOTTOM_PAIRS,
    )


if __name__ == "__main__":
    main()

