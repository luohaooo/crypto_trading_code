"""
Weighted moving average (WMA) factor generator.

Load an existing factor from ``strategy/backtest/factor_data/<factor_name>``,
apply an exponentially decayed weighted moving average across a configurable
time span, and save the result back to ``factor_data`` organised by month.

Example usage (edit the CONFIG_* block in __main__):

    python strategy/backtest/wma.py
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Compatibility for pickle protocol 5 on Python < 3.8
    import pickle5 as pickle  # type: ignore
except ImportError:  # pragma: no cover
    import pickle  # type: ignore

FACTOR_DATA_ROOT = Path(__file__).resolve().parent / "factor_data"


def _ensure_legacy_pandas_modules() -> None:
    """
    Older factor pickles may reference ``pandas.core.internals.managers`` which
    was refactored in later pandas releases. Inject a shim so pickle.load works.
    """
    module_name = "pandas.core.internals.managers"
    if module_name in sys.modules:
        return

    import pandas.core.internals as legacy_internals  # type: ignore

    shim = types.ModuleType(module_name)
    for attr in dir(legacy_internals):
        if not attr.startswith("__"):
            setattr(shim, attr, getattr(legacy_internals, attr))
    sys.modules[module_name] = shim


def read_pickle_compat(path: Path) -> pd.DataFrame:
    """Load a pickle file with fallbacks for protocol and pandas compat."""
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")

    try:
        return pd.read_pickle(path)
    except Exception:  # pylint: disable=broad-except
        _ensure_legacy_pandas_modules()
        with path.open("rb") as handle:
            return pickle.load(handle)


def month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the month start (naive timestamp) for a pandas Timestamp."""
    ts = pd.Timestamp(timestamp)
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def iter_month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    """Yield month starts for the inclusive range."""
    current = month_start(start)
    end_month = month_start(end)
    while current <= end_month:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def extract_factor_series(frame: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Pull the primary numeric factor series from a DataFrame.

    Prefers a column named 'factor'. If absent, uses the first numeric column.
    """
    if isinstance(frame, pd.Series):
        return frame, frame.name or "factor"

    if "factor" in frame.columns:
        return frame["factor"], "factor"

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to compute WMA.")

    column = numeric_cols[0]
    return frame[column], column


def build_weights(time_span_hours: int, decay: float) -> np.ndarray:
    """Create decay weights for a span (inclusive of current timestamp)."""
    if time_span_hours < 0:
        raise ValueError("time_span_hours must be non-negative")
    if not (0 <= decay):
        raise ValueError("decay must be non-negative")
    powers = np.arange(time_span_hours, -1, -1, dtype=np.float64)
    with np.errstate(over="ignore"):
        weights = np.power(decay, powers)
    return weights


def apply_weighted_average(series: pd.Series, weights: np.ndarray) -> pd.Series:
    """Apply weighted moving average to a single-symbol Series."""
    window = len(weights)
    values = series.to_numpy(dtype=np.float64, copy=False)
    result = np.full_like(values, fill_value=np.nan)

    for idx in range(window - 1, len(series)):
        window_slice = values[idx - window + 1 : idx + 1]
        mask = ~np.isnan(window_slice)
        if not mask.any():
            continue

        effective_weights = weights.copy()
        if mask.sum() != mask.size:
            effective_weights = effective_weights[mask]
            window_slice = window_slice[mask]

        denominator = effective_weights.sum()
        if math.isclose(denominator, 0.0):
            continue

        result[idx] = float(np.dot(window_slice, effective_weights) / denominator)

    return pd.Series(result, index=series.index, name=series.name)


@dataclass
class WMAConfig:
    factor_name: str
    start_time: str
    end_time: str
    time_span_hours: int
    decay: float
    output_factor_name: Optional[str] = None


class WeightedFactorGenerator:
    """Core engine that orchestrates factor loading, transformation, and saving."""

    def __init__(self, data_root: Path = FACTOR_DATA_ROOT) -> None:
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _factor_dir(self, name: str) -> Path:
        path = self.data_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _factor_file(self, name: str, timestamp: pd.Timestamp) -> Path:
        return self._factor_dir(name) / f"factor_{timestamp.year:04d}-{timestamp.month:02d}.pkl"

    def load_factor_months(
        self,
        factor_name: str,
        months: Sequence[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        frames: Dict[pd.Timestamp, pd.DataFrame] = {}
        source_dir = self._factor_dir(factor_name)

        for month_ts in months:
            file_path = source_dir / f"factor_{month_ts.year:04d}-{month_ts.month:02d}.pkl"
            if not file_path.exists():
                print(f"⚠️  Missing factor file: {file_path}")
                continue
            try:
                frames[month_ts] = read_pickle_compat(file_path)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️  Failed to load {file_path}: {exc}")

        return frames

    def compute(
        self,
        config: WMAConfig,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        start_ts = pd.to_datetime(config.start_time)
        end_ts = pd.to_datetime(config.end_time)
        if start_ts > end_ts:
            raise ValueError("start_time must be earlier than end_time")

        context_start = start_ts - timedelta(hours=config.time_span_hours)
        all_months = list(iter_month_starts(context_start, end_ts))
        if not all_months:
            raise ValueError("No months to process for the given range.")

        loaded_frames = self.load_factor_months(config.factor_name, all_months)
        if not loaded_frames:
            raise FileNotFoundError("No factor files loaded; check factor name and dates.")

        combined = pd.concat(loaded_frames.values()).sort_index()
        base_series, column_name = extract_factor_series(combined)

        if isinstance(base_series.index, pd.MultiIndex):
            if base_series.index.names[0] != "open_time":
                base_series.index = base_series.index.set_names("open_time", level=0)

        weights = build_weights(config.time_span_hours, config.decay)

        if isinstance(base_series.index, pd.MultiIndex) and "symbol" in base_series.index.names:
            results = []
            for symbol, symbol_series in base_series.groupby(level="symbol"):
                symbol_series = symbol_series.droplevel("symbol")
                wma_series = apply_weighted_average(symbol_series, weights)
                wma_series.index = pd.MultiIndex.from_product(
                    [wma_series.index, [symbol]],
                    names=["open_time", "symbol"],
                )
                results.append(wma_series)
            wma_full = pd.concat(results).sort_index()
        else:
            wma_full = apply_weighted_average(base_series, weights)

        # Restrict to requested window (including context for weighting)
        if isinstance(wma_full.index, pd.MultiIndex):
            mask = (wma_full.index.get_level_values("open_time") >= start_ts) & (
                wma_full.index.get_level_values("open_time") <= end_ts
            )
            wma_filtered = wma_full[mask]
        else:
            wma_filtered = wma_full.loc[start_ts:end_ts]

        if isinstance(wma_filtered, pd.Series):
            result_df = wma_filtered.to_frame("factor_value")
        else:
            result_df = wma_filtered.to_frame(name="factor_value")

        # Group by month for saving
        grouped: Dict[pd.Timestamp, pd.DataFrame] = {}
        if isinstance(result_df.index, pd.MultiIndex):
            grouper = result_df.groupby(pd.Grouper(level="open_time", freq="ME"))
            for month_key, frame in grouper:
                if frame.empty:
                    continue
                month_start_ts = month_start(month_key)
                grouped[month_start_ts] = frame.sort_index()
        else:
            grouper = result_df.groupby(pd.Grouper(freq="ME"))
            for month_key, frame in grouper:
                if frame.empty:
                    continue
                month_start_ts = month_start(month_key)
                grouped[month_start_ts] = frame.sort_index()

        return grouped

    def save_results(
        self,
        output_name: str,
        monthly_frames: Dict[pd.Timestamp, pd.DataFrame],
    ) -> List[Path]:
        output_files: List[Path] = []
        for month_ts, frame in monthly_frames.items():
            file_path = self._factor_file(output_name, month_ts)
            frame.to_pickle(file_path)
            output_files.append(file_path)
            print(f"💾 Saved {file_path} ({len(frame):,} rows)")
        return output_files


def run_weighted_factor(config: WMAConfig) -> List[Path]:
    """Convenience wrapper to compute and save the WMA factor."""
    generator = WeightedFactorGenerator()
    monthly_frames = generator.compute(config)
    output_name = config.output_factor_name or f"{config.factor_name}_wma"
    return generator.save_results(output_name, monthly_frames)


def main() -> None:
    """
    Update CONFIG_* values below and execute the file to generate a WMA factor.
    """

    CONFIG = WMAConfig(
        factor_name="cnn_10_03_72h_v4",  # Source factor directory name
        start_time="2025-04-01 00:00:00",
        end_time="2025-08-31 23:00:00",
        time_span_hours=2,
        decay=0.5,
        output_factor_name="cnn_10_03_72h_v4_wma_2h_0p5",  # Optional; defaults to '<factor_name>_wma'
    )

    print("🚀 Generating weighted moving average factor...")
    output_files = run_weighted_factor(CONFIG)
    print(f"✅ Completed WMA generation. Files written: {len(output_files)}")


if __name__ == "__main__":
    main()
