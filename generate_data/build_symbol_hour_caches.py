"""Split monthly filtered hourly caches into per-symbol pickle archives.

Load hourly OHLCV data from `/home/craz/crypto/crypto-data/filter_hour_cache`
and materialise per-symbol folders where each month is saved as an individual
pickle file. This mirrors the workflow from `generate_all_pickle_caches.ipynb`
but targets the filtered hourly cache produced earlier in the pipeline.
"""

import glob
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

SOURCE_DIR = Path("/home/craz/crypto/crypto-data/filter_hour_cache")
TARGET_ROOT = Path("/home/craz/crypto/crypto-data/filter_symbol_cache")


def get_available_months(pattern: str = "usdt_data_*.pkl") -> List[str]:
    """Return sorted YYYY-MM strings for each monthly filtered cache."""
    files = glob.glob(str(SOURCE_DIR / pattern))
    months: List[str] = []
    for file_path in files:
        filename = os.path.basename(file_path)
        year_month = filename.replace("usdt_data_", "").replace(".pkl", "")
        months.append(year_month)
    months.sort()
    return months


def load_monthly_frame(year_month: str) -> Optional[pd.DataFrame]:
    """Load a monthly filtered cache, handling pickle protocol compatibility."""
    file_path = SOURCE_DIR / f"usdt_data_{year_month}.pkl"

    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return None

    try:
        df = pd.read_pickle(file_path)
        print(f"✅ Loaded {year_month}: {len(df)} rows")
        return df
    except Exception as exc:
        if "pickle protocol" not in str(exc):
            print(f"❌ Error loading {year_month}: {exc}")
            return None

        print(f"⚠️ Pickle protocol mismatch for {year_month}, attempting fallback loaders...")
        try:
            with open(file_path, "rb") as handle:
                df = pickle.load(handle)
            print(f"✅ Loaded {year_month} with pickle.load: {len(df)} rows")
            return df
        except Exception as fallback_exc:
            print(f"❌ pickle.load failed: {fallback_exc}")
            return None


def _normalise_symbol_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe for a single symbol has a DatetimeIndex on open_time."""
    working = frame.copy()

    if "open_time" in working.columns:
        working["open_time"] = pd.to_datetime(working["open_time"])
        working = working.set_index("open_time")
    elif "open_time" not in working.index.names:
        raise KeyError("Expected 'open_time' column or index when normalising symbol dataframe.")
    else:
        working.index = pd.to_datetime(working.index)

    # Drop redundant symbol column if present (the directory already scopes the symbol)
    if "symbol" in working.columns and working["symbol"].nunique() == 1:
        working = working.drop(columns=["symbol"])

    return working.sort_index()


def split_month_by_symbol(year_month: str) -> Dict[str, int]:
    """Split a single month into per-symbol pickle files."""
    df = load_monthly_frame(year_month)
    if df is None or df.empty:
        print(f"❌ Skipping {year_month}: unable to load data.")
        return {}

    if df.index.names and "symbol" in df.index.names:
        grouped = df.groupby(level="symbol")
    elif "symbol" in df.columns:
        grouped = df.groupby("symbol")
    else:
        raise KeyError("Monthly frame must contain a 'symbol' column or index level.")

    symbol_counts: Dict[str, int] = {}

    for symbol, symbol_df in grouped:
        symbol_dir = TARGET_ROOT / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # If symbol was an index level, drop it so only open_time remains
        if symbol_df.index.names and "symbol" in symbol_df.index.names:
            symbol_df = symbol_df.droplevel("symbol")

        normalised = _normalise_symbol_dataframe(symbol_df)
        output_path = symbol_dir / f"{symbol}_{year_month}.pkl"
        normalised.to_pickle(output_path)
        symbol_counts[symbol] = len(normalised)
        print(f"   Saved {symbol}: {len(normalised):,} rows -> {output_path}")

    print(f"✅ Completed {year_month}: {len(symbol_counts)} symbols written.")
    return symbol_counts


def build_symbol_caches(
    months: Optional[Iterable[str]] = None,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """Process multiple months and return per-month symbol counts."""
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    if months is None:
        if start_month and end_month:
            months = month_range(start_month, end_month)
        elif start_month or end_month:
            raise ValueError("Both start_month and end_month must be provided together.")
        else:
            months = get_available_months()

    summary: Dict[str, Dict[str, int]] = {}
    months = list(months)

    if not months:
        print("⚠️ No months provided or discovered. Nothing to process.")
        return summary

    print(f"🚀 Building per-symbol caches for {len(months)} months...")

    for idx, month in enumerate(months, 1):
        print(f"\n[{idx:02d}/{len(months)}] Processing {month}")
        summary[month] = split_month_by_symbol(month)

    print("\n🎯 Completed per-symbol cache generation.")
    total_symbols = len({symbol for counts in summary.values() for symbol in counts})
    print(f"   Months processed: {len(summary)}")
    print(f"   Unique symbols output: {total_symbols}")
    print(f"   Output directory: {TARGET_ROOT}")

    return summary

def month_range(start_month: str, end_month: str) -> List[str]:
    """Generate a list of YYYY-MM strings from start to end (inclusive)."""
    start_year, start_mon = map(int, start_month.split("-"))
    end_year, end_mon = map(int, end_month.split("-"))

    if (start_year, start_mon) > (end_year, end_mon):
        raise ValueError(f"start_month {start_month} must be <= end_month {end_month}")

    months: List[str] = []
    current_year, current_mon = start_year, start_mon

    while (current_year < end_year) or (current_year == end_year and current_mon <= end_mon):
        months.append(f"{current_year:04d}-{current_mon:02d}")
        if current_mon == 12:
            current_year += 1
            current_mon = 1
        else:
            current_mon += 1

    return months


def main(start_month: Optional[str] = None, end_month: Optional[str] = None) -> None:
    """Entrypoint for command-line execution."""
    summary = build_symbol_caches(start_month=start_month, end_month=end_month)
    print("\nSummary (month → symbols processed):")
    for month, counts in summary.items():
        print(f"  {month}: {len(counts)} symbols")


if __name__ == "__main__":
    main(start_month="2024-01", end_month="2025-03")
    

