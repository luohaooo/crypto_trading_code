"""Future returns calculation pipeline extracted from calculate_future_returns.ipynb.

Use the exported functions to process hourly OHLCV caches and generate monthly
future-return datasets for downstream modeling.
"""

import glob
import os
import pickle
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = "/home/craz/crypto/crypto-data/filter_hour_cache/"
OUTPUT_PATH = "/home/craz/crypto/crypto-data/future_returns/"

TIME_HORIZONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    24,
    28,
    32,
    36,
    42,
    48,
    54,
    60,
    72,
    84,
    96,
    108,
    120,
    144,
]

os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_available_months(pattern: str = "usdt_data_*.pkl") -> List[str]:
    """Return sorted YYYY-MM strings for every cached hourly pickle."""
    files = glob.glob(os.path.join(INPUT_PATH, pattern))
    months: List[str] = []

    for file_path in files:
        filename = os.path.basename(file_path)
        year_month = filename.replace("usdt_data_", "").replace(".pkl", "")
        months.append(year_month)

    months.sort()
    return months


def load_monthly_data(year_month: str) -> Optional[pd.DataFrame]:
    """Load an hourly bundle while handling pickle protocol compatibility."""
    file_path = os.path.join(INPUT_PATH, f"usdt_data_{year_month}.pkl")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None

    try:
        df = pd.read_pickle(file_path)
        symbol_count = len(df.index.get_level_values("symbol").unique())
        print(f"✅ Loaded {year_month}: {len(df)} rows, {symbol_count} symbols")
        return df
    except (ValueError, ImportError) as exc:
        if "pickle protocol" not in str(exc):
            print(f"❌ Error loading {year_month}: {exc}")
            return None

        print(f"⚠️ Pickle protocol issue for {year_month}, trying fallback loaders...")

        try:
            with open(file_path, "rb") as handle:
                df = pickle.load(handle)
            symbol_count = len(df.index.get_level_values("symbol").unique())
            print(f"✅ Loaded {year_month} with pickle.load: {len(df)} rows, {symbol_count} symbols")
            return df
        except Exception as fallback_exc:
            print(f"❌ pickle.load failed: {fallback_exc}")

        try:
            import joblib  # type: ignore

            df = joblib.load(file_path)
            symbol_count = len(df.index.get_level_values("symbol").unique())
            print(f"✅ Loaded {year_month} with joblib: {len(df)} rows, {symbol_count} symbols")
            return df
        except ImportError:
            print("❌ joblib not available")
        except Exception as joblib_exc:
            print(f"❌ joblib.load failed: {joblib_exc}")

        print("❌ All loading methods failed. Regenerate caches with a compatible pickle protocol.")
        return None
    except Exception as unexpected_exc:  # pragma: no cover - defensive logging
        print(f"❌ Unexpected error loading {year_month}: {unexpected_exc}")
        return None


def get_next_month(year_month: str) -> str:
    """Return the next YYYY-MM string after the supplied month."""
    year, month = map(int, year_month.split("-"))

    if month == 12:
        return f"{year + 1:04d}-01"

    return f"{year:04d}-{month + 1:02d}"


def calculate_future_returns(df: pd.DataFrame, time_horizons: List[int]) -> pd.DataFrame:
    """Compute forward returns for every symbol at the requested horizons."""
    print(f"Calculating future returns for {len(time_horizons)} time horizons...")

    df_reset = df.reset_index()
    df_reset["open_time"] = pd.to_datetime(df_reset["open_time"])

    results: List[pd.DataFrame] = []
    symbols = df_reset["symbol"].unique()
    print(f"Processing {len(symbols)} symbols...")

    for idx, symbol in enumerate(symbols, 1):
        print(f"  [{idx:2d}/{len(symbols)}] Processing {symbol}...", end="")

        symbol_data = df_reset[df_reset["symbol"] == symbol].copy()
        symbol_data = symbol_data.sort_values("open_time")
        symbol_results = symbol_data[["open_time", "symbol", "open"]].copy()

        for horizon in time_horizons:
            future_times = symbol_data["open_time"] + pd.Timedelta(hours=horizon)
            future_prices: List[float] = []

            for current_price, future_time in zip(symbol_data["open"], future_times):
                future_row = symbol_data[symbol_data["open_time"] == future_time]

                if not future_row.empty and current_price > 0:
                    future_price = future_row["open"].iloc[0]
                    return_value = (future_price / current_price) - 1
                else:
                    return_value = np.nan

                future_prices.append(return_value)

            symbol_results[f"return_{horizon}h"] = future_prices

        results.append(symbol_results)
        print(" ✅")

    print("\nCombining results for all symbols...")

    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.set_index(["open_time", "symbol"])
    final_df = final_df.drop(columns=["open"])

    print(f"✅ Calculated returns for {len(final_df)} rows")
    print(f"Return columns: {[col for col in final_df.columns if col.startswith('return_')]}")

    return final_df


def process_monthly_returns(
    year_month: str,
    time_horizons: Optional[List[int]] = None,
) -> bool:
    """Generate and persist future returns for a single month."""
    horizons = time_horizons if time_horizons is not None else TIME_HORIZONS

    print("\n" + "=" * 60)
    print(f"Processing returns for {year_month}")
    print("=" * 60)

    current_df = load_monthly_data(year_month)
    if current_df is None:
        print(f"❌ Failed to load current month: {year_month}")
        return False

    next_month = get_next_month(year_month)
    next_df = load_monthly_data(next_month)

    if next_df is None:
        print(f"⚠️ Next month data not available: {next_month}")
        combined_df = current_df
    else:
        print(f"✅ Combining {year_month} and {next_month} data")
        combined_df = pd.concat([current_df, next_df])
        combined_df = combined_df.sort_index()
        print(f"Combined data shape: {combined_df.shape}")

    returns_df = calculate_future_returns(combined_df, horizons)

    current_month_start = pd.to_datetime(f"{year_month}-01")
    if year_month.endswith("12"):
        next_month_start = pd.to_datetime(f"{int(year_month[:4]) + 1}-01-01")
    else:
        next_month_number = int(year_month[-2:]) + 1
        next_month_start = pd.to_datetime(f"{year_month[:5]}{next_month_number:02d}-01")

    mask = (
        (returns_df.index.get_level_values("open_time") >= current_month_start)
        & (returns_df.index.get_level_values("open_time") < next_month_start)
    )
    current_month_returns = returns_df[mask]

    print(f"Filtered to current month: {len(current_month_returns)} rows")

    output_file = os.path.join(OUTPUT_PATH, f"future_returns_{year_month}.pkl")

    try:
        current_month_returns.to_pickle(output_file)
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Saved: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")

        return_columns = [col for col in current_month_returns.columns if col.startswith("return_")]
        non_null_counts = current_month_returns[return_columns].count()

        print("\nValidation:")
        print(f"- Shape: {current_month_returns.shape}")
        print(f"- Unique symbols: {len(current_month_returns.index.get_level_values('symbol').unique())}")
        print(f"- Non-null return counts (first 5): {non_null_counts.head().to_dict()}")

        return True
    except Exception as exc:
        print(f"❌ Error saving: {exc}")
        return False


def batch_process_returns(
    start_month: str,
    end_month: str,
    time_horizons: Optional[List[int]] = None,
    available_months: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Batch process future returns for a range of months."""
    horizons = time_horizons if time_horizons is not None else TIME_HORIZONS

    print("\n" + "=" * 80)
    print(f"BATCH PROCESSING: {start_month} to {end_month}")
    print("=" * 80)

    start_year, start_mon = map(int, start_month.split("-"))
    end_year, end_mon = map(int, end_month.split("-"))

    months_to_process: List[str] = []
    current_year, current_mon = start_year, start_mon

    while (current_year < end_year) or (current_year == end_year and current_mon <= end_mon):
        months_to_process.append(f"{current_year:04d}-{current_mon:02d}")
        if current_mon == 12:
            current_year += 1
            current_mon = 1
        else:
            current_mon += 1

    print(f"Months to process: {len(months_to_process)}")
    print(f"Month range: {months_to_process[0]} to {months_to_process[-1]}")

    if available_months is None:
        available_months = get_available_months()

    available_to_process = [month for month in months_to_process if month in available_months]
    unavailable = [month for month in months_to_process if month not in available_months]

    print(f"Available to process: {len(available_to_process)}")
    if unavailable:
        print(f"⚠️ Unavailable months: {unavailable}")

    results: Dict[str, List[str]] = {"successful": [], "failed": [], "skipped": unavailable}

    for idx, month in enumerate(available_to_process, 1):
        print(f"\n[{idx:2d}/{len(available_to_process)}] Processing {month}...")
        try:
            success = process_monthly_returns(month, horizons)
            if success:
                results["successful"].append(month)
                print(f"✅ {month} completed successfully")
            else:
                results["failed"].append(month)
                print(f"❌ {month} failed")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Exception processing {month}: {exc}")
            results["failed"].append(month)

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {len(results['successful'])} months")
    print(f"Failed: {len(results['failed'])} months")
    print(f"Skipped: {len(results['skipped'])} months")

    if results["successful"]:
        print("\n✅ Successfully processed:")
        for month in results["successful"]:
            output_file = os.path.join(OUTPUT_PATH, f"future_returns_{month}.pkl")
            if os.path.exists(output_file):
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  {month}: {file_size_mb:.2f} MB")

    if results["failed"]:
        print(f"\n❌ Failed to process: {results['failed']}")

    if results["skipped"]:
        print(f"\n⚠️ Skipped (unavailable): {results['skipped']}")

    return results

example_results = batch_process_returns('2025-08', '2025-08')



