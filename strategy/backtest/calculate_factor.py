"""
Factor Calculation and Storage Module

This module provides functionality to calculate trading factors on a monthly basis
and save them to pickle files for later use in backtesting.

Features:
- Monthly batch processing of factor calculations
- Automatic handling of historical data requirements (merges previous month data)
- Progress tracking with tqdm
- Organized output directory structure
- Memory-efficient processing (one month at a time)

Usage:
    See the main() function below for a complete example
"""

import sys
import os
import pickle
import pandas as pd
from typing import Optional, List
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from factors.base_factor import BaseFactor
from factors.ohlc_figure_factor import OHLCFigureFactor


def load_monthly_data(year: int, month: int, data_dir: str = "/home/craz/crypto/crypto-data/pickle_hour_cache") -> Optional[pd.DataFrame]:
    """
    Load hourly data for a specific month from pickle cache.

    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
        data_dir: Directory containing monthly pickle files

    Returns:
        pd.DataFrame: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
                     Returns None if file not found
    """
    try:
        filename = f"usdt_data_{year:04d}-{month:02d}.pkl"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Data file not found: {filepath}")
            return None

        # Load pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Ensure proper index format
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"Expected MultiIndex, got {type(data.index)}")

        # Verify index names
        expected_names = ['open_time', 'symbol']
        if data.index.names != expected_names:
            warnings.warn(f"Index names mismatch: expected {expected_names}, got {data.index.names}")

        return data

    except Exception as e:
        warnings.warn(f"Error loading data for {year}-{month:02d}: {e}")
        return None


def get_month_range(start_date: str, end_date: str) -> List[tuple]:
    """
    Generate a list of (year, month) tuples for the date range.

    Args:
        start_date: Start date in format 'YYYY-MM' or 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM' or 'YYYY-MM-DD'

    Returns:
        List of (year, month) tuples

    Example:
        >>> get_month_range('2025-01', '2025-03')
        [(2025, 1), (2025, 2), (2025, 3)]
    """
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Generate month range
    months = []
    current = start.replace(day=1)

    while current <= end:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


def calculate_and_save_factor_monthly(
    factor: BaseFactor,
    start_date: str,
    end_date: str,
    output_dir: str = "./factor_data",
    data_dir: str = "/home/craz/crypto/crypto-data/pickle_hour_cache",
    overwrite: bool = False
) -> dict:
    """
    Calculate factor values for all timestamps in the date range and save monthly.

    This function:
    1. Iterates through months in the specified date range
    2. Loads hourly data for each month (and previous month for context)
    3. Calculates factor values for each hour using factor.calculate()
    4. Saves results to monthly pickle files

    Args:
        factor: Factor instance (must implement BaseFactor.calculate)
        start_date: Start date in format 'YYYY-MM' or 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM' or 'YYYY-MM-DD'
        output_dir: Directory to save factor files (default: "./factor_data")
        data_dir: Directory containing source hourly data pickle files
        overwrite: If True, overwrite existing factor files; if False, skip existing months

    Returns:
        dict: Statistics about the calculation process
            - 'months_processed': Number of months successfully processed
            - 'months_skipped': Number of months skipped (missing data or already exists)
            - 'total_timestamps': Total number of timestamps processed
            - 'output_files': List of created output file paths

    Output Format:
        Each monthly file contains a DataFrame with:
        - MultiIndex: (open_time, symbol)
        - Column: factor value for that symbol at that timestamp

    Example:
        >>> factor = OHLCFigureFactor(model_path='model.pt')
        >>> stats = calculate_and_save_factor_monthly(
        ...     factor=factor,
        ...     start_date='2025-01',
        ...     end_date='2025-03'
        ... )
        >>> print(f"Processed {stats['months_processed']} months")
    """
    # Create output directory structure
    factor_output_dir = os.path.join(output_dir, factor.name)
    os.makedirs(factor_output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Factor Calculation: {factor.name}")
    print(f"{'='*80}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output Directory: {factor_output_dir}")
    print(f"Overwrite Mode: {overwrite}")
    print(f"{'='*80}\n")

    # Parse start and end dates for timestamp filtering
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Get month range
    months = get_month_range(start_date, end_date)
    print(f"Processing {len(months)} months: {months[0]} to {months[-1]}\n")

    # Statistics tracking
    stats = {
        'months_processed': 0,
        'months_skipped': 0,
        'total_timestamps': 0,
        'output_files': []
    }

    # Process each month
    for i, (year, month) in enumerate(months, 1):
        print(f"\n[{i}/{len(months)}] Processing {year}-{month:02d}")
        print("-" * 80)

        # Check if output file already exists
        output_filename = f"factor_{year:04d}-{month:02d}.pkl"
        output_filepath = os.path.join(factor_output_dir, output_filename)

        if os.path.exists(output_filepath) and not overwrite:
            print(f"[SKIP] File already exists: {output_filename}")
            stats['months_skipped'] += 1
            continue

        # Load current month data
        print(f"[LOAD] Loading current month data: {year}-{month:02d}")
        current_data = load_monthly_data(year, month, data_dir)

        if current_data is None:
            print(f"[ERROR] Failed to load data for {year}-{month:02d}")
            stats['months_skipped'] += 1
            continue

        print(f"       Loaded {len(current_data):,} rows, {len(current_data.index.get_level_values('symbol').unique())} symbols")

        # Load previous month data for context (factors may need historical data)
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        print(f"[LOAD] Loading previous month data for context: {prev_year}-{prev_month:02d}")
        prev_data = load_monthly_data(prev_year, prev_month, data_dir)

        if prev_data is not None:
            print(f"       Loaded {len(prev_data):,} rows")
            # Merge previous and current month data
            combined_data = pd.concat([prev_data, current_data], axis=0)
            # Remove duplicates if any, keeping later occurrences
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            print(f"       Combined data: {len(combined_data):,} rows")
        else:
            print(f"[WARN] Previous month data not available, using current month only")
            combined_data = current_data

        # Get all unique timestamps for current month (not previous month)
        current_timestamps = current_data.index.get_level_values('open_time').unique().sort_values()

        # Filter timestamps to only include those within user-specified date range
        current_timestamps = current_timestamps[(current_timestamps >= start_dt) & (current_timestamps <= end_dt)]

        if len(current_timestamps) == 0:
            print(f"[WARN] No timestamps in specified date range for {year}-{month:02d}")
            stats['months_skipped'] += 1
            continue

        print(f"\n[CALC] Processing {len(current_timestamps):,} timestamps (filtered by date range)...")

        # Calculate factors for each timestamp
        factor_results = []
        total_timestamps = len(current_timestamps)

        for idx, timestamp in enumerate(current_timestamps, 1):
            print(f"       [{idx}/{total_timestamps}] {timestamp}")
            try:
                # Call factor.calculate() with combined data (includes historical context)
                factor_values = factor.calculate(combined_data, timestamp)

                if factor_values is None or len(factor_values) == 0:
                    # No valid factor values for this timestamp
                    continue

                # Convert Series to DataFrame with timestamp
                factor_df = factor_values.to_frame(name='factor_value')
                factor_df.index.name = 'symbol'  # Ensure index has a name for reset_index()
                factor_df['open_time'] = timestamp
                factor_df = factor_df.reset_index()  # Move symbol from index to column
                factor_df = factor_df.set_index(['open_time', 'symbol'])

                factor_results.append(factor_df)

            except Exception as e:
                warnings.warn(f"Error calculating factor at {timestamp}: {e}")
                continue

        # Combine all results for this month
        if len(factor_results) == 0:
            print(f"\n[ERROR] No valid factor values calculated for {year}-{month:02d}")
            stats['months_skipped'] += 1
            continue

        monthly_factors = pd.concat(factor_results, axis=0)
        monthly_factors = monthly_factors.sort_index()

        print(f"\n[DONE] Calculated factors for {len(factor_results):,} timestamps")
        print(f"       Total rows: {len(monthly_factors):,}")
        print(f"       Symbols: {len(monthly_factors.index.get_level_values('symbol').unique())}")
        print(f"       Factor value range: [{monthly_factors['factor_value'].min():.6f}, {monthly_factors['factor_value'].max():.6f}]")

        # Save to pickle file
        print(f"\n[SAVE] Saving to: {output_filename}")
        with open(output_filepath, 'wb') as f:
            pickle.dump(monthly_factors, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"       Saved successfully")

        # Update statistics
        stats['months_processed'] += 1
        stats['total_timestamps'] += len(factor_results)
        stats['output_files'].append(output_filepath)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Months Processed: {stats['months_processed']}")
    print(f"Months Skipped: {stats['months_skipped']}")
    print(f"Total Timestamps: {stats['total_timestamps']:,}")
    print(f"Output Files: {len(stats['output_files'])}")
    print(f"{'='*80}\n")

    return stats


def load_factor_data(
    factor_name: str,
    start_date: str,
    end_date: str,
    factor_dir: str = "./factor_data"
) -> Optional[pd.DataFrame]:
    """
    Load saved factor data for a date range.

    Args:
        factor_name: Name of the factor (must match folder name)
        start_date: Start date in format 'YYYY-MM'
        end_date: End date in format 'YYYY-MM'
        factor_dir: Base directory containing factor data

    Returns:
        pd.DataFrame: Combined factor data with MultiIndex (open_time, symbol)

    Example:
        >>> factors = load_factor_data('OHLC_Figure_1h+2h+4h', '2025-01', '2025-03')
    """
    months = get_month_range(start_date, end_date)
    factor_data_dir = os.path.join(factor_dir, factor_name)

    all_data = []

    for year, month in months:
        filename = f"factor_{year:04d}-{month:02d}.pkl"
        filepath = os.path.join(factor_data_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Factor file not found: {filepath}")
            continue

        try:
            with open(filepath, 'rb') as f:
                monthly_data = pickle.load(f)
            all_data.append(monthly_data)
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue

    if len(all_data) == 0:
        warnings.warn(f"No factor data found for {factor_name} in range {start_date} to {end_date}")
        return None

    # Combine all months
    combined = pd.concat(all_data, axis=0)
    combined = combined.sort_index()

    return combined


def main():
    """
    Main function demonstrating how to use the factor calculation module.

    This example shows how to:
    1. Initialize a factor (OHLC Figure Factor)
    2. Calculate factors for a date range
    3. Save results to monthly pickle files
    4. Load saved factor data
    """
    print("\n" + "="*80)
    print("FACTOR CALCULATION EXAMPLE")
    print("="*80 + "\n")

    # Configuration
    # ===========================================================================

    # 1. Initialize the factor
    # -------------------------------------------------------------------------
    # Example: OHLC Figure Factor with trained model
    # Adjust the model_path to point to your trained model file

    MODEL_PATH = "/home/craz/crypto/crypto-trading/figure_model/model_saved/baseline_epoch_69_train_0.02644_val_0.02999.pt"

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("\nPlease update MODEL_PATH in the main() function to point to your trained model.")
        print("Available models in figure_model/model_saved/:")
        model_dir = "/home/craz/crypto/crypto-trading/figure_model/model_saved"
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return

    print("Initializing OHLC Figure Factor...")
    factor = OHLCFigureFactor(
        model_path=MODEL_PATH,
        device='auto',  # Use 'cuda' if GPU available, otherwise 'cpu'
        name="precious_ohlc_cnn",  # Auto-generate name based on timeframes
        lookback_periods=20,
        timeframes=['1h', '2h', '4h']  # Multi-timeframe analysis
    )
    print(f"[SUCCESS] Factor initialized: {factor.name}\n")

    # 2. Define date range
    # -------------------------------------------------------------------------
    # Calculate factors for specific date and time range
    # Note: Ensure you have hourly data files for these months in pickle_hour_cache

    START_DATE = '2024-01-01 00:00:00'  # Format: 'YYYY-MM-DD HH:MM:SS'
    END_DATE = '2025-03-31 23:00:00'    # Format: 'YYYY-MM-DD HH:MM:SS'

    # 3. Set output directory
    # -------------------------------------------------------------------------
    OUTPUT_DIR = "./factor_data"  # Factors will be saved to factor_data/{factor_name}/

    # 4. Calculate and save factors
    # -------------------------------------------------------------------------
    print(f"Starting factor calculation...")
    print(f"This may take a while depending on the date range and number of symbols.\n")

    stats = calculate_and_save_factor_monthly(
        factor=factor,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=OUTPUT_DIR,
        overwrite=False  # Set to True to recalculate existing months
    )


if __name__ == "__main__":
    main()
