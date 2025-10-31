"""
IC Analysis Module for Factor Performance Evaluation

This module provides comprehensive Information Coefficient (IC) analysis for trading factors.
It calculates IC, Rank IC, ICIR, and Rank ICIR metrics, and generates visualization reports.

Features:
- Load factor data and future returns data from monthly pickle files
- Calculate IC (Pearson correlation) and Rank IC (Spearman correlation)
- Compute ICIR (IC Information Ratio) for consistency measurement
- Generate bar chart visualizations for all return horizons
- Save detailed results and summary statistics
- Organized output directory structure

Usage:
    python ic_analysis.py

Or import as module:
    from ic_analysis import run_ic_analysis

    run_ic_analysis(
        factor_name='precious_ohlc_cnn',
        start_time='2024-01-01 00:00:00',
        end_time='2024-03-31 23:00:00'
    )
"""

import os
import sys
import pickle
import warnings
import json
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_monthly_returns_data(year: int, month: int,
                             returns_dir: str = "../../../crypto-data/future_returns") -> Optional[pd.DataFrame]:
    """
    Load future returns data for a specific month.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        returns_dir: Directory containing returns pickle files

    Returns:
        pd.DataFrame: Returns data with MultiIndex (open_time, symbol)
                     and columns for different return horizons (return_1h, return_2h, etc.)
    """
    try:
        filename = f"future_returns_{year:04d}-{month:02d}.pkl"
        filepath = os.path.join(returns_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Returns file not found: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Verify data structure
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"Expected MultiIndex, got {type(data.index)}")

        expected_names = ['open_time', 'symbol']
        if data.index.names != expected_names:
            warnings.warn(f"Index names mismatch: expected {expected_names}, got {data.index.names}")

        return data

    except Exception as e:
        warnings.warn(f"Error loading returns data for {year}-{month:02d}: {e}")
        return None


def load_monthly_factor_data(factor_name: str, year: int, month: int,
                           factor_dir: str = "./factor_data") -> Optional[pd.DataFrame]:
    """
    Load factor data for a specific month.

    Args:
        factor_name: Name of the factor (folder name)
        year: Year (e.g., 2024)
        month: Month (1-12)
        factor_dir: Base directory containing factor data

    Returns:
        pd.DataFrame: Factor data with MultiIndex (open_time, symbol)
                     and 'factor_value' column
    """
    try:
        factor_path = os.path.join(factor_dir, factor_name)
        filename = f"factor_{year:04d}-{month:02d}.pkl"
        filepath = os.path.join(factor_path, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Factor file not found: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Verify data structure
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"Expected MultiIndex, got {type(data.index)}")

        if 'factor_value' not in data.columns:
            raise ValueError(f"Expected 'factor_value' column, got {data.columns.tolist()}")

        return data

    except Exception as e:
        warnings.warn(f"Error loading factor data for {factor_name} {year}-{month:02d}: {e}")
        return None


def get_month_range(start_time: str, end_time: str) -> List[Tuple[int, int]]:
    """
    Generate list of (year, month) tuples for the time range.

    Args:
        start_time: Start timestamp in format 'YYYY-MM-DD HH:MM:SS'
        end_time: End timestamp in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        List of (year, month) tuples
    """
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    months = []
    current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current <= end_dt:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


def load_data_for_period(factor_name: str, start_time: str, end_time: str,
                        factor_dir: str = "./factor_data",
                        returns_dir: str = "../../../crypto-data/future_returns") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load and combine factor and returns data for the specified time period.

    Args:
        factor_name: Name of the factor
        start_time: Start timestamp
        end_time: End timestamp
        factor_dir: Factor data directory
        returns_dir: Returns data directory

    Returns:
        Tuple of (factor_df, returns_df) or (None, None) if loading fails
    """
    print(f"Loading data for period: {start_time} to {end_time}")

    # Get months to load
    months = get_month_range(start_time, end_time)
    print(f"Loading {len(months)} months: {months}")

    factor_data_list = []
    returns_data_list = []

    # Load data for each month
    for year, month in months:
        print(f"  Loading {year}-{month:02d}...")

        # Load factor data
        factor_monthly = load_monthly_factor_data(factor_name, year, month, factor_dir)
        if factor_monthly is not None:
            factor_data_list.append(factor_monthly)

        # Load returns data
        returns_monthly = load_monthly_returns_data(year, month, returns_dir)
        if returns_monthly is not None:
            returns_data_list.append(returns_monthly)

    # Combine monthly data
    if len(factor_data_list) == 0:
        warnings.warn(f"No factor data found for {factor_name}")
        return None, None

    if len(returns_data_list) == 0:
        warnings.warn("No returns data found")
        return None, None

    # Concatenate all months
    factor_df = pd.concat(factor_data_list, axis=0).sort_index()
    returns_df = pd.concat(returns_data_list, axis=0).sort_index()

    # Filter by time range
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    factor_timestamps = factor_df.index.get_level_values('open_time')
    returns_timestamps = returns_df.index.get_level_values('open_time')

    factor_mask = (factor_timestamps >= start_dt) & (factor_timestamps <= end_dt)
    returns_mask = (returns_timestamps >= start_dt) & (returns_timestamps <= end_dt)

    factor_df = factor_df[factor_mask]
    returns_df = returns_df[returns_mask]

    print(f"Loaded factor data: {len(factor_df):,} rows")
    print(f"Loaded returns data: {len(returns_df):,} rows")
    print(f"Factor timestamps: {factor_timestamps[factor_mask].min()} to {factor_timestamps[factor_mask].max()}")
    print(f"Returns timestamps: {returns_timestamps[returns_mask].min()} to {returns_timestamps[returns_mask].max()}")

    return factor_df, returns_df


def calculate_ic_for_timestamp(factor_values: pd.Series, returns_row: pd.Series, return_col: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate IC and Rank IC for a single timestamp.

    Args:
        factor_values: Factor values for symbols at this timestamp
        returns_row: Returns data for symbols at this timestamp
        return_col: Return column name (e.g., 'return_12h')

    Returns:
        Tuple of (ic, rank_ic) or (None, None) if calculation fails
    """
    try:
        # Align data by symbol
        common_symbols = factor_values.index.intersection(returns_row.index)

        if len(common_symbols) < 3:  # Need at least 3 symbols for meaningful correlation
            return None, None

        aligned_factors = factor_values.loc[common_symbols]
        aligned_returns = returns_row.loc[common_symbols]

        # Remove NaN values and filter factor values 
        factor_valid = pd.notna(aligned_factors) & np.isfinite(aligned_factors) 
        returns_valid = pd.notna(aligned_returns) & np.isfinite(aligned_returns)
        valid_mask = factor_valid & returns_valid

        if valid_mask.sum() < 3:
            return None, None

        clean_factors = aligned_factors[valid_mask]
        clean_returns = aligned_returns[valid_mask]

        # Calculate IC (Pearson correlation)
        if clean_factors.std() == 0 or clean_returns.std() == 0:
            ic = 0.0
        else:
            ic, _ = pearsonr(clean_factors, clean_returns)

        # Calculate Rank IC (Spearman correlation)
        if len(clean_factors.unique()) == 1 or len(clean_returns.unique()) == 1:
            rank_ic = 0.0
        else:
            rank_ic, _ = spearmanr(clean_factors, clean_returns)

        return ic, rank_ic

    except Exception as e:
        warnings.warn(f"Error calculating IC for {return_col}: {e}")
        return None, None


def calculate_ic_analysis(factor_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate IC and Rank IC for all timestamps and return horizons.

    Args:
        factor_df: Factor data with MultiIndex (open_time, symbol)
        returns_df: Returns data with MultiIndex (open_time, symbol)

    Returns:
        Dict with 'ic' and 'rank_ic' DataFrames
        Each DataFrame has timestamps as index and return horizons as columns
    """
    print("\nCalculating IC analysis...")

    # Get return columns (all columns starting with 'return_')
    return_columns = [col for col in returns_df.columns if col.startswith('return_')]
    return_columns.sort(key=lambda x: int(x.split('_')[1].replace('h', '')))  # Sort by hour number

    print(f"Found {len(return_columns)} return horizons: {return_columns}")

    # Get common timestamps
    factor_timestamps = set(factor_df.index.get_level_values('open_time'))
    returns_timestamps = set(returns_df.index.get_level_values('open_time'))
    common_timestamps = sorted(factor_timestamps.intersection(returns_timestamps))

    print(f"Found {len(common_timestamps)} common timestamps")

    if len(common_timestamps) == 0:
        raise ValueError("No common timestamps found between factor and returns data")

    # Initialize result dataframes
    ic_results = pd.DataFrame(index=common_timestamps, columns=return_columns, dtype=float)
    rank_ic_results = pd.DataFrame(index=common_timestamps, columns=return_columns, dtype=float)

    # Calculate IC for each timestamp and return horizon
    total_calculations = len(common_timestamps) * len(return_columns)
    calculation_count = 0

    for timestamp in common_timestamps:
        try:
            # Get factor values for this timestamp - need to handle MultiIndex properly
            factor_at_time = factor_df.loc[factor_df.index.get_level_values('open_time') == timestamp, 'factor_value']
            # Reset index to make symbol the index
            factor_at_time.index = factor_df.loc[factor_df.index.get_level_values('open_time') == timestamp].index.get_level_values('symbol')

            # Get returns for this timestamp
            returns_at_time = returns_df.loc[returns_df.index.get_level_values('open_time') == timestamp]
            # Reset index to make symbol the index
            returns_at_time.index = returns_df.loc[returns_df.index.get_level_values('open_time') == timestamp].index.get_level_values('symbol')

            for return_col in return_columns:
                calculation_count += 1
                if calculation_count % 1000 == 0:
                    print(f"  Progress: {calculation_count:,}/{total_calculations:,} ({100*calculation_count/total_calculations:.1f}%)")

                # Calculate IC and Rank IC
                ic, rank_ic = calculate_ic_for_timestamp(factor_at_time, returns_at_time[return_col], return_col)

                ic_results.loc[timestamp, return_col] = ic
                rank_ic_results.loc[timestamp, return_col] = rank_ic

        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")
            continue

    print(f"Completed {calculation_count:,} IC calculations")

    return {
        'ic': ic_results,
        'rank_ic': rank_ic_results
    }


def calculate_ic_statistics(ic_df: pd.DataFrame, rank_ic_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate ICIR and other summary statistics.

    Args:
        ic_df: IC results DataFrame
        rank_ic_df: Rank IC results DataFrame

    Returns:
        Dict with statistics for each return horizon
    """
    print("\nCalculating IC statistics...")

    return_columns = ic_df.columns

    # Calculate statistics for each return horizon
    ic_mean = ic_df.mean()
    ic_std = ic_df.std()
    icir = ic_mean / ic_std  # IC Information Ratio

    rank_ic_mean = rank_ic_df.mean()
    rank_ic_std = rank_ic_df.std()
    rank_icir = rank_ic_mean / rank_ic_std  # Rank IC Information Ratio

    statistics = {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_std': rank_ic_std,
        'rank_icir': rank_icir
    }

    # Print summary
    print("IC Statistics Summary:")
    print("-" * 50)
    for col in return_columns[:5]:  # Show first 5 horizons
        hours = col.replace('return_', '').replace('h', '')
        print(f"{hours}h: IC={ic_mean[col]:.4f}±{ic_std[col]:.4f} (ICIR={icir[col]:.2f}), "
              f"RankIC={rank_ic_mean[col]:.4f}±{rank_ic_std[col]:.4f} (RankICIR={rank_icir[col]:.2f})")

    if len(return_columns) > 5:
        print("...")

    return statistics


def create_ic_bar_chart(ic_series: pd.Series, title: str, output_path: str,
                       figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Create and save IC bar chart with uniform spacing.

    Args:
        ic_series: IC values for different return horizons
        title: Chart title
        output_path: Output file path
        figsize: Figure size (width, height)
    """
    # Extract hour numbers and prepare labels
    hours = [int(str(col).replace('return_', '').replace('h', '')) for col in ic_series.index]
    values = ic_series.values
    hour_labels = [f'{h}h' for h in hours]

    # Create bar chart with uniform spacing
    _, ax = plt.subplots(figsize=figsize)

    # Use uniform positions (0, 1, 2, 3, ...) instead of actual hour values
    positions = range(len(hours))

    # Color bars based on positive/negative values
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax.bar(positions, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Customize chart
    ax.set_xlabel('Return Horizon')
    ax.set_ylabel('IC Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Set x-axis ticks to show horizon labels with intelligent sampling
    total_positions = len(positions)

    if total_positions <= 15:
        # Show all labels if there are few horizons
        ax.set_xticks(positions)
        ax.set_xticklabels(hour_labels, rotation=45, fontsize=8)
    else:
        # Sample labels to avoid crowding (show about 10-12 labels)
        tick_step = max(1, total_positions // 10)
        tick_positions = list(range(0, total_positions, tick_step))
        if tick_positions[-1] != total_positions - 1:
            tick_positions.append(total_positions - 1)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels([hour_labels[i] for i in tick_positions], rotation=45, fontsize=8)

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved chart: {output_path}")


def create_monthly_ic_chart(ic_timeseries: pd.Series, title: str, output_path: str,
                            figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Create and save monthly aggregated IC bar chart.

    Args:
        ic_timeseries: IC time series data with datetime index
        title: Chart title
        output_path: Output file path
        figsize: Figure size (width, height)
    """
    # Ensure index is DatetimeIndex
    if not isinstance(ic_timeseries.index, pd.DatetimeIndex):
        ic_timeseries.index = pd.to_datetime(ic_timeseries.index)

    # Aggregate by month (calculate mean for each month)
    monthly_avg = ic_timeseries.groupby(pd.Grouper(freq='ME')).mean()

    # Remove NaN values (months with no data)
    monthly_avg = monthly_avg.dropna()

    if len(monthly_avg) == 0:
        print(f"No monthly data to plot for: {output_path}")
        return

    # Prepare data
    month_labels = [ts.strftime('%Y-%m') for ts in monthly_avg.index]
    values = monthly_avg.values
    positions = range(len(month_labels))

    # Create bar chart
    _, ax = plt.subplots(figsize=figsize)

    # Color bars based on positive/negative monthly average
    colors = ['#2E8B57' if v > 0 else '#DC143C' for v in values]
    bars = ax.bar(positions, values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)

    # Customize chart
    ax.set_xlabel('Month')
    ax.set_ylabel('IC Value (Monthly Average)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Set x-axis ticks
    ax.set_xticks(positions)

    # Rotate labels if there are many months
    rotation = 45 if len(month_labels) > 12 else 0
    ax.set_xticklabels(month_labels, rotation=rotation, fontsize=9)

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved monthly chart: {output_path}")


def save_detailed_results(ic_df: pd.DataFrame, rank_ic_df: pd.DataFrame,
                         statistics: Dict[str, pd.Series], output_dir: str) -> None:
    """
    Save detailed IC analysis results.

    Args:
        ic_df: IC results DataFrame
        rank_ic_df: Rank IC results DataFrame
        statistics: IC statistics
        output_dir: Output directory
    """
    print(f"\nSaving detailed results to: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save IC time series
    ic_file = os.path.join(output_dir, 'ic_timeseries.pkl')
    with open(ic_file, 'wb') as f:
        pickle.dump(ic_df, f)
    print(f"Saved IC time series: {ic_file}")

    # Save Rank IC time series
    rank_ic_file = os.path.join(output_dir, 'rank_ic_timeseries.pkl')
    with open(rank_ic_file, 'wb') as f:
        pickle.dump(rank_ic_df, f)
    print(f"Saved Rank IC time series: {rank_ic_file}")

    # Save summary statistics as JSON (only specified fields)
    summary_data = {}
    for return_col in ic_df.columns:
        hours = return_col.replace('return_', '').replace('h', '')
        summary_data[f"{hours}h"] = {
            'ic_mean': float(statistics['ic_mean'][return_col]) if not np.isnan(statistics['ic_mean'][return_col]) else None,
            'ic_std': float(statistics['ic_std'][return_col]) if not np.isnan(statistics['ic_std'][return_col]) else None,
            'icir': float(statistics['icir'][return_col]) if not np.isnan(statistics['icir'][return_col]) else None,
            'rank_ic_mean': float(statistics['rank_ic_mean'][return_col]) if not np.isnan(statistics['rank_ic_mean'][return_col]) else None,
            'rank_ic_std': float(statistics['rank_ic_std'][return_col]) if not np.isnan(statistics['rank_ic_std'][return_col]) else None,
            'rank_icir': float(statistics['rank_icir'][return_col]) if not np.isnan(statistics['rank_icir'][return_col]) else None
        }

    summary_file = os.path.join(output_dir, 'summary_statistics.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary statistics: {summary_file}")


def run_ic_analysis(factor_name: str, start_time: str, end_time: str,
                   factor_dir: str = "./factor_data",
                   returns_dir: str = "../../../crypto-data/future_returns",
                   output_base_dir: str = "./factor_report") -> None:
    """
    Run complete IC analysis for a factor.

    Args:
        factor_name: Name of the factor
        start_time: Start timestamp (format: 'YYYY-MM-DD HH:MM:SS')
        end_time: End timestamp (format: 'YYYY-MM-DD HH:MM:SS')
        factor_dir: Factor data directory
        returns_dir: Returns data directory
        output_base_dir: Base output directory
    """
    print("="*80)
    print(f"IC ANALYSIS: {factor_name}")
    print("="*80)
    print(f"Time Period: {start_time} to {end_time}")
    print(f"Factor Directory: {factor_dir}")
    print(f"Returns Directory: {returns_dir}")
    print(f"Output Directory: {output_base_dir}")
    print("="*80)

    try:
        # Load data
        factor_df, returns_df = load_data_for_period(factor_name, start_time, end_time, factor_dir, returns_dir)

        if factor_df is None or returns_df is None:
            raise ValueError("Failed to load data")

        # Calculate IC analysis
        ic_results = calculate_ic_analysis(factor_df, returns_df)
        ic_df = ic_results['ic']
        rank_ic_df = ic_results['rank_ic']

        # Calculate statistics
        statistics = calculate_ic_statistics(ic_df, rank_ic_df)

        # Create output directory structure
        start_date = pd.to_datetime(start_time).strftime('%Y%m%d')
        end_date = pd.to_datetime(end_time).strftime('%Y%m%d')
        output_dir = os.path.join(output_base_dir, factor_name, f"{start_date}_{end_date}", "ic_analysis")

        # Save detailed results
        save_detailed_results(ic_df, rank_ic_df, statistics, output_dir)

        # Generate charts for each return horizon
        print("\nGenerating IC charts...")

        # Create overall IC and Rank IC charts across all horizons
        overall_output_dir = os.path.join(output_dir, "ic_overall")
        os.makedirs(overall_output_dir, exist_ok=True)

        # IC mean chart across horizons
        ic_output_path = os.path.join(overall_output_dir, "ic.png")
        create_ic_bar_chart(
            statistics['ic_mean'],
            f'IC Analysis - {factor_name} - Mean IC Across Return Horizons',
            ic_output_path
        )

        # Rank IC mean chart across horizons
        rank_ic_output_path = os.path.join(overall_output_dir, "rank_ic.png")
        create_ic_bar_chart(
            statistics['rank_ic_mean'],
            f'Rank IC Analysis - {factor_name} - Mean Rank IC Across Return Horizons',
            rank_ic_output_path
        )

        # Generate individual horizon charts (time series for each horizon)
        return_columns = ic_df.columns
        for return_col in return_columns:
            hours = return_col.replace('return_', '').replace('h', '')

            # Create output directory for this horizon
            horizon_dir = os.path.join(output_dir, f"ic_{hours}h")
            os.makedirs(horizon_dir, exist_ok=True)

            # IC time series chart for this horizon
            ic_timeseries = ic_df[return_col].dropna()
            if len(ic_timeseries) > 0:
                plt.figure(figsize=(12, 6))

                # Prepare data for bar chart
                timestamps = ic_timeseries.index
                values = ic_timeseries.values

                # Color bars based on positive/negative values
                colors = ['#2E8B57' if v > 0 else '#DC143C' for v in values]

                # Create bar chart
                bars = plt.bar(range(len(timestamps)), values, color=colors, alpha=0.7,
                              edgecolor='black', linewidth=0.5)

                # Customize x-axis to show timestamps with intelligent sampling
                total_points = len(timestamps)

                # Determine optimal number of labels to display (max 15 labels)
                max_labels = min(15, total_points)

                if total_points <= max_labels:
                    # Show all timestamps if data points are few
                    tick_positions = range(len(timestamps))
                    tick_labels = [ts.strftime('%m-%d %H:%M') for ts in timestamps]
                else:
                    # Sample timestamps evenly
                    step = total_points // max_labels
                    tick_positions = list(range(0, total_points, step))

                    # Ensure we include the last point
                    if tick_positions[-1] != total_points - 1:
                        tick_positions.append(total_points - 1)

                    tick_labels = [timestamps[i].strftime('%m-%d %H:%M') for i in tick_positions]

                plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)

                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f'IC Time Series - {factor_name} - {hours}h Return Horizon')
                plt.xlabel('Timestamp')
                plt.ylabel('IC Value')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                ic_ts_path = os.path.join(horizon_dir, "ic_timeseries.png")
                plt.savefig(ic_ts_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Generate monthly aggregated IC chart
                ic_month_path = os.path.join(horizon_dir, "ic_month_timeseries.png")
                create_monthly_ic_chart(
                    ic_timeseries,
                    f'IC Monthly Average - {factor_name} - {hours}h Return Horizon',
                    ic_month_path
                )

            # Rank IC time series chart for this horizon
            rank_ic_timeseries = rank_ic_df[return_col].dropna()
            if len(rank_ic_timeseries) > 0:
                plt.figure(figsize=(12, 6))

                # Prepare data for bar chart
                timestamps = rank_ic_timeseries.index
                values = rank_ic_timeseries.values

                # Color bars based on positive/negative values
                colors = ['#2E8B57' if v > 0 else '#DC143C' for v in values]

                # Create bar chart
                bars = plt.bar(range(len(timestamps)), values, color=colors, alpha=0.7,
                              edgecolor='black', linewidth=0.5)

                # Customize x-axis to show timestamps with intelligent sampling
                total_points = len(timestamps)

                # Determine optimal number of labels to display (max 15 labels)
                max_labels = min(15, total_points)

                if total_points <= max_labels:
                    # Show all timestamps if data points are few
                    tick_positions = range(len(timestamps))
                    tick_labels = [ts.strftime('%m-%d %H:%M') for ts in timestamps]
                else:
                    # Sample timestamps evenly
                    step = total_points // max_labels
                    tick_positions = list(range(0, total_points, step))

                    # Ensure we include the last point
                    if tick_positions[-1] != total_points - 1:
                        tick_positions.append(total_points - 1)

                    tick_labels = [timestamps[i].strftime('%m-%d %H:%M') for i in tick_positions]

                plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)

                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f'Rank IC Time Series - {factor_name} - {hours}h Return Horizon')
                plt.xlabel('Timestamp')
                plt.ylabel('Rank IC Value')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                rank_ic_ts_path = os.path.join(horizon_dir, "rank_ic_timeseries.png")
                plt.savefig(rank_ic_ts_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Generate monthly aggregated Rank IC chart
                rank_ic_month_path = os.path.join(horizon_dir, "rank_ic_month_timeseries.png")
                create_monthly_ic_chart(
                    rank_ic_timeseries,
                    f'Rank IC Monthly Average - {factor_name} - {hours}h Return Horizon',
                    rank_ic_month_path
                )

        print("\n" + "="*80)
        print("IC ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Generated charts for {len(return_columns)} return horizons")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] IC analysis failed: {e}")
        raise


def main():
    """
    Main function for command-line usage.
    """
    # Example configuration
    factor_name = "cnn_10_03_72h_v3_wma_2h_0p5"
    start_time = "2025-04-01 00:00:00"
    end_time = "2025-08-31 23:00:00"

    print("IC Analysis Example")
    print(f"Factor: {factor_name}")
    print(f"Period: {start_time} to {end_time}")
    print()

    # Run analysis
    run_ic_analysis(factor_name, start_time, end_time)


if __name__ == "__main__":
    main()