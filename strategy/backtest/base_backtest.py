"""
Factor-Based Backtesting System

This module provides a comprehensive backtesting framework for factor-based trading strategies
with periodic rebalancing, performance analysis, and visualization.

Strategy:
- Periodic rebalancing at configurable intervals (e.g., every 5 hours)
- Long top-n symbols (highest factor values)
- Short bottom-n symbols (lowest factor values)
- Equal weighting within long and short portfolios
- 50% capital allocation to long, 50% to short

Features:
- Batch processing with parameter grid
- Detailed logging of each rebalance period
- Performance metrics (Sharpe, Calmar, Annual Return, Max Drawdown, Win Rate, P/L Ratio)
- PnL curve visualization (full period + monthly breakdown)
- Memory-efficient monthly data loading
"""

import os
import sys
import pickle
import json
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING MODULE
# ============================================================================

def load_pickle_safely(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load pickle file with Python 3.7 compatibility.

    Args:
        file_path: Path to pickle file

    Returns:
        DataFrame or None if loading fails
    """
    try:
        # Try pandas read_pickle first
        return pd.read_pickle(file_path)
    except (ValueError, ImportError) as e:
        if "pickle protocol" in str(e):
            # Fallback for protocol 5 incompatibility
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                warnings.warn(f"Failed to load {file_path}: {e2}")
                return None
        else:
            warnings.warn(f"Error loading {file_path}: {e}")
            return None
    except Exception as e:
        warnings.warn(f"Unexpected error loading {file_path}: {e}")
        return None


def load_factor_data(factor_name: str, start_date: str, end_date: str,
                     factor_base_dir: str = './factor_data') -> pd.DataFrame:
    """
    Load factor data for a date range.

    Args:
        factor_name: Name of the factor (subdirectory in factor_data)
        start_date: Start date in 'YYYY-MM' or 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM' or 'YYYY-MM-DD' format
        factor_base_dir: Base directory containing factor data

    Returns:
        DataFrame with MultiIndex (open_time, symbol) and 'factor_value' column
    """
    print(f"\n[LOAD] Loading factor data: {factor_name}")
    print(f"       Date range: {start_date} to {end_date}")

    # Parse dates to get month range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate month list
    months = pd.date_range(start=start_dt.replace(day=1),
                          end=end_dt.replace(day=1),
                          freq='MS')

    factor_dir = os.path.join(factor_base_dir, factor_name)

    if not os.path.exists(factor_dir):
        raise FileNotFoundError(f"Factor directory not found: {factor_dir}")

    # Load monthly files
    factor_dfs = []
    for month in months:
        filename = f"factor_{month.year:04d}-{month.month:02d}.pkl"
        filepath = os.path.join(factor_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Factor file not found: {filename}")
            continue

        df = load_pickle_safely(filepath)
        if df is not None:
            factor_dfs.append(df)
            print(f"       Loaded {filename}: {len(df)} rows")

    if not factor_dfs:
        raise ValueError(f"No factor data found for {factor_name}")

    # Combine all months
    combined = pd.concat(factor_dfs, axis=0)
    combined = combined.sort_index()

    # Filter to month range (not specific timestamps)
    # If start_date is 'YYYY-MM', include the entire month
    timestamps = combined.index.get_level_values('open_time')

    # Parse dates - if only YYYY-MM format, include full month
    if len(start_date) <= 7:  # YYYY-MM format
        month_start = pd.to_datetime(start_date + '-01')
        # Get last day of end month
        month_end = pd.to_datetime(end_date + '-01') + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        combined = combined[(timestamps >= month_start) & (timestamps <= month_end)]
    else:  # Full datetime format
        combined = combined[(timestamps >= start_dt) & (timestamps <= end_dt)]

    print(f"       Total: {len(combined)} rows, {len(combined.index.get_level_values('symbol').unique())} symbols")
    return combined


def load_returns_data(start_date: str, end_date: str,
                     returns_base_dir: str = '../../../crypto-data/future_returns') -> pd.DataFrame:
    """
    Load future returns data for a date range.

    Args:
        start_date: Start date in 'YYYY-MM' or 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM' or 'YYYY-MM-DD' format
        returns_base_dir: Base directory containing returns data

    Returns:
        DataFrame with MultiIndex (open_time, symbol) and return columns
    """
    print(f"\n[LOAD] Loading returns data")
    print(f"       Date range: {start_date} to {end_date}")

    # Parse dates to get month range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate month list
    months = pd.date_range(start=start_dt.replace(day=1),
                          end=end_dt.replace(day=1),
                          freq='MS')

    if not os.path.exists(returns_base_dir):
        raise FileNotFoundError(f"Returns directory not found: {returns_base_dir}")

    # Load monthly files
    returns_dfs = []
    for month in months:
        filename = f"future_returns_{month.year:04d}-{month.month:02d}.pkl"
        filepath = os.path.join(returns_base_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"Returns file not found: {filename}")
            continue

        df = load_pickle_safely(filepath)
        if df is not None:
            returns_dfs.append(df)
            print(f"       Loaded {filename}: {len(df)} rows")

    if not returns_dfs:
        raise ValueError("No returns data found")

    # Combine all months
    combined = pd.concat(returns_dfs, axis=0)
    combined = combined.sort_index()

    # Filter to month range (not specific timestamps)
    # If start_date is 'YYYY-MM', include the entire month
    timestamps = combined.index.get_level_values('open_time')

    # Parse dates - if only YYYY-MM format, include full month
    if len(start_date) <= 7:  # YYYY-MM format
        month_start = pd.to_datetime(start_date + '-01')
        # Get last day of end month
        month_end = pd.to_datetime(end_date + '-01') + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        combined = combined[(timestamps >= month_start) & (timestamps <= month_end)]
    else:  # Full datetime format
        combined = combined[(timestamps >= start_dt) & (timestamps <= end_dt)]

    print(f"       Total: {len(combined)} rows, {combined.shape[1]} return columns")
    return combined


# ============================================================================
# REBALANCING STRATEGY ENGINE
# ============================================================================

def run_single_backtest(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    rebalance_hours: int,
    top_n: int,
    bottom_n: int,
    start_time: str,
    end_time: str,
    log_file: str,
    fee_rate: float = 0.0005
) -> Tuple[pd.Series, List[Dict]]:
    """
    Run a single backtest with specified parameters.

    Args:
        factor_df: Factor data with MultiIndex (open_time, symbol)
        returns_df: Returns data with MultiIndex (open_time, symbol)
        rebalance_hours: Hours between rebalances (e.g., 5)
        top_n: Number of symbols to long
        bottom_n: Number of symbols to short
        start_time: Start timestamp 'YYYY-MM-DD HH:00:00'
        end_time: End timestamp 'YYYY-MM-DD HH:00:00'
        log_file: Path to save detailed log
        fee_rate: Transaction fee rate (default 0.2%)

    Returns:
        Tuple of (pnl_series, period_logs)
        - pnl_series: pd.Series with timestamp index and cumulative PnL values
        - period_logs: List of dictionaries with period details
    """
    print(f"\n[BACKTEST] Starting backtest")
    print(f"           Rebalance: {rebalance_hours}h, Top: {top_n}, Bottom: {bottom_n}")
    print(f"           Period: {start_time} to {end_time}")

    # Parse timestamps
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # Generate rebalance timestamps
    rebalance_timestamps = []
    current = start_dt
    while current <= end_dt:
        rebalance_timestamps.append(current)
        current += timedelta(hours=rebalance_hours)

    print(f"           Total rebalance periods: {len(rebalance_timestamps)}")

    # Check if we have the required return column
    return_col = f'return_{rebalance_hours}h'
    if return_col not in returns_df.columns:
        raise ValueError(f"Return column '{return_col}' not found in returns data. "
                        f"Available columns: {returns_df.columns.tolist()}")

    # Initialize
    pnl = 1.0  # Start with 1.0 (100%)
    pnl_history = []
    period_logs = []

    # Open log file
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("=" * 80 + "\n")
        log.write(f"BACKTEST LOG\n")
        log.write(f"Factor: {factor_df.columns[0] if len(factor_df.columns) > 0 else 'factor_value'}\n")
        log.write(f"Rebalance Period: {rebalance_hours} hours\n")
        log.write(f"Long Top: {top_n}, Short Bottom: {bottom_n}\n")
        log.write(f"Start Time: {start_time}\n")
        log.write(f"End Time: {end_time}\n")
        log.write(f"Fee Rate: {fee_rate * 100:.2f}%\n")
        log.write("=" * 80 + "\n\n")

        # Process each rebalance period
        for period_idx, timestamp in enumerate(rebalance_timestamps, 1):
            log.write(f"\nRebalance #{period_idx}: {timestamp}\n")
            log.write("-" * 80 + "\n")

            try:
                # Get factor values at this timestamp
                if timestamp not in factor_df.index.get_level_values('open_time'):
                    log.write(f"[SKIP] No factor data at {timestamp}, skipping period\n")
                    continue

                factor_values = factor_df.xs(timestamp, level='open_time')

                # Get returns at this timestamp
                if timestamp not in returns_df.index.get_level_values('open_time'):
                    log.write(f"[SKIP] No returns data at {timestamp}, skipping period\n")
                    continue

                returns_values = returns_df.xs(timestamp, level='open_time')[return_col]

                # Merge factor and returns
                merged = pd.DataFrame({
                    'factor': factor_values.iloc[:, 0] if isinstance(factor_values, pd.DataFrame) else factor_values,
                    'return': returns_values
                })

                # Remove NaN values
                merged = merged.dropna()


                # Sort by factor value (descending - higher is better)
                merged_sorted = merged.sort_values('factor', ascending=False)

                # Select top-n for long and bottom-n for short
                actual_top_n = min(top_n, len(merged_sorted))
                actual_bottom_n = min(bottom_n, len(merged_sorted))

                if actual_top_n == 0 or actual_bottom_n == 0:
                    log.write(f"[SKIP] Insufficient symbols (have {len(merged_sorted)}), skipping period\n")
                    continue

                long_positions = merged_sorted.head(actual_top_n)
                short_positions = merged_sorted.tail(actual_bottom_n)

                # Calculate returns
                long_return_mean = long_positions['return'].mean()
                short_return_mean = short_positions['return'].mean()

                # Period return: (long_return - short_return) / 2 - fee
                period_return = (long_return_mean - short_return_mean) / 2 - fee_rate
                # period_return = (-long_return_mean + short_return_mean) / 2 - fee_rate

                # Update PnL
                pnl *= (1 + period_return)

                # Log details
                log.write(f"\nLong Positions (Top {actual_top_n}):\n")
                for symbol, row in long_positions.iterrows():
                    log.write(f"  {symbol:12s}: factor={row['factor']:8.4f}, "
                            f"return={row['return']:7.4f} ({row['return']*100:6.2f}%)\n")

                log.write(f"\nShort Positions (Bottom {actual_bottom_n}):\n")
                for symbol, row in short_positions.iterrows():
                    log.write(f"  {symbol:12s}: factor={row['factor']:8.4f}, "
                            f"return={row['return']:7.4f} ({row['return']*100:6.2f}%)\n")

                log.write(f"\nLong Mean Return:  {long_return_mean:7.4f} ({long_return_mean*100:6.2f}%)\n")
                log.write(f"Short Mean Return: {short_return_mean:7.4f} ({short_return_mean*100:6.2f}%)\n")
                log.write(f"Period Return:     {period_return:7.4f} ({period_return*100:6.2f}%) [after {fee_rate*100:.2f}% fee]\n")
                log.write(f"Current PnL:       {pnl:.6f}\n")

                # Store history
                pnl_history.append({'timestamp': timestamp, 'pnl': pnl})

                # Store period log
                period_logs.append({
                    'period': period_idx,
                    'timestamp': timestamp,
                    'long_symbols': long_positions.index.tolist(),
                    'short_symbols': short_positions.index.tolist(),
                    'long_return_mean': long_return_mean,
                    'short_return_mean': short_return_mean,
                    'period_return': period_return,
                    'pnl': pnl
                })

            except Exception as e:
                log.write(f"L Error at {timestamp}: {e}\n")
                import traceback
                log.write(traceback.format_exc())
                continue

        # Final summary
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"BACKTEST COMPLETE\n")
        log.write(f"Total Periods Executed: {len(pnl_history)}\n")
        log.write(f"Final PnL: {pnl:.6f}\n")
        log.write(f"Total Return: {(pnl - 1) * 100:.2f}%\n")
        log.write("=" * 80 + "\n")

    # Convert to Series
    if pnl_history:
        pnl_series = pd.Series(
            [p['pnl'] for p in pnl_history],
            index=[p['timestamp'] for p in pnl_history],
            name='pnl'
        )
    else:
        pnl_series = pd.Series(dtype=float)

    print(f"           Completed: {len(pnl_history)} periods, Final PnL: {pnl:.6f}")

    return pnl_series, period_logs


# ============================================================================
# PERFORMANCE METRICS MODULE
# ============================================================================

def calculate_metrics(pnl_series: pd.Series, rebalance_hours: int) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        pnl_series: Series with timestamp index and PnL values
        rebalance_hours: Hours between rebalances (for annualization)

    Returns:
        Dictionary with performance metrics
    """
    if len(pnl_series) < 2:
        return {
            'error': 'Insufficient data for metrics calculation',
            'num_periods': len(pnl_series)
        }

    # Calculate returns
    returns = pnl_series.pct_change().dropna()

    if len(returns) == 0:
        return {
            'error': 'No valid returns for metrics calculation',
            'num_periods': len(pnl_series)
        }

    # Periods per year (assuming continuous hourly trading)
    periods_per_year = (365 * 24) / rebalance_hours

    # Basic statistics
    total_return = (pnl_series.iloc[-1] / pnl_series.iloc[0]) - 1

    # Time span
    time_span_hours = (pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 3600
    time_span_years = time_span_hours / (365 * 24)

    # Annual return
    if time_span_years > 0:
        annual_return = (1 + total_return) ** (1 / time_span_years) - 1
    else:
        annual_return = 0

    # Sharpe ratio
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return > 0:
        sharpe_ratio = mean_return / std_return * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0

    # Max drawdown
    cumulative = pnl_series
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    if abs(max_drawdown) > 0:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0

    # Win rate
    winning_periods = (returns > 0).sum()
    total_periods = len(returns)
    win_rate = winning_periods / total_periods if total_periods > 0 else 0

    # Profit/Loss ratio
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]

    if len(winning_returns) > 0 and len(losing_returns) > 0:
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        profit_loss_ratio = 0

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'num_periods': total_periods,
        'num_winning_periods': winning_periods,
        'num_losing_periods': total_periods - winning_periods,
        'time_span_hours': time_span_hours,
        'time_span_days': time_span_hours / 24,
        'rebalance_hours': rebalance_hours
    }

    return metrics


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def plot_pnl_curve(
    pnl_series: pd.Series,
    output_file: str,
    title: str = 'PnL Curve',
    show_drawdown: bool = True
) -> None:
    """
    Plot PnL curve with optional drawdown.

    Args:
        pnl_series: Series with timestamp index and PnL values
        output_file: Path to save the plot
        title: Plot title
        show_drawdown: Whether to show drawdown subplot
    """
    if len(pnl_series) < 2:
        print(f"[WARN] Insufficient data for plotting: {len(pnl_series)} points")
        return

    # Create figure
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    # Plot PnL curve
    ax1.plot(pnl_series.index, pnl_series.values, linewidth=2, color='#2E86AB', label='PnL')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial')
    ax1.fill_between(pnl_series.index, 1.0, pnl_series.values,
                     where=(pnl_series.values >= 1.0), alpha=0.3, color='green',
                     label='Profit', interpolate=True)
    ax1.fill_between(pnl_series.index, 1.0, pnl_series.values,
                     where=(pnl_series.values < 1.0), alpha=0.3, color='red',
                     label='Loss', interpolate=True)

    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('PnL (Cumulative)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add statistics text
    total_return = (pnl_series.iloc[-1] / pnl_series.iloc[0] - 1) * 100
    stats_text = f'Total Return: {total_return:.2f}%\nFinal PnL: {pnl_series.iloc[-1]:.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot drawdown if requested
    if show_drawdown:
        running_max = pnl_series.expanding().max()
        drawdown = (pnl_series - running_max) / running_max * 100

        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.5)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        max_dd = drawdown.min()
        ax2.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2f}%',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"       Saved plot: {output_file}")


def plot_monthly_pnl_curves(pnl_series: pd.Series, output_dir: str) -> None:
    """
    Plot separate PnL curves for each month.

    Args:
        pnl_series: Series with timestamp index and PnL values
        output_dir: Directory to save monthly plots
    """
    if len(pnl_series) < 2:
        print(f"[WARN] Insufficient data for monthly plotting: {len(pnl_series)} points")
        return

    # Group by month
    pnl_df = pnl_series.to_frame('pnl')
    pnl_df['year_month'] = pnl_df.index.to_period('M')

    months = pnl_df['year_month'].unique()

    print(f"\n[PLOT] Generating monthly PnL curves: {len(months)} months")

    for month in months:
        month_data = pnl_df[pnl_df['year_month'] == month]['pnl']

        if len(month_data) < 2:
            continue

        # Normalize to start at previous month's ending value or 1.0
        month_str = str(month)
        output_file = os.path.join(output_dir, f"pnl_curve_{month_str}.png")

        title = f"PnL Curve - {month_str}"
        plot_pnl_curve(month_data, output_file, title=title, show_drawdown=False)


# ============================================================================
# BATCH PROCESSING MODULE
# ============================================================================

def batch_backtest(
    factor_name: str,
    start_time: str,
    end_time: str,
    rebalance_hours_list: List[int] = [6, 12, 24],
    top_bottom_n_list: List[List[int]] = [[5, 5], [10, 10], [15, 15]],
    output_base_dir: str = './factor_report',
    factor_base_dir: str = './factor_data',
    returns_base_dir: str = '../../../crypto-data/future_returns'
) -> Dict:
    """
    Run batch backtesting with multiple parameter combinations.

    Args:
        factor_name: Name of factor (subdirectory in factor_data)
        start_time: Start timestamp 'YYYY-MM-DD HH:00:00'
        end_time: End timestamp 'YYYY-MM-DD HH:00:00'
        rebalance_hours_list: List of rebalance periods to test
        top_bottom_n_list: List of [top_n, bottom_n] pairs to test (e.g., [[5, 5], [10, 10]])
        output_base_dir: Base directory for output
        factor_base_dir: Base directory for factor data
        returns_base_dir: Base directory for returns data

    Returns:
        Dictionary with batch processing results
    """
    print("\n" + "=" * 80)
    print(f"BATCH BACKTEST")
    print("=" * 80)
    print(f"Factor: {factor_name}")
    print(f"Period: {start_time} to {end_time}")
    print(f"Rebalance Hours: {rebalance_hours_list}")
    print(f"Top-Bottom-N Pairs: {top_bottom_n_list}")
    print("=" * 80)

    # Load data once
    start_date = start_time.split()[0][:7]  # YYYY-MM
    end_date = end_time.split()[0][:7]

    factor_df = load_factor_data(factor_name, start_date, end_date, factor_base_dir)
    returns_df = load_returns_data(start_date, end_date, returns_base_dir)

    # Create base output directory
    date_range_str = f"{start_time.split()[0].replace('-', '')}_{end_time.split()[0].replace('-', '')}"
    base_output_dir = os.path.join(output_base_dir, factor_name, date_range_str, "base_backtest")
    os.makedirs(base_output_dir, exist_ok=True)

    # Track results
    results = {
        'factor_name': factor_name,
        'start_time': start_time,
        'end_time': end_time,
        'backtests': []
    }

    # Iterate over parameter combinations
    total_combinations = len(rebalance_hours_list) * len(top_bottom_n_list)
    combination_idx = 0

    for rebalance_hours in rebalance_hours_list:
        for top_n, bottom_n in top_bottom_n_list:
            combination_idx += 1

            print(f"\n[{combination_idx}/{total_combinations}] Testing: "
                  f"{rebalance_hours}h_top{top_n}_bottom{bottom_n}")

            # Create output directory for this combination
            combo_name = f"{rebalance_hours}h_top{top_n}_bottom{bottom_n}"
            combo_dir = os.path.join(base_output_dir, combo_name)
            os.makedirs(combo_dir, exist_ok=True)

            try:
                # Run backtest
                log_file = os.path.join(combo_dir, 'backtest_log.txt')
                pnl_series, period_logs = run_single_backtest(
                    factor_df=factor_df,
                    returns_df=returns_df,
                    rebalance_hours=rebalance_hours,
                    top_n=top_n,
                    bottom_n=bottom_n,
                    start_time=start_time,
                    end_time=end_time,
                    log_file=log_file
                )

                if len(pnl_series) == 0:
                    print(f"       [WARN] No valid backtest results")
                    continue

                # Save PnL series
                pnl_file = os.path.join(combo_dir, 'pnl_series.pkl')
                pnl_series.to_pickle(pnl_file)
                print(f"       Saved PnL series: {pnl_file}")

                # Calculate metrics
                print(f"       Calculating metrics...")
                metrics = calculate_metrics(pnl_series, rebalance_hours)

                # Save metrics
                metrics_file = os.path.join(combo_dir, 'metrics.json')
                with open(metrics_file, 'w') as f:
                    # Convert to JSON-serializable format
                    metrics_json = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                   for k, v in metrics.items()}
                    json.dump(metrics_json, f, indent=2)
                print(f"       Saved metrics: {metrics_file}")

                # Print key metrics
                if 'error' not in metrics:
                    print(f"       Total Return: {metrics['total_return']*100:.2f}%")
                    print(f"       Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    print(f"       Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                    print(f"       Win Rate: {metrics['win_rate']*100:.2f}%")

                # Plot full period PnL curve
                print(f"       Plotting PnL curves...")
                pnl_curve_file = os.path.join(combo_dir, 'pnl_curve_full.png')
                plot_pnl_curve(pnl_series, pnl_curve_file,
                             title=f'PnL Curve - {combo_name}',
                             show_drawdown=True)

                # Plot monthly PnL curves
                plot_monthly_pnl_curves(pnl_series, combo_dir)

                # Store result
                results['backtests'].append({
                    'rebalance_hours': rebalance_hours,
                    'top_n': top_n,
                    'bottom_n': bottom_n,
                    'output_dir': combo_dir,
                    'metrics': metrics,
                    'num_periods': len(pnl_series)
                })

                print(f"        Completed: {combo_name}")

            except Exception as e:
                print(f"       L Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save batch summary
    summary_file = os.path.join(base_output_dir, 'batch_summary.json')

    # Check if summary file already exists
    if os.path.exists(summary_file):
        # Load existing data
        with open(summary_file, 'r') as f:
            existing_data = json.load(f)

        # Prepare new backtests to append
        new_backtests = []
        for bt in results['backtests']:
            bt_json = {
                'rebalance_hours': bt['rebalance_hours'],
                'top_n': bt['top_n'],
                'bottom_n': bt['bottom_n'],
                'num_periods': bt['num_periods'],
                'output_dir': bt['output_dir']
            }
            # Add metrics if no error
            if 'error' not in bt['metrics']:
                bt_json['total_return'] = float(bt['metrics']['total_return'])
                bt_json['sharpe_ratio'] = float(bt['metrics']['sharpe_ratio'])
                bt_json['max_drawdown'] = float(bt['metrics']['max_drawdown'])
                bt_json['win_rate'] = float(bt['metrics']['win_rate'])

            new_backtests.append(bt_json)

        # Append new backtests to existing list
        existing_data['backtests'].extend(new_backtests)
        existing_data['total_backtests'] = len(existing_data['backtests'])

        results_json = existing_data
        print(f"\n[INFO] Appended {len(new_backtests)} new backtests to existing summary")
    else:
        # Create new summary file
        results_json = {
            'factor_name': results['factor_name'],
            'start_time': results['start_time'],
            'end_time': results['end_time'],
            'total_backtests': len(results['backtests']),
            'backtests': []
        }

        for bt in results['backtests']:
            bt_json = {
                'rebalance_hours': bt['rebalance_hours'],
                'top_n': bt['top_n'],
                'bottom_n': bt['bottom_n'],
                'num_periods': bt['num_periods'],
                'output_dir': bt['output_dir']
            }
            # Add metrics if no error
            if 'error' not in bt['metrics']:
                bt_json['total_return'] = float(bt['metrics']['total_return'])
                bt_json['sharpe_ratio'] = float(bt['metrics']['sharpe_ratio'])
                bt_json['max_drawdown'] = float(bt['metrics']['max_drawdown'])
                bt_json['win_rate'] = float(bt['metrics']['win_rate'])

            results_json['backtests'].append(bt_json)

        print(f"\n[INFO] Created new summary file with {len(results_json['backtests'])} backtests")

    # Write to file
    with open(summary_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'='*80}")
    print(f"BATCH BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"Total backtests: {len(results['backtests'])}")
    print(f"Output directory: {base_output_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}\n")

    return results

if __name__ == "__main__":
    # Example usage
    results = batch_backtest(
        factor_name='cnn_10_03_72h_v3_wma_2h_0p5',
        start_time='2025-04-02 00:00:00',
        end_time='2025-08-31 23:00:00',
        rebalance_hours_list=[16],
        top_bottom_n_list=[[i, i] for i in [2,4,6,8,10,12,16,20]],
        output_base_dir='./factor_report'
    )

    print(f"\nProcessed {len(results['backtests'])} parameter combinations")