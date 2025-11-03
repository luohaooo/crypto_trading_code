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


def load_ohlc_data(start_date: str, end_date: str,
                   ohlc_base_dir: str = '/home/craz/crypto/crypto-data/filter_hour_cache') -> pd.DataFrame:
    """
    Load hourly OHLC data for a date range.

    Args:
        start_date: Start date in 'YYYY-MM' or 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM' or 'YYYY-MM-DD' format
        ohlc_base_dir: Base directory containing hourly OHLC pickle files

    Returns:
        DataFrame with MultiIndex (open_time, symbol) and OHLC columns (open, high, low, close, volume)
    """
    print(f"\n[LOAD] Loading OHLC data")
    print(f"       Date range: {start_date} to {end_date}")

    # Parse dates to get month range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate month list
    months = pd.date_range(start=start_dt.replace(day=1),
                          end=end_dt.replace(day=1),
                          freq='MS')

    if not os.path.exists(ohlc_base_dir):
        raise FileNotFoundError(f"OHLC directory not found: {ohlc_base_dir}")

    # Load monthly files
    ohlc_dfs = []
    for month in months:
        filename = f"usdt_data_{month.year:04d}-{month.month:02d}.pkl"
        filepath = os.path.join(ohlc_base_dir, filename)

        if not os.path.exists(filepath):
            warnings.warn(f"OHLC file not found: {filename}")
            continue

        df = load_pickle_safely(filepath)
        if df is not None:
            ohlc_dfs.append(df)
            print(f"       Loaded {filename}: {len(df)} rows")

    if not ohlc_dfs:
        raise ValueError("No OHLC data found")

    # Combine all months
    combined = pd.concat(ohlc_dfs, axis=0)
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

def calculate_return_with_stop_loss(
    symbol: str,
    direction: str,
    entry_price: float,
    period_ohlc: pd.DataFrame,
    exit_price: float,
    stop_profit_pct: float,
    stop_loss_pct: float,
) -> Tuple[float, str, float]:
    """
    Calculate return with stop-loss and take-profit logic.

    Logic:
    - Priority: Stop-loss is checked FIRST (conservative strategy)
    - For Long: Stop-loss checks low, take-profit checks high
    - For Short: Stop-loss checks high, take-profit checks low
    - Uses boundary prices (stop-loss price or take-profit price)

    Args:
        symbol: Symbol name
        direction: 'long' or 'short'
        entry_price: Entry price (open price at entry timestamp)
        period_ohlc: OHLC data during rebalance period (hourly)
        exit_price: Default exit price (open price at next rebalance timestamp)
        stop_profit_pct: Take-profit percentage (0 = disabled)
        stop_loss_pct: Stop-loss percentage (0 = disabled)

    Returns:
        Tuple of (return, trigger_type, actual_exit_price)
        - return: Return percentage
        - trigger_type: 'stop_loss', 'take_profit', or 'normal'
        - actual_exit_price: Actual exit price used for calculation
    """

    if direction == 'long':
        # Calculate boundary prices for long position
        if stop_profit_pct > 0:
            take_profit_price = entry_price * (1 + stop_profit_pct)
        else:
            take_profit_price = float('inf')

        if stop_loss_pct > 0:
            stop_loss_price = entry_price * (1 - stop_loss_pct)
        else:
            stop_loss_price = 0

        # Check each hour in chronological order
        for timestamp, row in period_ohlc.iterrows():
            # Priority 1: Check stop-loss (conservative)
            if stop_loss_pct > 0 and row['low'] <= stop_loss_price:
                # Stop-loss triggered
                actual_exit_price = stop_loss_price
                ret = (actual_exit_price - entry_price) / entry_price
                return ret, 'stop_loss', actual_exit_price

            # Priority 2: Check take-profit
            if stop_profit_pct > 0 and row['high'] >= take_profit_price:
                # Take-profit triggered
                actual_exit_price = take_profit_price
                ret = (actual_exit_price - entry_price) / entry_price
                return ret, 'take_profit', actual_exit_price

        # Not triggered, use default exit price
        ret = (exit_price - entry_price) / entry_price
        return ret, 'normal', exit_price

    else:  # short
        # Calculate boundary prices for short position
        if stop_profit_pct > 0:
            take_profit_price = entry_price * (1 - stop_profit_pct)
        else:
            take_profit_price = 0

        if stop_loss_pct > 0:
            stop_loss_price = entry_price * (1 + stop_loss_pct)
        else:
            stop_loss_price = float('inf')

        # Check each hour in chronological order
        for timestamp, row in period_ohlc.iterrows():
            # Priority 1: Check stop-loss (conservative)
            if stop_loss_pct > 0 and row['high'] >= stop_loss_price:
                # Stop-loss triggered
                actual_exit_price = stop_loss_price
                ret = (entry_price - actual_exit_price) / entry_price
                return ret, 'stop_loss', actual_exit_price

            # Priority 2: Check take-profit
            if stop_profit_pct > 0 and row['low'] <= take_profit_price:
                # Take-profit triggered
                actual_exit_price = take_profit_price
                ret = (entry_price - actual_exit_price) / entry_price
                return ret, 'take_profit', actual_exit_price

        # Not triggered, use default exit price
        ret = (entry_price - exit_price) / entry_price
        return ret, 'normal', exit_price


def run_single_backtest(
    factor_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    rebalance_hours: int,
    top_n: int,
    bottom_n: int,
    start_time: str,
    end_time: str,
    log_file: str,
    fee_rate: float = 0.0005,
    stop_profit_pct: float = 0.0,
    stop_loss_pct: float = 0.0,
    leverage: float = 1.0
) -> Tuple[pd.Series, List[Dict]]:
    """
    Run a single backtest with specified parameters, supporting stop-loss and take-profit.

    Args:
        factor_df: Factor data with MultiIndex (open_time, symbol)
        ohlc_df: Hourly OHLC data with MultiIndex (open_time, symbol)
        rebalance_hours: Hours between rebalances (e.g., 5)
        top_n: Number of symbols to long
        bottom_n: Number of symbols to short
        start_time: Start timestamp 'YYYY-MM-DD HH:00:00'
        end_time: End timestamp 'YYYY-MM-DD HH:00:00'
        log_file: Path to save detailed log
        fee_rate: Transaction fee rate (default 0.2%)
        stop_profit_pct: Take-profit percentage (0 = disabled, e.g., 0.05 = 5%)
        stop_loss_pct: Stop-loss percentage (0 = disabled, e.g., 0.03 = 3%)
        leverage: Leverage multiplier (default 1.0, e.g., 2.0 = 2x leverage)

    Returns:
        Tuple of (pnl_series, period_logs)
        - pnl_series: pd.Series with timestamp index and cumulative PnL values
        - period_logs: List of dictionaries with period details
    """
    print(f"\n[BACKTEST] Starting backtest")
    print(f"           Rebalance: {rebalance_hours}h, Top: {top_n}, Bottom: {bottom_n}")
    print(f"           Period: {start_time} to {end_time}")
    print(f"           Stop Profit: {stop_profit_pct*100:.2f}%, Stop Loss: {stop_loss_pct*100:.2f}%")
    print(f"           Leverage: {leverage}x")

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
        log.write(f"Stop Profit: {stop_profit_pct * 100:.2f}%\n")
        log.write(f"Stop Loss: {stop_loss_pct * 100:.2f}%\n")
        log.write(f"Leverage: {leverage}x\n")
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

                # Check OHLC data availability at this timestamp
                if timestamp not in ohlc_df.index.get_level_values('open_time'):
                    log.write(f"[SKIP] No OHLC data at {timestamp}, skipping period\n")
                    continue

                # Get next rebalance timestamp for exit
                next_timestamp = timestamp + timedelta(hours=rebalance_hours)
                if next_timestamp not in ohlc_df.index.get_level_values('open_time'):
                    log.write(f"[SKIP] No OHLC data at next timestamp {next_timestamp}, skipping period\n")
                    continue

                # Get available symbols at current timestamp
                available_symbols = ohlc_df.xs(timestamp, level='open_time').index.tolist()

                # Filter factor values to only include symbols with OHLC data
                factor_values_filtered = factor_values[factor_values.index.isin(available_symbols)]

                if len(factor_values_filtered) == 0:
                    log.write(f"[SKIP] No valid symbols with both factor and OHLC data at {timestamp}, skipping period\n")
                    continue

                # Convert to DataFrame for easier handling
                merged = pd.DataFrame({
                    'factor': factor_values_filtered.iloc[:, 0] if isinstance(factor_values_filtered, pd.DataFrame) else factor_values_filtered
                })

                # Remove NaN values
                merged = merged.dropna()

                # # Filter out symbols with factor value equal to 0
                # original_count = len(merged)
                # merged = merged[merged['factor'] > 0.00001]
                # filtered_count = original_count - len(merged)

                # if filtered_count > 0:
                #      log.write(f"[FILTER] Removed {filtered_count} symbols with factor value = 0\n")

                # if len(merged) == 0:
                #     log.write(f"[SKIP] No valid data at {timestamp}, skipping period\n")
                #     continue

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

                # Calculate returns with stop-loss and take-profit logic
                long_returns = []
                short_returns = []
                long_triggers = []
                short_triggers = []
                long_entry_prices = []
                long_exit_prices = []
                short_entry_prices = []
                short_exit_prices = []

                # Process long positions
                for symbol in long_positions.index:
                    try:
                        # Get entry price (open at current timestamp)
                        entry_price = ohlc_df.loc[(timestamp, symbol), 'open']

                        # Get exit price (open at next timestamp)
                        exit_price = ohlc_df.loc[(next_timestamp, symbol), 'open']

                        # Get OHLC data during rebalance period
                        period_ohlc = ohlc_df.loc[
                            (ohlc_df.index.get_level_values('open_time') >= timestamp) &
                            (ohlc_df.index.get_level_values('open_time') < next_timestamp) &
                            (ohlc_df.index.get_level_values('symbol') == symbol)
                        ]

                        if len(period_ohlc) == 0:
                            log.write(f"[WARN] No period OHLC data for {symbol}, skipping\n")
                            continue

                        # Calculate return with stop-loss and take-profit
                        ret, trigger, actual_exit = calculate_return_with_stop_loss(
                            symbol, 'long', entry_price, period_ohlc,
                            exit_price, stop_profit_pct, stop_loss_pct
                        )

                        long_returns.append(ret)
                        long_triggers.append((symbol, trigger))
                        long_entry_prices.append(entry_price)
                        long_exit_prices.append(actual_exit)

                    except KeyError as e:
                        log.write(f"[WARN] Missing data for long symbol {symbol}: {e}\n")
                        continue

                # Process short positions
                for symbol in short_positions.index:
                    try:
                        # Get entry price (open at current timestamp)
                        entry_price = ohlc_df.loc[(timestamp, symbol), 'open']

                        # Get exit price (open at next timestamp)
                        exit_price = ohlc_df.loc[(next_timestamp, symbol), 'open']

                        # Get OHLC data during rebalance period
                        period_ohlc = ohlc_df.loc[
                            (ohlc_df.index.get_level_values('open_time') >= timestamp) &
                            (ohlc_df.index.get_level_values('open_time') < next_timestamp) &
                            (ohlc_df.index.get_level_values('symbol') == symbol)
                        ]

                        if len(period_ohlc) == 0:
                            log.write(f"[WARN] No period OHLC data for {symbol}, skipping\n")
                            continue

                        # Calculate return with stop-loss and take-profit
                        ret, trigger, actual_exit = calculate_return_with_stop_loss(
                            symbol, 'short', entry_price, period_ohlc,
                            exit_price, stop_profit_pct, stop_loss_pct
                        )

                        short_returns.append(ret)
                        short_triggers.append((symbol, trigger))
                        short_entry_prices.append(entry_price)
                        short_exit_prices.append(actual_exit)

                    except KeyError as e:
                        log.write(f"[WARN] Missing data for short symbol {symbol}: {e}\n")
                        continue

                # Check if we have valid returns
                if len(long_returns) == 0 or len(short_returns) == 0:
                    log.write(f"[SKIP] Insufficient valid returns (long: {len(long_returns)}, short: {len(short_returns)}), skipping period\n")
                    continue

                # Calculate mean returns
                long_return_mean = np.mean(long_returns)
                short_return_mean = np.mean(short_returns)

                # Period return
                period_return = (long_return_mean + short_return_mean) / 2 - fee_rate - (1 + (long_return_mean + short_return_mean) / 2) * fee_rate
                # period_return = (-long_return_mean + short_return_mean) / 2 - fee_rate

                # Update PnL with leverage
                pnl *= (1 + period_return * leverage)

                # Log details
                log.write(f"\nLong Positions (Top {actual_top_n}):\n")
                for i, symbol in enumerate(long_positions.index):
                    if i < len(long_returns):
                        factor_val = long_positions.loc[symbol, 'factor']
                        ret = long_returns[i]
                        trigger = long_triggers[i][1] if i < len(long_triggers) else 'unknown'
                        entry_p = long_entry_prices[i] if i < len(long_entry_prices) else 0.0
                        exit_p = long_exit_prices[i] if i < len(long_exit_prices) else 0.0
                        log.write(f"  {symbol:12s}: factor={factor_val:8.4f}, "
                                f"entry={entry_p:16.6f}, exit={exit_p:16.6f}, "
                                f"return={ret:7.4f} ({ret*100:6.2f}%), trigger={trigger}\n")

                log.write(f"\nShort Positions (Bottom {actual_bottom_n}):\n")
                for i, symbol in enumerate(short_positions.index):
                    if i < len(short_returns):
                        factor_val = short_positions.loc[symbol, 'factor']
                        ret = short_returns[i]
                        trigger = short_triggers[i][1] if i < len(short_triggers) else 'unknown'
                        entry_p = short_entry_prices[i] if i < len(short_entry_prices) else 0.0
                        exit_p = short_exit_prices[i] if i < len(short_exit_prices) else 0.0
                        log.write(f"  {symbol:12s}: factor={factor_val:8.4f}, "
                                f"entry={entry_p:16.6f}, exit={exit_p:16.6f}, "
                                f"return={ret:7.4f} ({ret*100:6.2f}%), trigger={trigger}\n")

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
    ohlc_base_dir: str = '/home/craz/crypto/crypto-data/filter_hour_cache',
    stop_profit_pct: float = 0.0,
    stop_loss_pct: float = 0.0,
    leverage: float = 1.0
) -> Dict:
    """
    Run batch backtesting with multiple parameter combinations, supporting stop-loss and take-profit.

    Args:
        factor_name: Name of factor (subdirectory in factor_data)
        start_time: Start timestamp 'YYYY-MM-DD HH:00:00'
        end_time: End timestamp 'YYYY-MM-DD HH:00:00'
        rebalance_hours_list: List of rebalance periods to test
        top_bottom_n_list: List of [top_n, bottom_n] pairs to test (e.g., [[5, 5], [10, 10]])
        output_base_dir: Base directory for output
        factor_base_dir: Base directory for factor data
        ohlc_base_dir: Base directory for OHLC data
        stop_profit_pct: Take-profit percentage (0 = disabled, e.g., 0.05 = 5%)
        stop_loss_pct: Stop-loss percentage (0 = disabled, e.g., 0.03 = 3%)
        leverage: Leverage multiplier (default 1.0, e.g., 2.0 = 2x leverage)

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
    print(f"Stop Profit: {stop_profit_pct*100:.2f}%, Stop Loss: {stop_loss_pct*100:.2f}%")
    print(f"Leverage: {leverage}x")
    print("=" * 80)

    # Load data once
    start_date = start_time.split()[0][:7]  # YYYY-MM
    end_date = end_time.split()[0][:7]

    factor_df = load_factor_data(factor_name, start_date, end_date, factor_base_dir)
    ohlc_df = load_ohlc_data(start_date, end_date, ohlc_base_dir)

    # Create base output directory with stop-profit/stop-loss info (hour precision)
    start_dt_full = pd.to_datetime(start_time)
    end_dt_full = pd.to_datetime(end_time)
    date_range_str = (
        f"{start_dt_full.strftime('%Y%m%d_%H%M')}_{end_dt_full.strftime('%Y%m%d_%H%M')}"
    )

    # Generate directory name with leverage, stop-profit, and stop-loss info
    if stop_profit_pct == 0.0 and stop_loss_pct == 0.0:
        stopprofit_dir = "backtest_no_stopprofit"
    else:
        profit_pct = int(stop_profit_pct * 100)
        loss_pct = int(stop_loss_pct * 100)
        stopprofit_dir = f"backtest_profit_{profit_pct}_stop_{loss_pct}"

    # Add leverage prefix
    lev_str = f"lev_{int(leverage)}" if leverage == int(leverage) else f"lev_{leverage}"
    stopprofit_dir = f"{lev_str}_{stopprofit_dir}"

    base_output_dir = os.path.join(
        output_base_dir,
        factor_name,
        "adjust_backtest",
        date_range_str,
        stopprofit_dir
    )
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
                    ohlc_df=ohlc_df,
                    rebalance_hours=rebalance_hours,
                    top_n=top_n,
                    bottom_n=bottom_n,
                    start_time=start_time,
                    end_time=end_time,
                    log_file=log_file,
                    stop_profit_pct=stop_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    leverage=leverage
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
    # Example usage with stop-loss, take-profit, and leverage
    for lev in [3]:
        for profit in [0,0.2,0.4]:
            for s in [f"2025-04-02 {hour:02d}:00:00" for hour in range(0, 8)]:
                results = batch_backtest(
                    factor_name='cnn_10_03_72h_v4_wma_10h_0.4',
                    start_time=s,
                    end_time='2025-08-31 23:00:00',
                    rebalance_hours_list=[16],
                    top_bottom_n_list=[[i, i] for i in [4]],
                    output_base_dir='./factor_report',
                    stop_profit_pct=profit,
                    stop_loss_pct=0.4,
                    leverage=lev
                )

    # print(f"\nProcessed {len(results['backtests'])} parameter combinations")
