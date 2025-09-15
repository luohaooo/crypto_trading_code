"""
OHLC Data Preprocessing Module

This module provides functions to preprocess OHLC (Open, High, Low, Close) data
from 1-minute intervals to various timeframes like 3min, 5min, 15min, etc.

The module is designed to work with data from load_usdt_symbols_from_month_cache()
and other similar data loading functions that return MultiIndex DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings

def aggregate_ohlc_data(
    data: pd.DataFrame,
    timeframe: str,
    symbols: Optional[List[str]] = None,
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None,
    volume_aggregation: str = 'sum'
) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLC data to specified timeframes.
    
    Args:
        data (pd.DataFrame): Input DataFrame with MultiIndex (open_time, symbol)
                            Columns: ['open', 'high', 'low', 'close', 'volume']
        timeframe (str): Target timeframe. Examples: '3min', '5min', '15min', '30min', '1h', '4h', '1d'
        symbols (List[str], optional): List of symbols to process. If None, processes all symbols.
        start_time (str/datetime, optional): Start time for filtering data
        end_time (str/datetime, optional): End time for filtering data
        volume_aggregation (str): How to aggregate volume ('sum' or 'mean'). Default: 'sum'
    
    Returns:
        pd.DataFrame: Aggregated OHLC data with same MultiIndex structure
    
    Example:
        # Load 1-minute data
        data_1min = load_usdt_symbols_from_month_cache(
            symbols=['BTCUSDT', 'ETHUSDT'],
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        # Aggregate to 5-minute data
        data_5min = aggregate_ohlc_data(data_1min, '5min')
        
        # Aggregate to 15-minute data for specific symbols
        data_15min = aggregate_ohlc_data(
            data_1min, 
            '15min', 
            symbols=['BTCUSDT'],
            start_time='2020-01-01 12:00:00'
        )
    """
    
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Validate input data structure
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("Data must have MultiIndex with levels (open_time, symbol)")
    
    if data.index.names != ['open_time', 'symbol']:
        raise ValueError("MultiIndex must have levels named ['open_time', 'symbol']")
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # print(f"Aggregating data from 1min to {timeframe}...")
    
    # Filter symbols if specified
    if symbols:
        available_symbols = data.index.get_level_values('symbol').unique().tolist()
        valid_symbols = [s for s in symbols if s in available_symbols]
        missing_symbols = [s for s in symbols if s not in available_symbols]
        
        if missing_symbols:
            print(f"Warning: Symbols not found in data: {missing_symbols}")
        
        if not valid_symbols:
            raise ValueError("None of the specified symbols were found in the data")
        
        data = data[data.index.get_level_values('symbol').isin(valid_symbols)]
        print(f"Filtered to {len(valid_symbols)} symbols")
    
    # Filter time range if specified
    if start_time or end_time:
        time_mask = pd.Series(True, index=data.index)
        
        if start_time:
            start_time = pd.to_datetime(start_time)
            time_mask &= data.index.get_level_values('open_time') >= start_time
        
        if end_time:
            end_time = pd.to_datetime(end_time)
            time_mask &= data.index.get_level_values('open_time') <= end_time
        
        data = data[time_mask]
        print(f"Filtered to time range: {data.index.get_level_values('open_time').min()} to {data.index.get_level_values('open_time').max()}")
    
    if len(data) == 0:
        raise ValueError("No data remaining after filtering")
    
    # Reset index to work with groupby
    data_reset = data.reset_index()
    
    # Convert timeframe to pandas offset
    try:
        freq = pd.Timedelta(timeframe)
    except ValueError:
        # Handle special cases like '1d', '1w', etc.
        freq = pd.tseries.frequencies.to_offset(timeframe)
    
    # print(f"Using frequency: {freq}")
    
    # Aggregate data for each symbol
    aggregated_dfs = []
    symbols_processed = data_reset['symbol'].unique()
    
    for symbol in symbols_processed:
        symbol_data = data_reset[data_reset['symbol'] == symbol].copy()
        symbol_data = symbol_data.set_index('open_time').sort_index()
        
        # Define aggregation functions
        agg_functions = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': volume_aggregation
        }
        
        # Resample data
        try:
            resampled = symbol_data.resample(freq, label='left', closed='left').agg(agg_functions)
            
            # Remove rows with NaN values (incomplete periods)
            resampled = resampled.dropna()
            
            # Add symbol back as column
            resampled['symbol'] = symbol
            
            # Reset index to get open_time as column
            resampled = resampled.reset_index()
            
            aggregated_dfs.append(resampled)
            
        except Exception as e:
            print(f"Warning: Failed to aggregate data for symbol {symbol}: {e}")
            continue
    
    if not aggregated_dfs:
        raise ValueError("Failed to aggregate data for any symbols")
    
    # Combine all symbols
    combined_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    # Recreate MultiIndex
    combined_df = combined_df.set_index(['open_time', 'symbol'])
    combined_df = combined_df.sort_index()
    
    # print(f"✅ Aggregation completed:")
    # print(f"   Input records: {len(data):,}")
    # print(f"   Output records: {len(combined_df):,}")
    # print(f"   Symbols processed: {len(symbols_processed)}")
    # print(f"   Timeframe: 1min → {timeframe}")
    # print(f"   Date range: {combined_df.index.get_level_values('open_time').min()} to {combined_df.index.get_level_values('open_time').max()}")
    
    return combined_df


def batch_aggregate_ohlc_data(
    data: pd.DataFrame,
    timeframes: List[str],
    symbols: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate OHLC data to multiple timeframes in batch.
    
    Args:
        data (pd.DataFrame): Input DataFrame with MultiIndex (open_time, symbol)
        timeframes (List[str]): List of target timeframes. Examples: ['3min', '5min', '15min']
        symbols (List[str], optional): List of symbols to process
        **kwargs: Additional arguments passed to aggregate_ohlc_data
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with timeframes as keys and aggregated data as values
    
    Example:
        data_1min = load_usdt_symbols_from_month_cache(...)
        
        # Aggregate to multiple timeframes
        aggregated = batch_aggregate_ohlc_data(
            data_1min, 
            timeframes=['5min', '15min', '1h'],
            symbols=['BTCUSDT', 'ETHUSDT']
        )
        
        data_5min = aggregated['5min']
        data_15min = aggregated['15min']
        data_1h = aggregated['1h']
    """
    
    results = {}
    
    # print(f"Batch aggregating data to {len(timeframes)} timeframes...")
    # print(f"Timeframes: {timeframes}")
    
    for timeframe in timeframes:
        try:
            print(f"\n--- Processing {timeframe} ---")
            aggregated = aggregate_ohlc_data(
                data=data,
                timeframe=timeframe,
                symbols=symbols,
                **kwargs
            )
            results[timeframe] = aggregated
            
        except Exception as e:
            print(f"❌ Error aggregating to {timeframe}: {e}")
            continue
    
    # print(f"\n✅ Batch aggregation completed: {len(results)}/{len(timeframes)} timeframes successful")
    
    return results


# def calculate_ohlc_indicators(
#     data: pd.DataFrame,
#     indicators: Optional[List[str]] = None
# ) -> pd.DataFrame:
#     """
#     Calculate additional technical indicators from OHLC data.
    
#     Args:
#         data (pd.DataFrame): OHLC data with MultiIndex (open_time, symbol)
#         indicators (List[str], optional): List of indicators to calculate.
#                                         Available: ['returns', 'log_returns', 'volatility', 'typical_price']
#                                         If None, calculates all available indicators.
    
#     Returns:
#         pd.DataFrame: Original data with additional indicator columns
    
#     Example:
#         data_with_indicators = calculate_ohlc_indicators(
#             data_5min, 
#             indicators=['returns', 'volatility']
#         )
#     """
    
#     if indicators is None:
#         indicators = ['returns', 'log_returns', 'volatility', 'typical_price']
    
#     print(f"Calculating indicators: {indicators}")
    
#     # Work with a copy to avoid modifying original data
#     result = data.copy()
    
#     # Calculate indicators for each symbol
#     symbols = result.index.get_level_values('symbol').unique()
    
#     for symbol in symbols:
#         symbol_data = result.xs(symbol, level='symbol')
        
#         if 'returns' in indicators:
#             # Simple returns
#             symbol_data['returns'] = symbol_data['close'].pct_change()
        
#         if 'log_returns' in indicators:
#             # Log returns
#             symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
        
#         if 'volatility' in indicators:
#             # Rolling volatility (20-period)
#             if 'returns' not in symbol_data.columns:
#                 returns = symbol_data['close'].pct_change()
#             else:
#                 returns = symbol_data['returns']
#             symbol_data['volatility'] = returns.rolling(window=20).std()
        
#         if 'typical_price' in indicators:
#             # Typical price (HLC/3)
#             symbol_data['typical_price'] = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
        
#         # Update the result DataFrame
#         for indicator in indicators:
#             if indicator in symbol_data.columns:
#                 result.loc[result.index.get_level_values('symbol') == symbol, indicator] = symbol_data[indicator].values
    
#     print(f"✅ Added {len(indicators)} indicators to {len(symbols)} symbols")
    
#     return result


# def resample_ohlc_data_advanced(
#     data: pd.DataFrame,
#     timeframe: str,
#     alignment: str = 'left',
#     label: str = 'left',
#     fill_method: Optional[str] = None,
#     min_periods: int = 1
# ) -> pd.DataFrame:
#     """
#     Advanced OHLC data resampling with more control over alignment and filling.
    
#     Args:
#         data (pd.DataFrame): Input OHLC data
#         timeframe (str): Target timeframe
#         alignment (str): How to align the resampled data ('left' or 'right')
#         label (str): How to label the resampled periods ('left' or 'right')  
#         fill_method (str, optional): Method to fill missing values ('ffill', 'bfill', None)
#         min_periods (int): Minimum number of observations required to have a value
    
#     Returns:
#         pd.DataFrame: Resampled OHLC data
#     """
    
#     print(f"Advanced resampling to {timeframe} with alignment={alignment}, label={label}")
    
#     # Reset index to work with groupby
#     data_reset = data.reset_index()
    
#     # Convert timeframe
#     try:
#         freq = pd.Timedelta(timeframe)
#     except ValueError:
#         freq = pd.tseries.frequencies.to_offset(timeframe)
    
#     aggregated_dfs = []
#     symbols_processed = data_reset['symbol'].unique()
    
#     for symbol in symbols_processed:
#         symbol_data = data_reset[data_reset['symbol'] == symbol].copy()
#         symbol_data = symbol_data.set_index('open_time').sort_index()
        
#         # Advanced resampling
#         resampled = symbol_data.resample(
#             freq, 
#             label=label, 
#             closed=alignment
#         ).agg({
#             'open': 'first',
#             'high': 'max', 
#             'low': 'min',
#             'close': 'last',
#             'volume': 'sum'
#         })
        
#         # Apply minimum periods filter
#         if min_periods > 1:
#             valid_periods = symbol_data.resample(freq, label=label, closed=alignment).count()['close'] >= min_periods
#             resampled = resampled[valid_periods]
        
#         # Fill missing values if specified
#         if fill_method:
#             resampled = resampled.fillna(method=fill_method)
#         else:
#             resampled = resampled.dropna()
        
#         # Add symbol back
#         resampled['symbol'] = symbol
#         resampled = resampled.reset_index()
        
#         aggregated_dfs.append(resampled)
    
#     # Combine and create MultiIndex
#     combined_df = pd.concat(aggregated_dfs, ignore_index=True)
#     combined_df = combined_df.set_index(['open_time', 'symbol']).sort_index()
    
#     print(f"✅ Advanced resampling completed: {len(combined_df):,} records")
    
#     return combined_df


# def validate_ohlc_data(data: pd.DataFrame, fix_errors: bool = False) -> Dict[str, Any]:
#     """
#     Validate OHLC data for common issues and optionally fix them.
    
#     Args:
#         data (pd.DataFrame): OHLC data to validate
#         fix_errors (bool): Whether to attempt fixing errors automatically
    
#     Returns:
#         Dict[str, Any]: Validation report with issues found and fixes applied
#     """
    
#     print("Validating OHLC data...")
    
#     report = {
#         'total_records': len(data),
#         'symbols': len(data.index.get_level_values('symbol').unique()),
#         'issues_found': [],
#         'fixes_applied': [],
#         'valid_records': 0,
#         'invalid_records': 0
#     }
    
#     # Check for basic OHLC relationships
#     issues = []
    
#     # High should be >= Open, Close, Low
#     high_issues = (data['high'] < data[['open', 'close', 'low']].max(axis=1))
#     if high_issues.any():
#         count = high_issues.sum()
#         issues.append(f"High < max(Open, Close, Low): {count} records")
#         if fix_errors:
#             data.loc[high_issues, 'high'] = data.loc[high_issues, ['open', 'close', 'low']].max(axis=1)
#             report['fixes_applied'].append(f"Fixed {count} high price issues")
    
#     # Low should be <= Open, Close, High
#     low_issues = (data['low'] > data[['open', 'close', 'high']].min(axis=1))
#     if low_issues.any():
#         count = low_issues.sum()
#         issues.append(f"Low > min(Open, Close, High): {count} records")
#         if fix_errors:
#             data.loc[low_issues, 'low'] = data.loc[low_issues, ['open', 'close', 'high']].min(axis=1)
#             report['fixes_applied'].append(f"Fixed {count} low price issues")
    
#     # Check for negative prices
#     negative_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
#     if negative_prices.any():
#         count = negative_prices.sum()
#         issues.append(f"Negative or zero prices: {count} records")
#         if fix_errors:
#             # Remove records with negative prices
#             data = data[~negative_prices]
#             report['fixes_applied'].append(f"Removed {count} records with negative prices")
    
#     # Check for missing values
#     missing_values = data.isnull().any(axis=1)
#     if missing_values.any():
#         count = missing_values.sum()
#         issues.append(f"Missing values: {count} records")
#         if fix_errors:
#             data = data.dropna()
#             report['fixes_applied'].append(f"Removed {count} records with missing values")
    
#     # Check for negative volume
#     negative_volume = (data['volume'] < 0)
#     if negative_volume.any():
#         count = negative_volume.sum()
#         issues.append(f"Negative volume: {count} records")
#         if fix_errors:
#             data.loc[negative_volume, 'volume'] = 0
#             report['fixes_applied'].append(f"Fixed {count} negative volume records")
    
#     report['issues_found'] = issues
#     report['valid_records'] = len(data) if fix_errors else len(data) - sum([
#         high_issues.sum(), low_issues.sum(), negative_prices.sum(), 
#         missing_values.sum(), negative_volume.sum()
#     ])
#     report['invalid_records'] = report['total_records'] - report['valid_records']
    
#     print(f"✅ Validation completed:")
#     print(f"   Total records: {report['total_records']:,}")
#     print(f"   Valid records: {report['valid_records']:,}")
#     print(f"   Invalid records: {report['invalid_records']:,}")
#     print(f"   Issues found: {len(report['issues_found'])}")
#     if report['fixes_applied']:
#         print(f"   Fixes applied: {len(report['fixes_applied'])}")
    
#     return report


# # Utility functions for common timeframe conversions
# COMMON_TIMEFRAMES = {
#     '1min': '1min',
#     '3min': '3min', 
#     '5min': '5min',
#     '15min': '15min',
#     '30min': '30min',
#     '1h': '1h',
#     '2h': '2h', 
#     '4h': '4h',
#     '8h': '8h',
#     '12h': '12h',
#     '1d': '1d',
#     '3d': '3d',
#     '1w': '1w',
#     '1M': '1M'
# }

# def get_supported_timeframes() -> List[str]:
#     """Get list of supported timeframes for aggregation."""
#     return list(COMMON_TIMEFRAMES.keys())


# def estimate_output_size(input_records: int, input_timeframe: str, output_timeframe: str) -> int:
#     """
#     Estimate the number of output records for a given aggregation.
    
#     Args:
#         input_records (int): Number of input records
#         input_timeframe (str): Input timeframe (e.g., '1min')
#         output_timeframe (str): Output timeframe (e.g., '5min')
    
#     Returns:
#         int: Estimated number of output records
#     """
    
#     try:
#         input_freq = pd.Timedelta(input_timeframe)
#         output_freq = pd.Timedelta(output_timeframe)
        
#         ratio = output_freq / input_freq
#         estimated = int(input_records / ratio)
        
#         return max(1, estimated)  # At least 1 record
        
#     except Exception:
#         return input_records  # Fallback to input size if calculation fails