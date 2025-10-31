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

