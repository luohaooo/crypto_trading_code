"""
Data loading utilities for cryptocurrency price data.

This module provides functions for:
- Loading cryptocurrency price data from CSV files
- Managing pickle file caches for performance optimization
- Getting available symbols and data ranges

The module prioritizes loading from pickle caches when available,
falling back to CSV files when needed.
"""

import pandas as pd
import os
import glob
from pathlib import Path

# Configuration
DATA_DIR = "/home/craz/crypto/crypto-data/future_data_2"
PICKLE_CACHE_DIR = "/home/craz/crypto/crypto-data/pickle_cache"

# Create pickle cache directory if it doesn't exist
Path(PICKLE_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def get_available_symbols():
    """Get list of all available crypto symbols"""
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])


def get_usdt_symbols():
    """Get list of all available crypto symbols ending with USDT"""
    all_symbols = get_available_symbols()
    return sorted([symbol for symbol in all_symbols if symbol.endswith('USDT')])


def get_pickle_path(symbol):
    """Get the pickle file path for a symbol"""
    return os.path.join(PICKLE_CACHE_DIR, f"{symbol}_full_data.pkl")


def is_pickle_cache_available(symbol):
    """Check if pickle cache file exists"""
    pickle_path = get_pickle_path(symbol)
    return os.path.exists(pickle_path)


def create_pickle_cache(symbol):
    """Create pickle cache for a symbol by loading all CSV data"""
    print(f"Creating pickle cache for {symbol}...")
    
    symbol_dir = os.path.join(DATA_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return None
    
    # Get all CSV files for the symbol
    csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
    csv_files.sort()
    
    if not csv_files:
        return None
    
    # Read and combine all data
    dfs = []
    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            
            # Progress indicator
            if len(csv_files) > 10 and i % 10 == 0:
                print(f"Processing file {i+1}/{len(csv_files)} for pickle cache")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return None
    
    # Combine and process data
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
    combined_df = combined_df.sort_values('open_time')
    
    try:
        # Save data to pickle using pandas method (consistent with notebook)
        pickle_path = get_pickle_path(symbol)
        combined_df.to_pickle(pickle_path)
        
        print(f"✅ Created pickle cache for {symbol}: {len(combined_df):,} records")
        return combined_df
        
    except Exception as e:
        print(f"Error creating pickle cache: {e}")
        # If pickle creation fails, return the DataFrame anyway
        return combined_df


def load_from_pickle_cache(symbol):
    """Load data from pickle cache using pandas method (consistent with notebook)"""
    pickle_path = get_pickle_path(symbol)
    
    try:
        df = pd.read_pickle(pickle_path)
        print(f"📦 Loaded {len(df):,} records from pickle cache for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error loading pickle cache: {e}")
        return None


def get_symbol_data(symbol, start_date=None, end_date=None):
    """
    Get price data for a symbol within date range - prioritizes pickle cache
    
    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'BTCUSDT')
        start_date (datetime.date, optional): Start date for filtering
        end_date (datetime.date, optional): End date for filtering
    
    Returns:
        pandas.DataFrame: OHLCV data with columns ['open_time', 'open', 'high', 'low', 'close', 'volume']
    """
    
    # Try to load from pickle cache first
    if is_pickle_cache_available(symbol):
        df = load_from_pickle_cache(symbol)
        if df is not None:
            # Apply date filtering if needed
            if start_date or end_date:
                if start_date:
                    df = df[df['open_time'].dt.date >= start_date]
                if end_date:
                    df = df[df['open_time'].dt.date <= end_date]
            return df
    
    # If no pickle cache available, load from CSV and create cache
    print(f"No pickle cache found for {symbol}, loading from CSV files...")
    df = create_pickle_cache(symbol)
    if df is None:
        return None
    
    # Apply date filtering if needed
    if start_date or end_date:
        if start_date:
            df = df[df['open_time'].dt.date >= start_date]
        if end_date:
            df = df[df['open_time'].dt.date <= end_date]
    
    print(f"Returning {len(df):,} data points for {symbol}")
    return df


def get_symbol_info(symbol):
    """
    Get basic information about a symbol's data files
    
    Args:
        symbol (str): The cryptocurrency symbol
        
    Returns:
        dict: Information about the symbol's data files including date range and file count
    """
    symbol_dir = os.path.join(DATA_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return None
    
    csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
    if not csv_files:
        return None
    
    csv_files.sort()
    
    # Get date range from filenames
    first_file = os.path.basename(csv_files[0]).split('_')[-1].replace('.csv', '')
    last_file = os.path.basename(csv_files[-1]).split('_')[-1].replace('.csv', '')
    
    return {
        'symbol': symbol,
        'first_date': first_file,
        'last_date': last_file,
        'total_files': len(csv_files),
        'has_pickle_cache': is_pickle_cache_available(symbol)
    }


def get_multiple_symbols_data(symbols, start_date=None, end_date=None):
    """
    Get price data for multiple symbols within date range and join them into a single DataFrame
    
    Args:
        symbols (list): List of cryptocurrency symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        start_date (datetime.date, optional): Start date for filtering
        end_date (datetime.date, optional): End date for filtering
    
    Returns:
        pandas.DataFrame: Combined OHLCV data with MultiIndex (open_time, symbol) for easy querying
    """
    all_dfs = []
    
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        df = get_symbol_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Add symbol column to identify which symbol each row belongs to
            df['symbol'] = symbol
            all_dfs.append(df)
        else:
            print(f"Warning: No data found for {symbol}")
    
    if not all_dfs:
        print("No data found for any of the requested symbols")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by open_time and symbol for consistent ordering
    combined_df = combined_df.sort_values(['open_time', 'symbol'])
    
    # Set MultiIndex with open_time first, then symbol for easy querying
    combined_df = combined_df.set_index(['open_time', 'symbol'])
    
    print(f"Combined data: {len(combined_df):,} records across {len(all_dfs)} symbols")
    return combined_df


def get_cache_status():
    """
    Get status of all pickle caches
    
    Returns:
        dict: Status of pickle cache directory including file counts and sizes
    """
    cache_files = glob.glob(os.path.join(PICKLE_CACHE_DIR, "*.pkl"))
    data_files = [f for f in cache_files if f.endswith('_full_data.pkl')]
    
    total_size = 0
    if data_files:
        total_size = sum(os.path.getsize(f) for f in data_files)
    
    return {
        'cache_directory': PICKLE_CACHE_DIR,
        'total_pickle_files': len(cache_files),
        'data_files': len(data_files),
        'total_cache_size_mb': total_size / (1024 * 1024),
        'total_cache_size_gb': total_size / (1024 * 1024 * 1024)
    }