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
import pickle
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional, Union

# Configuration
DATA_DIR = "/home/craz/crypto/crypto-data/future_data_2"
PICKLE_CACHE_DIR = "/home/craz/crypto/crypto-data/pickle_cache"
PICKLE_MONTH_CACHE_DIR = "/home/craz/crypto/crypto-data/pickle_month_cache"

# Create pickle cache directories if they don't exist
Path(PICKLE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(PICKLE_MONTH_CACHE_DIR).mkdir(parents=True, exist_ok=True)


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


def load_usdt_symbols_from_month_cache(
    symbols: Optional[List[str]] = None, 
    start_date: Optional[Union[str, date, datetime]] = None, 
    end_date: Optional[Union[str, date, datetime]] = None,
    use_fallback: bool = True
) -> pd.DataFrame:
    """
    Load USDT symbols data from monthly pickle cache files for a specific time range.
    
    This function loads data from the pickle files created by the monthly data processor,
    where each file contains all USDT symbols data for one month with MultiIndex (open_time, symbol).
    
    Args:
        symbols (List[str], optional): List of USDT symbols to filter. If None, loads all available symbols.
        start_date (str/date/datetime, optional): Start date for filtering (inclusive). 
                                                  Can be string in format 'YYYY-MM-DD' or date/datetime object.
        end_date (str/date/datetime, optional): End date for filtering (inclusive).
                                               Can be string in format 'YYYY-MM-DD' or date/datetime object.
        use_fallback (bool): If True, falls back to CSV loading if pickle fails due to version compatibility.
    
    Returns:
        pandas.DataFrame: Combined OHLCV data with MultiIndex (open_time, symbol)
                         Columns: ['open', 'high', 'low', 'close', 'volume']
    
    Example:
        # Load all USDT symbols for a specific month
        df = load_usdt_symbols_from_month_cache(start_date='2020-01-01', end_date='2020-01-31')
        
        # Load specific symbols for a date range
        df = load_usdt_symbols_from_month_cache(
            symbols=['BTCUSDT', 'ETHUSDT'], 
            start_date='2020-01-01', 
            end_date='2020-03-31'
        )
        
        # Access data for a specific symbol
        btc_data = df.xs('BTCUSDT', level='symbol')
    """
    
    # Convert date inputs to datetime objects for consistent processing
    def parse_date(date_input):
        if date_input is None:
            return None
        if isinstance(date_input, str):
            return datetime.strptime(date_input, '%Y-%m-%d').date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise ValueError(f"Unsupported date format: {type(date_input)}")
    
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    
    # Get available monthly pickle files
    month_pickle_files = glob.glob(os.path.join(PICKLE_MONTH_CACHE_DIR, "usdt_data_*.pkl"))
    month_pickle_files.sort()
    
    if not month_pickle_files:
        if use_fallback:
            print(f"No monthly pickle files found in {PICKLE_MONTH_CACHE_DIR}, falling back to CSV loading...")
            return load_usdt_symbols_from_csv_fallback(symbols, start_date, end_date)
        else:
            print(f"No monthly pickle files found in {PICKLE_MONTH_CACHE_DIR}")
            return pd.DataFrame()
    
    print(f"Found {len(month_pickle_files)} monthly pickle files")
    
    # Determine which months to load based on date range
    months_to_load = []
    
    if start_date is None and end_date is None:
        # Load all available months
        months_to_load = month_pickle_files
    else:
        # Filter months based on date range
        for file_path in month_pickle_files:
            file_name = os.path.basename(file_path)
            # Extract year-month from filename: usdt_data_YYYY-MM.pkl
            month_str = file_name.replace('usdt_data_', '').replace('.pkl', '')
            try:
                file_date = datetime.strptime(month_str + '-01', '%Y-%m-%d').date()
                
                # Check if this month overlaps with our date range
                month_start = file_date
                # Get last day of month
                if file_date.month == 12:
                    next_month = datetime(file_date.year + 1, 1, 1)
                else:
                    next_month = datetime(file_date.year, file_date.month + 1, 1)
                month_end = (next_month - pd.Timedelta(days=1)).date()
                
                # Check overlap with requested date range
                if start_date and end_date:
                    if month_end >= start_date and month_start <= end_date:
                        months_to_load.append(file_path)
                elif start_date:
                    if month_end >= start_date:
                        months_to_load.append(file_path)
                elif end_date:
                    if month_start <= end_date:
                        months_to_load.append(file_path)
                        
            except ValueError as e:
                print(f"Warning: Could not parse date from filename {file_name}: {e}")
                continue
    
    if not months_to_load:
        if use_fallback:
            print("No pickle files match the specified date range, falling back to CSV loading...")
            return load_usdt_symbols_from_csv_fallback(symbols, start_date, end_date)
        else:
            print("No pickle files match the specified date range")
            return pd.DataFrame()
    
    print(f"Loading {len(months_to_load)} month(s) of data...")
    
    # Load and combine data from selected months
    all_dataframes = []
    failed_files = []
    
    for file_path in months_to_load:
        try:
            # Try multiple loading methods for compatibility
            month_df = None
            
            # Method 1: pandas read_pickle
            try:
                month_df = pd.read_pickle(file_path)
            except Exception as e1:
                print(f"Pandas read_pickle failed for {os.path.basename(file_path)}: {e1}")
                
                # Method 2: standard pickle with different protocols
                for protocol in [None, 2, 3, 4]:
                    try:
                        if protocol is None:
                            with open(file_path, 'rb') as f:
                                month_df = pickle.load(f)
                        else:
                            with open(file_path, 'rb') as f:
                                month_df = pickle.load(f)
                        break
                    except Exception as e2:
                        continue
                
                if month_df is None:
                    print(f"All pickle methods failed for {os.path.basename(file_path)}")
                    failed_files.append(file_path)
                    continue
            
            file_name = os.path.basename(file_path)
            print(f"Loaded {file_name}: {len(month_df):,} rows")
            
            all_dataframes.append(month_df)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            failed_files.append(file_path)
            continue
    
    if not all_dataframes:
        if use_fallback and failed_files:
            print("Failed to load any pickle files, falling back to CSV loading...")
            return load_usdt_symbols_from_csv_fallback(symbols, start_date, end_date)
        else:
            print("Failed to load any data from pickle files")
            return pd.DataFrame()
    
    # Combine all monthly dataframes
    print("Combining monthly data...")
    combined_df = pd.concat(all_dataframes)
    
    # Apply date filtering if needed (more precise filtering on actual dates)
    if start_date or end_date:
        print("Applying date range filtering...")
        # Get the open_time index level
        date_index = combined_df.index.get_level_values('open_time')
        
        if start_date:
            # Convert datetime index to date for comparison
            mask_start = date_index.date >= start_date
            combined_df = combined_df[mask_start]
        
        if end_date:
            # Convert datetime index to date for comparison
            mask_end = date_index.date <= end_date
            combined_df = combined_df[mask_end]
    
    # Apply symbol filtering if specified
    if symbols:
        print(f"Filtering for symbols: {symbols}")
        # Filter for requested symbols only
        available_symbols = combined_df.index.get_level_values('symbol').unique().tolist()
        valid_symbols = [s for s in symbols if s in available_symbols]
        missing_symbols = [s for s in symbols if s not in available_symbols]
        
        if missing_symbols:
            print(f"Warning: Symbols not found in data: {missing_symbols}")
        
        if valid_symbols:
            combined_df = combined_df[combined_df.index.get_level_values('symbol').isin(valid_symbols)]
        else:
            print("None of the requested symbols were found in the data")
            if use_fallback:
                print("Falling back to CSV loading for missing symbols...")
                return load_usdt_symbols_from_csv_fallback(symbols, start_date, end_date)
            return pd.DataFrame()
    
    # Sort the final dataframe by index
    combined_df = combined_df.sort_index()
    
    unique_symbols = len(combined_df.index.get_level_values('symbol').unique())
    date_range = (
        combined_df.index.get_level_values('open_time').min(),
        combined_df.index.get_level_values('open_time').max()
    )
    
    print(f"✅ Loaded {len(combined_df):,} records for {unique_symbols} symbols")
    print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
    
    return combined_df


def load_usdt_symbols_from_csv_fallback(
    symbols: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Fallback function to load USDT symbols from CSV files when pickle loading fails.
    
    Args:
        symbols: List of USDT symbols to load
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        DataFrame with MultiIndex (open_time, symbol)
    """
    print("Using CSV fallback loading method...")
    
    # Get USDT symbols to load
    if symbols is None:
        symbols = get_usdt_symbols()
        print(f"Loading all {len(symbols)} USDT symbols from CSV...")
    else:
        # Filter for USDT symbols only
        usdt_symbols = [s for s in symbols if s.endswith('USDT')]
        if len(usdt_symbols) != len(symbols):
            non_usdt = [s for s in symbols if not s.endswith('USDT')]
            print(f"Warning: Ignoring non-USDT symbols: {non_usdt}")
        symbols = usdt_symbols
        print(f"Loading {len(symbols)} USDT symbols from CSV...")
    
    if not symbols:
        print("No USDT symbols to load")
        return pd.DataFrame()
    
    # Use existing function to load multiple symbols
    return get_multiple_symbols_data(symbols, start_date, end_date)


def get_month_cache_status():
    """
    Get status of monthly pickle cache files
    
    Returns:
        dict: Status of monthly pickle cache directory including file counts, date range, and sizes
    """
    cache_files = glob.glob(os.path.join(PICKLE_MONTH_CACHE_DIR, "usdt_data_*.pkl"))
    cache_files.sort()
    
    if not cache_files:
        return {
            'cache_directory': PICKLE_MONTH_CACHE_DIR,
            'total_files': 0,
            'total_size_mb': 0,
            'date_range': None,
            'available_months': []
        }
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in cache_files)
    
    # Extract date information
    available_months = []
    for file_path in cache_files:
        file_name = os.path.basename(file_path)
        month_str = file_name.replace('usdt_data_', '').replace('.pkl', '')
        available_months.append(month_str)
    
    return {
        'cache_directory': PICKLE_MONTH_CACHE_DIR,
        'total_files': len(cache_files),
        'total_size_mb': total_size / (1024 * 1024),
        'total_size_gb': total_size / (1024 * 1024 * 1024),
        'date_range': f"{available_months[0]} to {available_months[-1]}" if available_months else None,
        'available_months': available_months
    }