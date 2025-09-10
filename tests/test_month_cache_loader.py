#!/usr/bin/env python3
"""
Test script for the monthly cache loading function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_usdt_symbols_from_month_cache, get_month_cache_status

def test_month_cache_status():
    """Test getting cache status"""
    print("=== Testing Monthly Cache Status ===")
    status = get_month_cache_status()
    print(f"Cache directory: {status['cache_directory']}")
    print(f"Total files: {status['total_files']}")
    print(f"Total size: {status['total_size_mb']:.1f} MB")
    print(f"Date range: {status['date_range']}")
    print(f"Available months: {len(status['available_months'])}")
    if status['available_months']:
        print(f"First few months: {status['available_months'][:5]}")
    return status

def test_load_specific_month():
    """Test loading data for a specific month"""
    print("\n=== Testing Load Specific Month ===")
    try:
        # Test loading January 2020
        df = load_usdt_symbols_from_month_cache(
            start_date='2020-01-01', 
            end_date='2020-01-31'
        )
        if not df.empty:
            print(f"Successfully loaded data: {len(df):,} rows")
            print(f"Index levels: {df.index.names}")
            print(f"Columns: {df.columns.tolist()}")
            unique_symbols = len(df.index.get_level_values('symbol').unique())
            print(f"Unique symbols: {unique_symbols}")
            print("First few rows:")
            print(df.head())
            return df
        else:
            print("No data loaded")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_load_specific_symbols():
    """Test loading specific symbols"""
    print("\n=== Testing Load Specific Symbols ===")
    try:
        # Test loading BTC and ETH for a month
        df = load_usdt_symbols_from_month_cache(
            symbols=['BTCUSDT', 'ETHUSDT'],
            start_date='2020-01-01', 
            end_date='2020-01-31'
        )
        if not df.empty:
            print(f"Successfully loaded filtered data: {len(df):,} rows")
            symbols = df.index.get_level_values('symbol').unique().tolist()
            print(f"Symbols: {symbols}")
            
            # Test accessing specific symbol data
            if 'BTCUSDT' in symbols:
                btc_data = df.xs('BTCUSDT', level='symbol')
                print(f"BTCUSDT data: {len(btc_data):,} rows")
                print("BTC sample data:")
                print(btc_data.head(3))
            
            return df
        else:
            print("No data loaded for specific symbols")
            return None
    except Exception as e:
        print(f"Error loading specific symbols: {e}")
        return None

if __name__ == "__main__":
    print("Testing Monthly Cache Loading Functions")
    print("=" * 50)
    
    # Test cache status
    cache_status = test_month_cache_status()
    
    if cache_status['total_files'] == 0:
        print("\nNo monthly cache files found. Please run the monthly data processor notebook first.")
        sys.exit(1)
    
    # Test loading specific month
    month_data = test_load_specific_month()
    
    # Test loading specific symbols
    symbol_data = test_load_specific_symbols()
    
    print("\n=== Test Summary ===")
    print(f"Cache status: {'✅ OK' if cache_status['total_files'] > 0 else '❌ Failed'}")
    print(f"Month loading: {'✅ OK' if month_data is not None and not month_data.empty else '❌ Failed'}")
    print(f"Symbol filtering: {'✅ OK' if symbol_data is not None and not symbol_data.empty else '❌ Failed'}")
    print("\nAll tests completed!")