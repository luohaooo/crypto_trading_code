#!/usr/bin/env python3
"""
Quick test script for the monthly cache loading function - focused on specific symbols
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_usdt_symbols_from_month_cache, get_month_cache_status

def test_quick_symbol_loading():
    """Test loading just BTC and ETH for a specific period"""
    print("=== Testing Quick Symbol Loading ===")
    try:
        # Test loading only BTC and ETH for January 2020
        df = load_usdt_symbols_from_month_cache(
            symbols=['BTCUSDT', 'ETHUSDT'],
            start_date='2020-01-01', 
            end_date='2020-01-31'
        )
        
        if not df.empty:
            print(f"\n✅ Successfully loaded data: {len(df):,} rows")
            print(f"Index levels: {df.index.names}")
            print(f"Columns: {df.columns.tolist()}")
            symbols = df.index.get_level_values('symbol').unique().tolist()
            print(f"Loaded symbols: {symbols}")
            
            # Show sample data for BTC if available
            if 'BTCUSDT' in symbols:
                btc_data = df.xs('BTCUSDT', level='symbol')
                print(f"\nBTCUSDT sample data ({len(btc_data)} rows):")
                print(btc_data.head(3))
                
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_status():
    """Test cache status function"""
    print("=== Testing Cache Status ===")
    status = get_month_cache_status()
    print(f"Cache directory: {status['cache_directory']}")
    print(f"Total files: {status['total_files']}")
    print(f"Total size: {status['total_size_mb']:.1f} MB")
    print(f"Date range: {status['date_range']}")
    return status['total_files'] > 0

if __name__ == "__main__":
    print("Quick Test for Monthly Cache Loading Function")
    print("=" * 50)
    
    # Test cache status first
    cache_ok = test_cache_status()
    
    if not cache_ok:
        print("❌ No cache files found. Exiting...")
        sys.exit(1)
    
    # Quick symbol test
    symbol_ok = test_quick_symbol_loading()
    
    print(f"\n=== Test Results ===")
    print(f"Cache Status: {'✅' if cache_ok else '❌'}")
    print(f"Symbol Loading: {'✅' if symbol_ok else '❌'}")
    
    if symbol_ok:
        print("\n🎉 Function is working correctly with CSV fallback!")
    else:
        print("\n⚠️  Function needs debugging")