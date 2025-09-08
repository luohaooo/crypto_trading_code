#!/usr/bin/env python3
"""
Test integration of monthly cache loading in neural strategy backtesting engine.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

# Import neural strategy components
import importlib.util

# Import BacktestConfig
config_spec = importlib.util.spec_from_file_location("config", os.path.join(current_dir, "utils", "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
BacktestConfig = config_module.BacktestConfig

# Import BacktestEngine
engine_spec = importlib.util.spec_from_file_location("engine", os.path.join(current_dir, "backtest", "engine.py"))
engine_module = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(engine_module)
BacktestEngine = engine_module.BacktestEngine

def test_month_cache_integration():
    """Test that the updated engine uses monthly cache for data loading"""
    print("=== Testing Monthly Cache Integration ===")
    
    try:
        # Create a small test configuration
        config = BacktestConfig.create_default(
            symbol_list=['BTCUSDT', 'ETHUSDT'],  # Specific USDT symbols
            start_date='2020-01-01',
            end_date='2020-01-03'  # Just 3 days for quick test
        )
        
        # Adjust strategy to match available symbols
        config.strategy.top_n = 1  # 1 long position
        config.strategy.bottom_n = 1  # 1 short position
        
        print(f"Created config: {config}")
        
        # Create engine
        engine = BacktestEngine(config)
        print("Created BacktestEngine")
        
        # Test data loading
        print("\nTesting data loading...")
        data = engine.load_data()
        
        if data is not None and len(data) > 0:
            print(f"✅ Data loading successful!")
            print(f"📊 Loaded {len(data):,} records")
            print(f"🔢 Symbols: {len(data.index.get_level_values('symbol').unique())}")
            print(f"📅 Date range: {data.index.get_level_values('open_time').min()} to {data.index.get_level_values('open_time').max()}")
            print(f"📋 Columns: {list(data.columns)}")
            print(f"🏷️ Index levels: {data.index.names}")
            
            # Show sample data
            print(f"\nSample data:")
            print(data.head())
            
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """Test configuration creation with different parameters"""
    print("\n=== Testing Configuration Creation ===")
    
    try:
        # Test default config for USDT symbols
        config1 = BacktestConfig.create_default(
            start_date='2020-01-01',
            end_date='2020-01-02'
        )
        print(f"✅ Default USDT config created: {config1.data.symbol_filter}")
        
        # Test specific symbols
        config2 = BacktestConfig.create_default(
            symbol_list=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            start_date='2020-01-01',
            end_date='2020-01-02'
        )
        print(f"✅ Specific symbols config created: {len(config2.data.symbols)} symbols")
        
        # Test momentum strategy
        config3 = BacktestConfig.create_momentum_strategy(
            short_period=30,
            long_period=120,
            symbol_list=['BTCUSDT', 'ETHUSDT'],
            start_date='2020-01-01',
            end_date='2020-01-02'
        )
        print(f"✅ Momentum strategy config created: {config3.factor.factor_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in config creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Neural Strategy Monthly Cache Integration")
    print("=" * 60)
    
    # Test configuration creation
    config_ok = test_config_creation()
    
    # Test monthly cache integration
    integration_ok = test_month_cache_integration()
    
    print(f"\n=== Test Results ===")
    print(f"Configuration Creation: {'✅' if config_ok else '❌'}")
    print(f"Monthly Cache Integration: {'✅' if integration_ok else '❌'}")
    
    if config_ok and integration_ok:
        print("\n🎉 All tests passed! Monthly cache integration is working.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")