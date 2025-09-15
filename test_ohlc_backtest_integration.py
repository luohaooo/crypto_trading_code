#!/usr/bin/env python3
"""
OHLC Figure Factor Backtest Integration Test

Demonstrates how to use the OHLC Figure Factor within the backtesting engine.
This script shows the complete integration workflow from configuration to execution.
"""

import sys
import os
# from datetime import datetime, timedelta  # Not needed for this demo

# Add paths for imports
project_root = '/home/craz/crypto/crypto-trading'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'neural-strategy'))

from utils.config import BacktestConfig, StrategyConfig, FactorConfig, DataConfig
from backtest.engine import BacktestEngine

def create_ohlc_backtest_config():
    """Create configuration for OHLC Figure Factor backtesting."""
    
    # Data configuration
    data_config = DataConfig(
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT'],  # Test with major coins
        start_date='2024-06-01',
        end_date='2024-06-15',  # Short period for testing
        symbol_filter='usdt',
        min_data_points=500
    )
    
    # OHLC Figure Factor configuration
    factor_config = FactorConfig(
        name="OHLC_CNN_Neural",
        factor_type="ohlc_figure",
        lookback_periods=20,  # 20 periods for image generation
        params={
            'device': 'auto',  # Use GPU if available, otherwise CPU
            'timeframes': ['3min', '15min', '1h'],  # Multi-timeframe analysis
            'confidence_threshold': None  # No confidence filtering for now
        }
    )
    
    # Strategy configuration
    strategy_config = StrategyConfig(
        name="OHLC_Neutral_Strategy",
        strategy_type="neutral",
        initial_capital=100000.0,  # $100K starting capital
        commission_rate=0.001,  # 0.1% commission
        top_n=2,  # 2 long positions
        bottom_n=2,  # 2 short positions
        rebalance_frequency='4h'  # Rebalance every 4 hours
    )
    
    # Main backtest configuration
    backtest_config = BacktestConfig(
        strategy=strategy_config,
        factor=factor_config,
        data=data_config,
        initial_warmup_periods=100,  # Reduced for testing
        progress_reporting=True,
        save_trades=True
    )
    
    return backtest_config

def run_ohlc_backtest():
    """Run a complete OHLC Figure Factor backtest."""
    print("🚀 OHLC Figure Factor Backtesting Demo")
    print("=" * 60)
    
    try:
        # Create configuration
        print("📋 Creating backtest configuration...")
        config = create_ohlc_backtest_config()
        
        print(f"Strategy: {config.strategy.name}")
        print(f"Factor: {config.factor.name} ({config.factor.factor_type})")
        print(f"Symbols: {config.data.symbols}")
        print(f"Period: {config.data.start_date} to {config.data.end_date}")
        print(f"Capital: ${config.strategy.initial_capital:,.0f}")
        print(f"Positions: {config.strategy.top_n}L/{config.strategy.bottom_n}S")
        
        # Initialize backtest engine
        print("\n🔧 Initializing backtest engine...")
        engine = BacktestEngine(config)
        
        # Load data
        print("\n📊 Loading data...")
        engine.load_data()  # Load but don't store in variable since it's not used
        
        # Run backtest
        print("\n🎯 Starting backtest execution...")
        results = engine.run_backtest()
        
        # Display results
        print("\n📈 Backtest completed! Displaying results...")
        engine.print_results_summary()
        
        # Additional OHLC-specific analysis
        print("\n🔍 OHLC Factor Analysis:")
        factor = engine.strategy.factor
        model_info = factor.get_model_info()
        
        print(f"Model loaded: {model_info['model_loaded']}")
        print(f"Device: {model_info['device']}")
        print(f"Timeframes: {model_info['timeframes']}")
        print(f"Image size: {model_info['image_size']}")
        print(f"Cache size: {model_info['cache_size']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_configuration_examples():
    """Show different OHLC factor configuration examples."""
    print("\n📚 OHLC Figure Factor Configuration Examples")
    print("-" * 50)
    
    # Example 1: Basic OHLC factor
    print("\n1. Basic OHLC Factor:")
    basic_config = FactorConfig(
        name="OHLC_Basic",
        factor_type="ohlc_figure",
        lookback_periods=20
    )
    print(f"   {basic_config}")
    
    # Example 2: Single timeframe OHLC factor
    print("\n2. Single Timeframe OHLC Factor:")
    single_tf_config = FactorConfig(
        name="OHLC_15min_Only",
        factor_type="ohlc_figure",
        lookback_periods=30,
        params={
            'timeframes': ['15min'],
            'device': 'cpu'
        }
    )
    print(f"   {single_tf_config}")
    
    # Example 3: High-confidence OHLC factor
    print("\n3. High-Confidence OHLC Factor:")
    confident_config = FactorConfig(
        name="OHLC_High_Confidence",
        factor_type="ohlc_figure",
        lookback_periods=20,
        params={
            'timeframes': ['3min', '15min', '1h'],
            'confidence_threshold': 0.005,  # Only signals above 0.5%
            'device': 'cuda'
        }
    )
    print(f"   {confident_config}")

def main():
    """Main execution function."""
    # Show configuration examples
    demonstrate_configuration_examples()
    
    # Run actual backtest
    results = run_ohlc_backtest()
    
    if results:
        print("\n✅ OHLC Figure Factor integration test completed successfully!")
        print("\n🎉 The OHLC Figure Factor is now fully integrated into the backtesting engine!")
        print("\n💡 You can now use 'ohlc_figure' as a factor_type in your backtest configurations.")
    else:
        print("\n❌ Integration test failed. Check the error messages above.")

if __name__ == "__main__":
    main()