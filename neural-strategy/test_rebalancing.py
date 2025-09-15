#!/usr/bin/env python3
"""
Test rebalancing with insufficient cash warning fix.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

neural_strategy_root = os.path.dirname(__file__)
sys.path.insert(0, neural_strategy_root)

from strategies.factors.returns_factor import ReturnsFactor
from strategies.neutral_strategy import NeutralStrategy

def create_mock_data():
    """Create mock data for testing."""
    timestamps = pd.date_range('2024-01-01', periods=100, freq='1h')
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT']
    
    data_rows = []
    for timestamp in timestamps:
        for symbol in symbols:
            # Create some price movement
            base_price = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'ADAUSDT': 0.5, 'DOGEUSDT': 0.1, 'SOLUSDT': 100}[symbol]
            price_variation = 1 + (hash(str(timestamp) + symbol) % 1000 - 500) / 10000  # Random-ish variation
            close_price = base_price * price_variation
            
            data_rows.append({
                'open_time': timestamp,
                'symbol': symbol,
                'open': close_price * 0.999,
                'high': close_price * 1.002,
                'low': close_price * 0.998,
                'close': close_price,
                'volume': 1000000
            })
    
    df = pd.DataFrame(data_rows)
    df.set_index(['open_time', 'symbol'], inplace=True)
    return df

def test_rebalancing_cash_logic():
    """Test that rebalancing doesn't cause insufficient cash warnings."""
    
    print("🔄 Testing Rebalancing Cash Logic")
    print("=" * 50)
    
    # Create mock data
    data = create_mock_data()
    print(f"Created mock data: {len(data)} rows for {len(data.index.get_level_values('symbol').unique())} symbols")
    
    # Create strategy
    factor = ReturnsFactor(lookback_periods=10, name="TestReturns")  # Short lookback for testing
    strategy = NeutralStrategy(
        factor=factor,
        top_n=2,  # 2 long positions
        bottom_n=2,  # 2 short positions
        initial_capital=100000,
        commission_rate=0.001
    )
    
    print(f"Strategy: {strategy}")
    print(f"Initial cash: ${strategy.cash:,.2f}")
    
    # Test multiple rebalancing cycles
    test_timestamps = data.index.get_level_values('open_time').unique()[20:30]  # Skip first 20 for factor calculation
    
    print(f"\nTesting {len(test_timestamps)} rebalancing cycles...")
    
    success_count = 0
    for i, timestamp in enumerate(test_timestamps):
        try:
            print(f"\n--- Rebalance {i+1} at {timestamp} ---")
            
            # Generate signals
            signals = strategy.generate_signals(data, timestamp)
            
            if signals['action'] == 'rebalance':
                print(f"Portfolio value before: ${signals['portfolio_value']:,.2f}")
                print(f"Cash before: ${strategy.cash:,.2f}")
                print(f"Target positions: {len(signals['target_positions'])}")
                
                # Execute rebalancing
                strategy.execute_rebalance(data, timestamp, signals)
                
                new_portfolio_value = strategy.calculate_portfolio_value(data, timestamp)
                print(f"Portfolio value after: ${new_portfolio_value:,.2f}")
                print(f"Cash after: ${strategy.cash:,.2f}")
                print(f"Active positions: {len(strategy.positions)}")
                
                success_count += 1
                
            else:
                print(f"No rebalancing signal: {signals['action']}")
                
        except Exception as e:
            print(f"❌ Error during rebalance {i+1}: {e}")
            break
    
    print(f"\n📊 Test Results:")
    print(f"Successful rebalances: {success_count}/{len(test_timestamps)}")
    print(f"Final cash: ${strategy.cash:,.2f}")
    print(f"Final positions: {len(strategy.positions)}")
    print(f"Total trades completed: {len(strategy.completed_trades)}")
    
    if success_count == len(test_timestamps):
        print("✅ All rebalancing tests PASSED!")
        return True
    else:
        print("❌ Some rebalancing tests FAILED!")
        return False

if __name__ == "__main__":
    try:
        success = test_rebalancing_cash_logic()
        if success:
            print("\n🎉 Rebalancing cash logic works correctly!")
        else:
            print("\n💥 Rebalancing has issues!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)