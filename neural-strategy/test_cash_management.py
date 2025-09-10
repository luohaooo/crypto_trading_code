#!/usr/bin/env python3
"""
Test cash management logic for long/short positions.
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

from strategies.base_strategy import BaseStrategy, PositionType
from strategies.factors.returns_factor import ReturnsFactor
from strategies.neutral_strategy import NeutralStrategy

def test_cash_management():
    """Test that cash management works correctly for long and short positions."""
    
    print("🧪 Testing Cash Management Logic")
    print("=" * 50)
    
    # Create a simple strategy instance
    factor = ReturnsFactor(lookback_periods=60, name="TestFactor")
    strategy = NeutralStrategy(
        factor=factor,
        top_n=2,
        bottom_n=2,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    print(f"Initial cash: ${strategy.cash:,.2f}")
    print(f"Initial portfolio value: ${strategy.initial_capital:,.2f}")
    
    # Test 1: Open a long position
    print("\n📈 Test 1: Opening Long Position")
    long_position = strategy.open_position(
        symbol="BTCUSDT",
        position_type=PositionType.LONG,
        size=1.0,
        entry_price=50000.0,
        entry_time=pd.Timestamp('2024-01-01')
    )
    
    print(f"Opened long position: {long_position.size} shares at ${long_position.entry_price:,.2f}")
    print(f"Cash after long position: ${strategy.cash:,.2f}")
    expected_cash_after_long = 100000 - (1.0 * 50000 * (1 + 0.001))
    print(f"Expected cash: ${expected_cash_after_long:,.2f}")
    
    # Test 2: Open a short position
    print("\n📉 Test 2: Opening Short Position")
    short_position = strategy.open_position(
        symbol="ETHUSDT",
        position_type=PositionType.SHORT,
        size=10.0,
        entry_price=3000.0,
        entry_time=pd.Timestamp('2024-01-01')
    )
    
    print(f"Opened short position: {short_position.size} shares at ${short_position.entry_price:,.2f}")
    print(f"Cash after short position: ${strategy.cash:,.2f}")
    expected_cash_after_short = expected_cash_after_long + (10.0 * 3000 * (1 - 0.001))
    print(f"Expected cash: ${expected_cash_after_short:,.2f}")
    
    # Test 3: Close positions
    print("\n🔄 Test 3: Closing Positions")
    
    # Close long position (sell at different price)
    long_trade = strategy.close_position("BTCUSDT", 52000.0, pd.Timestamp('2024-01-02'))
    print(f"Closed long trade: P&L = ${long_trade.pnl:,.2f}")
    print(f"Cash after closing long: ${strategy.cash:,.2f}")
    
    # Close short position (buy back at different price)
    short_trade = strategy.close_position("ETHUSDT", 2800.0, pd.Timestamp('2024-01-02'))
    print(f"Closed short trade: P&L = ${short_trade.pnl:,.2f}")
    print(f"Final cash: ${strategy.cash:,.2f}")
    
    # Verify total P&L
    total_pnl = long_trade.pnl + short_trade.pnl
    final_value = strategy.cash
    expected_final = strategy.initial_capital + total_pnl
    
    print(f"\n📊 Summary:")
    print(f"Initial capital: ${strategy.initial_capital:,.2f}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Final cash: ${final_value:,.2f}")
    print(f"Expected final: ${expected_final:,.2f}")
    print(f"Difference: ${abs(final_value - expected_final):,.2f}")
    
    if abs(final_value - expected_final) < 0.01:
        print("✅ Cash management test PASSED!")
        return True
    else:
        print("❌ Cash management test FAILED!")
        return False

if __name__ == "__main__":
    try:
        success = test_cash_management()
        if success:
            print("\n🎉 All cash management tests passed!")
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)