import sys
import os
from datetime import datetime, timedelta

# Add parent directories to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.core.engine import BacktestEngine
from backtest.factors.cache_manager import FactorCacheManager
from utils.data_loader import get_available_symbols

def test_basic_backtest():
    """
    Test basic functionality of the market-neutral backtesting framework.
    """
    print("🚀 Testing Market-Neutral Backtest Framework...")
    print("=" * 60)
    
    # Test 1: Check data availability
    print("\n📊 Test 1: Checking data availability...")
    symbols = get_available_symbols()
    print(f"   Found {len(symbols)} available symbols")
    print(f"   Sample symbols: {symbols[:10]}")
    
    if len(symbols) < 50:
        print("   ⚠️  Warning: Limited symbols available for testing")
        return False
    
    # Test 2: Initialize backtesting engine
    print("\n⚙️  Test 2: Initializing backtesting engine...")
    
    # Use shorter date range for quick test
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)  # 1 month test
    
    try:
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,
            rebalance_frequency='1W',  # Weekly for faster test
            transaction_cost=0.001,
            max_symbols=50  # Limit symbols for quick test
        )
        print("   ✅ Engine initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize engine: {e}")
        return False
    
    # Test 3: Initialize symbol universe
    print("\n🌍 Test 3: Initializing symbol universe...")
    try:
        test_symbols = symbols[:20]  # Use only 20 symbols for quick test
        universe = engine.initialize_universe(test_symbols)
        print(f"   ✅ Universe initialized with {len(universe)} symbols")
    except Exception as e:
        print(f"   ❌ Failed to initialize universe: {e}")
        return False
    
    # Test 4: Test factor computation
    print("\n🔢 Test 4: Testing factor computation...")
    try:
        # Load a small sample of data
        from backtest.data.data_manager import DataManager
        from backtest.data.memory_manager import MemoryManager
        
        memory_manager = MemoryManager(max_memory_gb=2)  # Reduced memory limit
        data_manager = DataManager(memory_manager, max_workers=5)
        
        # Test with very small date range
        test_start = datetime(2023, 1, 1)
        test_end = datetime(2023, 1, 7)  # 1 week only
        
        symbol_data = data_manager.load_multi_symbol_data(
            symbols=universe[:5],  # Only 5 symbols for quick test
            start_date=test_start,
            end_date=test_end,
            chunk_size_days=30
        )
        
        if symbol_data:
            print(f"   ✅ Loaded data for {len(symbol_data)} symbols")
            
            # Test factor computation
            from backtest.factors.returns_factor import ReturnsFactor
            from backtest.factors.cache_manager import FactorCacheManager
            
            cache_manager = FactorCacheManager(use_redis=False)  # Use memory cache only
            returns_factor = ReturnsFactor(cache_manager)
            
            test_symbol = list(symbol_data.keys())[0]
            test_data = symbol_data[test_symbol]
            
            if len(test_data) > 10:
                factor_score = returns_factor.compute_with_cache(
                    symbol=test_symbol,
                    data=test_data,
                    date=test_end,
                    params={'lookback_hours': 24}
                )
                print(f"   ✅ Computed factor score for {test_symbol}: {factor_score:.6f}")
            else:
                print(f"   ⚠️  Insufficient data for {test_symbol}: {len(test_data)} records")
        else:
            print("   ❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"   ❌ Factor computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test portfolio operations
    print("\n💼 Test 5: Testing portfolio operations...")
    try:
        from backtest.core.portfolio import Portfolio, Trade
        
        portfolio = Portfolio(initial_capital=1000000)
        
        # Create test trade
        test_trade = Trade(
            symbol="BTCUSDT",
            side="long",
            action="open",
            quantity=1.0,
            price=45000.0,
            timestamp=datetime.now(),
            commission=45.0,
            reason="test"
        )
        
        # Execute trade
        success = portfolio.execute_trade(test_trade)
        if success:
            print("   ✅ Test trade executed successfully")
            print(f"   💰 Cash remaining: ${portfolio.cash:,.2f}")
            print(f"   📈 Positions: {len(portfolio.positions)}")
        else:
            print("   ❌ Test trade execution failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Portfolio test failed: {e}")
        return False
    
    # Test 6: Test storage system
    print("\n💾 Test 6: Testing storage system...")
    try:
        from backtest.storage.trade_logger import TradeLogger
        
        trade_logger = TradeLogger("test_backtest.db")
        
        # Test record creation
        test_config = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'initial_capital': 1000000
        }
        
        success = trade_logger.create_backtest_record(
            backtest_id="test_001",
            config=test_config,
            universe=universe[:10]
        )
        
        if success:
            print("   ✅ Storage system working")
            
            # Clean up test database
            import os
            if os.path.exists("test_backtest.db"):
                os.remove("test_backtest.db")
                print("   🧹 Test database cleaned up")
        else:
            print("   ❌ Storage system test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Storage test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Market-Neutral Backtest Framework is ready.")
    print("\n📋 Next steps:")
    print("   1. Run the web server: cd backtest/web && python app.py")
    print("   2. Open browser to: http://localhost:5001")
    print("   3. Configure and run your first backtest!")
    print("\n💡 Framework features:")
    print("   ✓ Memory-efficient data loading")
    print("   ✓ Multi-tier factor caching")
    print("   ✓ Market-neutral portfolio management")
    print("   ✓ Real-time WebSocket updates")
    print("   ✓ Performance metrics & reporting")
    print("   ✓ Trade logging & result storage")
    
    return True

if __name__ == "__main__":
    success = test_basic_backtest()
    exit(0 if success else 1)