#!/usr/bin/env python3
"""
Test script for OHLC Figure Factor

Simple test to validate the OHLC figure factor implementation
and integration with the existing framework.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add paths for imports
project_root = '/home/craz/crypto/crypto-trading'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'neural-strategy'))

def create_test_data():
    """Create synthetic OHLCV data for testing."""
    print("📊 Creating test data...")
    
    # Create more historical data (6 hours = 360 minutes) to support aggregation
    end_time = datetime.now().replace(second=0, microsecond=0)
    timestamps = [end_time - timedelta(minutes=i) for i in range(360)]
    timestamps.reverse()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    data_list = []
    for symbol in symbols:
        # Generate synthetic price data with some trend and volatility
        base_price = 50000 if symbol == 'BTCUSDT' else (3000 if symbol == 'ETHUSDT' else 1.5)
        
        # Create price series with random walk
        prices = [base_price]
        for i in range(1, len(timestamps)):
            # Add trend and noise
            trend = 0.0001 * i  # Small positive trend
            noise = np.random.normal(0, 0.01)
            price_change = prices[-1] * (trend + noise)
            new_price = max(0.01, prices[-1] + price_change)  # Ensure positive price
            prices.append(new_price)
        
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            
            # Generate realistic OHLC from price
            spread = price * 0.002  # 0.2% spread
            open_price = price + np.random.uniform(-spread, spread)
            close_price = price + np.random.uniform(-spread, spread) 
            
            high_price = max(open_price, close_price) + abs(np.random.normal(0, spread/2))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, spread/2))
            
            volume = np.random.uniform(100, 1000)
            
            data_list.append({
                'open_time': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    # Create MultiIndex DataFrame
    df = pd.DataFrame(data_list)
    df = df.set_index(['open_time', 'symbol']).sort_index()
    
    print(f"✅ Created test data: {len(df)} rows, {len(symbols)} symbols")
    print(f"📅 Time range: {timestamps[0]} to {timestamps[-1]}")
    
    return df

def test_ohlc_factor():
    """Test OHLC Figure Factor functionality."""
    print("\n🚀 Testing OHLC Figure Factor...")
    
    try:
        # Import the factor
        from strategies.factors import OHLCFigureFactor
        
        # Create test data
        data = create_test_data()
        
        # Initialize factor with debugging
        print("🔧 Initializing OHLC Figure Factor...")
        factor = OHLCFigureFactor(
            name="Test_OHLC_Factor",
            lookback_periods=20,
            timeframes=['3min', '15min', '1h']
        )
        
        # Get model info
        model_info = factor.get_model_info()
        print(f"📱 Model info: {model_info}")
        
        # Test with current timestamp (last timestamp in data)
        current_time = data.index.get_level_values('open_time').max()
        print(f"🕒 Testing at timestamp: {current_time}")
        
        # Check data availability at this timestamp
        current_data = data.loc[data.index.get_level_values('open_time') == current_time]
        symbols_at_time = current_data.index.get_level_values('symbol').unique().tolist()
        print(f"📊 Symbols available at timestamp: {symbols_at_time}")
        
        # Calculate factor scores
        print("🔢 Calculating factor scores...")
        scores = factor.calculate(data, current_time)
        
        print(f"✅ Factor calculation successful!")
        print(f"📊 Scores shape: {scores.shape}")
        print(f"📈 Factor scores:")
        if len(scores) > 0:
            for symbol, score in scores.items():
                print(f"  {symbol}: {score:.6f}")
        else:
            print("  (No scores generated)")
            
            # Debug: Try to understand why no scores
            print("\n🔍 Debugging empty scores...")
            print(f"  Data validation: {factor.validate_data(data, current_time)}")
            
            # Check individual symbol data
            for symbol in symbols_at_time:
                symbol_data = data.loc[data.index.get_level_values('symbol') == symbol]
                print(f"  {symbol}: {len(symbol_data)} data points")
            
            # Try a simpler timeframe
            simple_factor = OHLCFigureFactor(
                name="Simple_Test",
                lookback_periods=10,  # Reduced requirement
                timeframes=['15min']  # Single timeframe
            )
            simple_scores = simple_factor.calculate(data, current_time)
            print(f"  Simple factor scores: {simple_scores.shape}")
            if len(simple_scores) > 0:
                for symbol, score in simple_scores.items():
                    print(f"    {symbol}: {score:.6f}")
        
        # Test ranking functionality
        if len(scores) >= 2:
            print("\n🏆 Testing symbol ranking...")
            rankings = factor.rank_symbols(data, current_time, top_n=1, bottom_n=1)
            print(f"📈 Long candidates: {rankings['long']}")
            print(f"📉 Short candidates: {rankings['short']}")
        
        # Test cache functionality
        cache_info_before = factor.get_model_info()['cache_size']
        
        # Run again to test caching
        scores2 = factor.calculate(data, current_time)
        cache_info_after = factor.get_model_info()['cache_size']
        
        print(f"💾 Cache test - Before: {cache_info_before}, After: {cache_info_after}")
        
        # Clear cache
        factor.clear_cache()
        cache_info_cleared = factor.get_model_info()['cache_size']
        print(f"🧹 Cache cleared - Size: {cache_info_cleared}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the neural-strategy package is in the Python path")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ Model file error: {e}")
        print("💡 Make sure the trained model exists at the expected path")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print(f"📍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 OHLC Figure Factor Test Suite")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    success = test_ohlc_factor()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)