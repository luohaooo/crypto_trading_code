#!/usr/bin/env python3
"""
Simple test to verify the neural strategy framework works correctly.
"""

import sys
import os
from datetime import datetime

# Add paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

neural_strategy_root = os.path.dirname(__file__)
sys.path.insert(0, neural_strategy_root)

try:
    from utils.config import BacktestConfig
    from backtest.engine import BacktestEngine
    print("✅ All imports successful!")
    
    # Test configuration creation
    config = BacktestConfig.create_default(
        start_date='2024-01-01',
        end_date='2024-01-05'  # Very short period for testing
    )
    
    # Adjust for quick test
    config.strategy.top_n = 2
    config.strategy.bottom_n = 2
    config.factor.lookback_periods = 60  # Reduce lookback for testing
    config.initial_warmup_periods = 100
    
    print(f"✅ Configuration created: {config.strategy.name}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"❌ Configuration errors: {errors}")
        sys.exit(1)
    else:
        print("✅ Configuration validated successfully!")
    
    print("\n🎉 Neural Strategy Framework is ready to use!")
    print("Run `python example.py` to execute a full backtest example.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)