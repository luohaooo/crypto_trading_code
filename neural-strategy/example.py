"""
Neural Strategy Framework - Complete Example

This example demonstrates how to use the neural strategy framework
to run a complete backtest with factor-based neutral strategies.
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add paths for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Add neural-strategy to path  
neural_strategy_root = os.path.dirname(__file__)
sys.path.insert(0, neural_strategy_root)

# Core imports
from utils.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer, PerformanceVisualizer


def run_example_backtest():
    """
    Run a complete example backtest using the neural strategy framework.
    
    This example:
    1. Creates a default 4-hour returns-based neutral strategy
    2. Runs backtest on USDT pairs for 6 months
    3. Analyzes performance with comprehensive metrics
    4. Generates visualizations
    """
    
    print("🚀 Neural Strategy Framework - Example Backtest")
    print("=" * 80)
    
    # Step 1: Create configuration
    print("\n📋 Step 1: Creating backtest configuration...")
    
    config = BacktestConfig.create_default(
        symbol_list=None,  # Use all USDT symbols
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    
    # Customize configuration
    config.strategy.top_n = 5  # 5 long positions
    config.strategy.bottom_n = 5  # 5 short positions
    config.strategy.rebalance_frequency = '4h'  # Rebalance every 4 hours
    config.factor.lookback_periods = 240  # 4-hour lookback (240 minutes)
    config.initial_warmup_periods = 500  # Warmup period
    
    print(f"Configuration: {config}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"❌ Configuration errors: {errors}")
        return
    
    # Step 2: Initialize and run backtest
    print(f"\n⚙️ Step 2: Initializing backtest engine...")
    
    engine = BacktestEngine(config)
    
    try:
        # Run the complete backtest
        results = engine.run_backtest()
        
        # Print results summary
        engine.print_results_summary()
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return
    
    # Step 3: Advanced performance analysis
    print(f"\n📊 Step 3: Advanced performance analysis...")
    
    analyzer = PerformanceAnalyzer(results)
    
    # Generate comprehensive performance report
    performance_report = analyzer.create_performance_report()
    
    # Print advanced metrics
    advanced = performance_report['advanced_metrics']
    if 'error' not in advanced:
        print(f"\n🎯 Advanced Performance Metrics:")
        print(f"Sortino Ratio:          {advanced['sortino_ratio']:12.2f}")
        print(f"Calmar Ratio:           {advanced['calmar_ratio']:12.2f}")
        print(f"Hit Ratio:              {advanced['hit_ratio']*100:12.1f}%")
        print(f"Profit Factor:          {advanced['profit_factor']:12.2f}")
        print(f"VaR (95%):              {advanced['var_95']*100:12.2f}%")
        print(f"Max DD Duration:        {advanced['max_drawdown_duration']:12.0f} periods")
    
    # Factor effectiveness analysis
    factor_analysis = performance_report['factor_analysis']
    if 'error' not in factor_analysis:
        print(f"\n🔍 Factor Effectiveness Analysis:")
        print(f"Long Win Rate:          {factor_analysis['long_win_rate_pct']:12.1f}%")
        print(f"Short Win Rate:         {factor_analysis['short_win_rate_pct']:12.1f}%")
        print(f"Spread Return:          {factor_analysis['spread_return_pct']:12.2f}%")
        print(f"Factor Quality:         {factor_analysis['factor_effectiveness']:>12s}")
    
    # Step 4: Create visualizations
    print(f"\n📈 Step 4: Creating performance visualizations...")
    
    try:
        visualizer = PerformanceVisualizer(results)
        
        # Create all charts
        charts = visualizer.create_full_report(save_dir='neural-strategy/reports')
        
        print(f"✅ Generated {len(charts)} performance charts:")
        for chart_name in charts.keys():
            print(f"   - {chart_name}")
        
        # Show charts (optional - comment out if running headless)
        # visualizer.show_all_charts()
        
    except ImportError as e:
        print(f"⚠️  Could not create visualizations (missing matplotlib/seaborn): {e}")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
    
    print(f"\n🎉 Backtest completed successfully!")
    print(f"Results saved to neural-strategy/reports/")
    
    return results, performance_report


def run_momentum_strategy_example():
    """
    Run an example with momentum-based factor instead of returns.
    """
    
    print("🚀 Momentum Strategy Example")
    print("=" * 50)
    
    # Create momentum strategy configuration
    config = BacktestConfig.create_momentum_strategy(
        short_period=60,   # 1-hour short momentum
        long_period=240,   # 4-hour long momentum
        start_date='2024-01-01',
        end_date='2024-03-01'  # Shorter period for momentum
    )
    
    # Adjust strategy parameters
    config.strategy.top_n = 8
    config.strategy.bottom_n = 8
    config.strategy.rebalance_frequency = '2h'
    
    print(f"Momentum Strategy Config: {config.factor.name}")
    
    # Run backtest
    engine = BacktestEngine(config)
    
    try:
        results = engine.run_backtest()
        engine.print_results_summary()
        
        # Quick performance analysis
        analyzer = PerformanceAnalyzer(results)
        factor_analysis = analyzer.analyze_factor_effectiveness()
        
        print(f"\nMomentum Factor Analysis:")
        print(f"Spread Return: {factor_analysis.get('spread_return_pct', 0):.2f}%")
        print(f"Factor Quality: {factor_analysis.get('factor_effectiveness', 'Unknown')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Momentum strategy backtest failed: {e}")
        return None


def compare_strategies():
    """
    Compare different factor strategies side by side.
    """
    
    print("🚀 Strategy Comparison Example")
    print("=" * 50)
    
    strategies_config = [
        ('Returns_4H', BacktestConfig.create_default(start_date='2024-01-01', end_date='2024-04-01')),
        ('Momentum', BacktestConfig.create_momentum_strategy(start_date='2024-01-01', end_date='2024-04-01')),
        ('Vol_Adjusted', BacktestConfig.create_vol_adjusted_strategy(start_date='2024-01-01', end_date='2024-04-01'))
    ]
    
    results_comparison = {}
    
    for strategy_name, config in strategies_config:
        print(f"\nRunning {strategy_name} strategy...")
        
        try:
            engine = BacktestEngine(config)
            results = engine.run_backtest()
            
            results_comparison[strategy_name] = {
                'total_return': results['total_return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown_pct'],
                'win_rate': results['win_rate_pct']
            }
            
        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            continue
    
    # Print comparison
    print(f"\n📊 Strategy Comparison Results:")
    print(f"{'Strategy':<15} {'Return%':<10} {'Sharpe':<10} {'MaxDD%':<10} {'WinRate%':<10}")
    print("-" * 60)
    
    for strategy, metrics in results_comparison.items():
        print(f"{strategy:<15} {metrics['total_return']:<10.2f} {metrics['sharpe_ratio']:<10.2f} "
              f"{metrics['max_drawdown']:<10.2f} {metrics['win_rate']:<10.2f}")
    
    return results_comparison


if __name__ == "__main__":
    # Run different examples
    
    print("Choose example to run:")
    print("1. Complete backtest example (recommended)")
    print("2. Momentum strategy example")  
    print("3. Strategy comparison")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        run_example_backtest()
    elif choice == "2":
        run_momentum_strategy_example()
    elif choice == "3":
        compare_strategies()
    else:
        print("Running default complete backtest example...")
        run_example_backtest()