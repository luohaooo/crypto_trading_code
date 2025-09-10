# Neutral Strategy Backtesting Framework

A comprehensive implementation of the neutral strategy backtesting system as specified in the requirements. This framework provides factor-based ranking, market-neutral execution, and detailed performance analysis.

## 🚀 Quick Start

```python
# Run a complete example backtest
cd neural-strategy
python example.py
```

## 📁 Project Structure

```
neural-strategy/
├── strategies/                 # Strategy implementations
│   ├── factors/               # Factor calculation modules
│   │   ├── base_factor.py     # Abstract factor base class
│   │   └── returns_factor.py  # Returns-based factors
│   ├── base_strategy.py       # Abstract strategy base class
│   └── neutral_strategy.py    # Market-neutral strategy implementation
├── backtest/                  # Backtesting engine
│   ├── engine.py             # Core backtesting logic
│   └── performance.py        # Advanced performance analysis
├── utils/                     # Configuration and utilities
│   └── config.py             # Configuration management
└── example.py                # Usage examples
```

## 🎯 Strategy Overview

The neutral strategy implementation follows these mechanics:

1. **Factor Calculation**: Calculate factor scores for all symbols using configurable factors (returns, momentum, volatility-adjusted)
2. **Symbol Ranking**: Rank symbols by factor scores (high = long candidates, low = short candidates)  
3. **Position Allocation**: 
   - 50% capital → Top N symbols (long positions)
   - 50% capital → Bottom N symbols (short positions)
4. **Rebalancing**: Close all positions and reopen based on new rankings at specified intervals
5. **Performance Tracking**: Comprehensive P&L tracking, trade analysis, and risk metrics

## ⚙️ Key Features

### 🧮 Factor System
- **Returns Factor**: Price returns over configurable lookback periods
- **Momentum Factor**: Combined short-term and long-term momentum
- **Volatility-Adjusted Returns**: Sharpe-like ratio for consistent directional movement
- **Extensible**: Easy to add custom factors by extending `BaseFactor`

### 📊 Strategy Framework
- **Market-Neutral Execution**: Systematic long/short portfolio construction
- **Flexible Configuration**: Adjustable position counts, rebalancing frequency, commission rates
- **Portfolio Management**: Position tracking, P&L calculation, margin management
- **Risk Controls**: Cash management, position sizing, transaction cost modeling

### 🔧 Backtesting Engine
- **Time Window Iteration**: Systematic rebalancing across historical periods
- **Data Integration**: Uses existing `utils/data_loader.py` for efficient multi-symbol data loading
- **Performance Tracking**: Real-time equity curve, trade logging, execution monitoring
- **Error Handling**: Robust error handling for missing data and execution failures

### 📈 Performance Analytics
- **Basic Metrics**: Total return, Sharpe ratio, maximum drawdown, win rate
- **Advanced Metrics**: Sortino ratio, Calmar ratio, VaR, skewness, kurtosis
- **Factor Analysis**: Long/short basket performance, factor effectiveness measurement
- **Visualizations**: Equity curves, return distributions, trade analysis charts

## 🛠️ Usage Examples

### Basic Returns Strategy
```python
from utils.config import BacktestConfig
from backtest.engine import BacktestEngine

# Create default 4-hour returns strategy
config = BacktestConfig.create_default(
    start_date='2024-01-01',
    end_date='2024-06-01'
)

# Customize parameters
config.strategy.top_n = 10       # 10 long positions
config.strategy.bottom_n = 10    # 10 short positions  
config.strategy.rebalance_frequency = '4h'
config.factor.lookback_periods = 240  # 4-hour returns

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()
engine.print_results_summary()
```

### Momentum Strategy
```python
# Create momentum-based strategy
config = BacktestConfig.create_momentum_strategy(
    short_period=60,   # 1-hour momentum
    long_period=240,   # 4-hour momentum
    start_date='2024-01-01',
    end_date='2024-06-01'
)

engine = BacktestEngine(config)
results = engine.run_backtest()
```

### Advanced Performance Analysis
```python
from backtest.performance import PerformanceAnalyzer, PerformanceVisualizer

# Detailed performance analysis
analyzer = PerformanceAnalyzer(results)
performance_report = analyzer.create_performance_report()

# Advanced metrics
advanced_metrics = analyzer.calculate_advanced_metrics()
print(f"Sortino Ratio: {advanced_metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {advanced_metrics['calmar_ratio']:.2f}")

# Factor effectiveness
factor_analysis = analyzer.analyze_factor_effectiveness()
print(f"Long vs Short Spread: {factor_analysis['spread_return_pct']:.2f}%")

# Create visualizations
visualizer = PerformanceVisualizer(results)
charts = visualizer.create_full_report(save_dir='reports/')
```

## 🔧 Configuration Options

### Strategy Configuration
```python
strategy_config = StrategyConfig(
    name='MyNeutralStrategy',
    initial_capital=1000000.0,    # Starting capital
    commission_rate=0.001,        # 0.1% commission
    top_n=10,                     # Number of long positions
    bottom_n=10,                  # Number of short positions  
    rebalance_frequency='4h'      # Rebalancing interval
)
```

### Factor Configuration
```python
factor_config = FactorConfig(
    name='Returns_4H',
    factor_type='returns',        # 'returns', 'momentum', 'volatility_adjusted'
    lookback_periods=240,         # Lookback period in minutes
    params={'custom_param': 1.0}  # Factor-specific parameters
)
```

### Data Configuration
```python
data_config = DataConfig(
    symbols=None,                 # None for all USDT symbols
    start_date='2024-01-01',
    end_date='2024-06-01', 
    symbol_filter='usdt',         # 'usdt', 'all', or custom list
    min_data_points=1000          # Minimum data points per symbol
)
```

## 📊 Performance Metrics

### Basic Metrics
- Total Return, Annualized Return
- Volatility, Sharpe Ratio
- Maximum Drawdown
- Win Rate, Average Trade P&L

### Advanced Risk Metrics  
- Sortino Ratio (downside deviation)
- Calmar Ratio (return/max drawdown)
- Value at Risk (95%, 99%)
- Skewness, Kurtosis
- Hit Ratio, Profit Factor

### Factor Analysis
- Long vs Short basket performance
- Factor-return correlation
- Spread return analysis
- Factor effectiveness rating

## 🎯 Integration with Existing System

This framework seamlessly integrates with the existing crypto-trading project:

- **Data Loading**: Uses `utils/data_loader.py` with `get_multiple_symbols_data()` and `get_usdt_symbols()`  
- **MultiIndex Structure**: Leverages (open_time, symbol) indexing for efficient querying
- **Visualization**: Can integrate with existing Chart.js visualization system
- **Modular Design**: Follows CLAUDE.md planned architecture for easy extension

## 🔬 Testing and Validation

The framework includes comprehensive validation:

- Configuration validation with detailed error messages
- Data availability checks before factor calculation
- Position sizing validation and cash management
- Execution monitoring with success/failure tracking
- Performance metric calculation with error handling

## 🚀 Next Steps

1. **Run Examples**: Start with `python example.py` to see the framework in action
2. **Custom Factors**: Extend `BaseFactor` to create domain-specific factors
3. **Strategy Variants**: Modify `NeutralStrategy` for different allocation schemes  
4. **Web Integration**: Connect with existing visualization system for interactive results
5. **Real-time Trading**: Extend for live trading with real-time data feeds

## 📈 Expected Results

Based on the implementation, you can expect:

- **Professional-grade backtesting** with institutional-level performance metrics
- **Factor effectiveness analysis** to validate trading hypotheses  
- **Comprehensive risk analysis** including drawdown periods and tail risk
- **Extensible framework** for testing multiple factors and strategies
- **Production-ready code** with proper error handling and logging

The framework successfully implements all requirements from the neutral strategy specification while providing a robust foundation for advanced quantitative trading research.