# Factor-Based Backtesting System - User Guide

## Overview

This backtesting system implements a **long-short rebalancing strategy** based on factor signals. It provides:

- ✅ Periodic rebalancing with configurable intervals
- ✅ Batch processing with parameter grid search
- ✅ Comprehensive performance metrics
- ✅ Detailed logging of each rebalance period
- ✅ Full period and monthly PnL visualization
- ✅ Memory-efficient monthly data loading

## Quick Start

### Basic Usage

```python
from run_backtest import batch_backtest

# Run backtest for a single factor
results = batch_backtest(
    factor_name='precious_ohlc_cnn',           # Factor directory name
    start_time='2023-01-01 00:00:00',          # Start timestamp
    end_time='2023-03-31 23:00:00',            # End timestamp
    rebalance_hours_list=[6, 12, 24],          # Test 6h, 12h, 24h rebalancing
    top_n_list=[5, 10, 15],                    # Long top 5, 10, or 15 symbols
    bottom_n_list=[5, 10, 15],                 # Short bottom 5, 10, or 15 symbols
    output_base_dir='./factor_report'          # Output directory
)
```

### Advanced Usage

```python
# Custom parameters with single configuration
from run_backtest import (
    load_factor_data,
    load_returns_data,
    run_single_backtest,
    calculate_metrics,
    plot_pnl_curve
)

# Load data
factor_df = load_factor_data('my_factor', '2023-01', '2023-12')
returns_df = load_returns_data('2023-01', '2023-12')

# Run single backtest
pnl_series, logs = run_single_backtest(
    factor_df=factor_df,
    returns_df=returns_df,
    rebalance_hours=12,
    top_n=10,
    bottom_n=10,
    start_time='2023-01-01 00:00:00',
    end_time='2023-12-31 23:00:00',
    log_file='./my_backtest_log.txt',
    fee_rate=0.002  # 0.2% transaction fee
)

# Calculate metrics
metrics = calculate_metrics(pnl_series, rebalance_hours=12)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Annual Return: {metrics['annual_return']*100:.2f}%")

# Plot PnL curve
plot_pnl_curve(pnl_series, 'my_pnl_curve.png', title='Custom Backtest')
```

## Strategy Explanation

### Trading Logic

The system implements a **market-neutral long-short strategy**:

1. **Factor Ranking**: At each rebalance timestamp, rank all symbols by factor value
2. **Position Selection**:
   - **Long**: Select top-n symbols with highest factor values
   - **Short**: Select bottom-n symbols with lowest factor values
3. **Capital Allocation**:
   - 50% of capital → Long positions (equally weighted among top-n)
   - 50% of capital → Short positions (equally weighted among bottom-n)
4. **Holding Period**: Hold positions for `rebalance_hours` (e.g., 12 hours)
5. **Rebalancing**: At next timestamp, close all positions and repeat

### Return Calculation

For each rebalancing period:

```
Period Return = (Mean Return of Long Symbols - Mean Return of Short Symbols) / 2 - Fee Rate
```

Example with 12-hour rebalancing:
- At 12:00, select top-5 symbols for long and bottom-5 for short
- Hold until 00:00 (next day)
- Use `return_12h` column from returns data
- Calculate: `(avg_long_return - avg_short_return) / 2 - 0.002`

### PnL Accumulation

```
Initial PnL = 1.0
PnL(t) = PnL(t-1) × (1 + Period_Return(t))
```

## Data Requirements

### Factor Data

**Location**: `./factor_data/{factor_name}/factor_YYYY-MM.pkl`

**Format**:
```python
# MultiIndex DataFrame
Index: (open_time, symbol)
Columns: ['factor_value']

Example:
open_time            symbol
2023-01-01 00:00:00  BTCUSDT     0.8523
                     ETHUSDT     0.7234
2023-01-01 01:00:00  BTCUSDT     0.8611
                     ETHUSDT     0.7189
```

### Returns Data

**Location**: `../../crypto-data/future_returns/future_returns_YYYY-MM.pkl`

**Format**:
```python
# MultiIndex DataFrame
Index: (open_time, symbol)
Columns: ['return_1h', 'return_2h', ..., 'return_144h']

Example:
open_time            symbol     return_1h  return_12h  return_24h
2023-01-01 00:00:00  BTCUSDT    0.0123     0.0456      0.0789
                     ETHUSDT    0.0098     0.0234      0.0512
```

## Output Structure

For each parameter combination, the system creates:

```
./factor_report/
  └── {factor_name}/
      └── {start_date}_{end_date}/
          └── {rebalance_hours}h_top{top_n}_bottom{bottom_n}/
              ├── backtest_log.txt          # Detailed period-by-period log
              ├── pnl_series.pkl            # PnL time series data
              ├── metrics.json              # Performance metrics
              ├── pnl_curve_full.png        # Full period PnL chart
              ├── pnl_curve_2023-01.png     # Monthly PnL charts
              ├── pnl_curve_2023-02.png
              └── ...
```

## Output Files Explained

### 1. backtest_log.txt

Detailed log of each rebalance period:

```
Rebalance #1: 2023-01-01 12:00:00
────────────────────────────────────────────────────────────────

Long Positions (Top 5):
  BTCUSDT    : factor=  0.8523, return= 0.0234 (  2.34%)
  ETHUSDT    : factor=  0.7891, return= 0.0189 (  1.89%)
  ...

Short Positions (Bottom 5):
  XRPUSDT    : factor= -0.4521, return=-0.0156 ( -1.56%)
  ...

Long Mean Return:   0.0211 (  2.11%)
Short Mean Return: -0.0134 ( -1.34%)
Period Return:      0.0153 (  1.53%) [after 0.20% fee]
Current PnL:        1.0153
```

### 2. metrics.json

Performance metrics in JSON format:

```json
{
  "total_return": 0.2345,
  "annual_return": 0.8912,
  "sharpe_ratio": 2.134,
  "calmar_ratio": 4.567,
  "max_drawdown": -0.1234,
  "win_rate": 0.6234,
  "profit_loss_ratio": 1.789,
  "num_periods": 180,
  "num_winning_periods": 112,
  "num_losing_periods": 68,
  "time_span_hours": 2160.0,
  "time_span_days": 90.0,
  "rebalance_hours": 12
}
```

### 3. PnL Curves

- **pnl_curve_full.png**: Complete backtest period with drawdown subplot
- **pnl_curve_YYYY-MM.png**: Individual monthly performance

## Performance Metrics Explained

### Sharpe Ratio
```
Sharpe = (Mean Period Return / Std Dev of Returns) × √(Periods per Year)
```
- Measures risk-adjusted return
- Higher is better (> 1.0 is good, > 2.0 is excellent)

### Calmar Ratio
```
Calmar = Annual Return / |Max Drawdown|
```
- Return per unit of maximum risk
- Higher is better

### Annual Return
```
Annual Return = (Final PnL / Initial PnL) ^ (1 / Years) - 1
```
- Annualized return rate
- Expressed as decimal (0.30 = 30% per year)

### Max Drawdown
```
Max Drawdown = Min((Current PnL - Peak PnL) / Peak PnL)
```
- Maximum peak-to-trough decline
- Negative value (e.g., -0.15 = -15% drawdown)

### Win Rate
```
Win Rate = Number of Winning Periods / Total Periods
```
- Percentage of periods with positive returns
- Range: 0.0 to 1.0 (0.60 = 60% win rate)

### Profit/Loss Ratio
```
P/L Ratio = Mean Winning Return / |Mean Losing Return|
```
- Average win size relative to average loss
- Higher is better (> 1.0 means wins are larger than losses)

## Parameter Selection Guide

### Rebalance Hours

- **Short (4-8h)**: More trades, higher fees, captures short-term signals
- **Medium (12-24h)**: Balanced approach
- **Long (48-168h)**: Lower fees, captures longer trends

**Recommendation**: Test [6, 12, 24] initially

### Top-N and Bottom-N

- **Small (3-5)**: Concentrated positions, higher volatility
- **Medium (10-15)**: Balanced diversification
- **Large (20-30)**: Lower volatility, requires many symbols

**Recommendation**: Match to factor strength and symbol universe size

### Selection Strategy

1. Start with medium values: `rebalance_hours=12, top_n=10, bottom_n=10`
2. Run parameter sweep to find optimal combination
3. Analyze sensitivity to parameters
4. Select based on Sharpe ratio and max drawdown tolerance

## Common Issues & Solutions

### Issue: "No factor data found"
**Solution**: Ensure factor data exists in `./factor_data/{factor_name}/factor_YYYY-MM.pkl`

### Issue: "Return column not found"
**Solution**: The return column name must match rebalance hours. For 12h rebalancing, ensure `return_12h` exists in returns data.

### Issue: "Insufficient symbols"
**Solution**: Reduce `top_n` and `bottom_n` values, or ensure more symbols have valid factor values.

### Issue: "Pickle protocol error"
**Solution**: The code includes fallback loading for Python 3.7 compatibility. If still failing, regenerate pickle files with compatible protocol.

## Best Practices

1. **Start Small**: Test with 1-2 months before full backtest
2. **Validate Results**: Check log files for sanity (symbols selected, returns match expectations)
3. **Multiple Timeframes**: Test different rebalance periods
4. **Parameter Sweep**: Don't optimize on single parameter set
5. **Out-of-Sample**: Reserve recent period for validation
6. **Factor Analysis**: Check factor correlation with returns
7. **Fee Sensitivity**: Test different fee rates

## Example Workflows

### Workflow 1: Quick Test

```python
# Test single month with one parameter set
results = batch_backtest(
    factor_name='precious_ohlc_cnn',
    start_time='2023-01-01 00:00:00',
    end_time='2023-01-31 23:00:00',
    rebalance_hours_list=[12],
    top_n_list=[10],
    bottom_n_list=[10]
)
```

### Workflow 2: Parameter Sweep

```python
# Full parameter grid search
results = batch_backtest(
    factor_name='precious_ohlc_cnn',
    start_time='2023-01-01 00:00:00',
    end_time='2023-12-31 23:00:00',
    rebalance_hours_list=[6, 12, 24, 48],
    top_n_list=[5, 10, 15, 20],
    bottom_n_list=[5, 10, 15, 20]
)

# 4 × 4 × 4 = 64 combinations tested
```

### Workflow 3: Multiple Factors

```python
factors = ['precious_ohlc_cnn', 'momentum_factor', 'volatility_factor']

for factor in factors:
    results = batch_backtest(
        factor_name=factor,
        start_time='2023-01-01 00:00:00',
        end_time='2023-12-31 23:00:00',
        rebalance_hours_list=[12, 24],
        top_n_list=[10, 15],
        bottom_n_list=[10, 15]
    )
```

## Interpreting Results

### Good Performance Indicators

- ✅ Sharpe Ratio > 1.5
- ✅ Calmar Ratio > 2.0
- ✅ Win Rate > 55%
- ✅ P/L Ratio > 1.2
- ✅ Max Drawdown < -20%
- ✅ Consistent monthly returns

### Red Flags

- ❌ Sharpe Ratio < 0.5
- ❌ Max Drawdown < -40%
- ❌ Win Rate < 45%
- ❌ P/L Ratio < 0.8
- ❌ Irregular PnL curve (jumps/gaps)
- ❌ Single large winning period

### Next Steps After Backtest

1. **Review Logs**: Check symbol selection makes sense
2. **Analyze Drawdowns**: Identify what caused maximum drawdown
3. **Factor Validation**: Correlate factor values with actual returns
4. **Robustness**: Test on different time periods
5. **Live Simulation**: Paper trade before real capital

## Contact & Support

For questions about the backtesting system:
- Check log files first
- Review data format requirements
- Validate input parameters
- Test with small date ranges first