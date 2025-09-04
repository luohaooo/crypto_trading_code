# Cryptocurrency Trading System

This project is a comprehensive cryptocurrency trading system featuring:

## Current Status
- ✅ **Price Visualization System** - Interactive charts with advanced navigation (COMPLETED)
- 🚧 **Trading Strategy Development** - Strategy creation and testing framework (PLANNED)
- 🚧 **Backtesting Engine** - Historical strategy performance analysis (PLANNED)
- 🚧 **Web Interface for Backtesting** - User-friendly strategy testing interface (PLANNED)

## Quick Start

### Visualization Module (Currently Available)
```bash
cd visualization/
python app.py
```
Then visit `http://localhost:5000`

## Project Structure

```
/home/craz/crypto/crypto-trading/
├── visualization/              # Price visualization module ✅
├── strategies/                 # Trading strategy modules 🚧
├── backtest/                   # Backtesting engine 🚧
├── web/                        # Main web application 🚧
├── data/                       # Data handling utilities 🚧
├── utils/                      # Common utilities 🚧
└── tests/                      # Test suites 🚧
```

## Features

### Completed Features (Visualization)
- Interactive price and volume charts with dual Y-axes
- Time-based navigation with slider controls and keyboard shortcuts (A/D keys)
- Multiple chart types: Line, OHLC, and Candlestick with perfect alignment
- Data aggregation from 1-minute to 7-day intervals
- Zoom and pan functionality with Chart.js integration
- Responsive design with mobile-friendly controls

### Planned Features
- **Strategy Development**: Framework for creating and testing trading strategies
- **Backtesting Engine**: Historical performance testing with realistic trading simulation
- **Unified Web Interface**: Dashboard for strategy management and result analysis
- **Performance Metrics**: Sharpe ratio, drawdown, risk metrics, and more

## Dependencies

- Python (Flask, Pandas, datetime, glob, os)
- Chart.js with time adapters and zoom plugin
- CSV data files with OHLCV cryptocurrency data

## Documentation

See `CLAUDE.md` for detailed development guidance and architecture information.