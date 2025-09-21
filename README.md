# Cryptocurrency Trading System

This project is a comprehensive cryptocurrency trading system featuring:

## Current Status
- ✅ **Price Visualization System** - Interactive charts with advanced navigation (COMPLETED)
- ✅ **Neural Strategy Framework** - OHLC figure-based neural network strategies (COMPLETED)
- ✅ **Model Training Pipeline** - PyTorch-based model training with checkpointing (COMPLETED)
- ✅ **Backtesting Engine** - Historical strategy performance analysis with factor-based approach (COMPLETED)
- ✅ **Strategy Factors** - OHLC figure factors and returns-based factors (COMPLETED)
- 🚧 **Web Interface for Backtesting** - User-friendly strategy testing interface (PLANNED)

## Quick Start

### Visualization Module
```bash
cd visualization/
python app.py
```
Then visit `http://localhost:5000`

### Neural Strategy Training
```bash
cd figure_model/
python train_ohlc_model.py
```

### Strategy Backtesting
```bash
cd neural-strategy/
python ohlc_backtest.py
```

## Project Structure

```
/home/craz/crypto/crypto-trading/
├── visualization/              # Price visualization module ✅
├── figure_model/               # Neural network model training ✅
│   ├── train_ohlc_model.py    # Main training script
│   ├── ohlc_model.py          # PyTorch model definitions
│   ├── training_dataset.py    # Dataset classes and data loading
│   ├── ohlc2fig.py           # OHLC to figure conversion
│   └── model_checkpoint/      # Training checkpoints
├── neural-strategy/            # Neural strategy implementation ✅
│   ├── strategies/            # Strategy classes
│   │   ├── neutral_strategy.py # Market neutral strategy
│   │   └── factors/           # Factor implementations
│   ├── backtest/              # Backtesting engine
│   │   └── engine.py         # Core backtesting logic
│   ├── ohlc_backtest.py      # Main backtest script
│   └── utils/                # Utilities (config, notifications)
├── utils/                      # Common data utilities ✅
│   └── data_loader.py         # Data loading functions
├── strategies/                 # Trading strategy modules 🚧
├── web/                        # Main web application 🚧
└── tests/                      # Test suites 🚧
```

## Features

### Completed Features

#### Visualization Module
- Interactive price and volume charts with dual Y-axes
- Time-based navigation with slider controls and keyboard shortcuts (A/D keys)
- Multiple chart types: Line, OHLC, and Candlestick with perfect alignment
- Data aggregation from 1-minute to 7-day intervals
- Zoom and pan functionality with Chart.js integration
- Responsive design with mobile-friendly controls

#### Neural Strategy Framework
- PyTorch-based OHLC figure neural network models
- Automated training pipeline with checkpointing and validation
- OHLC data to figure conversion with preprocessing
- Model artifacts management and versioning
- Support for MSE and L1 loss functions

#### Backtesting Engine
- Factor-based strategy implementation
- Market neutral strategies with long/short positioning
- Portfolio rebalancing with configurable frequency
- Performance metrics calculation (returns, Sharpe ratio, etc.)
- Factor effectiveness analysis and spread calculation
- DingTalk notification integration for alerts

#### Strategy Factors
- OHLC Figure Factor: Neural network predictions on price patterns
- Returns Factor: Historical return-based signals
- Extensible factor framework for custom implementations

### Planned Features
- **Advanced Strategy Development**: Enhanced framework for creating and testing trading strategies
- **Web Interface**: Unified dashboard for strategy management and result analysis
- **Real-time Trading**: Live strategy execution and monitoring
- **Advanced Analytics**: Portfolio tracking, P&L calculations, risk metrics

## Dependencies

### Core Dependencies
- **Python**: Flask, Pandas, datetime, glob, os
- **Machine Learning**: PyTorch, NumPy, scikit-learn
- **Visualization**: Chart.js with time adapters and zoom plugin
- **Data**: CSV files with OHLCV cryptocurrency data
- **Notifications**: DingTalk webhook integration

### Data Sources
- Minute-level OHLCV data stored in `/home/craz/crypto/crypto-data/future_data_2/`
- Model checkpoints and saved models in `figure_model/model_checkpoint/`
- Training datasets with OHLC figure preprocessing

## Documentation

See `CLAUDE.md` for detailed development guidance and architecture information.