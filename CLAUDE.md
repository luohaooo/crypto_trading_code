# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive cryptocurrency trading project featuring:
1. **Price Visualization System** - Interactive charts with advanced navigation (✅ COMPLETED)
2. **Neural Strategy Framework** - OHLC figure-based neural network strategies (✅ COMPLETED)
3. **Model Training Pipeline** - PyTorch-based model training with checkpointing (✅ COMPLETED)
4. **Backtesting Engine** - Historical strategy performance analysis with factor-based approach (✅ COMPLETED)
5. **Strategy Factors** - OHLC figure factors and returns-based factors (✅ COMPLETED)
6. **Web Interface for Backtesting** - User-friendly strategy testing interface (🚧 PLANNED)

## Project Structure

```
/home/craz/crypto/crypto-trading/
├── visualization/              # Price visualization module (completed)
│   ├── app.py                 # Flask app for price charts
│   ├── templates/
│   │   └── index.html         # Visualization interface
│   └── static/                # CSS/JS assets (if needed)
├── figure_model/               # Neural network model training (completed)
│   ├── train_ohlc_model.py    # Main training script
│   ├── ohlc_model.py          # PyTorch model definitions
│   ├── training_dataset.py    # Dataset classes and data loading
│   ├── ohlc2fig.py           # OHLC to figure conversion
│   ├── ohlc_preprocessor.py   # Data preprocessing utilities
│   ├── generate_dataset_training.ipynb # Dataset generation notebook
│   ├── model_checkpoint/      # Training checkpoints by timestamp
│   └── model_saved/           # Final trained models
├── neural-strategy/            # Neural strategy implementation (completed)
│   ├── strategies/            # Strategy classes
│   │   ├── __init__.py
│   │   ├── base_strategy.py   # Base strategy class
│   │   ├── neutral_strategy.py # Market neutral strategy implementation
│   │   └── factors/           # Factor implementations
│   │       ├── __init__.py
│   │       ├── base_factor.py # Base factor class
│   │       ├── ohlc_figure_factor.py # Neural network factor
│   │       └── returns_factor.py # Returns-based factor
│   ├── backtest/              # Backtesting engine
│   │   ├── __init__.py
│   │   └── engine.py         # Core backtesting logic
│   ├── ohlc_backtest.py      # Main backtest script
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       └── dingding.py       # DingTalk notifications
├── utils/                      # Common data utilities (completed)
│   ├── __init__.py
│   └── data_loader.py         # Data loading functions
├── strategies/                 # Trading strategy modules (planned)
│   ├── __init__.py
│   ├── base_strategy.py       # Base strategy class
│   ├── ma_crossover.py        # Moving average strategy example
│   └── custom/                # User-defined strategies
├── backtest/                   # Backtesting engine (planned)
│   ├── __init__.py
│   ├── engine.py              # Core backtesting logic
│   ├── metrics.py             # Performance calculations
│   └── reports.py             # Result analysis
├── web/                        # Main web application (planned)
│   ├── app.py                 # Main Flask application
│   ├── templates/
│   │   ├── dashboard.html     # Main dashboard
│   │   ├── visualization.html # Price charts (embedded)
│   │   ├── strategy.html      # Strategy configuration
│   │   └── backtest.html      # Backtest results
│   └── static/                # Web assets
├── data/                       # Data handling utilities (planned)
│   ├── __init__.py
│   ├── loader.py              # Data loading functions
│   └── validator.py           # Data validation
└── tests/                      # Test suites (planned)
    ├── test_strategies.py
    ├── test_backtest.py
    └── test_visualization.py
```

## Current Implementation

### Completed Modules

#### Visualization Module
- **Backend**: Flask application serving cryptocurrency price data from CSV files
- **Frontend**: Interactive web interface with Chart.js-based visualizations
- **Data Source**: Minute-level OHLCV data stored in `/home/craz/crypto/crypto-data/future_data_2/`
- **Documentation**: Complete implementation guide in `price_visualize.md`

#### Neural Strategy Framework
- **Model Training**: PyTorch-based neural network training pipeline
- **Data Processing**: OHLC to figure conversion with preprocessing
- **Model Management**: Automated checkpointing, validation, and model saving
- **Architecture**: Support for various loss functions (MSE, L1) and model architectures

#### Backtesting Engine
- **Factor-Based Strategy**: Extensible factor framework for strategy development
- **Market Neutral Strategy**: Long/short positioning based on factor signals
- **Performance Analysis**: Returns calculation, Sharpe ratio, factor effectiveness
- **Portfolio Management**: Automated rebalancing with configurable frequency

#### Strategy Factors
- **OHLC Figure Factor**: Neural network predictions on OHLC price patterns
- **Returns Factor**: Historical return-based momentum signals
- **Base Factor Class**: Extensible framework for custom factor implementations

### Key Features Implemented

#### Visualization Features
- Interactive price and volume charts with dual Y-axes
- Time-based navigation with slider controls and keyboard shortcuts (A/D keys)
- Multiple chart types: Line view (purple close prices), OHLC view (color-coded O/H/L/C lines), and Candlestick view (green/red candles)
- Data aggregation from 1-minute to 7-day intervals
- Zoom and pan functionality with Chart.js integration
- Responsive design with mobile-friendly controls
- Perfect alignment between candlestick and volume bars

#### Neural Strategy Features
- PyTorch model training with automatic checkpointing
- OHLC data preprocessing and figure conversion
- Model validation and performance tracking
- Support for multiple loss functions and optimizers
- Timestamped model checkpoint management
- Model artifact versioning and storage

#### Backtesting Features
- Factor-based strategy implementation with extensible framework
- Market neutral portfolio construction (50% long, 50% short)
- Automated portfolio rebalancing (daily, weekly, monthly options)
- Performance metrics: total return, annualized return, Sharpe ratio
- Factor effectiveness analysis with spread calculation
- DingTalk notification integration for alerts and results

## Development Setup

**Current Stack**: Python + Flask + Chart.js + PyTorch

### Running the Applications

#### Visualization Module
```bash
cd visualization/
python app.py
```
- Server runs on `http://localhost:5000`
- Debug mode enabled for development
- Automatic reloading on code changes

#### Neural Strategy Training
```bash
cd figure_model/
python train_ohlc_model.py
```
- Trains OHLC figure-based neural network models
- Automatic checkpointing and validation
- Model artifacts saved to `model_checkpoint/` and `model_saved/`

#### Strategy Backtesting
```bash
cd neural-strategy/
python ohlc_backtest.py
```
- Runs factor-based backtesting with neural network predictions
- Configurable strategy parameters and rebalancing frequency
- Results output with performance metrics and factor analysis

### Dependencies
- **Python Core**: Flask, Pandas, datetime, glob, os, pickle
- **Machine Learning**: PyTorch, NumPy, scikit-learn
- **Frontend**: Chart.js, chartjs-adapter-date-fns, chartjs-plugin-zoom
- **Data**: CSV files with OHLCV cryptocurrency data
- **Notifications**: DingTalk webhook integration

### Data Sources
- **Price Data**: Minute-level OHLCV data in `/home/craz/crypto/crypto-data/future_data_2/`
- **Model Checkpoints**: Training checkpoints in `figure_model/model_checkpoint/`
- **Saved Models**: Final trained models in `figure_model/model_saved/`
- **Cache**: Preprocessed data cache in `/home/craz/crypto/crypto-data/pickle_cache/`

### Testing the Applications

#### Testing Visualization System
1. Start the Flask server: `cd visualization/ && python app.py`
2. Navigate to `http://localhost:5000`
3. Select a cryptocurrency symbol (BTCUSDT, ETHUSDT, etc.)
4. Choose date range and load data
5. Test navigation: slider, pan buttons (⬅️➡️), keyboard (A/D keys)
6. Toggle between Line, OHLC, and Candlestick chart types
7. Test data aggregation (1m to 7d intervals)

#### Testing Neural Strategy Training
1. Run training: `cd figure_model/ && python train_ohlc_model.py`
2. Monitor training progress and validation metrics
3. Check model checkpoints in `model_checkpoint/[timestamp]/`
4. Verify final model saved in `model_saved/`

#### Testing Strategy Backtesting
1. Ensure trained models are available in `figure_model/model_saved/`
2. Run backtest: `cd neural-strategy/ && python ohlc_backtest.py`
3. Review performance metrics and factor analysis results
4. Check DingTalk notifications (if configured)

## Completed & Planned Modules

### Completed Implementation

#### 1. Neural Strategy Framework ✅
- **Purpose**: OHLC figure-based neural network trading strategies
- **Components**:
  - PyTorch model training pipeline with automatic checkpointing
  - OHLC to figure conversion and preprocessing
  - Model validation and performance tracking
  - Timestamped checkpoint management and model versioning
- **Integration**: Connects with backtesting engine via OHLC Figure Factor

#### 2. Backtesting Engine ✅
- **Purpose**: Historical strategy performance testing with factor-based approach
- **Components**:
  - Core backtesting logic with realistic trading simulation
  - Factor-based strategy framework (neutral strategy implemented)
  - Performance metrics calculation (returns, Sharpe ratio, factor effectiveness)
  - Portfolio rebalancing with configurable frequency
  - DingTalk notification integration
- **Integration**: Uses neural network models via factor system

#### 3. Strategy Factors ✅
- **Purpose**: Modular factor-based signal generation
- **Components**:
  - Base factor class with standardized interface
  - OHLC Figure Factor using neural network predictions
  - Returns Factor for momentum-based signals
  - Extensible framework for custom factor implementations
- **Integration**: Plugs into backtesting engine for strategy execution

### Planned Implementation

#### 1. Advanced Trading Strategies Module 🚧
- **Purpose**: Enhanced strategy framework beyond neural factors
- **Components**:
  - Traditional technical indicator strategies (MA crossover, RSI, etc.)
  - Multi-factor strategy combinations
  - Strategy parameter optimization
  - Risk management and position sizing
- **Integration**: Will extend current factor-based approach

#### 2. Main Web Application 🚧
- **Purpose**: Unified interface for all modules
- **Components**:
  - Dashboard with portfolio overview and performance metrics
  - Strategy configuration interface for neural and traditional strategies
  - Integrated visualization (embedded from current module)
  - Backtest result analysis and comparison tools
  - Model training monitoring and management interface
- **Integration**: Will orchestrate all existing modules (visualization, neural strategy, backtesting)

## Architecture Notes

### Current Architecture

#### Visualization Layer
- **Data Layer**: CSV file-based storage with Pandas processing
- **API Layer**: Flask REST endpoints for symbols and OHLC data
- **Visualization Layer**: Chart.js with custom navigation controls
- **User Interface**: Single-page web application with responsive design

#### Neural Strategy Layer
- **Data Processing**: OHLC to figure conversion with preprocessing pipeline
- **Model Training**: PyTorch-based neural network training with checkpointing
- **Model Management**: Timestamped checkpoints and versioned model artifacts
- **Integration**: Factor-based interface for backtesting integration

#### Backtesting Layer
- **Strategy Framework**: Factor-based strategy implementation
- **Execution Engine**: Market neutral portfolio construction and rebalancing
- **Performance Analysis**: Comprehensive metrics and factor effectiveness analysis
- **Notification System**: DingTalk integration for alerts and results

### Planned Architecture (Full System)
- **Modular Design**: Separate modules for visualization, neural strategies, and backtesting (✅ implemented)
- **Shared Data Layer**: Common data access patterns across modules
- **API-First**: RESTful APIs for inter-module communication
- **Plugin System**: Extensible factor and strategy framework (✅ factor framework implemented)
- **Model Management**: Centralized model training, versioning, and deployment (✅ implemented)
- **Unified Interface**: Web dashboard integrating all components

### Key Components

#### Data Management
- **Price Data Processing**: Real-time aggregation and time-series manipulation
- **OHLC Figure Conversion**: Preprocessing pipeline for neural network input
- **Cache Management**: Pickle-based caching for preprocessed data
- **Model Artifacts**: Organized storage of training checkpoints and final models

#### User Interface
- **Navigation System**: Slider-based time window navigation (shows 20% of data at once)
- **Chart Management**: Dual-axis charts (price + volume) with synchronized navigation
- **Interactive Controls**: Keyboard shortcuts, zoom/pan, chart type switching

#### Strategy Execution
- **Factor Framework**: Modular signal generation with extensible base classes
- **Portfolio Management**: Market neutral construction with automated rebalancing
- **Performance Tracking**: Real-time metrics calculation and factor analysis

#### Error Handling & Monitoring
- **Robust Error Handling**: Comprehensive error handling for missing data and navigation edge cases
- **Training Monitoring**: Automatic checkpointing with validation loss tracking
- **Notification System**: DingTalk alerts for training completion and backtest results

### Performance Considerations

#### Visualization Performance
- No data sampling - preserves complete dataset integrity
- Client-side aggregation for smooth user interactions
- Optimized chart rendering with minimal point radius for large datasets
- Memory-efficient data structures for time-based navigation

#### Training Performance
- Automatic model checkpointing to prevent training loss
- GPU acceleration support for PyTorch models
- Efficient data loading with batch processing
- Cached preprocessing to avoid redundant computations

#### Backtesting Performance
- Vectorized operations for portfolio calculations
- Efficient factor computation with minimal data copying
- Configurable rebalancing frequency to balance accuracy and speed
- Memory-efficient factor storage and retrieval

## Color Scheme & Visual Design

**Line Mode:**
- Close Price: Purple (`rgba(147, 51, 234)`)
- Volume: Orange (`rgba(251, 146, 60)`)

**OHLC Mode:**
- Open: Yellow (`rgba(255, 206, 86)`)
- High: Green (`rgba(34, 197, 94)`)
- Low: Red (`rgba(239, 68, 68)`)
- Close: Purple (consistent with line mode)
- Volume: Orange (consistent with line mode)

**Candlestick Mode:**
- Bullish Candles: Green (`rgba(34, 197, 94)`) - when close >= open
- Bearish Candles: Red (`rgba(239, 68, 68)`) - when close < open
- Volume: Orange (consistent across all modes)

## Security Considerations

- All data is read-only from local CSV files
- No external API calls or credentials required
- Input validation for date ranges and symbol selection
- Error handling prevents injection attacks through URL parameters
- CORS and XSS protection through Flask defaults

## Migration Plan

### Phase 1: Project Restructuring (NEXT)
1. Create new folder structure
2. Move current files to `visualization/` directory
3. Update import paths and references
4. Test visualization module in new location

### Phase 2: Strategy Framework
1. Implement base strategy classes
2. Create sample strategies (MA crossover, RSI)
3. Add strategy parameter management
4. Build strategy testing utilities

### Phase 3: Backtesting Engine
1. Core backtesting logic implementation
2. Performance metrics calculation
3. Risk management features
4. Transaction cost modeling

### Phase 4: Unified Web Interface
1. Main Flask application development
2. Dashboard creation
3. Strategy configuration interface
4. Backtest result visualization
5. Module integration and testing

## Known Issues & Solutions

### Fixed Issues (Visualization Module)
- ✅ Slider data windowing causing data points to vanish - Fixed with time-based zoom navigation
- ✅ Keyboard controls A/D reversed - Fixed mapping (A=left/backward, D=right/forward)
- ✅ Pan buttons causing chart to vanish on first click - Added safety checks and fallback navigation
- ✅ Close price and volume color similarity - Improved color scheme with purple/orange contrast
- ✅ OHLC button non-responsive - Implemented working OHLC visualization with separate O/H/L/C lines
- ✅ Candlestick volume bar alignment - Fixed perfect alignment between candlestick and volume bars

### Current Behavior
- Navigation works through time-based zooming rather than data windowing
- All data points are preserved and accessible through slider navigation
- Fallback mechanisms ensure pan buttons work even before data is fully loaded
- Consistent color scheme across different chart types
- Perfect time alignment across all chart elements

## Testing & Quality Assurance

### Manual Testing Checklist (Visualization Module)
- [x] Symbol selection and data loading
- [x] Date range validation and quick time ranges
- [x] Slider navigation through entire time period
- [x] Pan button functionality (⬅️➡️)
- [x] Keyboard shortcuts (A/D keys)
- [x] Chart type switching (Line ↔ OHLC ↔ Candlestick)
- [x] Data aggregation levels (1m to 7d)
- [x] Zoom and reset functionality
- [x] Mobile responsiveness
- [x] Error handling for invalid inputs
- [x] Candlestick and volume bar alignment

### Performance Testing
- Large datasets (2+ years of minute data): ✅ Handled efficiently
- Navigation responsiveness: ✅ Smooth slider and pan operations
- Memory usage: ✅ Optimized for client-side processing
- Loading times: ✅ Progress indicators and chunked processing

## Future Development Roadmap

### Immediate Next Steps
1. **Project Restructuring**: Organize current code into modular structure
2. **Strategy Framework**: Build foundation for strategy development
3. **Backtesting Engine**: Implement core backtesting functionality
4. **Web Integration**: Create unified web interface

### Advanced Features (Long-term)
- **Technical Indicators**: Moving averages, RSI, MACD overlays on charts
- **Multiple Symbol Comparison**: Side-by-side or overlay comparisons
- **Export Functionality**: Chart images and data CSV exports
- **Real-time Integration**: WebSocket feeds for live price updates
- **Database Migration**: Replace CSV files with time-series database
- **Advanced Analytics**: Portfolio tracking, P&L calculations, risk metrics
- **Machine Learning**: Strategy optimization and market prediction
- **Paper Trading**: Real-time strategy testing without real money

### Scalability Considerations
- Current architecture supports single-user development environment
- For production: Consider Redis caching, database backend, load balancing
- API rate limiting and user authentication for multi-user deployment
- Microservices architecture for large-scale deployment

## Development Best Practices

### Code Organization
- **Modular Structure**: Separate modules for distinct functionality
- **Clear Separation**: Data processing, business logic, and presentation layers
- **Comprehensive Documentation**: README files and inline documentation
- **Version Control**: Git with clear commit messages and branching strategy

### Debugging
- Console logging for navigation operations
- Error messages displayed to users
- Progress indicators for long operations
- Browser developer tools integration for Chart.js debugging

### Maintenance
- Modular JavaScript functions for easy modification
- CSS variables for consistent styling
- Clear naming conventions for functions and variables
- Comprehensive error handling prevents system crashes
- Unit tests for critical functionality
- always write code and text in English even if I write prompt in Chinese.