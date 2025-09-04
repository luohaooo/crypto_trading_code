# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive cryptocurrency trading project featuring:
1. **Price Visualization System** - Interactive charts with advanced navigation (✅ COMPLETED)
2. **Trading Strategy Development** - Strategy creation and testing framework (🚧 PLANNED)
3. **Backtesting Engine** - Historical strategy performance analysis (🚧 PLANNED)
4. **Web Interface for Backtesting** - User-friendly strategy testing interface (🚧 PLANNED)

## Project Structure

```
/home/craz/crypto/crypto-trading/
├── visualization/              # Price visualization module (current)
│   ├── app.py                 # Flask app for price charts
│   ├── templates/
│   │   └── index.html         # Visualization interface
│   └── static/                # CSS/JS assets (if needed)
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
├── utils/                      # Common utilities (planned)
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── helpers.py             # Helper functions
└── tests/                      # Test suites (planned)
    ├── test_strategies.py
    ├── test_backtest.py
    └── test_visualization.py
```

## Current Implementation (Visualization Module)

### Completed Features
- **Backend**: Flask application serving cryptocurrency price data from CSV files
- **Frontend**: Interactive web interface with Chart.js-based visualizations
- **Data Source**: Minute-level OHLCV data stored in `/home/craz/crypto/crypto-data/future_data_2/`
- **Documentation**: Complete implementation guide in `price_visualize.md`

### Key Features Implemented
- Interactive price and volume charts with dual Y-axes
- Time-based navigation with slider controls and keyboard shortcuts (A/D keys)
- Multiple chart types: Line view (purple close prices), OHLC view (color-coded O/H/L/C lines), and Candlestick view (green/red candles)
- Data aggregation from 1-minute to 7-day intervals
- Zoom and pan functionality with Chart.js integration
- Responsive design with mobile-friendly controls
- Perfect alignment between candlestick and volume bars

## Development Setup

**Current Stack**: Python + Flask + Chart.js

### Running the Visualization Module
```bash
cd visualization/
python app.py
```
- Server runs on `http://localhost:5000`
- Debug mode enabled for development
- Automatic reloading on code changes

### Dependencies
- **Python**: Flask, Pandas, datetime, glob, os
- **Frontend**: Chart.js, chartjs-adapter-date-fns, chartjs-plugin-zoom
- **Data**: CSV files with OHLCV cryptocurrency data

### Testing the Visualization System
1. Start the Flask server: `cd visualization/ && python app.py`
2. Navigate to `http://localhost:5000`
3. Select a cryptocurrency symbol (BTCUSDT, ETHUSDT, etc.)
4. Choose date range and load data
5. Test navigation: slider, pan buttons (⬅️➡️), keyboard (A/D keys)
6. Toggle between Line, OHLC, and Candlestick chart types
7. Test data aggregation (1m to 7d intervals)

## Planned Modules

### 1. Trading Strategies Module
- **Purpose**: Define and manage trading strategies
- **Components**:
  - Base strategy class with common interfaces
  - Built-in strategies (MA crossover, RSI, etc.)
  - Custom strategy support
  - Parameter optimization
- **Integration**: Will connect with backtesting engine

### 2. Backtesting Engine
- **Purpose**: Historical strategy performance testing
- **Components**:
  - Core backtesting logic with realistic trading simulation
  - Performance metrics calculation (Sharpe ratio, drawdown, etc.)
  - Risk management features
  - Transaction cost modeling
- **Integration**: Will use visualization module for result charts

### 3. Main Web Application
- **Purpose**: Unified interface for all modules
- **Components**:
  - Dashboard with portfolio overview
  - Strategy configuration interface
  - Integrated visualization (embedded from current module)
  - Backtest result analysis and comparison
- **Integration**: Will orchestrate all other modules

## Architecture Notes

### Current Architecture (Visualization)
- **Data Layer**: CSV file-based storage with Pandas processing
- **API Layer**: Flask REST endpoints for symbols and OHLC data
- **Visualization Layer**: Chart.js with custom navigation controls
- **User Interface**: Single-page web application with responsive design

### Planned Architecture (Full System)
- **Modular Design**: Separate modules for visualization, strategies, backtesting
- **Shared Data Layer**: Common data access patterns across modules
- **API-First**: RESTful APIs for inter-module communication
- **Plugin System**: Extensible strategy and indicator framework

### Key Components
- **Data Processing**: Real-time aggregation and time-series manipulation
- **Navigation System**: Slider-based time window navigation (shows 20% of data at once)
- **Chart Management**: Dual-axis charts (price + volume) with synchronized navigation
- **Error Handling**: Robust error handling for missing data and navigation edge cases

### Performance Considerations
- No data sampling - preserves complete dataset integrity
- Client-side aggregation for smooth user interactions
- Optimized chart rendering with minimal point radius for large datasets
- Memory-efficient data structures for time-based navigation

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