"""
Configuration System

Centralized configuration management for neural strategy backtesting.
Provides parameter validation and default values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd


@dataclass
class FactorConfig:
    """Configuration for factor parameters"""
    name: str
    factor_type: str  # 'returns', 'momentum', 'volatility_adjusted'
    lookback_periods: int = 240  # Default 4 hours for 1-minute data
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.lookback_periods <= 0:
            raise ValueError("lookback_periods must be positive")
        
        valid_types = ['returns', 'momentum', 'volatility_adjusted', 'ohlc_figure']
        if self.factor_type not in valid_types:
            raise ValueError(f"factor_type must be one of {valid_types}")


@dataclass
class StrategyConfig:
    """Configuration for strategy parameters"""
    name: str
    strategy_type: str = 'neutral'
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001  # 0.1%
    top_n: int = 10  # Number of long positions
    bottom_n: int = 10  # Number of short positions
    rebalance_frequency: str = '1h'  # Pandas frequency string
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        if not 0 <= self.commission_rate <= 1:
            raise ValueError("commission_rate must be between 0 and 1")
        
        if self.top_n <= 0 or self.bottom_n <= 0:
            raise ValueError("top_n and bottom_n must be positive")
        
        # Validate rebalance frequency
        try:
            pd.Timedelta(self.rebalance_frequency)
        except ValueError:
            raise ValueError(f"Invalid rebalance_frequency: {self.rebalance_frequency}")


@dataclass
class DataConfig:
    """Configuration for data parameters"""
    symbols: Optional[List[str]] = None  # None means all available USDT symbols
    start_date: Optional[Union[str, datetime]] = None
    end_date: Optional[Union[str, datetime]] = None
    data_frequency: str = '1min'  # Data granularity
    symbol_filter: str = 'usdt'  # 'usdt', 'all', or custom list
    min_data_points: int = 1000  # Minimum data points required per symbol
    
    def __post_init__(self):
        """Validate and convert configuration after initialization"""
        # Convert string dates to datetime if needed
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date)
        
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date)
        
        # Validate date range
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("start_date must be before end_date")
        
        if self.min_data_points <= 0:
            raise ValueError("min_data_points must be positive")


@dataclass
class BacktestConfig:
    """Main backtesting configuration"""
    # Core components
    strategy: StrategyConfig
    factor: FactorConfig
    data: DataConfig
    
    # Backtesting parameters
    initial_warmup_periods: int = 1000  # Data points for factor initialization
    progress_reporting: bool = True
    save_trades: bool = True
    save_equity_curve: bool = True
    
    # Performance calculation
    benchmark_symbol: Optional[str] = 'BTCUSDT'  # For performance comparison
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calculation
    
    # Output settings
    output_dir: Optional[str] = None
    save_results: bool = False
    plot_results: bool = True
    
    def __post_init__(self):
        """Validate complete configuration"""
        if self.initial_warmup_periods < 0:
            raise ValueError("initial_warmup_periods must be non-negative")
        
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError("risk_free_rate must be between 0 and 1")
    
    @classmethod
    def create_default(cls, 
                      symbol_list: Optional[List[str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> 'BacktestConfig':
        """
        Create a default configuration with common settings.
        
        Args:
            symbol_list: List of symbols to trade (None for all USDT pairs)
            start_date: Start date string (e.g., '2023-01-01')
            end_date: End date string (e.g., '2024-01-01')
            
        Returns:
            Default BacktestConfig
        """
        # Default 4-hour returns factor
        factor_config = FactorConfig(
            name='Returns_4H',
            factor_type='returns',
            lookback_periods=240  # 4 hours in minutes
        )
        
        # Default neutral strategy
        strategy_config = StrategyConfig(
            name='NeutralStrategy_4H',
            strategy_type='neutral',
            initial_capital=1000000.0,
            commission_rate=0.001,
            top_n=10,
            bottom_n=10,
            rebalance_frequency='1h'
        )
        
        # Default data configuration
        data_config = DataConfig(
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            data_frequency='1min',
            symbol_filter='usdt' if symbol_list is None else 'custom',
            min_data_points=1000
        )
        
        return cls(
            strategy=strategy_config,
            factor=factor_config,
            data=data_config,
            initial_warmup_periods=500,
            progress_reporting=True,
            save_trades=True,
            save_equity_curve=True,
            benchmark_symbol='BTCUSDT',
            risk_free_rate=0.02,
            plot_results=True
        )
    
    @classmethod
    def create_momentum_strategy(cls,
                               short_period: int = 60,
                               long_period: int = 240,
                               **kwargs) -> 'BacktestConfig':
        """
        Create configuration for momentum-based strategy.
        
        Args:
            short_period: Short-term momentum period
            long_period: Long-term momentum period
            **kwargs: Additional parameters for create_default
            
        Returns:
            BacktestConfig for momentum strategy
        """
        config = cls.create_default(**kwargs)
        
        # Update factor to momentum
        config.factor = FactorConfig(
            name=f'Momentum_{short_period}m_{long_period}m',
            factor_type='momentum',
            lookback_periods=max(short_period, long_period),
            params={
                'short_period': short_period,
                'long_period': long_period
            }
        )
        
        # Update strategy name
        config.strategy.name = f'MomentumStrategy_{short_period}m_{long_period}m'
        
        return config
    
    @classmethod
    def create_vol_adjusted_strategy(cls,
                                   lookback_periods: int = 240,
                                   volatility_periods: int = 240,
                                   **kwargs) -> 'BacktestConfig':
        """
        Create configuration for volatility-adjusted returns strategy.
        
        Args:
            lookback_periods: Returns calculation period
            volatility_periods: Volatility calculation period
            **kwargs: Additional parameters for create_default
            
        Returns:
            BacktestConfig for volatility-adjusted strategy
        """
        config = cls.create_default(**kwargs)
        
        # Update factor to volatility-adjusted
        config.factor = FactorConfig(
            name=f'VolAdjReturns_{lookback_periods}m',
            factor_type='volatility_adjusted',
            lookback_periods=max(lookback_periods, volatility_periods),
            params={
                'lookback_periods': lookback_periods,
                'volatility_periods': volatility_periods
            }
        )
        
        # Update strategy name
        config.strategy.name = f'VolAdjStrategy_{lookback_periods}m'
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'strategy': {
                'name': self.strategy.name,
                'strategy_type': self.strategy.strategy_type,
                'initial_capital': self.strategy.initial_capital,
                'commission_rate': self.strategy.commission_rate,
                'top_n': self.strategy.top_n,
                'bottom_n': self.strategy.bottom_n,
                'rebalance_frequency': self.strategy.rebalance_frequency
            },
            'factor': {
                'name': self.factor.name,
                'factor_type': self.factor.factor_type,
                'lookback_periods': self.factor.lookback_periods,
                'params': self.factor.params
            },
            'data': {
                'symbols': self.data.symbols,
                'start_date': self.data.start_date.isoformat() if self.data.start_date else None,
                'end_date': self.data.end_date.isoformat() if self.data.end_date else None,
                'data_frequency': self.data.data_frequency,
                'symbol_filter': self.data.symbol_filter,
                'min_data_points': self.data.min_data_points
            },
            'backtest': {
                'initial_warmup_periods': self.initial_warmup_periods,
                'progress_reporting': self.progress_reporting,
                'save_trades': self.save_trades,
                'save_equity_curve': self.save_equity_curve,
                'benchmark_symbol': self.benchmark_symbol,
                'risk_free_rate': self.risk_free_rate,
                'output_dir': self.output_dir,
                'save_results': self.save_results,
                'plot_results': self.plot_results
            }
        }
    
    def validate(self) -> List[str]:
        """
        Validate the complete configuration and return any issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Individual component validation happens in __post_init__
            # Check for logical consistency across components
            
            # Factor lookback should be reasonable for rebalance frequency
            rebalance_minutes = pd.Timedelta(self.strategy.rebalance_frequency).total_seconds() / 60
            if self.factor.lookback_periods < rebalance_minutes:
                errors.append(
                    f"Factor lookback_periods ({self.factor.lookback_periods}) should be >= "
                    f"rebalance frequency in minutes ({rebalance_minutes})"
                )
            
            # Warmup periods should be sufficient for factor calculation
            if self.initial_warmup_periods < self.factor.lookback_periods:
                errors.append(
                    f"initial_warmup_periods ({self.initial_warmup_periods}) should be >= "
                    f"factor lookback_periods ({self.factor.lookback_periods})"
                )
            
            # Check if enough symbols for strategy
            total_positions = self.strategy.top_n + self.strategy.bottom_n
            if self.data.symbols and len(self.data.symbols) < total_positions:
                errors.append(
                    f"Not enough symbols ({len(self.data.symbols)}) for "
                    f"total positions required ({total_positions})"
                )
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def __str__(self) -> str:
        return (f"BacktestConfig(strategy={self.strategy.name}, "
                f"factor={self.factor.name}, "
                f"symbols={len(self.data.symbols) if self.data.symbols else 'all'}, "
                f"period={self.data.start_date} to {self.data.end_date})")


def load_config_from_dict(config_dict: Dict[str, Any]) -> BacktestConfig:
    """
    Load BacktestConfig from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        BacktestConfig object
    """
    strategy_config = StrategyConfig(**config_dict['strategy'])
    factor_config = FactorConfig(**config_dict['factor'])
    data_config = DataConfig(**config_dict['data'])
    
    backtest_params = config_dict['backtest']
    
    return BacktestConfig(
        strategy=strategy_config,
        factor=factor_config,
        data=data_config,
        **backtest_params
    )