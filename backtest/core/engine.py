import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_loader import get_available_symbols, get_symbol_data
from ..data.memory_manager import MemoryManager
from ..factors.cache_manager import FactorCacheManager
from .portfolio import Portfolio
from .rebalancer import Rebalancer


class BacktestEngine:
    """
    Main backtesting engine for market-neutral strategies.
    
    Orchestrates the entire backtesting process including:
    - Multi-symbol data loading and management
    - Factor computation and caching
    - Portfolio rebalancing and tracking
    - Performance calculation and logging
    """
    
    def __init__(self, 
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 1000000,
                 rebalance_frequency: str = '1D',
                 transaction_cost: float = 0.001,
                 max_symbols: int = 100):
        """
        Initialize the backtesting engine.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting  
            initial_capital: Initial portfolio capital
            rebalance_frequency: How often to rebalance ('1D', '1W', '1M')
            transaction_cost: Transaction cost as decimal (0.001 = 0.1%)
            max_symbols: Maximum number of symbols to process
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.max_symbols = max_symbols
        
        # Initialize components
        self.memory_manager = MemoryManager(max_memory_gb=8)
        self.factor_cache = FactorCacheManager()
        self.portfolio = Portfolio(initial_capital)
        self.rebalancer = Rebalancer(transaction_cost)
        
        # State tracking
        self.current_date = start_date
        self.symbol_universe = []
        self.backtest_results = []
        self.trade_log = []
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_universe(self, symbols: Optional[List[str]] = None) -> List[str]:
        """
        Initialize the symbol universe for backtesting.
        
        Args:
            symbols: Optional list of symbols. If None, uses all available symbols.
            
        Returns:
            List of symbols to use in backtesting
        """
        if symbols is None:
            all_symbols = get_available_symbols()
            self.symbol_universe = all_symbols[:self.max_symbols]
        else:
            self.symbol_universe = symbols[:self.max_symbols]
            
        self.logger.info(f"Initialized universe with {len(self.symbol_universe)} symbols")
        return self.symbol_universe
    
    def run_backtest(self, 
                     factor_config: Dict[str, Any],
                     strategy_config: Dict[str, Any],
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the complete backtesting process.
        
        Args:
            factor_config: Configuration for factor computation
            strategy_config: Configuration for trading strategy
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Generate rebalancing dates
        rebalance_dates = self._generate_rebalance_dates()
        total_periods = len(rebalance_dates)
        
        try:
            for i, rebalance_date in enumerate(rebalance_dates):
                self.current_date = rebalance_date
                
                # Progress update
                progress_pct = (i / total_periods) * 100
                if progress_callback:
                    progress_callback({
                        'progress': progress_pct,
                        'current_date': rebalance_date.isoformat(),
                        'period': i + 1,
                        'total_periods': total_periods
                    })
                
                # Execute rebalancing
                self._execute_rebalancing_period(
                    rebalance_date, 
                    factor_config, 
                    strategy_config
                )
                
                # Memory cleanup every 30 periods
                if i % 30 == 0:
                    self.memory_manager.cleanup_expired_data()
                
            # Calculate final performance metrics
            self.performance_metrics = self._calculate_performance_metrics()
            
            return {
                'status': 'completed',
                'performance_metrics': self.performance_metrics,
                'trade_log': self.trade_log,
                'portfolio_history': self.backtest_results,
                'total_periods': total_periods
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': {
                    'trade_log': self.trade_log,
                    'portfolio_history': self.backtest_results
                }
            }
    
    def _generate_rebalance_dates(self) -> List[datetime]:
        """
        Generate list of rebalancing dates based on frequency.
        
        Returns:
            List of datetime objects for rebalancing
        """
        dates = []
        current = self.start_date
        
        freq_map = {
            '1D': timedelta(days=1),
            '1W': timedelta(weeks=1), 
            '1M': timedelta(days=30),
            '3M': timedelta(days=90)
        }
        
        delta = freq_map.get(self.rebalance_frequency, timedelta(days=1))
        
        while current <= self.end_date:
            dates.append(current)
            current += delta
            
        return dates
    
    def _execute_rebalancing_period(self, 
                                   rebalance_date: datetime,
                                   factor_config: Dict[str, Any],
                                   strategy_config: Dict[str, Any]):
        """
        Execute a single rebalancing period.
        
        Args:
            rebalance_date: Date for this rebalancing period
            factor_config: Factor computation configuration
            strategy_config: Strategy configuration
        """
        # Load data for current period
        symbol_data = self._load_period_data(rebalance_date)
        
        if not symbol_data:
            self.logger.warning(f"No data available for {rebalance_date}")
            return
        
        # Compute factors
        factor_scores = self._compute_factor_scores(
            symbol_data, 
            rebalance_date, 
            factor_config
        )
        
        # Generate rebalancing trades
        trades = self.rebalancer.generate_rebalancing_trades(
            current_portfolio=self.portfolio,
            factor_scores=factor_scores,
            target_long_count=strategy_config.get('long_count', 25),
            target_short_count=strategy_config.get('short_count', 25),
            rebalance_date=rebalance_date
        )
        
        # Execute trades
        for trade in trades:
            self.portfolio.execute_trade(trade)
            self.trade_log.append({
                'timestamp': rebalance_date.isoformat(),
                'symbol': trade['symbol'],
                'side': trade['side'],
                'action': trade['action'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'value': trade['value'],
                'commission': trade.get('commission', 0)
            })
        
        # Update portfolio value
        portfolio_value = self.portfolio.calculate_total_value(symbol_data, rebalance_date)
        
        # Record portfolio state
        self.backtest_results.append({
            'timestamp': rebalance_date.isoformat(),
            'total_value': portfolio_value,
            'cash': self.portfolio.cash,
            'long_exposure': self.portfolio.get_long_exposure(symbol_data),
            'short_exposure': self.portfolio.get_short_exposure(symbol_data),
            'position_count': len(self.portfolio.positions)
        })
    
    def _load_period_data(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load data for all symbols around the given date.
        
        Args:
            date: Date to load data for
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbol_data = {}
        
        # Define date range for data loading
        start_date = (date - timedelta(days=30)).date()
        end_date = date.date()
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(get_symbol_data, symbol, start_date, end_date): symbol
                for symbol in self.symbol_universe
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        symbol_data[symbol] = df
                except Exception as e:
                    self.logger.warning(f"Failed to load data for {symbol}: {e}")
        
        return symbol_data
    
    def _compute_factor_scores(self, 
                              symbol_data: Dict[str, pd.DataFrame],
                              date: datetime,
                              factor_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute factor scores for all symbols.
        
        Args:
            symbol_data: Dictionary of symbol DataFrames
            date: Current date for factor computation
            factor_config: Factor computation configuration
            
        Returns:
            Dictionary mapping symbol to factor score
        """
        factor_scores = {}
        factor_type = factor_config.get('type', 'returns')
        
        for symbol, df in symbol_data.items():
            try:
                # Check cache first
                cache_key = f"{symbol}_{factor_type}_{date.strftime('%Y%m%d')}"
                cached_score = self.factor_cache.get(cache_key)
                
                if cached_score is not None:
                    factor_scores[symbol] = cached_score
                else:
                    # Compute factor score
                    if factor_type == 'returns':
                        lookback_hours = factor_config.get('lookback_hours', 96)  # 4 days
                        score = self._compute_returns_factor(df, date, lookback_hours)
                    else:
                        score = 0.0
                    
                    factor_scores[symbol] = score
                    self.factor_cache.set(cache_key, score)
                    
            except Exception as e:
                self.logger.warning(f"Failed to compute factor for {symbol}: {e}")
                factor_scores[symbol] = 0.0
        
        return factor_scores
    
    def _compute_returns_factor(self, 
                               df: pd.DataFrame, 
                               date: datetime, 
                               lookback_hours: int) -> float:
        """
        Compute returns-based factor score.
        
        Args:
            df: Price data DataFrame
            date: Current date
            lookback_hours: Hours to look back for return calculation
            
        Returns:
            Factor score (return over lookback period)
        """
        # Find the closest available timestamp
        target_time = date
        start_time = target_time - timedelta(hours=lookback_hours)
        
        # Filter data to the lookback period
        mask = (df['open_time'] >= start_time) & (df['open_time'] <= target_time)
        period_data = df[mask].copy()
        
        if len(period_data) < 2:
            return 0.0
        
        # Calculate return
        start_price = period_data['close'].iloc[0]
        end_price = period_data['close'].iloc[-1]
        
        if start_price == 0:
            return 0.0
            
        return (end_price - start_price) / start_price
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.backtest_results:
            return {}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(self.backtest_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate returns
        df['daily_return'] = df['total_value'].pct_change()
        df['cumulative_return'] = (df['total_value'] / self.initial_capital) - 1
        
        # Performance metrics
        total_return = df['cumulative_return'].iloc[-1]
        daily_returns = df['daily_return'].dropna()
        
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        
        # Maximum drawdown
        rolling_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(total_return / ((self.end_date - self.start_date).days / 365.25)),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': len(self.trade_log),
            'winning_trades': len([t for t in self.trade_log if t.get('pnl', 0) > 0]),
            'final_portfolio_value': float(df['total_value'].iloc[-1])
        }