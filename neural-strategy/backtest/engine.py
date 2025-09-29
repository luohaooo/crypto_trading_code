"""
Backtesting Engine

Core backtesting engine for running neural strategies with systematic rebalancing,
performance tracking, and comprehensive result analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import sys
import os
import warnings

# Add parent directories to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Add neural-strategy to path
neural_strategy_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, neural_strategy_root)

from strategies.base_strategy import BaseStrategy
from strategies.factors.base_factor import BaseFactor
from strategies.neutral_strategy import NeutralStrategy

# Import from main project utils using importlib
import importlib.util
data_loader_spec = importlib.util.spec_from_file_location("data_loader", os.path.join(project_root, "utils", "data_loader.py"))
data_loader_module = importlib.util.module_from_spec(data_loader_spec)
data_loader_spec.loader.exec_module(data_loader_module)
load_backtest_data = data_loader_module.load_backtest_data

# Import from neural-strategy utils using importlib
config_spec = importlib.util.spec_from_file_location("config", os.path.join(neural_strategy_root, "utils", "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
BacktestConfig = config_module.BacktestConfig


class BacktestEngine:
    """
    Core backtesting engine for neural strategies.
    
    Handles data loading, time window iteration, strategy execution,
    and performance tracking with comprehensive logging.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.data = None
        self.strategy = None
        self.results = {}
        
        # Execution state
        self.current_timestamp = None
        self.rebalance_timestamps = []
        self.execution_log = []
        
        # Performance tracking
        self.benchmark_returns = []
        self.strategy_returns = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data for backtesting using monthly cache exclusively.

        Returns:
            Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
        """
        print("Loading data from monthly cache...")

        # Load data using the new backtest-specific function
        data = load_backtest_data(
            symbols=self.config.data.symbols,
            start_date=self.config.data.start_date.strftime('%Y-%m-%d') if self.config.data.start_date else None,
            end_date=self.config.data.end_date.strftime('%Y-%m-%d') if self.config.data.end_date else None
        )
        
        if data is None or len(data) == 0:
            raise ValueError("No data loaded. Check symbol list and date range.")
        
        print(f"Loaded data: {len(data):,} records across {len(data.index.get_level_values('symbol').unique())} symbols")
        
        # Filter symbols with insufficient data
        symbol_counts = data.groupby(level='symbol').size()
        valid_symbols = symbol_counts[symbol_counts >= self.config.data.min_data_points].index.tolist()
        
        if len(valid_symbols) < len(data.index.get_level_values('symbol').unique()):
            all_symbols = data.index.get_level_values('symbol').unique().tolist()
            removed_symbols = set(all_symbols) - set(valid_symbols)
            print(f"Removed {len(removed_symbols)} symbols with insufficient data: {list(removed_symbols)[:10]}...")
            
            # Filter data to valid symbols only
            data = data.loc[data.index.get_level_values('symbol').isin(valid_symbols)]
        
        print(f"Final dataset: {len(data):,} records, {len(valid_symbols)} symbols")
        
        # Validate sufficient symbols for strategy
        total_positions = self.config.strategy.top_n + self.config.strategy.bottom_n
        if len(valid_symbols) < total_positions:
            raise ValueError(
                f"Insufficient symbols ({len(valid_symbols)}) for strategy requiring {total_positions} positions"
            )
        
        self.data = data
        return data
    
    def create_factor(self) -> BaseFactor:
        """
        Create factor instance based on configuration.
        
        Returns:
            Factor instance
        """
        from strategies.factors.returns_factor import ReturnsFactor, MomentumFactor, VolatilityAdjustedReturnsFactor
        from strategies.factors.ohlc_figure_factor import OHLCFigureFactor
        
        factor_type = self.config.factor.factor_type
        lookback = self.config.factor.lookback_periods
        params = self.config.factor.params
        
        if factor_type == 'returns':
            return ReturnsFactor(
                lookback_periods=lookback,
                name=self.config.factor.name
            )
        
        elif factor_type == 'momentum':
            short_period = params.get('short_period', 60)
            long_period = params.get('long_period', 240)
            return MomentumFactor(
                short_period=short_period,
                long_period=long_period,
                name=self.config.factor.name
            )
        
        elif factor_type == 'volatility_adjusted':
            lookback_periods = params.get('lookback_periods', lookback)
            volatility_periods = params.get('volatility_periods', lookback)
            return VolatilityAdjustedReturnsFactor(
                lookback_periods=lookback_periods,
                volatility_periods=volatility_periods,
                name=self.config.factor.name
            )
        
        elif factor_type == 'ohlc_figure':
            # OHLC Figure Factor parameters
            model_path = params.get('model_path', None)
            device = params.get('device', 'auto')
            timeframes = params.get('timeframes', ['3min', '15min', '1h'])
            confidence_threshold = params.get('confidence_threshold', None)
            
            return OHLCFigureFactor(
                model_path=model_path,
                device=device,
                name=self.config.factor.name,
                lookback_periods=lookback,
                timeframes=timeframes,
                confidence_threshold=confidence_threshold
            )
        
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")
    
    def create_strategy(self, factor: BaseFactor) -> BaseStrategy:
        """
        Create strategy instance based on configuration.
        
        Args:
            factor: Factor instance to use
            
        Returns:
            Strategy instance
        """
        if self.config.strategy.strategy_type == 'neutral':
            return NeutralStrategy(
                factor=factor,
                top_n=self.config.strategy.top_n,
                bottom_n=self.config.strategy.bottom_n,
                initial_capital=self.config.strategy.initial_capital,
                commission_rate=self.config.strategy.commission_rate,
                name=self.config.strategy.name
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.config.strategy.strategy_type}")
    
    def generate_rebalance_timestamps(self) -> List[pd.Timestamp]:
        """
        Generate timestamps for strategy rebalancing.
        
        Returns:
            List of rebalancing timestamps
        """
        if self.data is None:
            raise ValueError("Data must be loaded before generating timestamps")
        
        # Get all available timestamps
        all_timestamps = self.data.index.get_level_values('open_time').unique().sort_values()
        
        # Skip warmup period
        warmup_start_idx = max(0, self.config.initial_warmup_periods)
        available_timestamps = all_timestamps[warmup_start_idx:]
        
        if len(available_timestamps) == 0:
            raise ValueError("No timestamps available after warmup period")
        
        # Generate rebalance schedule based on frequency
        rebalance_freq = pd.Timedelta(self.config.strategy.rebalance_frequency)
        
        rebalance_timestamps = []
        last_rebalance = available_timestamps[0]
        rebalance_timestamps.append(last_rebalance)
        
        for timestamp in available_timestamps[1:]:
            if timestamp - last_rebalance >= rebalance_freq:
                rebalance_timestamps.append(timestamp)
                last_rebalance = timestamp
        
        print(f"Generated {len(rebalance_timestamps)} rebalance timestamps from {rebalance_timestamps[0]} to {rebalance_timestamps[-1]}")
        
        self.rebalance_timestamps = rebalance_timestamps
        return rebalance_timestamps
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtesting process.
        
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting backtest: {self.config}")
        print("=" * 80)
        
        # Load and validate data
        if self.data is None:
            self.load_data()
        
        # Create factor and strategy
        factor = self.create_factor()
        self.strategy = self.create_strategy(factor)
        
        print(f"Strategy: {self.strategy}")
        print(f"Factor: {factor}")
        
        # Generate rebalancing schedule
        timestamps = self.generate_rebalance_timestamps()

        # Preload images for OHLC figure factors to improve performance
        if hasattr(factor, 'preload_images') and hasattr(factor, 'is_preloaded'):
            print(f"\n🚀 检测到OHLC图像因子，开始预加载图像以提升性能...")
            try:
                factor.preload_images(self.data, timestamps)
                preload_stats = factor.get_preload_stats()
                print(f"✅ 预加载完成: {preload_stats}")
            except Exception as e:
                print(f"⚠️ 预加载失败，将使用原始方法: {e}")

        # Execute backtesting
        print(f"\nExecuting backtest over {len(timestamps)} rebalance periods...")
        print("-" * 50)
        
        failed_rebalances = 0
        
        for i, timestamp in enumerate(timestamps):
            try:
                self.current_timestamp = timestamp
                
                # Generate trading signals
                signals = self.strategy.generate_signals(self.data, timestamp)
                
                if signals['action'] == 'rebalance':
                    # Execute rebalancing
                    self.strategy.execute_rebalance(self.data, timestamp, signals)

                    # Calculate current portfolio value
                    portfolio_value = self.strategy.calculate_portfolio_value(self.data, timestamp)

                    # Print detailed rebalance information
                    print(f"\n📊 调仓执行 #{i+1} | {timestamp.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   组合价值: ${portfolio_value:12,.0f}")

                    # Print position changes
                    long_symbols = signals.get('long_symbols', [])
                    short_symbols = signals.get('short_symbols', [])

                    if long_symbols:
                        print(f"   📈 做多仓位 ({len(long_symbols)}): {', '.join(long_symbols[:5])}")
                        if len(long_symbols) > 5:
                            print(f"      + {len(long_symbols) - 5} 个其他...")

                    if short_symbols:
                        print(f"   📉 做空仓位 ({len(short_symbols)}): {', '.join(short_symbols[:5])}")
                        if len(short_symbols) > 5:
                            print(f"      + {len(short_symbols) - 5} 个其他...")

                    # Print recent trades if any
                    recent_trades = [t for t in self.strategy.completed_trades if
                                   hasattr(t, 'close_time') and t.close_time == timestamp]
                    if recent_trades:
                        total_pnl = sum(t.pnl for t in recent_trades)
                        print(f"   💰 本次平仓: {len(recent_trades)} 笔交易, P&L: ${total_pnl:+,.0f}")

                    print(f"   🎯 总持仓: {len(self.strategy.positions)} 个")

                    # Log execution
                    self.execution_log.append({
                        'timestamp': timestamp,
                        'rebalance_id': i,
                        'action': 'rebalance_success',
                        'portfolio_value': portfolio_value,
                        'num_long': len(long_symbols),
                        'num_short': len(short_symbols),
                        'num_positions': len(self.strategy.positions),
                        'long_symbols': long_symbols,
                        'short_symbols': short_symbols,
                        'recent_trades_pnl': sum(t.pnl for t in recent_trades) if recent_trades else 0
                    })

                    # Progress reporting (less frequent now since we print each rebalance)
                    if self.config.progress_reporting and (i % 20 == 0 or i == len(timestamps) - 1):
                        progress_pct = (i + 1) / len(timestamps) * 100

                        print(f"\n🔄 回测进度: {progress_pct:5.1f}% | "
                              f"已执行 {i+1}/{len(timestamps)} 次调仓")
                
                elif signals['action'] == 'no_signal':
                    failed_rebalances += 1

                    # Print info for failed rebalances occasionally
                    if i % 50 == 0:  # Every 50 attempts
                        reason = signals.get('reason', 'unknown')
                        print(f"⚠️  调仓跳过 #{i+1} | {timestamp.strftime('%Y-%m-%d %H:%M')} | 原因: {reason}")

                    self.execution_log.append({
                        'timestamp': timestamp,
                        'rebalance_id': i,
                        'action': 'no_signal',
                        'reason': signals.get('reason', 'unknown')
                    })
                
            except Exception as e:
                print(f"Error at timestamp {timestamp}: {e}")
                failed_rebalances += 1
                
                self.execution_log.append({
                    'timestamp': timestamp,
                    'rebalance_id': i,
                    'action': 'error',
                    'error': str(e)
                })
                continue
        
        # Close all final positions
        if len(self.strategy.positions) > 0:
            final_timestamp = timestamps[-1]
            final_trades = self.strategy.close_all_positions(self.data, final_timestamp)
            print(f"\nClosed {len(final_trades)} final positions")
        
        print(f"\nBacktest completed!")
        print(f"Successful rebalances: {len(timestamps) - failed_rebalances}/{len(timestamps)}")
        
        # Calculate results
        self.results = self.calculate_results()
        
        return self.results
    
    def calculate_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.
        
        Returns:
            Dictionary with performance metrics and analysis
        """
        if self.strategy is None:
            raise ValueError("Strategy must be run before calculating results")
        
        print("\nCalculating performance metrics...")
        
        # Basic strategy performance
        basic_performance = self.strategy.get_performance_summary()
        
        # Equity curve analysis
        if len(self.strategy.equity_curve) > 1:
            equity_df = pd.DataFrame(self.strategy.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['portfolio_return'] = equity_df['portfolio_value'].pct_change()
            
            # Performance metrics
            total_return = (equity_df['portfolio_value'].iloc[-1] - equity_df['portfolio_value'].iloc[0]) / equity_df['portfolio_value'].iloc[0]
            
            # Annualized metrics
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = days / 365.25 if days > 0 else 1
            
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
            
            # Risk metrics
            returns = equity_df['portfolio_return'].dropna()
            if len(returns) > 1:
                volatility = returns.std()
                sharpe_ratio = (annualized_return - self.config.risk_free_rate) / (volatility * np.sqrt(252)) if volatility > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                
        else:
            annualized_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            equity_df = pd.DataFrame()
        
        # Trade analysis
        trades = self.strategy.completed_trades
        trade_analysis = self.analyze_trades(trades)
        
        # Execution analysis
        execution_analysis = self.analyze_execution()
        
        results = {
            # Basic metrics
            'initial_capital': basic_performance['initial_capital'],
            'final_portfolio_value': basic_performance['final_portfolio_value'],
            'total_return_pct': basic_performance['total_return_pct'],
            'annualized_return_pct': annualized_return * 100,
            
            # Risk metrics
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            
            # Trade metrics
            'total_trades': basic_performance['total_trades'],
            'win_rate_pct': basic_performance['win_rate_pct'],
            'avg_trade_pnl': basic_performance['avg_trade_pnl'],
            
            # Detailed analysis
            'trade_analysis': trade_analysis,
            'execution_analysis': execution_analysis,
            'equity_curve': equity_df,
            'all_trades': trades,
            'config': self.config.to_dict()
        }
        
        return results
    
    def analyze_trades(self, trades: List) -> Dict[str, Any]:
        """
        Analyze completed trades for detailed statistics.
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Dictionary with trade analysis
        """
        if not trades:
            return {'message': 'No trades to analyze'}
        
        # Basic statistics
        pnls = [trade.pnl for trade in trades]
        returns = [trade.return_pct for trade in trades]
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        # Position type analysis
        long_trades = [t for t in trades if t.position_type.value == 'long']
        short_trades = [t for t in trades if t.position_type.value == 'short']
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': (len(winning_trades) / len(trades)) * 100,
            
            'avg_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate_pct': (len([t for t in long_trades if t.pnl > 0]) / len(long_trades)) * 100 if long_trades else 0,
            'short_win_rate_pct': (len([t for t in short_trades if t.pnl > 0]) / len(short_trades)) * 100 if short_trades else 0,
            
            'avg_holding_period': np.mean([t.holding_period.total_seconds() / 3600 for t in trades]),  # Hours
        }
    
    def analyze_execution(self) -> Dict[str, Any]:
        """
        Analyze execution quality and rebalancing statistics.
        
        Returns:
            Dictionary with execution analysis
        """
        if not self.execution_log:
            return {'message': 'No execution data to analyze'}
        
        successful_rebalances = [log for log in self.execution_log if log['action'] == 'rebalance_success']
        failed_rebalances = [log for log in self.execution_log if log['action'] in ['no_signal', 'error']]
        
        execution_rate = len(successful_rebalances) / len(self.execution_log) * 100 if self.execution_log else 0
        
        return {
            'total_rebalance_attempts': len(self.execution_log),
            'successful_rebalances': len(successful_rebalances),
            'failed_rebalances': len(failed_rebalances),
            'execution_rate_pct': execution_rate,
            
            'avg_positions_per_rebalance': np.mean([log.get('num_positions', 0) for log in successful_rebalances]) if successful_rebalances else 0,
            'target_positions': self.config.strategy.top_n + self.config.strategy.bottom_n,
            
            'rebalance_frequency': self.config.strategy.rebalance_frequency,
            'first_rebalance': self.rebalance_timestamps[0] if self.rebalance_timestamps else None,
            'last_rebalance': self.rebalance_timestamps[-1] if self.rebalance_timestamps else None,
        }
    
    def print_results_summary(self) -> str:
        """
        Print a formatted summary of backtest results and return as string.

        Returns:
            String containing the formatted backtest results
        """
        if not self.results:
            message = "No results to display. Run backtest first."
            print(message)
            return message

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("BACKTEST RESULTS SUMMARY")
        lines.append("=" * 80)

        lines.append(f"\nPERFORMANCE METRICS")
        lines.append(f"Initial Capital:        ${self.results['initial_capital']:12,.0f}")
        lines.append(f"Final Portfolio Value:  ${self.results['final_portfolio_value']:12,.0f}")
        lines.append(f"Total Return:           {self.results['total_return_pct']:12.2f}%")
        lines.append(f"Annualized Return:      {self.results['annualized_return_pct']:12.2f}%")
        lines.append(f"Volatility:             {self.results['volatility_pct']:12.2f}%")
        lines.append(f"Sharpe Ratio:           {self.results['sharpe_ratio']:12.2f}")
        lines.append(f"Max Drawdown:           {self.results['max_drawdown_pct']:12.2f}%")

        lines.append(f"\nTRADING ACTIVITY")
        lines.append(f"Total Trades:           {self.results['total_trades']:12,}")
        lines.append(f"Win Rate:               {self.results['win_rate_pct']:12.1f}%")
        lines.append(f"Avg Trade P&L:          ${self.results['avg_trade_pnl']:12.2f}")

        trade_analysis = self.results['trade_analysis']
        if 'message' not in trade_analysis:
            lines.append(f"Best Trade:             ${trade_analysis['best_trade']:12.2f}")
            lines.append(f"Worst Trade:            ${trade_analysis['worst_trade']:12.2f}")
            lines.append(f"Avg Holding Period:     {trade_analysis['avg_holding_period']:12.1f} hours")

        lines.append(f"\nEXECUTION QUALITY")
        exec_analysis = self.results['execution_analysis']
        if 'message' not in exec_analysis:
            lines.append(f"Execution Rate:         {exec_analysis['execution_rate_pct']:12.1f}%")
            lines.append(f"Successful Rebalances:  {exec_analysis['successful_rebalances']:12,}")
            lines.append(f"Failed Rebalances:      {exec_analysis['failed_rebalances']:12,}")
            lines.append(f"Avg Positions:          {exec_analysis['avg_positions_per_rebalance']:12.1f}")
            lines.append(f"Target Positions:       {exec_analysis['target_positions']:12}")

        lines.append(f"\nSTRATEGY CONFIGURATION")
        lines.append(f"Strategy:               {self.config.strategy.name}")
        lines.append(f"Factor:                 {self.config.factor.name}")
        lines.append(f"Long Positions:         {self.config.strategy.top_n}")
        lines.append(f"Short Positions:        {self.config.strategy.bottom_n}")
        lines.append(f"Commission Rate:        {self.config.strategy.commission_rate:.3f}")
        lines.append(f"Rebalance Frequency:    {self.config.strategy.rebalance_frequency}")

        lines.append("\n" + "=" * 80)

        # Create the full summary string
        summary_string = "\n".join(lines)

        # Print to console as before
        print(summary_string)

        # Return the string
        return summary_string