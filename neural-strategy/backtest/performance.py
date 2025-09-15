"""
Performance Metrics and Reporting

Advanced performance analytics, visualization, and reporting for backtesting results.
Includes Sharpe ratio, drawdown analysis, factor attribution, and comprehensive charts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    
    Provides advanced metrics, risk analysis, and visualization capabilities
    for strategy evaluation and comparison.
    """
    
    def __init__(self, results: Dict[str, Any], benchmark_data: Optional[pd.DataFrame] = None):
        """
        Initialize performance analyzer.
        
        Args:
            results: Backtest results dictionary
            benchmark_data: Optional benchmark price data for comparison
        """
        self.results = results
        self.benchmark_data = benchmark_data
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('all_trades', [])
        
    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics beyond basic returns.
        
        Returns:
            Dictionary with advanced performance metrics
        """
        if self.equity_curve.empty:
            return {'error': 'No equity curve data available'}
        
        returns = self.equity_curve['portfolio_return'].dropna()
        if len(returns) < 2:
            return {'error': 'Insufficient return data'}
        
        # Basic statistics
        total_return = self.results.get('total_return_pct', 0) / 100
        annual_return = self.results.get('annualized_return_pct', 0) / 100
        volatility = returns.std()
        sharpe_ratio = self.results.get('sharpe_ratio', 0)
        
        # Advanced risk metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Sortino ratio (return over downside deviation)
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration analysis
        drawdown_periods = self._analyze_drawdown_periods(drawdown)
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Value at Risk (95% and 99%)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Expected Shortfall / Conditional VaR
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Beta and correlation to benchmark (if available)
        beta, correlation, tracking_error = self._calculate_benchmark_metrics(returns)
        
        # Hit ratio (proportion of positive periods)
        hit_ratio = (returns > 0).mean()
        
        # Profit factor (gross profit / gross loss)
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdown_periods['avg_drawdown'],
            'max_drawdown_duration': drawdown_periods['max_duration'],
            'avg_drawdown_duration': drawdown_periods['avg_duration'],
            
            # VaR metrics
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            
            # Benchmark metrics
            'beta': beta,
            'correlation_to_benchmark': correlation,
            'tracking_error': tracking_error,
            'information_ratio': (annual_return - (beta * 0.1)) / tracking_error if tracking_error > 0 else 0  # Assume 10% benchmark return
        }
    
    def _analyze_drawdown_periods(self, drawdown: pd.Series) -> Dict[str, float]:
        """
        Analyze drawdown periods for duration and depth statistics.
        
        Args:
            drawdown: Drawdown time series
            
        Returns:
            Dictionary with drawdown period statistics
        """
        # Identify drawdown periods (where drawdown < -0.01%)
        in_drawdown = drawdown < -0.0001
        
        # Find start and end of drawdown periods
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
        
        periods = []
        start_idx = None
        
        for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if is_start:
                start_idx = i
            elif is_end and start_idx is not None:
                period_drawdown = drawdown.iloc[start_idx:i+1]
                periods.append({
                    'duration': len(period_drawdown),
                    'max_drawdown': period_drawdown.min()
                })
                start_idx = None
        
        if not periods:
            return {'avg_drawdown': 0, 'max_duration': 0, 'avg_duration': 0}
        
        return {
            'avg_drawdown': np.mean([p['max_drawdown'] for p in periods]),
            'max_duration': max([p['duration'] for p in periods]),
            'avg_duration': np.mean([p['duration'] for p in periods])
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate benchmark-related metrics.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Tuple of (beta, correlation, tracking_error)
        """
        if self.benchmark_data is None or self.benchmark_data.empty:
            return 0.0, 0.0, 0.0
        
        try:
            # Calculate benchmark returns for same period
            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
            
            # Align with strategy returns
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0, 0.0, 0.0
            
            aligned_strategy = returns.loc[common_index]
            aligned_benchmark = benchmark_returns.loc[common_index]
            
            # Calculate correlation
            correlation = aligned_strategy.corr(aligned_benchmark)
            
            # Calculate beta
            covariance = aligned_strategy.cov(aligned_benchmark)
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
            # Calculate tracking error
            excess_returns = aligned_strategy - aligned_benchmark
            tracking_error = excess_returns.std()
            
            return beta, correlation, tracking_error
            
        except Exception as e:
            print(f"Warning: Could not calculate benchmark metrics: {e}")
            return 0.0, 0.0, 0.0
    
    def analyze_factor_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of the factor used in the strategy.
        
        Returns:
            Dictionary with factor analysis metrics
        """
        if not self.trades:
            return {'error': 'No trades available for factor analysis'}
        
        # Analyze long vs short performance
        long_trades = [t for t in self.trades if t.position_type.value == 'long']
        short_trades = [t for t in self.trades if t.position_type.value == 'short']
        
        long_pnl = [t.pnl for t in long_trades]
        short_pnl = [t.pnl for t in short_trades]
        
        # Factor effectiveness metrics
        long_win_rate = (len([p for p in long_pnl if p > 0]) / len(long_pnl)) * 100 if long_pnl else 0
        short_win_rate = (len([p for p in short_pnl if p > 0]) / len(short_pnl)) * 100 if short_pnl else 0
        
        # Spread analysis (long performance - short performance)
        avg_long_return = np.mean([t.return_pct for t in long_trades]) if long_trades else 0
        avg_short_return = np.mean([t.return_pct for t in short_trades]) if short_trades else 0
        spread_return = avg_long_return - avg_short_return
        
        return {
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate_pct': long_win_rate,
            'short_win_rate_pct': short_win_rate,
            'long_avg_pnl': np.mean(long_pnl) if long_pnl else 0,
            'short_avg_pnl': np.mean(short_pnl) if short_pnl else 0,
            'long_avg_return_pct': avg_long_return,
            'short_avg_return_pct': avg_short_return,
            'spread_return_pct': spread_return,
            'factor_effectiveness': 'Good' if spread_return > 0 else 'Poor'
        }
    
    def create_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive performance report with all metrics.
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Complete performance report dictionary
        """
        report = {
            'basic_metrics': {
                'initial_capital': self.results.get('initial_capital', 0),
                'final_value': self.results.get('final_portfolio_value', 0),
                'total_return_pct': self.results.get('total_return_pct', 0),
                'annualized_return_pct': self.results.get('annualized_return_pct', 0),
                'volatility_pct': self.results.get('volatility_pct', 0),
                'sharpe_ratio': self.results.get('sharpe_ratio', 0),
                'max_drawdown_pct': self.results.get('max_drawdown_pct', 0)
            },
            'advanced_metrics': self.calculate_advanced_metrics(),
            'trade_analysis': self.results.get('trade_analysis', {}),
            'factor_analysis': self.analyze_factor_effectiveness(),
            'execution_analysis': self.results.get('execution_analysis', {}),
            'config': self.results.get('config', {})
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert non-serializable objects
                serializable_report = self._make_serializable(report)
                json.dump(serializable_report, f, indent=2, default=str)
        
        return report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        else:
            return obj


class PerformanceVisualizer:
    """
    Create visualizations for backtesting results.
    
    Provides comprehensive charts including equity curves, drawdowns,
    trade analysis, and factor performance visualization.
    """
    
    def __init__(self, results: Dict[str, Any], benchmark_data: Optional[pd.DataFrame] = None):
        """
        Initialize performance visualizer.
        
        Args:
            results: Backtest results dictionary
            benchmark_data: Optional benchmark data for comparison
        """
        self.results = results
        self.benchmark_data = benchmark_data
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('all_trades', [])
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (15, 8), 
                         show_benchmark: bool = True) -> plt.Figure:
        """
        Plot strategy equity curve with optional benchmark comparison.
        
        Args:
            figsize: Figure size tuple
            show_benchmark: Whether to show benchmark comparison
            
        Returns:
            matplotlib Figure object
        """
        if self.equity_curve.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No equity curve data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main equity curve
        dates = self.equity_curve.index
        values = self.equity_curve['portfolio_value']
        
        ax1.plot(dates, values, linewidth=2, label='Strategy', color='#2E86AB')
        
        # Benchmark comparison
        if show_benchmark and self.benchmark_data is not None and not self.benchmark_data.empty:
            # Normalize benchmark to same starting value
            benchmark_normalized = self.benchmark_data['close'] / self.benchmark_data['close'].iloc[0] * self.results['initial_capital']
            ax1.plot(benchmark_normalized.index, benchmark_normalized.values, 
                    linewidth=2, label='Benchmark', color='#A23B72', alpha=0.7)
        
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Format y-axis with currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown subplot
        if 'portfolio_return' in self.equity_curve.columns:
            returns = self.equity_curve['portfolio_return'].dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax2.fill_between(drawdown.index, drawdown.values * 100, 0, 
                           alpha=0.7, color='#F18F01', label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot returns distribution analysis.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        if self.equity_curve.empty or 'portfolio_return' not in self.equity_curve.columns:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No returns data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
        
        returns = self.equity_curve['portfolio_return'].dropna() * 100  # Convert to percentage
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(returns.mean(), color='#F18F01', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax1.axvline(returns.median(), color='#A23B72', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')
        ax1.set_title('Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
        ax3.plot(rolling_vol.index, rolling_vol.values, color='#F18F01')
        ax3.set_title('30-Day Rolling Volatility', fontweight='bold')
        ax3.set_ylabel('Annualized Volatility (%)')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + returns / 100).cumprod() * 100 - 100
        ax4.plot(cumulative_returns.index, cumulative_returns.values, color='#2E86AB')
        ax4.set_title('Cumulative Returns', fontweight='bold')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive trade analysis.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        if not self.trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trade data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Trade P&L distribution
        pnls = [trade.pnl for trade in self.trades]
        ax1.hist(pnls, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(np.mean(pnls), color='#F18F01', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('Trade P&L ($)')
        ax1.set_ylabel('Number of Trades')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Win/Loss by position type
        long_trades = [t for t in self.trades if t.position_type.value == 'long']
        short_trades = [t for t in self.trades if t.position_type.value == 'short']
        
        long_wins = len([t for t in long_trades if t.pnl > 0])
        long_losses = len(long_trades) - long_wins
        short_wins = len([t for t in short_trades if t.pnl > 0])
        short_losses = len(short_trades) - short_wins
        
        categories = ['Long Wins', 'Long Losses', 'Short Wins', 'Short Losses']
        values = [long_wins, long_losses, short_wins, short_losses]
        colors = ['#2E86AB', '#A23B72', '#2E86AB', '#A23B72']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Wins/Losses by Position Type', fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3)
        
        # Trade timeline
        trade_times = [trade.entry_time for trade in self.trades]
        trade_pnls = [trade.pnl for trade in self.trades]
        
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        ax3.scatter(trade_times, trade_pnls, c=colors, alpha=0.6, s=30)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Trade P&L Timeline', fontweight='bold')
        ax3.set_xlabel('Trade Entry Time')
        ax3.set_ylabel('Trade P&L ($)')
        ax3.grid(True, alpha=0.3)
        
        # Holding period distribution
        holding_periods = [trade.holding_period.total_seconds() / 3600 for trade in self.trades]  # Hours
        ax4.hist(holding_periods, bins=20, alpha=0.7, color='#F18F01', edgecolor='black')
        ax4.set_title('Holding Period Distribution', fontweight='bold')
        ax4.set_xlabel('Holding Period (Hours)')
        ax4.set_ylabel('Number of Trades')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_full_report(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create complete visual report with all charts.
        
        Args:
            save_dir: Optional directory to save charts
            
        Returns:
            Dictionary of chart names to Figure objects
        """
        charts = {
            'equity_curve': self.plot_equity_curve(),
            'returns_distribution': self.plot_returns_distribution(),
            'trade_analysis': self.plot_trade_analysis()
        }
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for name, fig in charts.items():
                fig.savefig(os.path.join(save_dir, f'{name}.png'), 
                           dpi=300, bbox_inches='tight', facecolor='white')
        
        return charts
    
    def show_all_charts(self):
        """Display all charts."""
        self.create_full_report()
        plt.show()