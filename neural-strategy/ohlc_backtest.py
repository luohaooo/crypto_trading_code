import sys
import os
# from datetime import datetime, timedelta  # Not needed for this demo

# Add paths for imports
project_root = '/home/craz/crypto/crypto-trading'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'neural-strategy'))

from utils.config import BacktestConfig, StrategyConfig, FactorConfig, DataConfig
from backtest.engine import BacktestEngine
from utils.dingding import send_dingtalk_message



data_config = DataConfig(
        symbols=None,
        start_date='2025-07-01',
        end_date='2025-07-31', 
        min_data_points=1200
    )

factor_config = FactorConfig(
    name="OHLC_CNN_Neural",
    factor_type="ohlc_figure",
    lookback_periods=20,  # 20 periods for image generation
    params={
        'model_path': '/home/craz/crypto/crypto-trading/figure_model/model_saved/f_2025-02_2025-05_mse_1d.pt',  # Path to trained model file (.pt) - set to None for random predictions
        'device': 'auto',  # Use GPU if available, otherwise CPU
        'timeframes': ['3min', '15min', '1h'],  # Multi-timeframe analysis
        'confidence_threshold': None  # No confidence filtering for now
    }
)

# Strategy configuration
strategy_config = StrategyConfig(
    name="OHLC_Neutral_Strategy",
    strategy_type="neutral",
    initial_capital=100000.0,  # $100K starting capital
    commission_rate=0.001,  # 0.1% commission
    top_n     =  10,  # 2 long positions
    bottom_n  =  10,  # 2 short positions
    rebalance_frequency='1d'  # Rebalance every 4 hours
)

# Main backtest configuration
backtest_config = BacktestConfig(
    strategy=strategy_config,
    factor=factor_config,
    data=data_config,
    initial_warmup_periods=1200,  
    progress_reporting=True,
    save_trades=True
)


engine = BacktestEngine(backtest_config)

engine.load_data()

results = engine.run_backtest()

engine.print_results_summary()
        

print("\n🔍 OHLC Factor Analysis:")
factor = engine.strategy.factor
model_info = factor.get_model_info()

