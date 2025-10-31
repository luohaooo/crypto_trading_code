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


for model_path in [
    '/home/craz/crypto/crypto-trading/figure_model/model_saved/baseline_epoch_52_train_0.02124_val_0.02302.pt',
    ]:
    for [s, e] in [
                            ['2025-01-01', '2025-01-31'],
                            # ['2025-02-01', '2025-02-28'],
                            # ['2025-03-01', '2025-03-31'],
                            # ['2025-04-01', '2025-04-30'],
                            # ['2025-05-01', '2025-05-31'],
                            # ['2025-06-01', '2025-06-30'],
                            # ['2024-07-01', '2024-07-31'],
                           ]:
        for top_k in [2]:

            data_config = DataConfig(
                    symbols=None,
                    start_date=s,
                    end_date=e, 
                    min_data_points=4800
                )

            factor_config = FactorConfig(
                name="OHLC_CNN_Neural",
                factor_type="ohlc_figure",
                lookback_periods=20,  # 20 periods for image generation
                params={
                    'model_path': model_path,  # Path to trained model file (.pt) - set to None for random predictions
                    'device': 'auto',  # Use GPU if available, otherwise CPU
                    'timeframes': ['1h', '2h', '4h'],  # Multi-timeframe analysis
                    'confidence_threshold': None  # No confidence filtering for now
                }
            )

            # Strategy configuration
            strategy_config = StrategyConfig(
                name="OHLC_Neutral_Strategy",
                strategy_type="neutral",
                initial_capital=100000.0,  # $100K starting capital
                commission_rate=0.001,  # 0.1% commission
                top_n     =  top_k,  # 2 long positions
                bottom_n  =  top_k,  # 2 short positions
                rebalance_frequency='4h'  # Rebalance every 4 hours
            )

        # Main backtest configuration
            backtest_config = BacktestConfig(
                strategy=strategy_config,
                factor=factor_config,
                data=data_config,
                initial_warmup_periods=4800,  
                progress_reporting=True,
                save_trades=True
            )


            engine = BacktestEngine(backtest_config)

            engine.load_data()

            results = engine.run_backtest()

            summary_string = engine.print_results_summary()

            send_dingtalk_message(f"Model:{model_path}\n, start_date={s}, end_date={e}, top_k={top_k}\n"+summary_string)
                

        # print("\n🔍 OHLC Factor Analysis:")
        # factor = engine.strategy.factor
        # model_info = factor.get_model_info()

