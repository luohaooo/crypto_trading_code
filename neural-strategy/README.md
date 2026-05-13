# Neutral Strategy Backtesting

`neural-strategy` 是离线多币种中性策略回测框架。它从 `utils/data_loader.py` 读取本地分钟级 OHLCV 数据，按因子排序构建多空组合，并输出收益、交易、风险指标和可选图表。

## 目录结构

```text
neural-strategy/
├── ohlc_backtest.py             # OHLC 神经网络因子回测入口
├── example.py                   # returns/momentum 示例
├── strategies/
│   ├── base_strategy.py         # 策略和仓位抽象
│   ├── neutral_strategy.py      # 多空中性组合逻辑
│   └── factors/
│       ├── base_factor.py
│       ├── returns_factor.py    # returns、momentum、volatility adjusted
│       └── ohlc_figure_factor.py
├── backtest/
│   ├── engine.py                # 回测循环、调仓和结果汇总
│   └── performance.py           # 风险指标和 matplotlib/seaborn 图表
└── utils/
    ├── config.py                # BacktestConfig/StrategyConfig/FactorConfig/DataConfig
    └── dingding.py              # 钉钉通知
```

## 运行

从仓库根目录安装依赖后执行：

```bash
python neural-strategy/ohlc_backtest.py
```

也可以进入目录运行示例：

```bash
cd neural-strategy
python example.py
```

回测依赖 `utils/data_loader.py` 中配置的本地 CSV 或 pickle 缓存。没有数据时，脚本无法生成有效结果。

## 策略机制

1. 在每个调仓窗口读取目标交易对数据。
2. 使用 returns、momentum、volatility-adjusted returns 或 OHLC figure factor 计算因子。
3. 按因子值排序，前 `top_n` 分配多头，后 `bottom_n` 分配空头。
4. 按 `rebalance_frequency` 平仓并重新建仓。
5. 记录资金曲线、交易明细、手续费、回撤、Sharpe、Sortino、Calmar、VaR 等指标。

## 配置示例

```python
from utils.config import BacktestConfig
from backtest.engine import BacktestEngine

config = BacktestConfig.create_default(
    symbol_list=["BTCUSDT", "ETHUSDT"],
    start_date="2024-01-01",
    end_date="2024-06-01",
)
config.strategy.top_n = 5
config.strategy.bottom_n = 5
config.strategy.rebalance_frequency = "1h"
config.factor.lookback_periods = 240

engine = BacktestEngine(config)
results = engine.run_backtest()
engine.print_results_summary()
```

## 扩展因子

新增因子时继承 `strategies/factors/base_factor.py` 的 `BaseFactor`，实现因子计算接口，并在回测配置中选择对应 `factor_type` 或直接注入 factor 实例。共享数据读取逻辑应继续复用仓库根目录的 `utils/data_loader.py`。
