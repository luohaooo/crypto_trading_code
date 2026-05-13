# Neural Strategy Trading

`binance_implement/neural_strategy` 是当前神经网络 OHLC 因子的币安 USDT 永续合约自动交易入口，支持 testnet 和实盘环境。

## 文件说明

```text
automated_trading.py          # 主循环：调度、数据处理、因子计算、下单、监控
config.py                     # API、杠杆、调仓周期、保护单、模型路径等配置
trading_executor.py           # ccxt 交易执行、仓位、保证金模式、保护单
optimized_data_processor.py   # 拉取和整理交易所 OHLC 数据
fs_factor_calculator.py       # 当前使用的 OHLC 图像因子计算器
optimized_factor_calculator.py# 备用因子计算器
ohlc_model.py                 # 基线模型结构
ohlc_model_v2.py              # 当前因子模型结构
trading_utils/
├── logger.py                 # 日志初始化
├── monitor.py                # 交易周期监控
└── dingding.py               # 钉钉通知
```

## 运行

从仓库根目录安装依赖：

```bash
pip install -r requirements.txt
```

配置 testnet：

```bash
export BINANCE_TESTNET_API_KEY="your_key"
export BINANCE_TESTNET_API_SECRET="your_secret"
python binance_implement/neural_strategy/automated_trading.py
```

配置实盘：

```bash
export BINANCE_API_KEY_SH="your_key"
export BINANCE_API_SECRET_SH="your_secret"
python binance_implement/neural_strategy/automated_trading.py
```

## 当前交易流程

1. 初始化 `TradingConfig`、logger、monitor、executor、data processor 和 factor calculator。
2. 按 `EXECUTION_MODE` 选择固定间隔、每日或 16 小时周期执行。
3. 同步交易所仓位，移除已经被保护单或外部操作平掉的仓位。
4. 拉取最近 `LOOKBACK_HOURS` 的 OHLC 数据。
5. 使用 `figure_model/model_saved/` 下的 PyTorch checkpoint 计算多时间框架 OHLC 图像因子。
6. 按因子排序选择 `TOP_N_LONG` 和 `TOP_N_SHORT`。
7. 设置保证金模式和杠杆，使用市价单建仓。
8. 根据配置创建止盈止损保护单，并写入日志/钉钉通知。

## 关键配置

主要参数在 `config.py` 的 `TradingConfig`：

- `EXECUTION_MODE`: `interval`、`daily` 或 `16h`
- `REBALANCE_INTERVAL`: 固定间隔模式的秒数
- `TOP_N_LONG` / `TOP_N_SHORT`: 多空标的数量
- `LEVERAGE`: 杠杆倍数
- `MARGIN_TYPE`: `ISOLATED` 或 `CROSSED`
- `ENABLE_PROTECTIVE_ORDERS`: 是否启用止盈止损
- `TAKE_PROFIT_RATIO` / `STOP_LOSS_RATIO`: 保护单比例
- `MODEL_PATH`: 因子模型 checkpoint 路径

## 风险提示

这是实盘交易代码。运行前必须确认 API 权限、账户余额、杠杆、保证金模式、保护单参数和日志路径。首次改动后建议只在 testnet 或小资金环境验证。
