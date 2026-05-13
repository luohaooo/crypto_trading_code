# Crypto Trading Code

这是一个面向币安 USDT 永续合约的量化交易代码库，核心功能包括实时交易执行、神经网络 OHLC 因子计算、离线回测、模型训练、账户统计和数据缓存加载。旧价格可视化模块已经移除。

## 主要模块

```text
binance_implement/
├── neural_strategy/        # 当前神经网络因子实盘/测试网交易入口
├── whole_strategy/         # 完整交易执行版本，包含保护单和因子执行逻辑
├── only_in/                # 只入场策略及保护单检查相关逻辑
├── find_in_out/            # 入场/出场信号实验脚本
├── account_stat/           # 多账户资产统计、余额监控和报表输出
└── test_websocket/         # WebSocket 分钟数据测试脚本

figure_model/               # OHLC 图像数据生成、模型训练和 checkpoint
neural-strategy/            # 离线中性策略回测框架
utils/                      # CSV/pickle/month-cache 数据加载工具
tests/                      # 缓存、资金费率、保护单状态等脚本化检查
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

运行当前神经网络交易系统：

```bash
python binance_implement/neural_strategy/automated_trading.py
```

运行离线中性策略回测：

```bash
python neural-strategy/ohlc_backtest.py
```

训练 OHLC 基线模型：

```bash
python figure_model/train_ohlc_model.py
```

检查月度缓存加载：

```bash
python tests/test_month_cache_loader.py
python tests/quick_test_month_cache.py
```

## 数据要求

`utils/data_loader.py` 默认读取：

- CSV 数据目录：`/home/craz/crypto/crypto-data/future_data_2`
- 全量 pickle 缓存：`/home/craz/crypto/crypto-data/pickle_cache`
- 月度 pickle 缓存：`/home/craz/crypto/crypto-data/pickle_month_cache`

迁移数据时优先修改 `utils/data_loader.py` 顶部的路径常量，避免在策略或回测脚本中硬编码路径。

## 实盘配置

交易相关配置集中在各策略目录的 `config.py` 中。当前 `binance_implement/neural_strategy/config.py` 支持 testnet 和实盘：

- Testnet：`BINANCE_TESTNET_API_KEY`、`BINANCE_TESTNET_API_SECRET`
- 实盘：`BINANCE_API_KEY_SH`、`BINANCE_API_SECRET_SH`

不要把真实 API key 写入仓库。执行实盘脚本前先确认杠杆、保证金模式、止盈止损和调仓周期。

## 回测与模型

- `neural-strategy/` 使用 `BacktestConfig`、`NeutralStrategy`、factor 模块和 `BacktestEngine` 执行多币种多空中性回测。
- `figure_model/` 将 OHLC 序列转换为图像特征，训练 PyTorch 模型，并将 checkpoint 保存到 `figure_model/model_saved/`。
- `binance_implement/neural_strategy/` 和 `binance_implement/whole_strategy/` 会加载模型 checkpoint 计算实时因子。

## 注意事项

- 本仓库包含实盘交易代码，运行前必须确认环境变量、账户权限、保证金模式和保护单配置。
- 日志、csv、pdf 和 `.pyc` 等运行产物不要作为功能变更提交，分享日志前需要清理账户和仓位信息。
- 当前测试主要是可执行脚本，不是完整 pytest 套件；运行结果需要人工查看 stdout 和日志。
