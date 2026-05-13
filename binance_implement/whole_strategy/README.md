# Whole Strategy Trading

`binance_implement/whole_strategy` 是更完整的自动交易实现，复用神经网络 OHLC 因子，并包含保证金模式、保护单、仓位同步和较完整的交易执行辅助逻辑。

## 文件说明

```text
automated_trading.py           # 主交易循环
config.py                      # 策略、API、风险和日志配置
trading_executor.py            # ccxt/REST 下单、仓位、保护单、保证金模式
optimized_data_processor.py    # 实时 OHLC 数据处理
previous_data_processoe.py     # 历史数据处理版本
fs_factor_calculator.py        # OHLC 图像因子计算
ohlc_model_v2.py
ohlc_model_v3.py               # 因子模型结构
trading_utils/                 # logger、monitor、dingding
test_strategy/calculate_factor.py
```

## 运行

```bash
python binance_implement/whole_strategy/automated_trading.py
```

运行前检查 `config.py` 中的执行模式、杠杆、保证金模式、保护单和模型路径，并通过环境变量提供 Binance API key。

## 与 neural_strategy 的区别

- `whole_strategy` 的 `trading_executor.py` 包含更多 REST 辅助逻辑和保护单处理。
- `neural_strategy` 是当前更明确的神经因子交易入口，文档和运行路径更集中。
- 两者都依赖 `figure_model` 中的 OHLC 图像转换和模型 checkpoint。

## 调试建议

- 优先在 testnet 或低风险账户运行。
- 保留单独终端查看日志。
- 不要提交运行生成的 `.pyc`、日志、CSV 或 PDF 报表。
