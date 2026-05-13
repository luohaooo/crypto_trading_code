# 中性策略说明

中性策略在每个调仓窗口内同时持有多头和空头，用因子排序决定交易标的。

## 核心流程

1. 读取指定时间范围内的多币种 OHLCV 数据。
2. 计算每个币种的因子值，例如 4 小时收益率、动量、波动率调整收益或 OHLC 图像模型预测值。
3. 按因子从高到低排序。
4. 将一半资金平均分配给前 `n` 名做多。
5. 将另一半资金平均分配给后 `n` 名做空。
6. 到下一个调仓窗口时平掉旧仓位，再按最新因子重新开仓。
7. 汇总资金曲线、交易明细、手续费、最大回撤、Sharpe 等指标。

## 当前实现位置

- 策略逻辑：`strategies/neutral_strategy.py`
- 因子基类：`strategies/factors/base_factor.py`
- 收益/动量因子：`strategies/factors/returns_factor.py`
- OHLC 图像因子：`strategies/factors/ohlc_figure_factor.py`
- 回测引擎：`backtest/engine.py`
- 绩效分析：`backtest/performance.py`
- 配置对象：`utils/config.py`

## 运行入口

```bash
python neural-strategy/ohlc_backtest.py
python neural-strategy/example.py
```

回测需要本地数据缓存可用，路径由仓库根目录的 `utils/data_loader.py` 控制。
