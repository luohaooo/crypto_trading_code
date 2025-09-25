请在@binance_implement/notebook_test/ 中，使用ccxt库（可以在../ccxt 中查看）执行以下操作，请给testnet或者实盘的选择接口。

每隔五分钟：

1. 将所有仓位平仓
2. 检测是否全部平仓，确认
3. 打印当前合约保证金
4. 参考@/neural-strategy/strategies/factors/ohlc_figure_factor.py 分别提取每个币种前80h的ohlc数据，通过函数转化为图片后，通过模型转化输出为因子
5. 对因子值排序，前10名作为做多的币种，后10名作为做空的币种
6. 设置杠杆为1，将保证金平均分为20份，并依次开仓
7. 检测是否全部开仓，打印仓位

请给我实现的思路，不需要实现任何代码。