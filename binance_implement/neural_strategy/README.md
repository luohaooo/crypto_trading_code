# 自动化交易系统

基于神经网络OHLC因子的币安期货自动化交易系统，支持testnet和实盘环境。

## 🚀 功能特性

- **智能因子计算**: 集成神经网络OHLC图像因子，支持多时间框架分析 (1h, 2h, 4h)
- **自动化交易**: 每5分钟执行一轮完整的交易周期
- **风险控制**: 1倍杠杆，资金等分，最小保证金检查
- **环境切换**: 支持testnet测试环境和实盘环境
- **实时监控**: 详细的日志记录和性能监控
- **容错机制**: 异常处理和自动恢复

## 📋 交易流程

每个5分钟周期执行以下步骤：

1. **平仓**: 关闭所有现有仓位
2. **验证**: 确认所有仓位已平仓
3. **余额查询**: 获取当前合约保证金
4. **数据处理**: 提取80小时OHLC数据，转换为多时间框架图像
5. **因子计算**: 使用神经网络模型预测并计算因子值
6. **选币策略**: 因子排序，前10名做多，后10名做空
7. **开仓交易**: 设置1倍杠杆，将保证金平均分为20份开仓
8. **仓位验证**: 确认开仓完成并打印仓位信息

## 🛠️ 环境要求

### Python版本
- Python 3.7+

### 依赖包
```bash
pip install ccxt pandas numpy torch asyncio
```

### API配置

#### Testnet环境
```bash
export BINANCE_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_TESTNET_API_SECRET="your_testnet_secret"
```

#### 实盘环境
```bash
export BINANCE_API_KEY="your_live_api_key"
export BINANCE_API_SECRET="your_live_secret"
```

### 神经网络模型
确保神经网络模型文件位于正确路径：
```
figure_model/model_saved/baseline_epoch_17_train_0.00479_val_0.00397.pt
```

## 🚦 快速开始

### 1. 系统测试
运行组件测试以验证系统配置：
```bash
python test_system.py
```

### 2. Testnet环境运行
```bash
# 方法1: 使用示例脚本
python examples/run_testnet.py

# 方法2: 使用主脚本
python automated_trading.py
# 选择 1 (Testnet)
```

### 3. 实盘环境运行
```bash
# 方法1: 使用示例脚本
python examples/run_live.py

# 方法2: 使用主脚本
python automated_trading.py
# 选择 2 (实盘)
```

## 📂 项目结构

```
notebook_test/
├── automated_trading.py          # 主交易脚本
├── trading_executor.py           # 交易执行器
├── data_processor.py            # 数据处理器
├── factor_calculator.py         # 因子计算器
├── test_system.py              # 系统测试脚本
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── logger.py              # 日志工具
│   └── monitor.py             # 监控工具
└── examples/                   # 示例脚本
    ├── run_testnet.py         # Testnet运行示例
    └── run_live.py            # 实盘运行示例
```

## ⚙️ 配置说明

配置参数在 `../config.py` 中定义：

### 基础交易配置
- `REBALANCE_INTERVAL`: 300秒 (5分钟交易周期)
- `TOP_N_LONG`: 10 (做多币种数量)
- `TOP_N_SHORT`: 10 (做空币种数量)
- `LEVERAGE`: 1 (杠杆倍数)
- `LOOKBACK_HOURS`: 80 (历史数据回看小时数)

### 神经网络配置
- `TIMEFRAMES`: ['1h', '2h', '4h'] (多时间框架)
- `LOOKBACK_PERIODS`: 20 (OHLC回看周期数)

### 风险控制配置
- `MAX_POSITION_PCT`: 0.05 (单仓位最大占比 5%)
- `MIN_MARGIN_BALANCE`: 100 (最小保证金余额 USDT)
- `MAX_DAILY_TRADES`: 100 (每日最大交易次数)

## 📊 监控和日志

### 日志文件
- 自动生成带时间戳的日志文件
- 控制台和文件双重输出
- 详细的交易操作记录

### 性能监控
- 交易周期成功率统计
- 平均执行时间监控
- 系统健康状态检查
- 实时余额变化跟踪

## ⚠️ 风险提示

### Testnet环境
- 使用虚拟资金，安全可靠
- 建议充分测试策略参数
- 验证所有功能正常运行

### 实盘环境
- **使用真实资金，存在损失风险**
- 建议从小额资金开始
- 持续监控系统运行状态
- 设置合理的止损机制

## 🔧 故障排除

### 常见问题

1. **API连接失败**
   - 检查API密钥是否正确配置
   - 确认网络连接正常
   - 验证API权限设置

2. **神经网络模型加载失败**
   - 检查模型文件路径
   - 确认依赖包已安装
   - 查看详细错误日志

3. **数据获取失败**
   - 检查网络连接
   - 确认交易对有效性
   - 查看API请求限制

4. **交易执行失败**
   - 检查账户余额充足
   - 确认交易权限开启
   - 查看订单执行日志

### 调试模式
在测试环境中，可以通过修改日志级别来获取更详细的调试信息：
```python
self.config.LOG_LEVEL = 'DEBUG'
```

## 📈 性能优化

### 数据处理优化
- 并行获取多个交易对数据
- 缓存机制减少重复计算
- 批量处理图像生成

### 内存管理
- 定期清理缓存数据
- 限制预加载图像数量
- 优化数据结构使用

### 网络优化
- 合理设置请求间隔
- 使用连接池管理
- 实现请求重试机制

## 🤝 支持和反馈

如果遇到问题或有改进建议，请：

1. 查看日志文件获取详细错误信息
2. 运行测试脚本验证各组件状态
3. 检查配置参数是否合理
4. 参考故障排除指南

## 📄 许可证

本项目仅供学习和研究使用。使用者需要：

- 遵守相关法律法规
- 自行承担交易风险
- 不得用于非法用途

## 🔄 更新日志

### v1.0.0 (2024-09-23)
- 完整的自动化交易系统实现
- 神经网络OHLC因子集成
- 支持testnet和实盘环境
- 完善的监控和日志系统
- 详细的文档和示例

---

**⚠️ 重要提醒**: 量化交易存在风险，过往表现不能保证未来收益。请在充分理解和测试后再使用实盘环境。