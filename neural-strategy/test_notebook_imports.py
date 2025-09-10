#!/usr/bin/env python3
"""
测试 Jupyter notebook 导入逻辑
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# 模拟 Jupyter notebook 的路径设置
neural_strategy_root = '/home/craz/crypto/crypto-trading/neural-strategy'
project_root = os.path.abspath(os.path.join(neural_strategy_root, '..'))

print(f"🐍 Python version: {sys.version}")
print(f"📁 Project root: {project_root}")
print(f"📁 Neural strategy root: {neural_strategy_root}")

# 添加路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if neural_strategy_root not in sys.path:
    sys.path.insert(0, neural_strategy_root)

# 使用动态导入
import importlib.util

def import_neural_strategy_module(module_name, file_path):
    """动态导入神经策略模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print("\n🔧 开始导入核心模块...")

# 导入配置模块
try:
    config_module = import_neural_strategy_module(
        "config", 
        os.path.join(neural_strategy_root, "utils", "config.py")
    )
    BacktestConfig = config_module.BacktestConfig
    print("✅ BacktestConfig 导入成功")
except Exception as e:
    print(f"❌ 配置模块导入失败: {e}")
    raise

# 导入回测引擎
try:
    engine_module = import_neural_strategy_module(
        "engine", 
        os.path.join(neural_strategy_root, "backtest", "engine.py")
    )
    BacktestEngine = engine_module.BacktestEngine
    print("✅ BacktestEngine 导入成功")
except Exception as e:
    print(f"❌ 回测引擎导入失败: {e}")
    raise

# 导入性能分析模块
try:
    performance_module = import_neural_strategy_module(
        "performance", 
        os.path.join(neural_strategy_root, "backtest", "performance.py")
    )
    PerformanceAnalyzer = performance_module.PerformanceAnalyzer
    PerformanceVisualizer = performance_module.PerformanceVisualizer
    print("✅ PerformanceAnalyzer & PerformanceVisualizer 导入成功")
except Exception as e:
    print(f"❌ 性能分析模块导入失败: {e}")
    raise

print("\n🎉 所有核心模块导入成功！")

# 测试配置创建
print("\n🧪 测试配置创建...")
config = BacktestConfig.create_default(
    start_date='2024-01-01',
    end_date='2024-01-02'
)
config.strategy.top_n = 2
config.strategy.bottom_n = 2
config.factor.lookback_periods = 60
config.initial_warmup_periods = 100

print(f"配置创建成功: {config.strategy.name}")

# 验证配置
errors = config.validate()
if errors:
    print(f"❌ 配置错误: {errors}")
else:
    print("✅ 配置验证通过！")

print("\n🎊 Jupyter notebook 导入测试完成！")
print("现在可以在 Jupyter notebook 中使用这些模块了。")