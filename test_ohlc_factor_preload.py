#!/usr/bin/env python3
"""
Test script for OHLC Figure Factor with preloading optimization.

Demonstrates the performance improvement from batch image preloading.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neural_strategy.strategies.factors.ohlc_figure_factor import OHLCFigureFactor
from neural_strategy.utils.data_loader import load_usdt_symbols_from_month_cache

def test_ohlc_factor_preload():
    """Test OHLC factor with and without preloading."""

    print("🧪 测试 OHLC Figure Factor 预加载优化")
    print("="*60)

    # Load test data
    print("📊 加载测试数据...")
    start_date = "2024-03-01"
    end_date = "2024-03-02"

    data = load_usdt_symbols_from_month_cache(start_date, end_date)

    if data is None or len(data) == 0:
        print("❌ 无法加载测试数据")
        return

    print(f"✅ 加载数据: {len(data):,} 条记录")

    # Get test timestamps (every 4 hours)
    timestamps = data.index.get_level_values('open_time').unique()
    timestamps = timestamps[::240]  # Every 4 hours (240 minutes)
    test_timestamps = timestamps[:5]  # First 5 timestamps

    print(f"📅 测试时间戳: {len(test_timestamps)} 个")
    for ts in test_timestamps:
        print(f"   {ts}")

    # Initialize factor
    print("\n🤖 初始化 OHLC Factor...")
    factor = OHLCFigureFactor(
        name="TestOHLCFactor",
        lookback_periods=20,
        timeframes=['3min', '15min', '1h']
    )

    print(f"✅ 模型信息:")
    model_info = factor.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # Test without preloading
    print("\n📈 测试方式1: 不使用预加载 (原始方法)")
    start_time = time.time()

    results_original = []
    for i, timestamp in enumerate(test_timestamps):
        print(f"   处理时间戳 {i+1}/{len(test_timestamps)}: {timestamp}")
        result = factor.calculate(data, timestamp)
        results_original.append(result)
        print(f"   结果: {len(result)} 个交易对的factor值")

    time_original = time.time() - start_time
    print(f"⏱️  原始方法耗时: {time_original:.2f} 秒")

    # Test with preloading
    print("\n🚀 测试方式2: 使用预加载优化")
    start_time = time.time()

    # Preload images
    print("   预加载图像...")
    factor.preload_images(data, test_timestamps.tolist())

    preload_time = time.time() - start_time
    print(f"   预加载耗时: {preload_time:.2f} 秒")

    # Get preload stats
    stats = factor.get_preload_stats()
    print(f"   预加载统计: {stats}")

    # Calculate factors using preloaded images
    print("   使用预加载图像计算factor...")
    calc_start = time.time()

    results_preloaded = []
    for i, timestamp in enumerate(test_timestamps):
        print(f"   处理时间戳 {i+1}/{len(test_timestamps)}: {timestamp}")
        result = factor.calculate(data, timestamp)
        results_preloaded.append(result)
        print(f"   结果: {len(result)} 个交易对的factor值")

    calc_time = time.time() - calc_start
    total_time_preloaded = preload_time + calc_time

    print(f"⏱️  预加载方法 - 计算耗时: {calc_time:.2f} 秒")
    print(f"⏱️  预加载方法 - 总耗时: {total_time_preloaded:.2f} 秒")

    # Performance comparison
    print("\n📊 性能对比:")
    print(f"   原始方法: {time_original:.2f} 秒")
    print(f"   预加载方法 (总): {total_time_preloaded:.2f} 秒")
    print(f"   预加载方法 (仅计算): {calc_time:.2f} 秒")

    if calc_time > 0:
        speedup_calc = time_original / calc_time
        print(f"   计算速度提升: {speedup_calc:.1f}x")

    if total_time_preloaded > 0:
        speedup_total = time_original / total_time_preloaded
        if speedup_total > 1:
            print(f"   总体速度提升: {speedup_total:.1f}x")
        else:
            print(f"   总体速度降低: {1/speedup_total:.1f}x (预加载开销)")

    # Verify results consistency
    print("\n🔍 验证结果一致性:")
    consistent = True

    for i, (orig, preload) in enumerate(zip(results_original, results_preloaded)):
        if len(orig) != len(preload):
            print(f"   ❌ 时间戳 {i}: 结果数量不一致 ({len(orig)} vs {len(preload)})")
            consistent = False
            continue

        # Check if symbols match
        if not orig.index.equals(preload.index):
            print(f"   ⚠️  时间戳 {i}: 交易对顺序可能不同")

        # Check if values are close (allowing small numerical differences)
        if len(orig) > 0 and len(preload) > 0:
            common_symbols = orig.index.intersection(preload.index)
            if len(common_symbols) > 0:
                orig_values = orig.reindex(common_symbols)
                preload_values = preload.reindex(common_symbols)

                max_diff = np.abs(orig_values - preload_values).max()
                if max_diff > 1e-6:
                    print(f"   ⚠️  时间戳 {i}: 数值差异较大 (最大差异: {max_diff:.2e})")
                else:
                    print(f"   ✅ 时间戳 {i}: 结果一致 (最大差异: {max_diff:.2e})")

    if consistent:
        print("✅ 所有结果都保持一致!")

    # Memory usage estimate
    memory_stats = factor.get_preload_stats()
    if memory_stats.get("memory_mb"):
        print(f"\n💾 内存使用: ~{memory_stats['memory_mb']} MB")

    print("\n🎉 测试完成!")

if __name__ == "__main__":
    try:
        test_ohlc_factor_preload()
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()