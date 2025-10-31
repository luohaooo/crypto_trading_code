import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import gc
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add project root to path
parent = Path.cwd().parent.resolve()
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

# Import required modules
from utils.data_loader import load_usdt_symbols_from_month_cache, get_usdt_symbols
from figure_model.ohlc_preprocessor import aggregate_ohlc_data, batch_aggregate_ohlc_data
from figure_model.ohlc2fig import ohlc_to_image_without_volume, ohlc_to_image_with_volume

RETURN_HORIZONS = [1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,32,36,42,48,54,60,72,84,96,108,120,144]

TIMEFRAMES = ['1h', '2h', '4h']

TIMEFRAMES_SPAN_MINUTES = {'1h':60, '2h': 120, '4h': 240}  # 时间框架对应分钟数

WINDOW_SIZE = 20  # 每个图像的OHLC条数
IMAGE_HEIGHT = 64  # 图像高度
SAMPLE_INTERVAL_HOURS = 4  # 采样间隔(小时)

# 数据加载时间缓冲区
BEFORE_TIME_SPAN = timedelta(hours=80)  
AFTER_TIME_SPAN = timedelta(hours=168)  

# 输出目录
OUTPUT_DIR = Path('/home/craz/crypto/model_training/ohlc_img_dataset_4')
OUTPUT_DIR.mkdir(exist_ok=True)

def month_iter(start_year, start_month, end_year, end_month):
    """生成月份迭代器"""
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield datetime(y, m, 1).date()
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

def get_sampling_timestamps(start_time: datetime, end_time: datetime, interval_hours: int) -> List[datetime]:
    """生成采样时间戳列表"""
    timestamps = []
    current = start_time 
    while current <= end_time:
        timestamps.append(current)
        current += timedelta(hours=interval_hours)
    return timestamps

def extract_ohlc_window(data: pd.DataFrame, symbol: str, end_time: datetime, 
                       window_size: int = 20) -> Optional[pd.DataFrame]:
    """提取指定符号和结束时间的OHLC窗口数据"""
    try:
        # 获取该符号的数据
        symbol_data = data.xs(symbol, level='symbol')
        
        # 找到结束时间在时间序列中的位置
        timestamps = symbol_data.index
        
        # 找到最接近end_time的时间戳
        end_idx = timestamps.get_indexer([end_time], method='nearest')[0]
        if end_idx == -1 or end_idx < window_size:
            return None
        
        # 提取窗口数据
        start_idx = end_idx - window_size
        window_data = symbol_data.iloc[start_idx:end_idx]
        
        if len(window_data) != window_size:
            return None
            
        return window_data
        
    except Exception as e:
        return None

def calculate_future_returns(data_1h: pd.DataFrame, symbol: str, current_time: datetime,
                           horizons: List[int]) -> Optional[np.ndarray]:
    """计算未来收益率"""
    try:
        # 获取该符号的1小时数据
        symbol_data = data_1h.xs(symbol, level='symbol')
        timestamps = symbol_data.index
        
        # 找到当前时间最近的价格
        current_idx = timestamps.get_indexer([current_time], method='nearest')[0]
        if current_idx == -1:
            return None
        
        current_price = symbol_data.iloc[current_idx]['open']
        
        # 计算各个时间窗口的未来收益率
        returns = []
        for horizon_hours in horizons:
            future_time = current_time + timedelta(hours=horizon_hours)
            
            # 找到未来时间最接近的数据点
            future_candidates = timestamps[timestamps >= future_time]
            if len(future_candidates) == 0:
                return None  # 没有足够的未来数据
            
            # 使用第一个候选时间戳（最接近的未来时间点）
            future_timestamp = future_candidates[0]
            future_idx = timestamps.get_loc(future_timestamp)  # 使用get_loc代替get_indexer
            
            future_price = symbol_data.iloc[future_idx]['open']
            return_pct = (future_price - current_price) / current_price
            returns.append(return_pct)
        
        return np.array(returns, dtype=np.float32)
        
    except Exception as e:
        print(f"计算未来收益率时出错: {e}")
        return None

def generate_ohlc_image(window_data: pd.DataFrame, height: int = 64, 
                       include_volume: bool = True) -> np.ndarray:
    """将OHLC数据转换为图像"""
    if include_volume:
        return ohlc_to_image_with_volume(window_data, window=len(window_data), height=height)
    else:
        return ohlc_to_image_without_volume(window_data, window=len(window_data), height=height)

def process_single_month(year: int, month: int, 
                        symbols_limit: Optional[int] = None,
                        samples_per_symbol_limit: Optional[int] = None) -> Dict[str, Dict]:
    """处理单个月份的数据生成 - 确保所有时间框架数据对齐"""
    
    print(f"\n{'='*60}")
    print(f"处理 {year:04d}-{month:02d}")
    print(f"{'='*60}")
    
    # 计算月份日期范围
    import calendar
    first_day = datetime(year, month, 1).date()
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_num).date()
    
    month_start = datetime.combine(first_day, datetime.min.time())
    month_end = datetime.combine(last_day, datetime.max.time())
    
    print(f"月份范围: {month_start} 到 {month_end}")
    
    # 加载数据(包含前后缓冲区用于计算收益率)
    data_start = month_start - BEFORE_TIME_SPAN
    data_end = month_end + AFTER_TIME_SPAN
    
    print(f"加载数据范围: {data_start} 到 {data_end}")
    
    try:
        # 加载月度数据
        month_data = load_usdt_symbols_from_month_cache(
            start_date=data_start.strftime('%Y-%m-%d'),
            end_date=data_end.strftime('%Y-%m-%d')
        )
        
        if month_data is None or len(month_data) == 0:
            print("❌ 未能加载到数据")
            return {}
            
        print(f"✅ 成功加载数据: {len(month_data):,} 条记录")
        
        # 获取可用的交易对列表（在释放数据前）
        symbols_file = Path("/home/craz/crypto/crypto-data/available_symbol") / f"{year:04d}-{month:02d}_symbols.pkl"
        available_symbols: List[str] = []

        if symbols_file.exists():
            try:
                loaded_symbols = pd.read_pickle(symbols_file)
                if isinstance(loaded_symbols, (list, tuple, set, np.ndarray, pd.Index)):
                    available_symbols = list(loaded_symbols)
                elif isinstance(loaded_symbols, pd.DataFrame):
                    if 'symbol' in loaded_symbols.columns:
                        available_symbols = loaded_symbols['symbol'].astype(str).tolist()
                    else:
                        available_symbols = loaded_symbols.squeeze().astype(str).tolist()
                else:
                    available_symbols = list(loaded_symbols)
            except Exception as exc:
                print(f"⚠️ 无法从 {symbols_file} 读取交易对列表: {exc}")

        if not available_symbols:
            available_symbols = month_data.index.get_level_values('symbol').unique().tolist()

        available_symbols = [
            symbol for symbol in available_symbols
            if symbol in month_data.index.get_level_values('symbol').unique()
        ]

        if symbols_limit:
            available_symbols = available_symbols[:symbols_limit]
        print(f"处理 {len(available_symbols)} 个交易对")
        
        # 聚合到不同时间框架
        print("聚合数据到不同时间框架...")
        aggregated_data = batch_aggregate_ohlc_data(month_data, timeframes=TIMEFRAMES)
        
        # 释放原始数据内存
        del month_data
        gc.collect()
        
        if not aggregated_data:
            print("❌ 数据聚合失败")
            return {}
            
        print(f"✅ 聚合完成，包含时间框架: {list(aggregated_data.keys())}")
        
        # 确保有1小时数据用于计算收益率
        if '1h' not in aggregated_data:
            print("❌ 缺少1小时数据，无法计算收益率")
            return {}
        
        data_1h = aggregated_data['1h']
        
        # 生成采样时间戳
        sampling_timestamps = get_sampling_timestamps(
            month_start, month_end, SAMPLE_INTERVAL_HOURS
        )
        print(f"生成 {len(sampling_timestamps)} 个采样时间点")
        
        # 新的对齐处理方式：先确定有效样本，然后为所有时间框架生成相同的样本
        print(f"\n🎯 开始数据对齐处理...")
        
        # Step 1: 确定有效的 (symbol, timestamp) 组合
        valid_samples = []
        
        print("第一步：确定所有时间框架都有效的样本组合...")
        for symbol in available_symbols:
            symbol_samples = 0
            
            for timestamp in sampling_timestamps:
                if samples_per_symbol_limit and symbol_samples >= samples_per_symbol_limit:
                    break
                
                # 检查所有时间框架是否都有有效数据
                all_timeframes_valid = True
                
                # 检查是否可以计算未来收益率
                future_returns = calculate_future_returns(data_1h, symbol, timestamp, RETURN_HORIZONS)
                if future_returns is None:
                    continue
                
                # 检查每个时间框架是否都有有效的OHLC窗口
                for timeframe in TIMEFRAMES:
                    if timeframe not in aggregated_data:
                        all_timeframes_valid = False
                        break
                    
                    window_data = extract_ohlc_window(
                        aggregated_data[timeframe], symbol, timestamp, WINDOW_SIZE
                    )
                    if window_data is None:
                        all_timeframes_valid = False
                        break
                
                # 只有当所有时间框架都有效时，才添加到有效样本列表
                if all_timeframes_valid:
                    valid_samples.append({
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'future_returns': future_returns
                    })
                    symbol_samples += 1
        
        print(f"✅ 找到 {len(valid_samples)} 个对齐的有效样本")
        
        if len(valid_samples) == 0:
            print("❌ 没有找到有效的对齐样本")
            return {}
        
        # Step 2: 为所有时间框架生成相同的图像样本
        results = {'images': {}, 'labels': None, 'metadata': None}
        
        # 提取统一的标签和元数据（所有时间框架共享）
        labels_array = np.array([sample['future_returns'] for sample in valid_samples])
        shared_metadata = [{
            'symbol': sample['symbol'],
            'timestamp': sample['timestamp'],
            'year': year,
            'month': month
        } for sample in valid_samples]
        
        results['labels'] = labels_array
        results['metadata'] = shared_metadata
        
        print(f"✅ 统一标签数组形状: {labels_array.shape}")
        
        # 为每个时间框架生成图像
        for timeframe in TIMEFRAMES:
            print(f"\n--- 处理时间框架: {timeframe} ---")
            
            tf_data = aggregated_data[timeframe]
            images = []
            successful_samples = 0
            
            # 使用相同的有效样本列表
            for sample_info in tqdm(valid_samples, desc=f"生成{timeframe}图像"):
                symbol = sample_info['symbol']
                timestamp = sample_info['timestamp']
                
                # 提取OHLC窗口数据（我们已经验证过这是有效的）
                window_data = extract_ohlc_window(tf_data, symbol, timestamp, WINDOW_SIZE)

                # if timeframe == '4h':
                #     print('-------------')
                #     print(symbol, timestamp)
                #     print(window_data)
                
                # 生成图像
                try:
                    image = generate_ohlc_image(
                        window_data, IMAGE_HEIGHT, include_volume=True
                    )
                    
                    images.append(image)
                    successful_samples += 1
                    
                except Exception as e:
                    print(f"图像生成错误 {symbol} @ {timestamp}: {e}")
                    # 注意：这里不应该发生错误，因为我们已经验证过数据有效性
                    continue
            
            print(f"✅ {timeframe}: {successful_samples}/{len(valid_samples)} 个样本成功生成")
            
            if len(images) > 0:
                # 转换为张量
                try:
                    images_tensor = torch.from_numpy(np.stack(images)).unsqueeze(1).float()  # (N, 1, H, W)
                    
                    print(f"张量形状 - 图像: {images_tensor.shape}")
                    
                    results['images'][timeframe] = {
                        'images': images_tensor,
                        'config': {
                            'year': year,
                            'month': month,
                            'timeframe': timeframe,
                            'window_size': WINDOW_SIZE,
                            'image_height': IMAGE_HEIGHT,
                            'sample_interval_hours': SAMPLE_INTERVAL_HOURS,
                            'return_horizons': RETURN_HORIZONS,
                            'num_samples': len(images)
                        }
                    }
                except ImportError as e:
                    print(f"⚠️ PyTorch未安装，使用NumPy保存: {e}")
                    # 使用NumPy数组作为备选
                    images_array = np.stack(images)[:, np.newaxis, :, :]  # (N, 1, H, W)
                    
                    print(f"NumPy数组形状 - 图像: {images_array.shape}")
                    
                    results['images'][timeframe] = {
                        'images': images_array,
                        'config': {
                            'year': year,
                            'month': month,
                            'timeframe': timeframe,
                            'window_size': WINDOW_SIZE,
                            'image_height': IMAGE_HEIGHT,
                            'sample_interval_hours': SAMPLE_INTERVAL_HOURS,
                            'return_horizons': RETURN_HORIZONS,
                            'num_samples': len(images)
                        }
                    }
            else:
                print(f"⚠️  {timeframe}: 没有生成任何样本")
        
        # 验证所有时间框架的样本数量是否一致
        sample_counts = [len(results['images'][tf]['images']) for tf in TIMEFRAMES if tf in results['images']]
        if len(set(sample_counts)) == 1:
            print(f"\n✅ 数据对齐验证通过：所有时间框架都有 {sample_counts[0]} 个样本")
        else:
            print(f"\n⚠️ 数据对齐警告：样本数量不一致 {dict(zip(TIMEFRAMES, sample_counts))}")
        
        # 清理内存
        del aggregated_data
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ 处理月份时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}

def save_monthly_results(year: int, month: int, results: Dict[str, Dict]):
    """保存月度结果 - 新的保存方式"""
    if not results or 'images' not in results:
        print(f"⚠️  {year:04d}-{month:02d}: 没有数据需要保存")
        return
    
    print(f"\n保存 {year:04d}-{month:02d} 数据集...")
    
    # 1. 保存每个时间框架的图像
    for timeframe, data in results['images'].items():
        images_filename = f"images_{timeframe}_{year:04d}-{month:02d}"
        
        try:
            # 尝试使用torch保存
            if hasattr(data['images'], 'numpy'):
                torch.save({
                    'images': data['images'],
                    'metadata': results['metadata'],
                    'config': data['config']
                }, OUTPUT_DIR / f"{images_filename}.pt")
                images_filepath = OUTPUT_DIR / f"{images_filename}.pt"
            else:
                # 使用NumPy保存
                np.savez_compressed(OUTPUT_DIR / f"{images_filename}.npz",
                                  images=data['images'],
                                  metadata=results['metadata'],
                                  config=data['config'])
                images_filepath = OUTPUT_DIR / f"{images_filename}.npz"
        except:
            # 备用方案：使用NumPy
            np.savez_compressed(OUTPUT_DIR / f"{images_filename}.npz",
                              images=data['images'],
                              metadata=results['metadata'],
                              config=data['config'])
            images_filepath = OUTPUT_DIR / f"{images_filename}.npz"
        
        # 显示图像文件信息
        images_size_mb = images_filepath.stat().st_size / (1024 * 1024)
        num_samples = len(data['images'])
        print(f"  ✅ 图像文件 {timeframe}: {images_filepath.name} ({num_samples} 样本, {images_size_mb:.1f} MB)")
    
    # 2. 按收益时间窗口保存标签（每个时间窗口一个文件）
    if results['labels'] is not None:
        labels_array = results['labels']  # 形状: (N, 16)
        
        print(f"\n保存标签文件 - 按收益时间窗口分离:")
        for i, horizon_hours in enumerate(RETURN_HORIZONS):
            # 提取第i列（对应第i个时间窗口的收益率）
            horizon_labels = labels_array[:, i]  # 形状: (N,)
            
            # 转换时间窗口为天数表示（如果>=24小时）

            horizon_str = f"{horizon_hours}h"
            
            labels_filename = f"labels_{horizon_str}_{year:04d}-{month:02d}"
            
            try:
                # 尝试使用torch保存
                horizon_tensor = torch.from_numpy(horizon_labels).float()
                torch.save({
                    'labels': horizon_tensor,
                    'metadata': results['metadata'],
                    'config': {
                        'year': year,
                        'month': month,
                        'horizon_hours': horizon_hours,
                        'horizon_str': horizon_str,
                        'num_samples': len(horizon_labels)
                    }
                }, OUTPUT_DIR / f"{labels_filename}.pt")
                labels_filepath = OUTPUT_DIR / f"{labels_filename}.pt"
            except:
                # 备用方案：使用NumPy
                np.savez_compressed(OUTPUT_DIR / f"{labels_filename}.npz",
                                  labels=horizon_labels,
                                  metadata=results['metadata'],
                                  config={
                                      'year': year,
                                      'month': month,
                                      'horizon_hours': horizon_hours,
                                      'horizon_str': horizon_str,
                                      'num_samples': len(horizon_labels)
                                  })
                labels_filepath = OUTPUT_DIR / f"{labels_filename}.npz"
            
            # 显示标签文件信息
            labels_size_kb = labels_filepath.stat().st_size / 1024
            print(f"    {horizon_str}: {labels_filepath.name} ({len(horizon_labels)} 样本, {labels_size_kb:.1f} KB)")

# 批量处理多个月份的数据 - 生产环境使用
# 注意: 这个单元格会处理大量数据，运行时间很长，请根据需要调整参数

BATCH_PROCESS = True  # 设置为True以启用批量处理
START_YEAR, START_MONTH = 2025, 4
END_YEAR, END_MONTH = 2025, 8

if BATCH_PROCESS:
    print(f"开始批量处理: {START_YEAR}-{START_MONTH:02d} 到 {END_YEAR}-{END_MONTH:02d}")
    print("⚠️  这将处理大量数据，可能需要数小时时间")
    
    successful_months = 0
    failed_months = 0
    
    for month_date in month_iter(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        year, month = month_date.year, month_date.month
        
        try:
            # 处理完整月份 (不限制交易对和样本数)
            monthly_results = process_single_month(year, month)
            
            if monthly_results:
                save_monthly_results(year, month, monthly_results)
                successful_months += 1
                print(f"✅ {year}-{month:02d} 处理成功")
            else:
                failed_months += 1
                print(f"❌ {year}-{month:02d} 处理失败")
                
        except Exception as e:
            failed_months += 1
            print(f"❌ {year}-{month:02d} 出现异常: {e}")
            continue
    
    print(f"\n批量处理完成!")
    print(f"成功: {successful_months} 个月")
    print(f"失败: {failed_months} 个月")
    
    # 最终文件统计
    final_files = list(OUTPUT_DIR.glob("*.pt")) + list(OUTPUT_DIR.glob("*.npz"))
    total_size_mb = sum(f.stat().st_size for f in final_files) / (1024 * 1024)
    print(f"\n生成文件: {len(final_files)} 个")
    print(f"总大小: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
else:
    print("批量处理已禁用。要启用批量处理，请将 BATCH_PROCESS 设置为 True")
