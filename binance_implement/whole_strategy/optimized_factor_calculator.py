"""
优化的因子计算器
使用torch模型对单个symbol进行因子计算，避免大数据量处理

功能：
- 预加载torch模型
- 逐个symbol处理，获取多时间框架数据
- 使用ohlc_to_image_with_volume转换为图像
- 通过神经网络模型推理获取因子值
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, List
from datetime import datetime
from ohlc_model import create_model

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

figure_model_root = os.path.join(project_root, 'figure_model')
sys.path.insert(0, figure_model_root)


from figure_model.ohlc2fig import ohlc_to_image_with_volume



class OptimizedFactorCalculator:
    """优化的因子计算器，基于torch模型的逐symbol处理"""

    def __init__(self, config, logger):
        """
        初始化优化因子计算器

        Args:
            config: 交易配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger

        # 模型配置
        self.timeframes = ['1h', '2h', '4h']  # 默认时间框架
        self.limit = 82  # 每个时间框架的数据点数
        self.image_height = 64
        self.image_width = 60  # 20个周期 * 3像素/周期

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型相关
        self.model = None
        self.model_path = None

        # 数据处理器
        self.data_processor = None

    def initialize(self):
        """初始化因子计算器"""
        try:
            # 从配置获取模型路径
            self.model_path = self.config.MODEL_PATH

            # 加载torch模型
            self._load_model()

            self.logger.info("[OK] 优化因子计算器初始化成功")
            self.logger.info(f"   设备: {self.device}")
            self.logger.info(f"   时间框架: {self.timeframes}")
            self.logger.info(f"   模型路径: {self.model_path}")

        except Exception as e:
            self.logger.error(f"[ERROR] 优化因子计算器初始化失败: {e}")
            raise

    def _load_model(self):
        """加载torch模型"""
        try:
            if not self.model_path:
                raise ValueError("模型路径未配置，请在 config.py 中设置 MODEL_PATH")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}\n"
                                      f"请检查 config.py 中的 MODEL_PATH 配置是否正确")

            # 创建模型架构
            self.model = create_model(use_parallel=torch.cuda.device_count() > 1)

            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 假设checkpoint直接是state dict
                self.model.load_state_dict(checkpoint)

            # 移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ 成功加载OHLC CNN模型: {self.model_path}")
            print(f"📱 使用设备: {self.device}")

        except Exception as e:
            self.logger.error(f"[ERROR] 加载模型失败: {e}")
            self.model = None
            raise

    def calculate_factors(self, data_processor) -> Optional[pd.Series]:
        """
        计算所有交易对的因子值

        Args:
            data_processor: 优化的数据处理器实例

        Returns:
            pd.Series: 因子值序列，index为symbol
        """
        if self.model is None:
            self.logger.error("[ERROR] 模型未加载，无法计算因子")
            return None

        try:
            print(f"\n🚀 开始计算因子...")
            print(f"时间框架: {self.timeframes}")
            print(f"每框架数据点: {self.limit}")

            # 获取所有活跃交易对的数据
            all_symbols_data = data_processor.get_all_symbols_factors(
                timeframes=self.timeframes,
                limit=self.limit
            )

            if not all_symbols_data:
                self.logger.error("[ERROR] 没有获取到有效的交易对数据")
                return None

            symbols = list(all_symbols_data.keys())
            print(f"✅ 成功获取 {len(symbols)} 个交易对的数据")

            # 逐个symbol计算因子值
            factor_values = {}
            successful_count = 0

            print(f"\n🎯 开始逐个计算因子值...")
            for symbol in tqdm(symbols, desc="计算因子", unit="交易对"):
                if symbol == "USDC/USDT:USDT":
                    continue
                try:
                    factor_value = self._calculate_single_symbol_factor(all_symbols_data[symbol])

                    if factor_value is not None:
                        factor_values[symbol] = factor_value
                        successful_count += 1

                except Exception as e:
                    # 静默跳过失败的交易对
                    continue

            if not factor_values:
                self.logger.error("[ERROR] 没有计算出任何有效因子值")
                return None

            # 转换为Series
            factor_series = pd.Series(factor_values, name='neural_factor')

            print(f"\n✅ 因子计算完成:")
            print(f"   成功计算: {successful_count} 个交易对")
            print(f"   因子值范围: {factor_series.min():.6f} ~ {factor_series.max():.6f}")

            return factor_series

        except Exception as e:
            self.logger.error(f"[ERROR] 计算因子失败: {e}")
            return None

    def _calculate_single_symbol_factor(self,
                                      symbol_data: pd.DataFrame) -> Optional[float]:
        """
        计算单个交易对的因子值

        Args:
            symbol: 交易对符号
            symbol_data: 该交易对的多时间框架数据

        Returns:
            float: 因子值，如果计算失败返回None
        """
        try:
            

            df_0 = symbol_data.iloc[-80:].copy()
            df_1 = symbol_data.iloc[-81:-1].copy()


            # 计算两个子因子
            sub_factor_0 = self._calculate_sub_factor(df_0)
            sub_factor_1 = self._calculate_sub_factor(df_1)
            return (sub_factor_0 + 0.9 * sub_factor_1) / 1.9

        except Exception as e:
            return None

    def _calculate_sub_factor(self, symbol_data: pd.DataFrame) -> Optional[float]:
        """
        计算子因子值

        Args:
            df: 单个时间框架的OHLCV数据

        Returns:
            float: 子因子值，如果计算失败返回None
        """
                    # 为每个时间框架生成图像
        images = []
        for timeframe in self.timeframes:
            if timeframe == '1h':
                df = symbol_data[-20:].copy()
            elif timeframe == '2h':
                df = aggregate_bars(symbol_data[-20*2:].copy(), window_hours=2).copy()
            elif timeframe == '4h':
                df = aggregate_bars(symbol_data[-20*4:].copy(), window_hours=4).copy()
        
            # 使用ohlc_to_image_with_volume转换为图像
            image = ohlc_to_image_with_volume(
                df,
                window=20,
                height=60,
                open_col='open',
                high_col='high',
                low_col='low',
                close_col='close',
                volume_col='volume'
            )

            images.append(image)

        # 将三个时间框架的图像叠在一起 (3, 64, 60)
        stacked_images = np.stack(images, axis=0)

        # 转换为torch tensor并添加batch维度 (1, 3, 64, 60)
        image_tensor = torch.from_numpy(stacked_images).float().unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # 使用模型进行推理
        with torch.no_grad():
            prediction = self.model(image_tensor)
            factor_value = prediction.cpu().numpy().flatten()[0]

        return float(factor_value)

    
        

    def set_model_path(self, model_path: str):
        """设置模型路径（已弃用，请使用config.py配置）"""
        import warnings
        warnings.warn("set_model_path 方法已弃用，请在 config.py 中配置 MODEL_PATH",
                     DeprecationWarning, stacklevel=2)
        self.model_path = model_path

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'timeframes': self.timeframes,
            'limit': self.limit,
            'image_size': (self.image_height, self.image_width),
            'model_loaded': self.model is not None
        }

    def cleanup(self):
        """清理资源"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("[OK] 优化因子计算器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理优化因子计算器失败: {e}")


def aggregate_bars(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    df = df.copy()
    # Create aggregated bars using rolling window
    # Generate lookback_periods bars in chronological order
    aggregated_bars = []

    # Generate bars from front to back (chronological order)
    for i in range(20):
        # Calculate indices for this bar
        start_idx = i * window_hours
        end_idx = start_idx + window_hours

        window = df.iloc[start_idx:end_idx]

        # Aggregate this window into one bar
        # open_time is the FIRST hour of the window (window start time)
        bar = {
            'open_time': window['open_time'].iloc[0],  # Use the first hour's timestamp as bar time
            'open': window['open'].iloc[0],
            'high': window['high'].max(),
            'low': window['low'].min(),
            'close': window['close'].iloc[-1],
            'volume': window['volume'].sum()
        }
        aggregated_bars.append(bar)

    # Convert to DataFrame
    aggregated = pd.DataFrame(aggregated_bars)
    aggregated = aggregated.set_index('open_time')

    return aggregated
