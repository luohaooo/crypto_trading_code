"""
因子计算器模块
集成神经网络OHLC图像因子，计算交易信号

功能：
- 集成现有的OHLCFigureFactor
- 多时间框架OHLC图像生成
- 神经网络预测和因子计算
- 因子排序和筛选
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import warnings

# 添加神经策略路径
neural_strategy_path = os.path.join(os.path.dirname(__file__), '..', '..', 'neural-strategy')
sys.path.insert(0, neural_strategy_path)

# 导入OHLC因子
try:
    from strategies.factors.ohlc_figure_factor import OHLCFigureFactor
except ImportError as e:
    print(f"警告: 无法导入OHLCFigureFactor: {e}")
    OHLCFigureFactor = None


class FactorCalculator:
    """因子计算器类"""

    def __init__(self, config, logger):
        """
        初始化因子计算器

        Args:
            config: 交易配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.ohlc_factor = None

    async def initialize(self):
        """初始化因子计算器"""
        try:
            if OHLCFigureFactor is None:
                raise ImportError("OHLCFigureFactor未能正确导入")

            # 初始化OHLC图像因子
            self.logger.info("[NEURAL] 初始化神经网络因子...")

            self.ohlc_factor = OHLCFigureFactor(
                model_path=self.config.MODEL_PATH,
                timeframes=self.config.TIMEFRAMES,  # ['1h', '2h', '4h']
                lookback_periods=self.config.LOOKBACK_PERIODS,  # 20
                device='auto'
            )

            # 打印模型信息
            model_info = self.ohlc_factor.get_model_info()
            self.logger.info("[MODEL] 神经网络模型信息:")
            self.logger.info(f"   模型路径: {model_info.get('model_path', 'N/A')}")
            self.logger.info(f"   设备: {model_info.get('device', 'N/A')}")
            self.logger.info(f"   时间框架: {model_info.get('timeframes', [])}")
            self.logger.info(f"   回看周期: {model_info.get('lookback_periods', 0)}")
            self.logger.info(f"   图像尺寸: {model_info.get('image_size', (0, 0))}")
            self.logger.info(f"   模型已加载: {model_info.get('model_loaded', False)}")

            if not model_info.get('model_loaded', False):
                self.logger.warning("[WARNING] 神经网络模型未正确加载，将使用随机因子")

            self.logger.info("[OK] 因子计算器初始化成功")

        except Exception as e:
            self.logger.error(f"[ERROR] 因子计算器初始化失败: {e}")
            self.logger.warning("[WARNING] 将使用备用随机因子计算器")
            self.ohlc_factor = None

    async def calculate_factors(self, data: pd.DataFrame,
                              timestamp: datetime) -> Optional[pd.Series]:
        """
        计算交易因子

        Args:
            data: 历史OHLC数据 (MultiIndex: open_time, symbol)
            timestamp: 当前时间戳

        Returns:
            pd.Series: 因子值序列，index为symbol
        """
        try:
            self.logger.info("[CALC] 开始计算交易因子...")

            if data is None or len(data) == 0:
                self.logger.error("[ERROR] 输入数据为空")
                return None

            # 获取当前可用的交易对
            available_symbols = data.index.get_level_values('symbol').unique().tolist()
            self.logger.info(f"[SYMBOLS] 可用交易对: {len(available_symbols)} 个")

            if len(available_symbols) == 0:
                self.logger.error("[ERROR] 没有可用的交易对")
                return None

            # 检查数据时间范围
            time_range = data.index.get_level_values('open_time')
            self.logger.info(f"[TIME] 数据时间范围: {time_range.min()} ~ {time_range.max()}")
            self.logger.info(f"[DATA] 总数据点: {len(data):,}")

            # 使用神经网络因子计算
            if self.ohlc_factor is not None:
                factors = await self._calculate_neural_factors(data, timestamp, available_symbols)
            else:
                self.logger.error("[ERROR] 神经网络因子不可用，无法计算因子")
                return None

            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 因子计算失败")
                return None

            self.logger.info(f"[OK] 因子计算完成: {len(factors)} 个有效因子")
            self.logger.info(f"[STATS] 因子值范围: {factors.min():.6f} ~ {factors.max():.6f}")

            return factors

        except Exception as e:
            self.logger.error(f"[ERROR] 因子计算异常: {e}")
            return None

    async def _calculate_neural_factors(self, data: pd.DataFrame, timestamp: datetime,
                                      symbols: List[str]) -> Optional[pd.Series]:
        """使用神经网络计算因子"""
        try:
            self.logger.info("[NEURAL] 使用神经网络计算因子...")

            # 验证数据充足性
            if not self.ohlc_factor.validate_data(data, timestamp):
                self.logger.error("[ERROR] 数据不足，无法使用神经网络因子")
                return None

            # 预加载图像以提高性能
            self.logger.info("[IMAGE] 预加载OHLC图像...")
            timestamps_to_preload = [timestamp]

            # 执行预加载（异步）
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.ohlc_factor.preload_images,
                data,
                timestamps_to_preload
            )

            # 检查预加载统计
            preload_stats = self.ohlc_factor.get_preload_stats()
            if preload_stats.get('status') == 'preloaded':
                self.logger.info(f"[OK] 预加载完成: {preload_stats.get('total_images', 0)} 张图像")
            else:
                self.logger.warning("[WARNING] 图像预加载失败，使用实时计算")

            # 计算因子
            self.logger.info("[PREDICT] 执行神经网络预测...")
            factors = await asyncio.get_event_loop().run_in_executor(
                None,
                self.ohlc_factor.calculate,
                data,
                timestamp
            )

            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 神经网络因子计算失败")
                return None

            self.logger.info(f"[OK] 神经网络因子计算成功: {len(factors)} 个")
            return factors

        except Exception as e:
            self.logger.error(f"[ERROR] 神经网络因子计算异常: {e}")
            return None

    def get_factor_statistics(self, factors: pd.Series) -> Dict:
        """获取因子统计信息"""
        try:
            if factors is None or len(factors) == 0:
                return {"error": "no_factors"}

            stats = {
                "count": len(factors),
                "mean": float(factors.mean()),
                "std": float(factors.std()),
                "min": float(factors.min()),
                "max": float(factors.max()),
                "median": float(factors.median()),
                "q25": float(factors.quantile(0.25)),
                "q75": float(factors.quantile(0.75))
            }

            return stats

        except Exception as e:
            self.logger.error(f"[ERROR] 计算因子统计失败: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """清理资源"""
        try:
            if self.ohlc_factor is not None:
                self.ohlc_factor.clear_cache()

            self.logger.info("[OK] 因子计算器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理因子计算器失败: {e}")

    def is_neural_factor_available(self) -> bool:
        """检查神经网络因子是否可用"""
        return self.ohlc_factor is not None