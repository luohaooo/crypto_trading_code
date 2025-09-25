"""
数据处理器模块
负责从交易所获取OHLC数据，进行多时间框架聚合和预处理

功能：
- 获取历史OHLC数据（80小时）
- 多时间框架数据聚合
- 数据格式转换和清理
"""

import asyncio
import ccxt.pro as ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from tqdm import tqdm


class DataProcessor:
    """数据处理器类"""

    def __init__(self, config, logger):
        """
        初始化数据处理器

        Args:
            config: 交易配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.exchange = None
        self.data_cache = {}

    async def initialize(self):
        """初始化数据处理器"""
        try:
            # 创建CCXT交易所实例（用于数据获取）
            self.exchange = ccxt.binance(self.config.get_ccxt_config())

            # 测试数据连接
            await self._test_data_connection()

            self.logger.info("[OK] 数据处理器初始化成功")

        except Exception as e:
            self.logger.error(f"[ERROR] 数据处理器初始化失败: {e}")
            raise

    async def _test_data_connection(self):
        """测试数据连接"""
        try:
            # 测试获取一个样本数据
            test_symbol = 'BTC/USDT:USDT'
            current_time = datetime.now()
            since = int((current_time - timedelta(hours=1)).timestamp() * 1000)

            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=test_symbol,
                timeframe='1h',  # 使用1小时时间框架
                since=since,
                limit=10
            )

            if ohlcv and len(ohlcv) > 0:
                self.logger.info("[OK] 数据连接测试成功")
            else:
                raise Exception("无法获取测试数据")

        except Exception as e:
            self.logger.error(f"[ERROR] 数据连接测试失败: {e}")
            raise

    async def get_historical_data(self, hours: int = 80) -> Optional[pd.DataFrame]:
        """
        获取历史OHLC数据

        Args:
            hours: 获取多少小时的历史数据

        Returns:
            pd.DataFrame: 多币种OHLC数据，MultiIndex (open_time, symbol)
        """
        try:
            self.logger.info(f"[DATA] 开始获取 {hours} 小时历史数据...")

            # 获取活跃交易对
            active_symbols = await self._get_active_trading_symbols()
            if not active_symbols:
                self.logger.error("[ERROR] 没有找到活跃交易对")
                return None

            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # 使用进度条逐个获取数据
            all_data = []

            print(f"获取 {len(active_symbols)} 个交易对的历史数据...")

            # 使用tqdm显示API请求进度
            for symbol in tqdm(active_symbols, desc="获取数据", unit="交易对"):
                try:
                    result = await self._fetch_symbol_ohlcv(symbol, start_time, end_time)

                    if result is not None and len(result) > 0:
                        result['symbol'] = symbol
                        all_data.append(result)

                except Exception as e:
                    # 静默跳过失败的交易对
                    continue

            if not all_data:
                self.logger.error("[ERROR] 没有获取到任何有效数据")
                return None

            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)

            # 转换为MultiIndex格式
            combined_data['open_time'] = pd.to_datetime(combined_data['open_time'])
            combined_data = combined_data.set_index(['open_time', 'symbol']).sort_index()

            # 数据充足性验证 - 必须严格满足80小时数据要求
            required_data_points = hours  # 80小时 = 80个1小时数据点

            valid_symbols = self.get_symbols_with_sufficient_data(combined_data, required_data_points)

            if not valid_symbols:
                self.logger.error("[ERROR] 没有交易对满足80小时完整数据要求")
                return None

            # 只保留数据充足的交易对
            combined_data = combined_data[combined_data.index.get_level_values('symbol').isin(valid_symbols)]

            self.logger.info(f"[OK] 数据获取完成:")
            self.logger.info(f"   时间范围: {combined_data.index.get_level_values('open_time').min()} ~ {combined_data.index.get_level_values('open_time').max()}")
            self.logger.info(f"   满足80小时要求的交易对数: {len(combined_data.index.get_level_values('symbol').unique())}")
            self.logger.info(f"   总记录数: {len(combined_data):,}")

            return combined_data

        except Exception as e:
            self.logger.error(f"[ERROR] 获取历史数据失败: {e}")
            return None

    async def _get_active_trading_symbols(self) -> List[str]:
        """获取活跃的USDT永续合约交易对"""
        try:
            # 获取所有市场信息
            markets = await self.exchange.load_markets()

            # 筛选USDT永续合约
            usdt_futures = []
            for symbol, market in markets.items():
                if (market.get('type') == 'swap' and
                    market.get('quote') == 'USDT' and
                    market.get('active', False)):
                    usdt_futures.append(symbol)

            # 按名称排序（不限制数量）
            usdt_futures = sorted(usdt_futures)

            return usdt_futures

        except Exception as e:
            self.logger.error(f"[ERROR] 获取交易对列表失败: {e}")
            return []

    async def _fetch_symbol_ohlcv(self, symbol: str, start_time: datetime,
                                end_time: datetime) -> Optional[pd.DataFrame]:
        """
        获取单个交易对的OHLCV数据

        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            pd.DataFrame: OHLCV数据
        """
        try:
            # 转换为毫秒时间戳
            since = int(start_time.timestamp() * 1000)

            # 直接获取80小时的1小时数据（80个数据点）
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='1h',
                since=since,
                limit=80  # 80小时 = 80个1小时数据点
            )

            if not ohlcv:
                return None

            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)

            # 过滤时间范围
            df = df[(df['open_time'] >= start_time) & (df['open_time'] <= end_time)]

            # 确保数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 移除异常数据
            df = df.dropna()

            return df

        except Exception as e:
            self.logger.warning(f"[WARNING] 获取 {symbol} 数据异常: {e}")
            return None

    def aggregate_to_timeframes(self, data: pd.DataFrame,
                              timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        将1小时数据聚合到多个时间框架

        Args:
            data: 原始1小时数据
            timeframes: 目标时间框架列表 ['1h', '2h', '4h']

        Returns:
            Dict: 各时间框架的聚合数据
        """
        try:
            aggregated_data = {}
            print(f"聚合数据到时间框架: {timeframes}")

            for timeframe in timeframes:
                try:
                    # 针对1小时基础数据的优化聚合
                    if timeframe == '1h':
                        # 1小时数据无需聚合，直接使用原始数据
                        aggregated_data[timeframe] = data.copy()
                        continue

                    # 映射时间框架到pandas频率
                    freq_map = {
                        '2h': '120min',
                        '4h': '240min'
                    }

                    if timeframe not in freq_map:
                        self.logger.warning(f"[WARNING] 不支持的时间框架: {timeframe}")
                        continue

                    freq = freq_map[timeframe]

                    # 重置索引进行聚合
                    temp_data = data.reset_index()
                    temp_data = temp_data.set_index('open_time')

                    # 按symbol分组并聚合
                    symbol_groups = temp_data.groupby('symbol')
                    aggregated_list = []

                    # 使用tqdm显示聚合进度
                    for symbol, group in tqdm(symbol_groups, desc=f"聚合{timeframe}", leave=False):
                        try:
                            # 聚合OHLCV数据
                            resampled = group.resample(freq, label='left', closed='left').agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            }).dropna()

                            if len(resampled) > 0:
                                resampled['symbol'] = symbol
                                resampled = resampled.reset_index().set_index(['open_time', 'symbol'])
                                aggregated_list.append(resampled)

                        except Exception as e:
                            pass
                            continue

                    if aggregated_list:
                        aggregated_data[timeframe] = pd.concat(aggregated_list).sort_index()

                except Exception as e:
                    self.logger.error(f"[ERROR] 处理时间框架 {timeframe} 失败: {e}")
                    continue

            return aggregated_data

        except Exception as e:
            self.logger.error(f"[ERROR] 数据聚合失败: {e}")
            return {}

    def get_symbols_with_sufficient_data(self, data: pd.DataFrame,
                                       min_periods: int = 20) -> List[str]:
        """
        获取有足够数据的交易对列表

        Args:
            data: OHLCV数据
            min_periods: 最少需要的数据点数

        Returns:
            List[str]: 符合条件的交易对列表
        """
        try:
            # 统计每个交易对的数据点数
            symbol_counts = data.groupby('symbol').size()

            # 筛选有足够数据的交易对
            valid_symbols = symbol_counts[symbol_counts >= min_periods].index.tolist()

            return valid_symbols

        except Exception as e:
            self.logger.error(f"[ERROR] 统计交易对数据失败: {e}")
            return []

    def clear_cache(self):
        """清理数据缓存"""
        self.data_cache.clear()

    async def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                await self.exchange.close()
            self.clear_cache()

        except Exception as e:
            self.logger.error(f"[ERROR] 清理数据处理器失败: {e}")