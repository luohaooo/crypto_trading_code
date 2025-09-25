"""
优化的数据处理器
专门为高效的按symbol处理设计，直接获取不同时间框架的数据

功能：
- 直接获取指定时间框架的OHLC数据
- 避免大量历史数据聚合
- 支持多时间框架数据获取
- 高效的单symbol处理
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from tqdm import tqdm


class OptimizedDataProcessor:
    """优化的数据处理器，专注于高效的单symbol多时间框架数据获取"""

    def __init__(self, config, logger):
        """
        初始化优化数据处理器

        Args:
            config: 交易配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.exchange = None

    def initialize(self):
        """初始化数据处理器"""
        try:
            # 创建CCXT交易所实例（同步版本）
            self.exchange = ccxt.binance(self.config.get_ccxt_config())

            # 测试连接
            self._test_connection()

            self.logger.info("[OK] 优化数据处理器初始化成功")

        except Exception as e:
            self.logger.error(f"[ERROR] 优化数据处理器初始化失败: {e}")
            raise

    def _test_connection(self):
        """测试API连接"""
        try:
            test_symbol = 'BTC/USDT:USDT'

            # 记录API调用时间 - 使用最简单的调用方式
            start_time = datetime.now()
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=test_symbol,
                timeframe='1h',
                limit=5  # 移除 since 参数，使用最简单的调用
            )
            api_duration = (datetime.now() - start_time).total_seconds()

            if ohlcv and len(ohlcv) > 0:
                self.logger.info(f"[OK] API连接测试成功，耗时: {api_duration:.3f}秒")
                print(f"🔗 API连接测试: {api_duration:.3f}秒 (优化后)")
            else:
                raise Exception("无法获取测试数据")

        except Exception as e:
            self.logger.error(f"[ERROR] API连接测试失败: {e}")
            raise

    def get_active_symbols(self) -> List[str]:
        """获取活跃的USDT永续合约交易对"""
        try:
            # 记录获取市场数据时间
            start_time = datetime.now()
            markets = self.exchange.load_markets()
            market_duration = (datetime.now() - start_time).total_seconds()

            usdt_futures = []
            for symbol, market in markets.items():
                if (market.get('type') == 'swap' and
                    market.get('quote') == 'USDT' and
                    market.get('active', False)):
                    usdt_futures.append(symbol)

            total_duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"[OK] 获取活跃交易对完成，耗时: {total_duration:.3f}秒")
            print(f"📊 获取 {len(usdt_futures)} 个活跃交易对: {total_duration:.3f}秒")

            return sorted(usdt_futures)

        except Exception as e:
            self.logger.error(f"[ERROR] 获取活跃交易对失败: {e}")
            return []

    def get_symbol_multiframe_data(self, symbol: str,
                                 timeframes: List[str],
                                 limit: int = 20) -> Optional[Dict[str, pd.DataFrame]]:
        """
        获取单个交易对的多时间框架数据

        Args:
            symbol: 交易对符号
            timeframes: 时间框架列表，如 ['1h', '2h', '4h']
            limit: 每个时间框架获取的数据点数量

        Returns:
            Dict[str, pd.DataFrame]: 各时间框架的OHLCV数据，如果失败返回None
        """
        try:
            result = {}

            for timeframe in timeframes:
                try:
                    # 记录单个API调用时间
                    api_start_time = datetime.now()
                    # 直接获取指定时间框架的数据
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit
                    )
                    api_duration = (datetime.now() - api_start_time).total_seconds()

                    if not ohlcv or len(ohlcv) < limit:
                        # 如果数据不足，跳过这个交易对
                        return None

                    # 转换为DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.drop('timestamp', axis=1)

                    # 确保数据类型
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # 移除异常数据
                    df = df.dropna()

                    # 确保有足够的数据
                    if len(df) < limit:
                        return None

                    # 只保留最新的limit条数据
                    df = df.tail(limit).copy()
                    result[timeframe] = df

                    # 记录成功的API调用时间（可选，用于调试）
                    # print(f"    {symbol} {timeframe}: {api_duration:.3f}s")

                except Exception as e:
                    # 如果任何时间框架失败，整个symbol失败
                    return None

            return result if len(result) == len(timeframes) else None

        except Exception as e:
            return None

    def get_all_symbols_factors(self, timeframes: List[str] = ['1h', '2h', '4h'],
                              limit: int = 20) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        """
        获取所有活跃交易对的多时间框架数据

        Args:
            timeframes: 时间框架列表
            limit: 每个时间框架的数据点数

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {symbol: {timeframe: df}} 格式的数据
        """
        try:
            # 获取活跃交易对
            symbols_start_time = datetime.now()
            active_symbols = self.get_active_symbols()
            symbols_duration = (datetime.now() - symbols_start_time).total_seconds()

            if not active_symbols:
                self.logger.error("[ERROR] 没有找到活跃交易对")
                return None

            print(f"获取 {len(active_symbols)} 个交易对的多时间框架数据...")
            print(f"时间框架: {timeframes}, 每个框架获取 {limit} 个数据点")

            all_data = {}
            successful_count = 0
            total_api_calls = 0
            total_api_time = 0.0

            # 使用tqdm显示进度
            for symbol in tqdm(active_symbols, desc="获取数据", unit="交易对"):
                try:
                    data_start_time = datetime.now()
                    symbol_data = self.get_symbol_multiframe_data(symbol, timeframes, limit)
                    data_duration = (datetime.now() - data_start_time).total_seconds()

                    if symbol_data is not None:
                        all_data[symbol] = symbol_data
                        successful_count += 1
                        total_api_calls += len(timeframes)  # 每个symbol需要3个API调用
                        total_api_time += data_duration

                except Exception as e:
                    # 静默跳过失败的交易对
                    continue

            if successful_count == 0:
                self.logger.error("[ERROR] 没有获取到任何有效数据")
                return None

            total_duration = (datetime.now() - symbols_start_time).total_seconds()

            print(f"\n✅ 数据获取完成:")
            print(f"   成功获取: {successful_count} 个交易对")
            print(f"   时间框架: {timeframes}")
            print(f"   每个框架数据点: {limit}")
            print(f"   总API调用次数: {total_api_calls}")
            print(f"   平均每次API调用: {total_api_time/total_api_calls:.3f}秒")
            print(f"   总耗时: {total_duration:.2f}秒")
            print(f"   处理速度: {successful_count/total_duration:.1f} 交易对/秒")

            return all_data

        except Exception as e:
            self.logger.error(f"[ERROR] 获取所有交易对数据失败: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                self.exchange.close()
            self.logger.info("[OK] 优化数据处理器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理优化数据处理器失败: {e}")