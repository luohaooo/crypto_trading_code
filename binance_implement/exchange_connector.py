"""
Binance 交易所连接模块
支持 testnet 和实盘环境，提供统一的API接口
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 直接使用系统安装的ccxt
try:
    import ccxt
except ImportError as e:
    print(f"❌ ccxt库导入失败: {e}")
    print("请安装ccxt: pip install ccxt")
    sys.exit(1)

from config import TradingConfig


class BinanceConnector:
    """Binance交易所连接器"""

    def __init__(self, config: TradingConfig):
        """
        初始化连接器

        Args:
            config: 交易配置
        """
        self.config = config
        self.exchange = None
        self.markets = None
        self.logger = self._setup_logger()

        # 连接状态
        self.is_connected = False
        self.last_heartbeat = None

        # 错误统计
        self.error_count = 0
        self.max_errors = 10

    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f'BinanceConnector_{self.config.ENV_NAME}')
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))

        if not logger.handlers:
            # 文件处理器
            fh = logging.FileHandler(self.config.LOG_FILE)
            fh.setLevel(logging.DEBUG)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def connect(self) -> bool:
        """
        连接到Binance交易所

        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info(f"🔗 正在连接到Binance {self.config.ENV_NAME}...")

            # 创建交易所实例
            self.exchange = ccxt.binance(self.config.get_ccxt_config())

            # 设置沙盒模式（testnet）
            if self.config.use_testnet:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("🧪 启用沙盒模式 (Testnet)")

            # 加载市场数据
            self.logger.info("📊 正在加载市场数据...")
            self.markets = self.exchange.load_markets()

            # 测试连接
            balance = self.exchange.fetch_balance()
            self.logger.info(f"✅ 连接成功! 账户类型: {balance.get('info', {}).get('accountType', 'Unknown')}")

            # 更新连接状态
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            self.error_count = 0

            return True

        except Exception as e:
            self.logger.error(f"❌ 连接失败: {e}")
            self.is_connected = False
            return False

    def test_connection(self) -> bool:
        """
        测试连接状态

        Returns:
            bool: 连接是否正常
        """
        try:
            if not self.exchange:
                return False

            # 简单的ping测试
            server_time = self.exchange.fetch_time()
            self.last_heartbeat = datetime.now()

            self.logger.debug(f"📡 心跳检测成功，服务器时间: {server_time}")
            return True

        except Exception as e:
            self.logger.warning(f"⚠️ 连接测试失败: {e}")
            self.error_count += 1

            if self.error_count >= self.max_errors:
                self.logger.error(f"❌ 连续失败{self.max_errors}次，标记为断开连接")
                self.is_connected = False

            return False

    def reconnect(self) -> bool:
        """
        重新连接

        Returns:
            bool: 重连是否成功
        """
        self.logger.info("🔄 正在重新连接...")
        self.is_connected = False
        self.exchange = None

        # 等待一段时间后重连
        time.sleep(5)
        return self.connect()

    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息

        Returns:
            Dict: 账户信息
        """
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions()

            # 过滤非零仓位
            active_positions = [pos for pos in positions if float(pos['size']) != 0]

            # 计算总权益
            total_equity = float(balance['USDT']['total']) if 'USDT' in balance else 0

            account_info = {
                'total_equity': total_equity,
                'available_balance': float(balance['USDT']['free']) if 'USDT' in balance else 0,
                'margin_balance': float(balance['USDT']['total']) if 'USDT' in balance else 0,
                'position_count': len(active_positions),
                'positions': active_positions,
                'account_type': balance.get('info', {}).get('accountType', 'Unknown'),
                'can_trade': balance.get('info', {}).get('canTrade', False),
                'timestamp': datetime.now()
            }

            return account_info

        except Exception as e:
            self.logger.error(f"❌ 获取账户信息失败: {e}")
            raise

    def get_futures_symbols(self, min_volume_24h: float = 1000000) -> List[str]:
        """
        获取期货交易对列表

        Args:
            min_volume_24h: 最小24小时交易量过滤

        Returns:
            List[str]: 符合条件的交易对列表
        """
        try:
            if not self.markets:
                self.markets = self.exchange.load_markets()

            # 获取24小时ticker数据
            tickers = self.exchange.fetch_tickers()

            futures_symbols = []
            for symbol, market in self.markets.items():
                # 只选择USDT永续合约
                if (market['type'] == 'swap' and
                    market['quote'] == 'USDT' and
                    market['active'] and
                    symbol in tickers):

                    ticker = tickers[symbol]
                    volume_24h = ticker['quoteVolume'] or 0

                    # 过滤低交易量的币种
                    if volume_24h >= min_volume_24h:
                        futures_symbols.append(symbol)

            self.logger.info(f"📊 找到 {len(futures_symbols)} 个活跃期货交易对")
            return sorted(futures_symbols)

        except Exception as e:
            self.logger.error(f"❌ 获取交易对失败: {e}")
            return []

    def get_ohlcv_data(self, symbol: str, timeframe: str = '1m',
                       limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """
        获取OHLCV历史数据

        Args:
            symbol: 交易对符号
            timeframe: 时间框架 ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: 数据条数限制
            since: 开始时间戳

        Returns:
            pd.DataFrame: OHLCV数据
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('open_time')
            df = df.drop('timestamp', axis=1)

            self.logger.debug(f"📈 获取 {symbol} {timeframe} 数据: {len(df)} 条")
            return df

        except Exception as e:
            self.logger.error(f"❌ 获取{symbol} OHLCV数据失败: {e}")
            return pd.DataFrame()

    def get_historical_data_batch(self, symbols: List[str],
                                hours_back: int = 80) -> pd.DataFrame:
        """
        批量获取多个交易对的历史数据

        Args:
            symbols: 交易对列表
            hours_back: 回看小时数

        Returns:
            pd.DataFrame: 多交易对OHLCV数据，MultiIndex (open_time, symbol)
        """
        try:
            self.logger.info(f"📊 批量获取 {len(symbols)} 个交易对的历史数据...")

            # 计算开始时间
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            since = int(start_time.timestamp() * 1000)

            all_data = []
            successful_symbols = []

            for i, symbol in enumerate(symbols):
                try:
                    self.logger.debug(f"获取 {symbol} 数据 ({i+1}/{len(symbols)})")

                    # 获取1分钟数据
                    df = self.get_ohlcv_data(symbol, '1m', limit=hours_back*60, since=since)

                    if not df.empty:
                        df['symbol'] = symbol
                        df = df.reset_index().set_index(['open_time', 'symbol'])
                        all_data.append(df)
                        successful_symbols.append(symbol)

                    # 避免频率限制
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.warning(f"⚠️ 获取 {symbol} 数据失败: {e}")
                    continue

            if all_data:
                combined_df = pd.concat(all_data, axis=0).sort_index()
                self.logger.info(f"✅ 成功获取 {len(successful_symbols)} 个交易对的数据")
                return combined_df
            else:
                self.logger.warning("❌ 没有成功获取任何数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ 批量获取历史数据失败: {e}")
            return pd.DataFrame()

    def check_connection_health(self) -> Dict[str, Any]:
        """
        检查连接健康状态

        Returns:
            Dict: 连接状态信息
        """
        health = {
            'is_connected': self.is_connected,
            'last_heartbeat': self.last_heartbeat,
            'error_count': self.error_count,
            'exchange_status': None,
            'timestamp': datetime.now()
        }

        if self.exchange:
            try:
                # 测试连接
                server_time = self.exchange.fetch_time()
                health['exchange_status'] = 'healthy'
                health['server_time'] = server_time
                health['time_diff'] = abs(time.time() * 1000 - server_time)

            except Exception as e:
                health['exchange_status'] = f'error: {e}'

        return health

    def close(self):
        """关闭连接"""
        self.logger.info("🔌 关闭Binance连接")
        self.is_connected = False
        self.exchange = None


def test_connection():
    """测试连接功能"""
    print("🧪 测试Binance连接模块...")

    # 测试testnet连接
    testnet_config = TradingConfig(use_testnet=True)
    testnet_config.print_config()

    if not testnet_config.validate_config():
        print("❌ 配置验证失败")
        return

    connector = BinanceConnector(testnet_config)

    if connector.connect():
        print("✅ Testnet连接成功")

        # 测试账户信息
        try:
            account_info = connector.get_account_info()
            print(f"💰 账户余额: {account_info['available_balance']:.4f} USDT")
            print(f"📊 活跃仓位: {account_info['position_count']} 个")
        except Exception as e:
            print(f"❌ 获取账户信息失败: {e}")

        # 测试获取交易对
        symbols = connector.get_futures_symbols(min_volume_24h=500000)
        print(f"📈 找到 {len(symbols)} 个活跃交易对")
        if symbols:
            print(f"前5个交易对: {symbols[:5]}")

        # 测试历史数据获取
        if symbols:
            test_symbol = symbols[0]
            print(f"📊 测试获取 {test_symbol} 历史数据...")
            df = connector.get_ohlcv_data(test_symbol, '1m', limit=10)
            if not df.empty:
                print(f"✅ 成功获取 {len(df)} 条数据")
                print(f"最新价格: {df['close'].iloc[-1]:.4f}")
            else:
                print("❌ 未获取到数据")

        # 健康检查
        health = connector.check_connection_health()
        print(f"🔍 连接健康状态: {health['exchange_status']}")

        connector.close()
    else:
        print("❌ Testnet连接失败")


if __name__ == "__main__":
    test_connection()