"""
交易执行器模块
负责所有与Binance API的交互，包括：
- 仓位管理（开仓/平仓）
- 账户信息查询
- 订单执行和监控
"""

import ccxt
import math
import time
from datetime import datetime
from typing import Optional, Dict, List

# 导入钉钉通知模块
try:
    from trading_utils.dingding import send_dingtalk_message
    DINGDING_AVAILABLE = True
except ImportError:
    DINGDING_AVAILABLE = False


class TradingExecutor:
    """交易执行器类"""

    def __init__(self, config, logger):
        """
        初始化交易执行器

        Args:
            config: 交易配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.use_testnet = config.use_testnet  # 添加环境检测属性
        self.exchange = None
        self.active_symbols = []

    def initialize(self):
        """初始化交易所连接"""
        try:
            # 创建CCXT交易所实例
            self.exchange = ccxt.binance(self.config.get_ccxt_config())

            # 测试连接
            self._test_connection()

            # 获取可交易的币种列表
            self._load_active_symbols()

            self.logger.info("[OK] 交易执行器初始化成功")

        except Exception as e:
            self.logger.error(f"[ERROR] 交易执行器初始化失败: {e}")
            raise

    def _test_connection(self):
        """测试API连接"""
        try:
            # 获取账户信息
            balance = self.exchange.fetch_balance()
            self.logger.info(f"[OK] API连接成功 - {self.config.ENV_NAME}")

            # 打印账户概要
            total_balance = balance.get('USDT', {}).get('total', 0)
            self.logger.info(f"[BALANCE] USDT总余额: {total_balance:.2f}")

        except Exception as e:
            self.logger.error(f"[ERROR] API连接测试失败: {e}")
            raise

    def _load_active_symbols(self):
        """加载活跃交易对"""
        try:
            # 获取所有市场信息
            markets = self.exchange.load_markets()

            # 筛选USDT永续合约
            usdt_futures = []
            for symbol, market in markets.items():
                if (market.get('type') == 'swap' and
                    market.get('quote') == 'USDT' and
                    market.get('active', False)):
                    usdt_futures.append(symbol)

            self.active_symbols = sorted(usdt_futures)
            self.logger.info(f"[INFO] 找到 {len(self.active_symbols)} 个活跃USDT永续合约")

            # 只显示前10个作为示例
            sample_symbols = self.active_symbols[:10]
            self.logger.info(f"示例交易对: {', '.join(sample_symbols)}")

        except Exception as e:
            self.logger.error(f"[ERROR] 加载交易对失败: {e}")
            raise

    def send_balance_notification(self):
        """发送当前余额和时间的钉钉通知"""
        try:
            # 获取当前余额
            balance = self.get_account_balance()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if balance is not None:
                message = f"📊 账户余额报告\n⏰ 时间: {current_time}\n💰 USDT余额: {balance:.2f}\n🔧 环境: {'测试网' if self.config.use_testnet else '实盘'}"
                self._send_dingding_notification(message)
                self.logger.info(f"[BALANCE] 余额通知已发送: {balance:.2f} USDT")
            else:
                error_msg = f"❌ 余额查询失败\n⏰ 时间: {current_time}\n🔧 环境: {'测试网' if self.config.use_testnet else '实盘'}"
                self._send_dingding_notification(error_msg)
                self.logger.error("[ERROR] 余额查询失败，已发送错误通知")

        except Exception as e:
            self.logger.error(f"[ERROR] 发送余额通知失败: {e}")

    def get_account_balance(self) -> Optional[float]:
        """
        获取账户USDT余额

        Returns:
            float: USDT余额，失败返回None
        """
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            return float(usdt_balance)

        except Exception as e:
            self.logger.error(f"[ERROR] 获取账户余额失败: {e}")
            return None

    def close_all_positions(self) -> bool:
        """
        平仓所有现有仓位 - 环境路由方法

        Returns:
            bool: 平仓是否成功
        """
        if self.use_testnet:
            return self.close_all_positions_testnet()
        else:
            return self.close_all_positions_live()

    def close_all_positions_testnet(self) -> bool:
        """
        模拟盘平仓所有现有仓位

        Returns:
            bool: 平仓是否成功
        """
        try:
            self.logger.info("[TESTNET CLOSE] 开始平仓所有仓位...")

            # 获取当前仓位
            positions = self.exchange.fetch_positions()

            # 筛选有仓位的合约
            active_positions = []
            for position in positions:
                if position['contracts'] != 0:  # 有仓位
                    active_positions.append(position)

            if not active_positions:
                self.logger.info("[TESTNET OK] 当前无持仓，无需平仓")
                return True

            self.logger.info(f"[TESTNET POSITIONS] 发现 {len(active_positions)} 个持仓需要平仓")

            # 批量平仓
            close_orders = []
            failed_positions = []

            for position in active_positions:
                try:
                    symbol = position['symbol']
                    size = abs(position['contracts'])
                    side = 'sell' if position['side'] == 'long' else 'buy'

                    self.logger.info(f"[TESTNET CLOSE] 平仓 {symbol}: {side} {size}")

                    # 模拟盘平仓使用最简化参数
                    order = self._create_market_order_testnet(
                        symbol=symbol,
                        side=side,
                        amount=size
                        # 模拟盘不传递任何额外参数
                    )

                    if order:
                        close_orders.append(order)
                        self.logger.info(f"[TESTNET SUCCESS] 平仓订单创建成功 {symbol}: {order.get('id', 'Unknown')}")
                    else:
                        self.logger.error(f"[TESTNET ERROR] 平仓失败 {symbol}")
                        failed_positions.append(symbol)

                    # 避免请求过于频繁
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"[TESTNET ERROR] 平仓 {symbol} 异常: {e}")
                    failed_positions.append(symbol)
                    continue

            # 记录平仓结果
            if failed_positions:
                self.logger.warning(f"[TESTNET WARNING] {len(failed_positions)} 个仓位平仓失败: {failed_positions}")

            # 等待订单执行
            if close_orders:
                self.logger.info(f"[TESTNET WAIT] 等待 {len(close_orders)} 个平仓订单执行...")
                time.sleep(2)

            self.logger.info("[TESTNET OK] 平仓操作完成")
            return True

        except Exception as e:
            self.logger.error(f"[TESTNET ERROR] 平仓失败: {e}")
            return False

    def close_all_positions_live(self) -> bool:
        """
        实盘平仓所有现有仓位

        Returns:
            bool: 平仓是否成功
        """
        try:
            self.logger.info("[LIVE CLOSE] 开始平仓所有仓位...")

            # 获取当前仓位
            positions = self.exchange.fetch_positions()

            # 筛选有仓位的合约
            active_positions = []
            for position in positions:
                if position['contracts'] != 0:  # 有仓位
                    active_positions.append(position)

            if not active_positions:
                self.logger.info("[LIVE OK] 当前无持仓，无需平仓")
                return True

            self.logger.info(f"[LIVE POSITIONS] 发现 {len(active_positions)} 个持仓需要平仓")

            # 批量平仓
            close_orders = []
            failed_positions = []

            for position in active_positions:
                try:
                    symbol = position['symbol']
                    size = abs(position['contracts'])
                    side = 'sell' if position['side'] == 'long' else 'buy'

                    self.logger.info(f"[LIVE CLOSE] 平仓 {symbol}: {side} {size}")

                    # 确定平仓的positionSide
                    position_side = 'LONG' if position['side'] == 'long' else 'SHORT'

                    # 实盘使用完整参数
                    order = self._create_market_order_live(
                        symbol=symbol,
                        side=side,
                        amount=size,
                        params={'reduceOnly': True, 'positionSide': position_side}
                    )

                    if order:
                        close_orders.append(order)
                        self.logger.info(f"[LIVE SUCCESS] 平仓订单创建成功 {symbol}: {order.get('id', 'Unknown')}")
                    else:
                        self.logger.error(f"[LIVE ERROR] 平仓失败 {symbol}")
                        failed_positions.append(symbol)

                    # 避免请求过于频繁
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"[LIVE ERROR] 平仓 {symbol} 异常: {e}")
                    failed_positions.append(symbol)
                    continue

            # 记录平仓结果
            if failed_positions:
                self.logger.warning(f"[LIVE WARNING] {len(failed_positions)} 个仓位平仓失败: {failed_positions}")

            # 等待订单执行
            if close_orders:
                self.logger.info(f"[LIVE WAIT] 等待 {len(close_orders)} 个平仓订单执行...")
                time.sleep(2)

            self.logger.info("[LIVE OK] 平仓操作完成")
            return True

        except Exception as e:
            self.logger.error(f"[LIVE ERROR] 平仓失败: {e}")
            return False

    def verify_positions_closed(self) -> bool:
        """
        验证所有仓位是否已平仓

        Returns:
            bool: 是否全部平仓
        """
        try:

            positions = self.exchange.fetch_positions()

            # 检查是否还有持仓
            remaining_positions = []
            for position in positions:
                if position['contracts'] != 0:
                    remaining_positions.append(position)

            if remaining_positions:
                self.logger.warning(f"[WARNING] 仍有 {len(remaining_positions)} 个未平仓位:")
                for pos in remaining_positions:
                    self.logger.warning(f"   {pos['symbol']}: {pos['side']} {abs(pos['contracts'])}")
                return False
            else:
                self.logger.info("[OK] 确认所有仓位已平仓")
                return True

        except Exception as e:
            self.logger.error(f"[ERROR] 验证平仓状态失败: {e}")
            return False

    def open_positions(self, long_symbols: List[str], short_symbols: List[str],
                      total_balance: float) -> bool:
        """
        开仓交易

        Args:
            long_symbols: 做多标的列表
            short_symbols: 做空标的列表
            total_balance: 总可用余额

        Returns:
            bool: 开仓是否成功
        """
        try:
            all_symbols = long_symbols + short_symbols
            total_positions = len(all_symbols)

            if total_positions == 0:
                self.logger.info("[OK] 无需开仓标的")
                return True

            # 计算每个仓位的资金
            position_value =  0.99 * total_balance / total_positions # 防止资金不足
            self.logger.info(f"[BALANCE] 总余额: {total_balance:.2f} USDT")
            self.logger.info(f"[INFO] 总仓位数: {total_positions}")
            self.logger.info(f"[VALUE] 单仓价值: {position_value:.2f} USDT")

            # 设置杠杆
            self._set_leverage_for_symbols(all_symbols)

            # 开多头仓位
            long_success = self._open_long_positions(long_symbols, position_value)

            # 开空头仓位
            short_success = self._open_short_positions(short_symbols, position_value)

            return long_success and short_success

        except Exception as e:
            error_msg = f"开仓过程异常！错误: {str(e)}"
            self.logger.error(f"[ERROR] 开仓失败: {e}")
            self._send_dingding_notification(error_msg)
            return False

    def _set_leverage_for_symbols(self, symbols: List[str]):
        """为交易对设置杠杆"""
        self.logger.info(f"[LEVERAGE] 设置杠杆为 {self.config.LEVERAGE}x...")

        for symbol in symbols:
            try:
                self.exchange.set_leverage(self.config.LEVERAGE, symbol)
                time.sleep(0.05)  # 避免请求过快

            except Exception as e:
                self.logger.warning(f"[WARNING] 设置 {symbol} 杠杆失败: {e}")
                continue

    def _open_long_positions(self, symbols: List[str], position_value: float) -> bool:
        """开多头仓位 - 环境路由方法"""
        if self.use_testnet:
            return self._open_long_positions_testnet(symbols, position_value)
        else:
            return self._open_long_positions_live(symbols, position_value)

    def _open_long_positions_testnet(self, symbols: List[str], position_value: float) -> bool:
        """模拟盘开多头仓位"""
        if not symbols:
            return True

        self.logger.info(f"[TESTNET LONG] 开多头仓位: {len(symbols)} 个")

        success_count = 0
        for symbol in symbols:
            try:
                # 获取当前价格
                ticker = self.exchange.fetch_ticker(symbol)
                if not ticker:
                    self.logger.error(f"[TESTNET ERROR] 无法获取 {symbol} 价格信息，跳过")
                    continue

                current_price = ticker['last']

                # 计算下单数量
                quantity = position_value / current_price

                # 获取交易对精度
                market = self.exchange.markets[symbol]
                amount_precision = market['precision']['amount']

                # 处理精度：如果是小数，计算小数位数
                if isinstance(amount_precision, float):
                    # 计算小数位数
                    decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                    quantity = round(quantity, decimal_places)
                    # 确保不小于最小精度
                    if quantity < amount_precision:
                        quantity = amount_precision
                else:
                    # 如果是整数精度，使用原来的逻辑
                    precision_power = int(amount_precision)
                    quantity = math.floor(quantity * (10 ** precision_power)) / (10 ** precision_power)

                self.logger.info(f"[TESTNET BUY] 做多 {symbol}: {quantity} @ {current_price}")

                # 模拟盘使用简化参数（不需要positionSide）
                order = self._create_market_order_testnet(
                    symbol=symbol,
                    side='buy',
                    amount=quantity
                    # 模拟盘不传递 positionSide 参数
                )

                if order:
                    success_count += 1
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[TESTNET] 多头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[TESTNET ERROR] 做多 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[TESTNET OK] 多头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0

    def _open_long_positions_live(self, symbols: List[str], position_value: float) -> bool:
        """实盘开多头仓位"""
        if not symbols:
            return True

        self.logger.info(f"[LIVE LONG] 开多头仓位: {len(symbols)} 个")

        success_count = 0
        for symbol in symbols:
            try:
                # 获取当前价格
                ticker = self.exchange.fetch_ticker(symbol)
                if not ticker:
                    self.logger.error(f"[LIVE ERROR] 无法获取 {symbol} 价格信息，跳过")
                    continue

                current_price = ticker['last']

                # 计算下单数量
                quantity = position_value / current_price

                # 获取交易对精度
                market = self.exchange.markets[symbol]
                amount_precision = market['precision']['amount']

                # 处理精度：如果是小数，计算小数位数
                if isinstance(amount_precision, float):
                    # 计算小数位数
                    decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                    quantity = round(quantity, decimal_places)
                    # 确保不小于最小精度
                    if quantity < amount_precision:
                        quantity = amount_precision
                else:
                    # 如果是整数精度，使用原来的逻辑
                    precision_power = int(amount_precision)
                    quantity = math.floor(quantity * (10 ** precision_power)) / (10 ** precision_power)

                self.logger.info(f"[LIVE BUY] 做多 {symbol}: {quantity} @ {current_price}")

                # 实盘严格使用positionSide参数
                order = self._create_market_order_live(
                    symbol=symbol,
                    side='buy',
                    amount=quantity,
                    params={'positionSide': 'LONG'}
                )

                if order:
                    success_count += 1
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[LIVE] 多头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[LIVE ERROR] 做多 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[LIVE OK] 多头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0

    def _open_short_positions(self, symbols: List[str], position_value: float) -> bool:
        """开空头仓位 - 环境路由方法"""
        if self.use_testnet:
            return self._open_short_positions_testnet(symbols, position_value)
        else:
            return self._open_short_positions_live(symbols, position_value)

    def _open_short_positions_testnet(self, symbols: List[str], position_value: float) -> bool:
        """模拟盘开空头仓位"""
        if not symbols:
            return True

        self.logger.info(f"[TESTNET SHORT] 开空头仓位: {len(symbols)} 个")

        success_count = 0
        for symbol in symbols:
            try:
                # 获取当前价格
                ticker = self.exchange.fetch_ticker(symbol)
                if not ticker:
                    self.logger.error(f"[TESTNET ERROR] 无法获取 {symbol} 价格信息，跳过")
                    continue

                current_price = ticker['last']

                # 计算下单数量
                quantity = position_value / current_price

                # 获取交易对精度
                market = self.exchange.markets[symbol]
                amount_precision = market['precision']['amount']

                # 处理精度：如果是小数，计算小数位数
                if isinstance(amount_precision, float):
                    # 计算小数位数
                    decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                    quantity = round(quantity, decimal_places)
                    # 确保不小于最小精度
                    if quantity < amount_precision:
                        quantity = amount_precision
                else:
                    # 如果是整数精度，使用原来的逻辑
                    precision_power = int(amount_precision)
                    quantity = math.floor(quantity * (10 ** precision_power)) / (10 ** precision_power)

                self.logger.info(f"[TESTNET SELL] 做空 {symbol}: {quantity} @ {current_price}")

                # 模拟盘使用简化参数（不需要positionSide）
                order = self._create_market_order_testnet(
                    symbol=symbol,
                    side='sell',
                    amount=quantity
                    # 模拟盘不传递 positionSide 参数
                )

                if order:
                    success_count += 1
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[TESTNET] 空头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[TESTNET ERROR] 做空 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[TESTNET OK] 空头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0

    def _open_short_positions_live(self, symbols: List[str], position_value: float) -> bool:
        """实盘开空头仓位"""
        if not symbols:
            return True

        self.logger.info(f"[LIVE SHORT] 开空头仓位: {len(symbols)} 个")

        success_count = 0
        for symbol in symbols:
            try:
                # 获取当前价格
                ticker = self.exchange.fetch_ticker(symbol)
                if not ticker:
                    self.logger.error(f"[LIVE ERROR] 无法获取 {symbol} 价格信息，跳过")
                    continue

                current_price = ticker['last']

                # 计算下单数量
                quantity = position_value / current_price

                # 获取交易对精度
                market = self.exchange.markets[symbol]
                amount_precision = market['precision']['amount']

                # 处理精度：如果是小数，计算小数位数
                if isinstance(amount_precision, float):
                    # 计算小数位数
                    decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                    quantity = round(quantity, decimal_places)
                    # 确保不小于最小精度
                    if quantity < amount_precision:
                        quantity = amount_precision
                else:
                    # 如果是整数精度，使用原来的逻辑
                    precision_power = int(amount_precision)
                    quantity = math.floor(quantity * (10 ** precision_power)) / (10 ** precision_power)

                self.logger.info(f"[LIVE SELL] 做空 {symbol}: {quantity} @ {current_price}")

                # 实盘严格使用positionSide参数
                order = self._create_market_order_live(
                    symbol=symbol,
                    side='sell',
                    amount=quantity,
                    params={'positionSide': 'SHORT'}
                )

                if order:
                    success_count += 1
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[LIVE] 空头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[LIVE ERROR] 做空 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[LIVE OK] 空头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0

    def get_current_positions(self) -> Dict[str, Dict]:
        """
        获取当前所有仓位

        Returns:
            Dict: 仓位信息字典 {symbol: position_info}
        """
        try:
            positions = self.exchange.fetch_positions()

            current_positions = {}
            for position in positions:
                if position['contracts'] != 0:
                    symbol = position['symbol']
                    current_positions[symbol] = {
                        'size': position['contracts'],
                        'side': position['side'],
                        'entry_price': position['entryPrice'],
                        'mark_price': position['markPrice'],
                        'pnl': position['unrealizedPnl'],
                        'percentage': position['percentage']
                    }

            return current_positions

        except Exception as e:
            self.logger.error(f"[ERROR] 获取仓位信息失败: {e}")
            return {}

    def get_active_symbols(self) -> List[str]:
        """获取活跃交易对列表"""
        return self.active_symbols

    def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                # 对于同步ccxt，不需要close()方法
                self.exchange = None
            self.logger.info("[OK] 交易执行器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理交易执行器失败: {e}")

    def _create_market_order_safe(self, symbol: str, side: str, amount: float,
                                 params: dict = None) -> dict:
        """
        安全的市价单创建方法 - 环境路由版本 (保持向后兼容)

        Args:
            symbol: 交易对
            side: 买卖方向 ('buy' 或 'sell')
            amount: 交易数量
            params: 额外参数

        Returns:
            dict: 订单信息，失败时返回None
        """
        # 路由到对应的环境特定方法
        if self.use_testnet:
            return self._create_market_order_testnet(symbol, side, amount, params)
        else:
            return self._create_market_order_live(symbol, side, amount, params)

    def _create_market_order_testnet(self, symbol: str, side: str, amount: float,
                                   params: dict = None) -> dict:
        """
        模拟盘环境的市价单创建方法

        Args:
            symbol: 交易对
            side: 买卖方向 ('buy' 或 'sell')
            amount: 交易数量
            params: 额外参数

        Returns:
            dict: 订单信息，失败时返回None
        """
        try:
            # 参数验证
            if not symbol or not side or amount <= 0:
                self.logger.error(f"[TESTNET ERROR] 无效参数: symbol={symbol}, side={side}, amount={amount}")
                return None

            # 确保交易对在市场列表中
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                self.logger.error(f"[TESTNET ERROR] 市场数据未加载")
                return None

            if symbol not in self.exchange.markets:
                self.logger.error(f"[TESTNET ERROR] 交易对 {symbol} 不在市场列表中")
                return None

            # 检查最小交易量
            market = self.exchange.markets[symbol]
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if min_amount and amount < min_amount:
                self.logger.error(f"[TESTNET ERROR] {symbol} 交易量 {amount} 小于最小值 {min_amount}")
                return None

            self.logger.info(f"[TESTNET MARKET] 创建市价单 {symbol}: {side} {amount}")

            # 模拟盘简化参数策略 - 直接移除有问题的参数
            safe_params = params if params is not None else {}

            # 模拟盘直接使用最简化的参数，避免不必要的错误
            try:
                # 第一次尝试：移除模拟盘不支持的参数
                testnet_params = {}  # 模拟盘使用最基础的参数
                # 只保留必要的参数，移除 reduceOnly 和 positionSide

                order = self.exchange.create_market_order(
                    symbol=symbol, side=side, amount=amount, params=testnet_params
                )

                # 记录成功信息
                if order:
                    order_id = order.get('id', 'Unknown')
                    filled = order.get('filled', 0)
                    self.logger.info(f"[TESTNET SUCCESS] {symbol} 市价单创建成功 ID:{order_id} 成交:{filled}")

                return order

            except Exception as e:
                error_str = str(e)
                self.logger.error(f"[TESTNET ERROR] {symbol} 订单创建失败: {error_str}")

                # 如果基础参数也失败，则无法修复
                if "-4061" in error_str:
                    self.logger.error(f"[TESTNET ERROR] {symbol} 基础参数仍然失败，可能需要检查账户设置")
                elif "-4164" in error_str:
                    self.logger.error(f"[TESTNET ERROR] {symbol} 订单金额太小，请增加订单金额")

                raise e

        except Exception as e:
            self.logger.error(f"[TESTNET ERROR] {symbol} 市价单创建异常: {e}")
            return None

    def _create_market_order_live(self, symbol: str, side: str, amount: float,
                                 params: dict = None) -> dict:
        """
        实盘环境的市价单创建方法

        Args:
            symbol: 交易对
            side: 买卖方向 ('buy' 或 'sell')
            amount: 交易数量
            params: 额外参数

        Returns:
            dict: 订单信息，失败时返回None
        """
        try:
            # 参数验证
            if not symbol or not side or amount <= 0:
                self.logger.error(f"[LIVE ERROR] 无效参数: symbol={symbol}, side={side}, amount={amount}")
                return None

            # 确保交易对在市场列表中
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                self.logger.error(f"[LIVE ERROR] 市场数据未加载")
                return None

            if symbol not in self.exchange.markets:
                self.logger.error(f"[LIVE ERROR] 交易对 {symbol} 不在市场列表中")
                return None

            # 检查最小交易量
            market = self.exchange.markets[symbol]
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if min_amount and amount < min_amount:
                self.logger.error(f"[LIVE ERROR] {symbol} 交易量 {amount} 小于最小值 {min_amount}")
                return None

            self.logger.info(f"[LIVE MARKET] 创建市价单 {symbol}: {side} {amount}")

            # 实盘严格参数策略
            safe_params = params if params is not None else {}

            # 第一次尝试：使用完整参数
            try:
                order = self.exchange.create_market_order(
                    symbol=symbol, side=side, amount=amount, params=safe_params
                )

                # 记录成功信息
                if order:
                    order_id = order.get('id', 'Unknown')
                    filled = order.get('filled', 0)
                    self.logger.info(f"[LIVE SUCCESS] {symbol} 市价单创建成功 ID:{order_id} 成交:{filled}")

                return order

            except Exception as e:
                error_str = str(e)

                # 实盘的智能重试逻辑
                if "-1106" in error_str and "reduceOnly" in safe_params:
                    self.logger.warning(f"[LIVE WARNING] {symbol} reduceOnly参数不被接受 (错误: -1106)")
                    self.logger.info(f"[LIVE RETRY] {symbol} 尝试不带reduceOnly参数重新创建订单")

                    try:
                        # 移除reduceOnly参数重试
                        retry_params = safe_params.copy()
                        retry_params.pop('reduceOnly', None)

                        order = self.exchange.create_market_order(
                            symbol=symbol, side=side, amount=amount, params=retry_params
                        )

                        if order:
                            order_id = order.get('id', 'Unknown')
                            filled = order.get('filled', 0)
                            self.logger.info(f"[LIVE SUCCESS] {symbol} 不带reduceOnly参数创建成功 ID:{order_id} 成交:{filled}")

                        return order

                    except Exception as e2:
                        self.logger.error(f"[LIVE ERROR] {symbol} 重试后仍失败: {e2}")
                        raise e  # 抛出原始错误

                elif "-4061" in error_str:
                    self.logger.error(f"[LIVE ERROR] {symbol} 持仓方向错误 (-4061): 请检查positionSide参数")
                    raise e

                elif "-4164" in error_str:
                    self.logger.error(f"[LIVE ERROR] {symbol} 订单金额太小 (-4164): 请增加订单金额")
                    raise e

                else:
                    self.logger.error(f"[LIVE ERROR] {symbol} 订单创建失败: {error_str}")
                    raise e  # 抛出原始错误

        except Exception as e:
            self.logger.error(f"[LIVE ERROR] {symbol} 市价单创建异常: {e}")
            return None

    def _send_dingding_notification(self, message: str):
        """发送钉钉通知"""
        if DINGDING_AVAILABLE:
            try:
                env_prefix = "[TESTNET]" if self.config.use_testnet else "[LIVE]"
                full_message = f"{env_prefix} {message}"
                send_dingtalk_message(full_message)
                self.logger.info(f"[DINGDING] 通知已发送: {message}")
            except Exception as e:
                self.logger.error(f"[DINGDING ERROR] 发送钉钉通知失败: {e}")
        else:
            self.logger.warning("[DINGDING] 钉钉模块不可用，跳过通知")

