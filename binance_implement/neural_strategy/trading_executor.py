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
from typing import Optional, List

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

                    # 模拟盘平仓使用分批交易支持
                    order = self._create_market_order_testnet(
                        symbol=symbol,
                        side=side,
                        amount=size
                        # 模拟盘不传递任何额外参数，分批逻辑在_create_market_order_testnet内部处理
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

                    # 实盘平仓使用分批交易支持
                    order = self._create_market_order_live(
                        symbol=symbol,
                        side=side,
                        amount=size,
                        params={'reduceOnly': True, 'positionSide': position_side}
                        # 分批逻辑在_create_market_order_live内部处理
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

    def close_specific_positions(self, positions_dict: dict) -> bool:
        """
        平仓指定仓位 - 环境路由方法

        Args:
            positions_dict: 仓位字典，格式: {symbol: {'side': 'long'/'short', 'quantity': float}}

        Returns:
            bool: 平仓是否成功
        """
        if self.use_testnet:
            return self.close_specific_positions_testnet(positions_dict)
        else:
            return self.close_specific_positions_live(positions_dict)

    def close_specific_positions_testnet(self, positions_dict: dict) -> bool:
        """
        模拟盘平仓指定仓位

        Args:
            positions_dict: 仓位字典，格式: {symbol: {'side': 'long'/'short', 'quantity': float}}

        Returns:
            bool: 平仓是否成功
        """
        try:
            if not positions_dict:
                self.logger.info("[TESTNET OK] 无指定仓位需要平仓")
                return True

            self.logger.info(f"[TESTNET CLOSE] 开始平仓指定仓位: {len(positions_dict)} 个")

            close_orders = []
            failed_positions = []

            for symbol, position_info in positions_dict.items():
                try:
                    side = position_info['side']
                    quantity = position_info['quantity']

                    # 确定平仓方向 (做多仓位用sell平仓，做空仓位用buy平仓)
                    close_side = 'sell' if side == 'long' else 'buy'

                    self.logger.info(f"[TESTNET CLOSE] 平仓 {symbol}: {close_side} {quantity} (原{side}仓位)")

                    # 模拟盘平仓
                    order = self._create_market_order_testnet(
                        symbol=symbol,
                        side=close_side,
                        amount=abs(quantity)
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

            success_count = len(positions_dict) - len(failed_positions)
            self.logger.info(f"[TESTNET OK] 平仓操作完成，成功: {success_count}/{len(positions_dict)}")
            return len(failed_positions) == 0

        except Exception as e:
            self.logger.error(f"[TESTNET ERROR] 指定仓位平仓失败: {e}")
            return False

    def close_specific_positions_live(self, positions_dict: dict) -> bool:
        """
        实盘平仓指定仓位

        Args:
            positions_dict: 仓位字典，格式: {symbol: {'side': 'long'/'short', 'quantity': float}}

        Returns:
            bool: 平仓是否成功
        """
        try:
            if not positions_dict:
                self.logger.info("[LIVE OK] 无指定仓位需要平仓")
                return True

            self.logger.info(f"[LIVE CLOSE] 开始平仓指定仓位: {len(positions_dict)} 个")

            close_orders = []
            failed_positions = []

            for symbol, position_info in positions_dict.items():
                try:
                    side = position_info['side']
                    quantity = position_info['quantity']

                    # 确定平仓方向 (做多仓位用sell平仓，做空仓位用buy平仓)
                    close_side = 'sell' if side == 'long' else 'buy'

                    self.logger.info(f"[LIVE CLOSE] 平仓 {symbol}: {close_side} {quantity} (原{side}仓位)")

                    # 实盘平仓
                    order = self._create_market_order_live(
                        symbol=symbol,
                        side=close_side,
                        amount=abs(quantity)
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

            success_count = len(positions_dict) - len(failed_positions)
            self.logger.info(f"[LIVE OK] 平仓操作完成，成功: {success_count}/{len(positions_dict)}")
            return len(failed_positions) == 0

        except Exception as e:
            self.logger.error(f"[LIVE ERROR] 指定仓位平仓失败: {e}")
            return False

    def open_positions(self, long_symbols: List[str], short_symbols: List[str],
                      total_balance: float) -> tuple[bool, dict]:
        """
        开仓交易

        Args:
            long_symbols: 做多标的列表
            short_symbols: 做空标的列表
            total_balance: 总可用余额

        Returns:
            tuple[bool, dict]: (开仓是否成功, 开仓位置字典)
                开仓位置字典格式: {symbol: {'side': 'long'/'short', 'quantity': float}}
        """
        opened_positions = {}
        try:
            all_symbols = long_symbols + short_symbols
            total_positions = len(all_symbols)

            if total_positions == 0:
                self.logger.info("[OK] 无需开仓标的")
                return True, opened_positions

            # 计算每个仓位的资金
            position_value =  0.99 * total_balance / total_positions # 防止资金不足
            self.logger.info(f"[BALANCE] 总余额: {total_balance:.2f} USDT")
            self.logger.info(f"[INFO] 总仓位数: {total_positions}")
            self.logger.info(f"[VALUE] 单仓价值: {position_value:.2f} USDT")

            # 设置杠杆
            self._set_leverage_for_symbols(all_symbols)

            # 开多头仓位
            long_success, long_positions = self._open_long_positions(long_symbols, position_value)
            opened_positions.update(long_positions)

            # 开空头仓位
            short_success, short_positions = self._open_short_positions(short_symbols, position_value)
            opened_positions.update(short_positions)

            success = long_success and short_success
            return success, opened_positions

        except Exception as e:
            error_msg = f"开仓过程异常！错误: {str(e)}"
            self.logger.error(f"[ERROR] 开仓失败: {e}")
            self._send_dingding_notification(error_msg)
            return False, opened_positions

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

    def _open_long_positions(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """开多头仓位 - 环境路由方法"""
        if self.use_testnet:
            return self._open_long_positions_testnet(symbols, position_value)
        else:
            return self._open_long_positions_live(symbols, position_value)

    def _open_long_positions_testnet(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """模拟盘开多头仓位"""
        opened_positions = {}
        if not symbols:
            return True, opened_positions

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
                    # 记录成功开仓的位置
                    opened_positions[symbol] = {
                        'side': 'long',
                        'quantity': quantity
                    }
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[TESTNET] 多头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[TESTNET ERROR] 做多 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[TESTNET OK] 多头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0, opened_positions

    def _open_long_positions_live(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """实盘开多头仓位"""
        opened_positions = {}
        if not symbols:
            return True, opened_positions

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
                    # 记录成功开仓的位置
                    opened_positions[symbol] = {
                        'side': 'long',
                        'quantity': quantity
                    }
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[LIVE] 多头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[LIVE ERROR] 做多 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[LIVE OK] 多头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0, opened_positions

    def _open_short_positions(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """开空头仓位 - 环境路由方法"""
        if self.use_testnet:
            return self._open_short_positions_testnet(symbols, position_value)
        else:
            return self._open_short_positions_live(symbols, position_value)

    def _open_short_positions_testnet(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """模拟盘开空头仓位"""
        opened_positions = {}
        if not symbols:
            return True, opened_positions

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
                    # 记录成功开仓的位置
                    opened_positions[symbol] = {
                        'side': 'short',
                        'quantity': quantity
                    }
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[TESTNET] 空头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[TESTNET ERROR] 做空 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[TESTNET OK] 空头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0, opened_positions

    def _open_short_positions_live(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """实盘开空头仓位"""
        opened_positions = {}
        if not symbols:
            return True, opened_positions

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
                    # 记录成功开仓的位置
                    opened_positions[symbol] = {
                        'side': 'short',
                        'quantity': quantity
                    }
                else:
                    # 订单失败，发送钉钉通知
                    error_msg = f"[LIVE] 空头订单失败！{symbol} 数量: {quantity} @ {current_price}"
                    self._send_dingding_notification(error_msg)

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[LIVE ERROR] 做空 {symbol} 失败: {e}")
                continue

        self.logger.info(f"[LIVE OK] 空头开仓: {success_count}/{len(symbols)} 成功")
        return success_count > 0, opened_positions

    def get_active_symbols(self) -> List[str]:
        """获取活跃交易对列表"""
        return self.active_symbols

    def _split_large_order(self, symbol: str, side: str, total_amount: float,
                          max_amount: float, max_batches: int = None) -> List[float]:
        """
        分割大订单为多个小订单

        Args:
            symbol: 交易对
            side: 买卖方向
            total_amount: 总交易数量
            max_amount: 单次最大交易数量
            max_batches: 最大分批数量（保留参数兼容性，但不再限制）

        Returns:
            List[float]: 分批后的数量列表
        """
        # 检查是否启用分批交易
        if not getattr(self.config, 'ENABLE_BATCH_TRADING', True):
            self.logger.warning(f"[WARNING] {symbol} 分批交易已禁用，无法处理大订单")
            return [total_amount]

        if total_amount <= max_amount:
            return [total_amount]

        # 计算需要分成多少批（确保完全执行原始数量）
        batches_needed = math.ceil(total_amount / max_amount)

        self.logger.info(f"[SPLIT INFO] {symbol} 大订单分批: 总量 {total_amount}，需要 {batches_needed} 批")

        # 平均分配数量，确保总量不变
        base_amount = total_amount / batches_needed
        remaining_amount = total_amount

        # 获取精度信息进行舍入
        market = self.exchange.markets[symbol]
        amount_precision = market['precision']['amount']

        batch_amounts = []

        # 检查最小分批大小（基于单批，而非总量）
        min_batch_ratio = getattr(self.config, 'MIN_BATCH_SIZE_RATIO', 0.1)
        min_batch_size = max_amount * min_batch_ratio  # 基于最大单批大小计算最小批次

        for i in range(batches_needed):
            if i == batches_needed - 1:
                # 最后一批使用剩余数量
                batch_amount = remaining_amount
            else:
                batch_amount = base_amount

            # 应用精度处理
            if isinstance(amount_precision, float):
                decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                batch_amount = round(batch_amount, decimal_places)
            else:
                precision_power = int(amount_precision)
                batch_amount = math.floor(batch_amount * (10 ** precision_power)) / (10 ** precision_power)

            # 确保每批数量不超过最大限制
            if batch_amount > max_amount:
                batch_amount = max_amount

            # 确保批次大小合理（除了最后一批）
            if batch_amount > 0:
                if i == batches_needed - 1:
                    # 最后一批，无论多小都要执行
                    batch_amounts.append(batch_amount)
                elif batch_amount >= min_batch_size:
                    # 中间批次，必须满足最小大小要求
                    batch_amounts.append(batch_amount)
                    remaining_amount -= batch_amount
                else:
                    # 如果中间批次太小，合并到上一批
                    if len(batch_amounts) > 0:
                        # 但要确保合并后不超过最大限制
                        combined_amount = batch_amounts[-1] + batch_amount
                        if combined_amount <= max_amount:
                            batch_amounts[-1] = combined_amount
                            remaining_amount -= batch_amount
                        else:
                            # 无法合并，单独作为一批
                            batch_amounts.append(batch_amount)
                            remaining_amount -= batch_amount
                    else:
                        # 第一批，即使小也要执行
                        batch_amounts.append(batch_amount)
                        remaining_amount -= batch_amount

        # 验证总量
        total_split = sum(batch_amounts)
        if abs(total_split - total_amount) > 0.0001:  # 允许微小的精度误差
            self.logger.warning(f"[SPLIT WARNING] {symbol} 分批总量 {total_split} 与原始总量 {total_amount} 不匹配")

        self.logger.info(f"[SPLIT] {symbol} 分批交易: 总量 {total_amount} 分为 {len(batch_amounts)} 批: {batch_amounts}")
        return batch_amounts

    def _validate_order_amount(self, symbol: str, amount: float) -> tuple[bool, str, float]:
        """
        验证订单数量是否在允许范围内

        Args:
            symbol: 交易对
            amount: 交易数量

        Returns:
            tuple: (是否有效, 错误信息, 调整后的数量)
        """
        try:
            market = self.exchange.markets[symbol]
            limits = market.get('limits', {}).get('amount', {})

            min_amount = limits.get('min', 0)
            max_amount = limits.get('max', float('inf'))

            # 检查最小数量
            if min_amount and amount < min_amount:
                return False, f"交易量 {amount} 小于最小值 {min_amount}", min_amount

            # 检查最大数量
            if max_amount and max_amount != float('inf') and amount > max_amount:
                return False, f"交易量 {amount} 大于最大值 {max_amount}", max_amount

            return True, "", amount

        except Exception as e:
            return False, f"验证交易量时出错: {e}", amount

    def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                # 对于同步ccxt，不需要close()方法
                self.exchange = None
            self.logger.info("[OK] 交易执行器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理交易执行器失败: {e}")

    def _create_market_order_testnet(self, symbol: str, side: str, amount: float,
                                   params: dict = None) -> dict:
        """
        模拟盘环境的市价单创建方法（支持分批交易）

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

            # 验证订单数量
            is_valid, error_msg, adjusted_amount = self._validate_order_amount(symbol, amount)
            if not is_valid:
                # 检查是否是数量过大的问题
                if "大于最大值" in error_msg:
                    self.logger.warning(f"[TESTNET WARNING] {symbol} {error_msg}，尝试分批交易")
                    market = self.exchange.markets[symbol]
                    max_amount = market.get('limits', {}).get('amount', {}).get('max', float('inf'))

                    # 分批处理
                    batch_amounts = self._split_large_order(symbol, side, amount, max_amount)
                    return self._execute_batch_orders_testnet(symbol, side, batch_amounts, params)
                else:
                    self.logger.error(f"[TESTNET ERROR] {symbol} {error_msg}")
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

                # 特殊处理 -4005 错误（数量过大）
                if "-4005" in error_str:
                    self.logger.warning(f"[TESTNET WARNING] {symbol} 数量过大，尝试分批交易")
                    market = self.exchange.markets[symbol]
                    max_amount = market.get('limits', {}).get('amount', {}).get('max', amount * 0.5)  # 使用一半作为估计
                    batch_amounts = self._split_large_order(symbol, side, amount, max_amount)
                    return self._execute_batch_orders_testnet(symbol, side, batch_amounts, params)
                elif "-4061" in error_str:
                    self.logger.error(f"[TESTNET ERROR] {symbol} 基础参数仍然失败，可能需要检查账户设置")
                elif "-4164" in error_str:
                    self.logger.error(f"[TESTNET ERROR] {symbol} 订单金额太小，请增加订单金额")

                raise e

        except Exception as e:
            self.logger.error(f"[TESTNET ERROR] {symbol} 市价单创建异常: {e}")
            return None

    def _execute_batch_orders_testnet(self, symbol: str, side: str, batch_amounts: List[float],
                                    params: dict = None) -> dict:
        """
        执行分批订单（模拟盘）

        Args:
            symbol: 交易对
            side: 买卖方向
            batch_amounts: 分批数量列表
            params: 额外参数

        Returns:
            dict: 合并的订单信息
        """
        try:
            total_filled = 0
            order_ids = []
            successful_batches = 0

            for i, batch_amount in enumerate(batch_amounts, 1):
                try:
                    self.logger.info(f"[TESTNET BATCH] {symbol} 执行第 {i}/{len(batch_amounts)} 批: {batch_amount}")

                    # 创建单批订单
                    batch_order = self.exchange.create_market_order(
                        symbol=symbol, side=side, amount=batch_amount, params={}
                    )

                    if batch_order:
                        order_id = batch_order.get('id', 'Unknown')
                        filled = batch_order.get('filled', 0)
                        total_filled += filled
                        order_ids.append(order_id)
                        successful_batches += 1

                        self.logger.info(f"[TESTNET SUCCESS] 第 {i} 批成功 ID:{order_id} 成交:{filled}")

                        # 批次间等待时间使用配置参数
                        wait_time = getattr(self.config, 'BATCH_WAIT_TIME', 0.1)
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"[TESTNET ERROR] 第 {i} 批订单创建失败")

                except Exception as e:
                    self.logger.error(f"[TESTNET ERROR] 第 {i} 批订单异常: {e}")
                    continue

            # 构造合并的订单信息
            if successful_batches > 0:
                merged_order = {
                    'id': f"BATCH_{'-'.join(order_ids)}",
                    'symbol': symbol,
                    'side': side,
                    'amount': sum(batch_amounts),
                    'filled': total_filled,
                    'status': 'closed' if successful_batches == len(batch_amounts) else 'partial',
                    'info': f"分批交易: {successful_batches}/{len(batch_amounts)} 批成功"
                }

                self.logger.info(f"[TESTNET BATCH COMPLETE] {symbol} 分批交易完成: {successful_batches}/{len(batch_amounts)} 批成功，总成交: {total_filled}")
                return merged_order
            else:
                self.logger.error(f"[TESTNET ERROR] {symbol} 所有分批订单都失败")
                return None

        except Exception as e:
            self.logger.error(f"[TESTNET ERROR] {symbol} 分批交易异常: {e}")
            return None

    def _create_market_order_live(self, symbol: str, side: str, amount: float,
                                 params: dict = None) -> dict:
        """
        实盘环境的市价单创建方法（支持分批交易）

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

            # 验证订单数量
            is_valid, error_msg, adjusted_amount = self._validate_order_amount(symbol, amount)
            if not is_valid:
                # 检查是否是数量过大的问题
                if "大于最大值" in error_msg:
                    self.logger.warning(f"[LIVE WARNING] {symbol} {error_msg}，尝试分批交易")
                    market = self.exchange.markets[symbol]
                    max_amount = market.get('limits', {}).get('amount', {}).get('max', float('inf'))

                    # 分批处理
                    batch_amounts = self._split_large_order(symbol, side, amount, max_amount)
                    return self._execute_batch_orders_live(symbol, side, batch_amounts, params)
                else:
                    self.logger.error(f"[LIVE ERROR] {symbol} {error_msg}")
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

                # 特殊处理 -4005 错误（数量过大）
                if "-4005" in error_str:
                    self.logger.warning(f"[LIVE WARNING] {symbol} 数量过大，尝试分批交易")
                    market = self.exchange.markets[symbol]
                    max_amount = market.get('limits', {}).get('amount', {}).get('max', amount * 0.5)  # 使用一半作为估计
                    batch_amounts = self._split_large_order(symbol, side, amount, max_amount)
                    return self._execute_batch_orders_live(symbol, side, batch_amounts, params)

                # 实盘的智能重试逻辑
                elif "-1106" in error_str and "reduceOnly" in safe_params:
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

    def _execute_batch_orders_live(self, symbol: str, side: str, batch_amounts: List[float],
                                  params: dict = None) -> dict:
        """
        执行分批订单（实盘）

        Args:
            symbol: 交易对
            side: 买卖方向
            batch_amounts: 分批数量列表
            params: 额外参数

        Returns:
            dict: 合并的订单信息
        """
        try:
            total_filled = 0
            order_ids = []
            successful_batches = 0
            safe_params = params if params is not None else {}

            for i, batch_amount in enumerate(batch_amounts, 1):
                try:
                    self.logger.info(f"[LIVE BATCH] {symbol} 执行第 {i}/{len(batch_amounts)} 批: {batch_amount}")

                    # 创建单批订单，使用原始参数
                    batch_order = self.exchange.create_market_order(
                        symbol=symbol, side=side, amount=batch_amount, params=safe_params
                    )

                    if batch_order:
                        order_id = batch_order.get('id', 'Unknown')
                        filled = batch_order.get('filled', 0)
                        total_filled += filled
                        order_ids.append(order_id)
                        successful_batches += 1

                        self.logger.info(f"[LIVE SUCCESS] 第 {i} 批成功 ID:{order_id} 成交:{filled}")

                        # 批次间等待时间使用配置参数
                        wait_time = getattr(self.config, 'BATCH_WAIT_TIME', 0.1)
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"[LIVE ERROR] 第 {i} 批订单创建失败")

                except Exception as e:
                    error_str = str(e)
                    self.logger.error(f"[LIVE ERROR] 第 {i} 批订单异常: {e}")

                    # 对单批订单也尝试参数修复
                    if "-1106" in error_str and "reduceOnly" in safe_params:
                        try:
                            retry_params = safe_params.copy()
                            retry_params.pop('reduceOnly', None)

                            batch_order = self.exchange.create_market_order(
                                symbol=symbol, side=side, amount=batch_amount, params=retry_params
                            )

                            if batch_order:
                                order_id = batch_order.get('id', 'Unknown')
                                filled = batch_order.get('filled', 0)
                                total_filled += filled
                                order_ids.append(order_id)
                                successful_batches += 1
                                self.logger.info(f"[LIVE SUCCESS] 第 {i} 批重试成功 ID:{order_id} 成交:{filled}")
                                wait_time = getattr(self.config, 'BATCH_WAIT_TIME', 0.1)
                                time.sleep(wait_time)
                        except Exception as e2:
                            self.logger.error(f"[LIVE ERROR] 第 {i} 批重试后仍失败: {e2}")

                    continue

            # 构造合并的订单信息
            if successful_batches > 0:
                merged_order = {
                    'id': f"BATCH_{'-'.join(order_ids)}",
                    'symbol': symbol,
                    'side': side,
                    'amount': sum(batch_amounts),
                    'filled': total_filled,
                    'status': 'closed' if successful_batches == len(batch_amounts) else 'partial',
                    'info': f"分批交易: {successful_batches}/{len(batch_amounts)} 批成功"
                }

                self.logger.info(f"[LIVE BATCH COMPLETE] {symbol} 分批交易完成: {successful_batches}/{len(batch_amounts)} 批成功，总成交: {total_filled}")
                return merged_order
            else:
                self.logger.error(f"[LIVE ERROR] {symbol} 所有分批订单都失败")
                return None

        except Exception as e:
            self.logger.error(f"[LIVE ERROR] {symbol} 分批交易异常: {e}")
            return None

    def _send_dingding_notification(self, message: str):
        """发送钉钉通知"""
        if DINGDING_AVAILABLE:
            try:
                env_prefix = "[TESTNET]" if self.config.use_testnet else "[LIVE]"
                # 获取日志文件名
                log_filename = getattr(self.config, 'LOG_FILENAME', None)
                if log_filename:
                    log_info = f"[{log_filename}]"
                else:
                    # 如果没有LOG_FILENAME，从LOG_FILE路径中提取文件名
                    import os
                    log_filename = os.path.basename(self.config.LOG_FILE).replace('.log', '')
                    log_info = f"[{log_filename}]"

                full_message = f"{env_prefix}{log_info} {message}"
                send_dingtalk_message(full_message)
                self.logger.info(f"[DINGDING] 通知已发送: {message}")
            except Exception as e:
                self.logger.error(f"[DINGDING ERROR] 发送钉钉通知失败: {e}")
        else:
            self.logger.warning("[DINGDING] 钉钉模块不可用，跳过通知")

