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
import hashlib
import hmac
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from urllib.parse import urlencode

import requests

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
        self.exchange = None
        self.active_symbols = []
        self.margin_configured_symbols: Set[str] = set()

        # API频率控制
        self.last_api_call_time = 0
        self.api_call_interval = 0.1  # 最小间隔100ms

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
            # 获取所有市场信息（默认已经在config中设置为期货环境）
            markets = self.exchange.load_markets()

            # 仅保留USDT计价、处于TRADING状态的永续合约
            usdt_perp_symbols = []
            for symbol, market in markets.items():
                is_contract = market.get('contract', False)
                is_usdt_quote = market.get('quote') == 'USDT'
                is_perp = market.get('swap', False) or (
                    market.get('info', {}).get('contractType', '').lower() == 'perpetual'
                )
                is_active = market.get('active', True)
                status = market.get('info', {}).get('status', '').upper()

                if (
                    is_contract
                    and is_usdt_quote
                    and is_perp
                    and is_active
                    and (not status or status == 'TRADING')
                ):
                    usdt_perp_symbols.append(symbol)

            self.active_symbols = sorted(set(usdt_perp_symbols))
            self.logger.info(f"[INFO] 找到 {len(self.active_symbols)} 个活跃USDT永续合约")

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
                message = f" 账户余额报告\n 时间: {current_time}\n USDT余额: {balance:.2f}"
                self._send_dingding_notification(message)
                self.logger.info(f"[BALANCE] 余额通知已发送: {balance:.2f} USDT")
            else:
                error_msg = f"❌ 余额查询失败\n 时间: {current_time}\n 环境: {self.config.ENV_NAME}"
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
        平仓所有现有仓位

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

                    # 实盘平仓直接使用市价单
                    order = self._create_market_order_live(
                        symbol=symbol,
                        side=side,
                        amount=size,
                        params={'positionSide': position_side}
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
        平仓指定仓位

        Args:
            positions_dict: 仓位字典，格式: {symbol: {'side': 'long'/'short', 'quantity': float}}

        Returns:
            bool: 平仓是否成功
        """
        return self.close_specific_positions_live(positions_dict)

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
            summary_lines: List[str] = []
            missing_profit_symbols: List[str] = []
            total_profit_loss = 0.0

            for symbol, position_info in positions_dict.items():
                try:
                    side = position_info['side']
                    quantity = position_info['quantity']
                    open_price = position_info.get('open_price', 0)  # 获取开仓价格，如果没有则默认为0

                    # 取消关联的止盈止损订单
                    self._cancel_protective_orders(symbol, position_info)

                    # 获取当前价格用于计算收益
                    try:
                        ticker = self._fetch_ticker_with_rate_limit(symbol)
                        current_price = ticker['last'] if ticker else 0
                    except Exception as e:
                        self.logger.error(f"[LIVE ERROR] 无法获取 {symbol} 当前价格: {e}")
                        current_price = 0

                    # 计算预期收益（只有在有开仓价格和当前价格时才计算）
                    profit_loss = 0
                    profit_percentage = 0
                    if open_price > 0 and current_price > 0:
                        if side == 'long':
                            # 多头：(当前价格 - 开仓价格) / 开仓价格 * 100%
                            profit_percentage = ((current_price - open_price) / open_price) * 100
                            profit_loss = (current_price - open_price) * quantity
                        else:  # short
                            # 空头：(开仓价格 - 当前价格) / 开仓价格 * 100%
                            profit_percentage = ((open_price - current_price) / open_price) * 100
                            profit_loss = (open_price - current_price) * quantity

                    # 确定平仓方向 (做多仓位用sell平仓，做空仓位用buy平仓)
                    close_side = 'sell' if side == 'long' else 'buy'

                    # 记录平仓信息和收益
                    if open_price > 0 and current_price > 0:
                        self.logger.info(f"[LIVE CLOSE] 平仓 {symbol}: {close_side} {quantity} (原{side}仓位)")
                        self.logger.info(f"[LIVE PROFIT] {symbol} 开仓价格: {open_price:.8f}, 当前价格: {current_price:.8f}")
                        self.logger.info(f"[LIVE PROFIT] {symbol} 预期收益: {profit_percentage:.2f}% (约 {profit_loss:.8f} USDT)")
                    else:
                        self.logger.info(f"[LIVE CLOSE] 平仓 {symbol}: {close_side} {quantity} (原{side}仓位)")
                        self.logger.warning(f"[LIVE WARNING] {symbol} 缺少价格信息，无法计算收益")

                    # 确定positionSide参数
                    position_side = 'LONG' if side == 'long' else 'SHORT'

                    # 实盘平仓
                    order = self._create_market_order_live(
                        symbol=symbol,
                        side=close_side,
                        amount=abs(quantity),
                        params={'positionSide': position_side}
                    )

                    if order:
                        close_orders.append(order)
                        self.logger.info(f"[LIVE SUCCESS] 平仓订单创建成功 {symbol}: {order.get('id', 'Unknown')}")

                        if open_price > 0 and current_price > 0:
                            direction_text = "做多" if side == 'long' else "做空"
                            summary_lines.append(
                                f"{symbol} | {direction_text} | 数量 {quantity:.4f} | 开仓 {open_price:.4f} | 平仓 {current_price:.4f} | 收益 {profit_percentage:.2f}% ({profit_loss:.2f} USDT)"
                            )
                            total_profit_loss += profit_loss
                        else:
                            missing_profit_symbols.append(symbol)
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

            if summary_lines:
                summary_message = "[实盘] 平仓汇总\n"
                summary_message += "\n\n".join(summary_lines)
                summary_message += f"\n\n总收益: {total_profit_loss:.2f} USDT"
                if missing_profit_symbols:
                    summary_message += f"\n以下交易缺少价格信息无法计算收益: {', '.join(missing_profit_symbols)}"
                self._send_dingding_notification(summary_message)
            elif missing_profit_symbols:
                message = "[实盘] 平仓汇总\n所有平仓缺少价格或收益信息，未能计算收益。\n"
                message += f"涉及交易对: {', '.join(missing_profit_symbols)}"
                self._send_dingding_notification(message)

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
                开仓位置字典格式:
                {
                    symbol: {
                        'side': 'long'/'short',
                        'quantity': float,
                        'open_price': float,
                        'position_side': 'LONG'/'SHORT',
                        'take_profit_order_id': Optional[str],
                        'take_profit_price': Optional[float],
                        'stop_loss_order_id': Optional[str],
                        'stop_loss_price': Optional[float],
                    }
                }
        """
        opened_positions = {}
        try:
            all_symbols = long_symbols + short_symbols
            total_positions = len(all_symbols)

            if total_positions == 0:
                self.logger.info("[OK] 无需开仓标的")
                return True, opened_positions

            # 计算每个仓位的资金
            position_value = 0.98 * total_balance * self.config.LEVERAGE / total_positions # 防止资金不足
            self.logger.info(f"[BALANCE] 总余额: {total_balance:.2f} USDT")
            self.logger.info(f"[INFO] 总仓位数: {total_positions}")
            self.logger.info(f"[VALUE] 单仓价值: {position_value:.2f} USDT")

            # 设置保证金模式
            self._set_margin_type_for_symbols(all_symbols)

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

    def _set_margin_type_for_symbols(self, symbols: List[str]):
        """为交易对设置保证金模式"""
        if not self.config.ENABLE_MARGIN_TYPE_SETTING:
            self.logger.info("[MARGIN] 保证金模式设置已禁用，跳过")
            return

        # 避免重复设置导致 -4047（存在未完成订单无法切换保证金模式）
        pending_symbols = [s for s in symbols if s not in self.margin_configured_symbols]
        if not pending_symbols:
            self.logger.debug("[MARGIN] 所有交易对的保证金模式已设置，跳过")
            return

        self.logger.info(f"[MARGIN] 设置保证金模式为 {self.config.MARGIN_TYPE}...")
        success_count = 0
        failed_symbols = []

        for symbol in pending_symbols:
            retry_count = 0
            while retry_count < self.config.MARGIN_TYPE_RETRY_COUNT:
                try:
                    # 获取市场信息进行symbol格式转换
                    market = self.exchange.market(symbol)
                    binance_symbol = market['id']  # 获取Binance原生格式 (如: BTCUSDT)

                    # 使用CCXT直接调用Binance API
                    response = self.exchange.fapiprivate_post_margintype({
                        'symbol': binance_symbol,  # 使用转换后的格式
                        'marginType': self.config.MARGIN_TYPE,
                    })
                    self.logger.info(f"[MARGIN] {symbol} ({binance_symbol}) 保证金模式设置为 {self.config.MARGIN_TYPE} ✓")
                    success_count += 1
                    self.margin_configured_symbols.add(symbol)
                    time.sleep(0.05)  # 避免请求过快
                    break  # 成功则跳出重试循环
                except Exception as e:
                    error_msg = str(e).lower()

                    # 检查是否是市场信息获取失败（symbol格式问题）
                    if ('market' in error_msg and ('not found' in error_msg or 'not loaded' in error_msg)) or \
                       'exchangeerror' in type(e).__name__.lower():
                        self.logger.warning(f"[WARNING] {symbol} 市场信息获取失败，可能不支持futures交易，跳过保证金模式设置")
                        failed_symbols.append(symbol)
                        break

                    # 检查是否已经是目标保证金模式
                    if 'no need to change margin type' in error_msg or 'margin type is not modified' in error_msg:
                        self.logger.info(f"[MARGIN] {symbol} 已经是 {self.config.MARGIN_TYPE} 模式")
                        success_count += 1
                        self.margin_configured_symbols.add(symbol)
                        break

                    if 'margin type cannot be changed if there exists open orders' in error_msg or '-4047' in error_msg:
                        self.logger.warning(f"[MARGIN] {symbol} 存在挂单，无法切换保证金模式，本次跳过")
                        failed_symbols.append(symbol)
                        break

                    retry_count += 1
                    if retry_count < self.config.MARGIN_TYPE_RETRY_COUNT:
                        self.logger.warning(f"[WARNING] 设置 {symbol} 保证金模式失败 (重试 {retry_count}/{self.config.MARGIN_TYPE_RETRY_COUNT}): {e}")
                        time.sleep(0.1)  # 重试前等待
                    else:
                        self.logger.error(f"[ERROR] 设置 {symbol} 保证金模式最终失败: {e}")
                        failed_symbols.append(symbol)

        # 汇总结果
        self.logger.info(f"[MARGIN] 保证金模式设置完成: {success_count}/{len(pending_symbols)} 成功")
        if failed_symbols:
            self.logger.warning(f"[MARGIN] 失败的交易对: {failed_symbols}")

    def _open_long_positions(self, symbols: List[str], position_value: float) -> tuple[bool, dict]:
        """开多头仓位"""
        return self._open_long_positions_live(symbols, position_value)

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
                ticker = self._fetch_ticker_with_rate_limit(symbol)
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
                    scale = 10 ** decimal_places if decimal_places > 0 else 1
                    quantity = math.floor(quantity * scale) / scale
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
                    # 记录成功开仓的位置（包含开仓价格）
                    opened_positions[symbol] = {
                        'side': 'long',
                        'quantity': quantity,
                        'open_price': current_price,
                        'position_side': 'LONG'
                    }
                    protective_orders = self._place_protective_orders(symbol, 'long', quantity, current_price)
                    opened_positions[symbol].update(protective_orders)
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
        """开空头仓位"""
        return self._open_short_positions_live(symbols, position_value)

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
                ticker = self._fetch_ticker_with_rate_limit(symbol)
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
                    # 记录成功开仓的位置（包含开仓价格）
                    opened_positions[symbol] = {
                        'side': 'short',
                        'quantity': quantity,
                        'open_price': current_price,
                        'position_side': 'SHORT'
                    }
                    protective_orders = self._place_protective_orders(symbol, 'short', quantity, current_price)
                    opened_positions[symbol].update(protective_orders)
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

    def _calculate_protective_prices(self, side: str, entry_price: float) -> tuple[Optional[float], Optional[float]]:
        """根据仓位方向计算止盈止损价格"""
        if entry_price <= 0:
            return None, None

        tp_ratio = getattr(self.config, 'TAKE_PROFIT_RATIO', 0)
        sl_ratio = getattr(self.config, 'STOP_LOSS_RATIO', 0)

        take_profit_price = None
        stop_loss_price = None

        # 比例为0时表示不启用对应的保护单
        if tp_ratio:
            if side == 'long':
                take_profit_price = entry_price * (1 + tp_ratio)
            else:
                take_profit_price = entry_price * (1 - tp_ratio)

        if sl_ratio:
            if side == 'long':
                stop_loss_price = entry_price * (1 - sl_ratio)
            else:
                stop_loss_price = entry_price * (1 + sl_ratio)

        if take_profit_price and take_profit_price <= 0:
            take_profit_price = None
        if stop_loss_price and stop_loss_price <= 0:
            stop_loss_price = None

        return take_profit_price, stop_loss_price

    def _create_algo_protective_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        position_side: str,
        quantity: float,
        trigger_price: float,
        working_type: str
    ) -> Optional[Dict[str, Any]]:
        """使用Binance Algo Order API创建止盈/止损单"""
        if trigger_price is None or trigger_price <= 0:
            return None

        def _fallback_symbol_id(sym: str) -> str:
            cleaned = sym.replace('/', '')
            if ':' in cleaned:
                cleaned = cleaned.split(':')[0]
            return cleaned

        try:
            market = self.exchange.market(symbol)
            symbol_id = market.get('id') or _fallback_symbol_id(symbol)
        except Exception:
            symbol_id = _fallback_symbol_id(symbol)

        url = f"{self.config.BASE_URL}/fapi/v1/algoOrder"
        side_upper = 'BUY' if side.lower() == 'buy' else 'SELL'
        timestamp = int(time.time() * 1000)
        suffix = str(timestamp)[-10:]  # keep client id short enough for Binance limit
        order_code = 'tp' if 'TAKE' in order_type.upper() else 'sl'
        client_algo_id = f"prot_{order_code}_{suffix}"
        base_payload = {
            'symbol': symbol_id,
            'side': side_upper,
            'positionSide': position_side,
            'algoType': 'CONDITIONAL',
            'type': order_type,
            'quantity': str(quantity),
            'triggerPrice': str(trigger_price),
            'workingType': working_type,
            'priceProtect': 'TRUE',
            'clientAlgoId': client_algo_id,
            'recvWindow': 5000,
            'timestamp': timestamp,
        }

        query_string = urlencode(base_payload)
        signature = hmac.new(
            self.config.API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        payload = dict(base_payload)
        payload['signature'] = signature
        headers = {
            'X-MBX-APIKEY': self.config.API_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
        except requests.RequestException as e:
            raise RuntimeError(f"Algo order request failed: {e}")

        try:
            data = response.json()
        except ValueError:
            data = response.text or ''

        if not response.ok:
            raise RuntimeError(
                f"Algo order HTTP {response.status_code}: {data}"
            )

        if isinstance(data, dict) and 'code' in data and data.get('code') not in (0, '0'):
            msg = data.get('msg', 'Unknown error')
            raise RuntimeError(f"Algo order error {data.get('code')}: {msg}")

        algo_id = None
        if isinstance(data, dict):
            algo_id = data.get('algoId') or data.get('orderId')
            client_algo_id_resp = data.get('clientAlgoId') or client_algo_id
        else:
            client_algo_id_resp = client_algo_id

        if algo_id is None and client_algo_id_resp:
            algo_id = client_algo_id_resp

        return {
            'id': str(algo_id) if algo_id is not None else None,
            'algo_id': str(algo_id) if algo_id is not None else None,
            'client_algo_id': client_algo_id_resp,
            'info': data
        }

    def _cancel_algo_order(
        self,
        algo_id: Optional[str] = None,
        client_algo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """取消已创建的Algo订单"""
        if not algo_id and not client_algo_id:
            raise ValueError("algo_id 或 client_algo_id 必须提供一个")

        url = f"{self.config.BASE_URL}/fapi/v1/algoOrder"
        timestamp = int(time.time() * 1000)
        base_payload = {
            'recvWindow': 5000,
            'timestamp': timestamp,
        }
        if algo_id:
            base_payload['algoid'] = str(algo_id)
        if client_algo_id:
            base_payload['clientalgoid'] = client_algo_id

        query_string = urlencode(base_payload)
        signature = hmac.new(
            self.config.API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        payload = dict(base_payload)
        payload['signature'] = signature
        headers = {
            'X-MBX-APIKEY': self.config.API_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.delete(url, headers=headers, params=payload, timeout=10)
        except requests.RequestException as e:
            raise RuntimeError(f"Algo cancel request failed: {e}")

        try:
            data = response.json()
        except ValueError:
            data = response.text or ''

        if not response.ok:
            raise RuntimeError(f"Algo cancel HTTP {response.status_code}: {data}")

        if isinstance(data, dict) and 'code' in data:
            code_value = data.get('code')
            if code_value not in (0, '0', 200, '200'):
                msg = data.get('msg', 'Unknown error')
                raise RuntimeError(f"Algo cancel error {code_value}: {msg}")

        if isinstance(data, dict):
            return data
        return {'info': data}

    def _place_protective_orders(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float
    ) -> Dict[str, Optional[float]]:
        """
        为新仓位创建止盈止损订单

        Returns:
            dict: 包含保护单信息的字典
        """
        if not getattr(self.config, 'ENABLE_PROTECTIVE_ORDERS', False):
            return {}

        if quantity <= 0 or entry_price <= 0:
            self.logger.warning(f"[PROTECT] {symbol} 保护单跳过，数量或价格无效 quantity={quantity}, entry={entry_price}")
            return {}

        take_profit_price, stop_loss_price = self._calculate_protective_prices(side, entry_price)
        position_side = 'LONG' if side == 'long' else 'SHORT'
        results: Dict[str, Optional[float]] = {}

        try:
            if take_profit_price:
                precise_tp = float(self.exchange.price_to_precision(symbol, take_profit_price))
            else:
                precise_tp = None
            if stop_loss_price:
                precise_sl = float(self.exchange.price_to_precision(symbol, stop_loss_price))
            else:
                precise_sl = None
            precise_quantity = float(self.exchange.amount_to_precision(symbol, quantity))
        except Exception as e:
            self.logger.error(f"[PROTECT ERROR] {symbol} 价格精度转换失败: {e}")
            return results
        if precise_quantity <= 0:
            self.logger.warning(f"[PROTECT] {symbol} 保护单跳过，数量精度调整后无效 quantity={quantity}")
            return results

        working_type = getattr(self.config, 'PROTECTIVE_WORKING_TYPE', 'MARK_PRICE')

        # 止盈单
        if precise_tp:
            tp_side = 'sell' if side == 'long' else 'buy'
            try:
                tp_order = self._create_algo_protective_order(
                    symbol=symbol,
                    order_type='TAKE_PROFIT_MARKET',
                    side=tp_side,
                    position_side=position_side,
                    quantity=precise_quantity,
                    trigger_price=precise_tp,
                    working_type=working_type
                )
                if tp_order:
                    tp_id = tp_order.get('algo_id') or tp_order.get('id')
                    tp_client_id = tp_order.get('client_algo_id')
                    if tp_id:
                        results['take_profit_order_id'] = tp_id
                    if tp_client_id:
                        results['take_profit_client_id'] = tp_client_id
                    results['take_profit_price'] = precise_tp
                    self.logger.info(f"[PROTECT] {symbol} 止盈单创建成功 ID:{tp_id} 触发价:{precise_tp}")
                time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"[PROTECT ERROR] {symbol} 止盈单创建失败: {e}")

        # 止损单
        if precise_sl:
            sl_side = 'sell' if side == 'long' else 'buy'
            try:
                sl_order = self._create_algo_protective_order(
                    symbol=symbol,
                    order_type='STOP_MARKET',
                    side=sl_side,
                    position_side=position_side,
                    quantity=precise_quantity,
                    trigger_price=precise_sl,
                    working_type=working_type
                )
                if sl_order:
                    sl_id = sl_order.get('algo_id') or sl_order.get('id')
                    sl_client_id = sl_order.get('client_algo_id')
                    if sl_id:
                        results['stop_loss_order_id'] = sl_id
                    if sl_client_id:
                        results['stop_loss_client_id'] = sl_client_id
                    results['stop_loss_price'] = precise_sl
                    self.logger.info(f"[PROTECT] {symbol} 止损单创建成功 ID:{sl_id} 触发价:{precise_sl}")
                time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"[PROTECT ERROR] {symbol} 止损单创建失败: {e}")

        return results

    def _cancel_protective_orders(self, symbol: str, position_info: dict, known_statuses: Optional[Dict[str, str]] = None):
        """取消与仓位关联的止盈止损订单"""
        if not getattr(self.config, 'ENABLE_PROTECTIVE_ORDERS', False):
            return

        side = position_info.get('side')
        if side not in ['long', 'short']:
            return

        position_side = 'LONG' if side == 'long' else 'SHORT'
        params = {'positionSide': position_side}

        for key, label in [('take_profit_order_id', '止盈'), ('stop_loss_order_id', '止损')]:
            order_id = position_info.get(key)
            client_order_id = position_info.get(f"{key.replace('_order_id', '')}_client_id")
            if not order_id:
                continue
            status = None
            if known_statuses and order_id in known_statuses:
                status = known_statuses[order_id]
            if status and status.lower() in ['closed', 'filled']:
                continue  # 已成交订单无需取消
            try:
                self._cancel_algo_order(algo_id=str(order_id), client_algo_id=client_order_id)
                self.logger.info(f"[PROTECT] {symbol} 已取消{label}单 {order_id}")
            except Exception as e:
                self.logger.warning(
                    f"[PROTECT WARNING] {symbol} Algo接口取消{label}单失败 ({order_id}): {e}"
                )
                try:
                    self.exchange.cancel_order(order_id, symbol, params)
                    self.logger.info(f"[PROTECT] {symbol} 已通过常规接口取消{label}单 {order_id}")
                except Exception as inner_e:
                    self.logger.warning(
                        f"[PROTECT WARNING] {symbol} 取消{label}单失败 ({order_id}): {inner_e}"
                    )

    def _cancel_single_protective_order(
        self,
        symbol: str,
        order_id: Optional[str],
        client_order_id: Optional[str],
        label: str,
        position_side: Optional[str],
        status_lookup: Optional[Dict[str, str]]
    ) -> None:
        """取消指定的保护单（用于另一侧触发后的对侧撤单）"""
        if not order_id and not client_order_id:
            return

        status = None
        if status_lookup:
            if order_id and str(order_id) in status_lookup:
                status = status_lookup[str(order_id)]
            elif client_order_id and str(client_order_id) in status_lookup:
                status = status_lookup[str(client_order_id)]

        normalized = status.upper() if isinstance(status, str) else ''
        skip_statuses = {'TRIGGERED', 'FINISHED', 'FILLED', 'CLOSED', 'CANCELED', 'CANCELLED'}
        if normalized in skip_statuses:
            return  # 已结束或已被撤销的订单无需重复取消

        params = {}
        if position_side in ['long', 'short']:
            params['positionSide'] = 'LONG' if position_side == 'long' else 'SHORT'

        try:
            self._cancel_algo_order(algo_id=str(order_id) if order_id else None, client_algo_id=client_order_id)
            self.logger.info(f"[PROTECT] {symbol} 对侧触发，已取消{label}单 {order_id or client_order_id}")
        except Exception as e:
            self.logger.warning(
                f"[PROTECT WARNING] {symbol} Algo接口取消{label}单失败 ({order_id or client_order_id}): {e}"
            )
            if order_id:
                try:
                    self.exchange.cancel_order(order_id, symbol, params)
                    self.logger.info(f"[PROTECT] {symbol} 已通过常规接口取消{label}单 {order_id}")
                except Exception as inner_e:
                    self.logger.warning(
                        f"[PROTECT WARNING] {symbol} 取消{label}单失败 ({order_id}): {inner_e}"
                    )

    def _respect_rate_limit(self):
        """统一的速率限制控制"""
        elapsed = time.time() - self.last_api_call_time
        if elapsed < self.api_call_interval:
            time.sleep(self.api_call_interval - elapsed)
        self.last_api_call_time = time.time()

    def _fetch_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        """获取订单状态"""
        try:
            self._respect_rate_limit()
            order = self.exchange.fetch_order(order_id, symbol)
            self.last_api_call_time = time.time()
            status = order.get('status')
            self.logger.debug(f"[PROTECT DEBUG] {symbol} 订单 {order_id} 状态: {status}")
            return status
        except Exception as e:
            self.logger.warning(f"[PROTECT WARNING] 获取 {symbol} 订单状态失败 ({order_id}): {e}")
            return None

    def _get_position_contracts(self, symbol: str, position_side: Optional[str]) -> Optional[float]:
        """获取交易所当前持仓数量"""
        try:
            self._respect_rate_limit()
            positions = self.exchange.fetch_positions([symbol])
            self.last_api_call_time = time.time()
            target_side = position_side.lower() if position_side else None
            for pos in positions:
                if pos.get('symbol') != symbol:
                    continue
                side_text = pos.get('side')
                info_side = pos.get('info', {}).get('positionSide')
                matches = False
                if target_side:
                    if side_text and side_text.lower() == target_side:
                        matches = True
                    elif info_side and info_side.lower() == target_side:
                        matches = True
                else:
                    matches = True
                if matches:
                    contracts = pos.get('contracts')
                    if contracts is not None:
                        return abs(float(contracts))
            return None
        except Exception as e:
            self.logger.warning(f"[PROTECT WARNING] 获取 {symbol} 持仓信息失败: {e}")
            return None

    def refresh_positions_status(self, positions: Dict[str, dict]) -> List[Dict[str, str]]:
        """
        刷新仓位状态，识别已通过止盈止损平仓的仓位

        Returns:
            List[Dict[str, str]]: [{ 'symbol': str, 'reason': str }]
        """
        if not positions:
            return []

        symbols = list(positions.keys())
        algo_orders, status_lookup = self._fetch_current_algo_orders(symbols)
        if algo_orders is None or status_lookup is None:
            status_lookup = {}

        removed = []
        for symbol, info in list(positions.items()):
            reason = None
            statuses: Dict[str, str] = {}
            triggered_orders = []

            order_entries = [
                ('take_profit_order_id', 'take_profit_client_id', '止盈', 'take_profit'),
                ('stop_loss_order_id', 'stop_loss_client_id', '止损', 'stop_loss')
            ]
            triggered_statuses = {'TRIGGERED', 'FINISHED', 'FILLED', 'CLOSED', 'CANCELED', 'CANCELLED'}

            # 先检测是否有止盈/止损触发
            for order_id_key, client_id_key, label, short_key in order_entries:
                order_id = info.get(order_id_key)
                client_order_id = info.get(client_id_key)
                if not order_id and not client_order_id:
                    continue

                status = status_lookup.get(str(order_id)) if order_id else None
                if status is None and client_order_id:
                    status = status_lookup.get(str(client_order_id))
                lookup_key = str(order_id or client_order_id)
                if status:
                    statuses[lookup_key] = status
                else:
                    statuses[lookup_key] = 'UNKNOWN'

                normalized = status.upper() if isinstance(status, str) else None
                if normalized in triggered_statuses:
                    reason = f"{label}单已触发 (状态: {status})"
                    triggered_orders.append(short_key)
                elif status is None:
                    self.logger.warning(
                        f"[PROTECT WARNING] {symbol} 未在Algo列表中找到{label}单 {lookup_key}，保留仓位并等待下一轮检查"
                    )

            if triggered_orders:
                # 撤销对侧保护单
                position_side = info.get('side')
                for order_id_key, client_id_key, label, short_key in order_entries:
                    if short_key in triggered_orders:
                        continue
                    opposite_order_id = info.get(order_id_key)
                    opposite_client_id = info.get(client_id_key)
                    self._cancel_single_protective_order(
                        symbol=symbol,
                        order_id=opposite_order_id,
                        client_order_id=opposite_client_id,
                        label=label,
                        position_side=position_side,
                        status_lookup=status_lookup
                    )

                self.logger.info(f"[PROTECT] 检测到保护单触发，标记仓位 {symbol} ({reason or '保护单已触发'})")
                removed.append({'symbol': symbol, 'reason': reason or '保护单已触发'})
            else:
                self.logger.info(f"[PROTECT] {symbol} 未检测到保护单触发，状态: {statuses or 'N/A'}")
        if not removed:
            self.logger.info("[PROTECT] 本轮未检测到任何保护单触发")
        return removed

    def _fetch_current_algo_orders(self, symbols: List[str]) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, str]]]:
        """获取当前交易所的Algo订单列表（按symbol分页查询），并构建状态映射"""
        if not symbols:
            return [], {}

        url = f"{self.config.BASE_URL}/fapi/v1/allAlgoOrders"
        headers = {'X-MBX-APIKEY': self.config.API_KEY}
        all_orders: List[Dict[str, Any]] = []
        status_lookup: Dict[str, str] = {}
        page_size = 50

        def _to_symbol_id(sym: str) -> str:
            try:
                market = self.exchange.market(sym)
                return market.get('id') or sym.replace('/', '')
            except Exception:
                return sym.replace('/', '')

        for sym in symbols:
            symbol_id = _to_symbol_id(sym)
            page = 1
            while True:
                timestamp = int(time.time() * 1000)
                payload = {
                    'recvWindow': 5000,
                    'timestamp': timestamp,
                    'page': page,
                    'pageSize': page_size,
                    'symbol': symbol_id,
                }
                query = urlencode(payload)
                signature = hmac.new(
                    self.config.API_SECRET.encode('utf-8'),
                    query.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params = dict(payload)
                params['signature'] = signature

                try:
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                except requests.RequestException as exc:
                    self.logger.warning(f"[PROTECT WARNING] 获取 {sym} Algo订单列表失败 (page {page}): {exc}")
                    break

                try:
                    data = response.json()
                except ValueError:
                    self.logger.warning(f"[PROTECT WARNING] {sym} Algo订单列表响应无法解析为JSON (page {page})")
                    break

                if not response.ok:
                    self.logger.warning(f"[PROTECT WARNING] {sym} Algo订单列表接口返回错误 (page {page}): {data}")
                    break

                orders: List[Dict[str, Any]] = []
                if isinstance(data, list):
                    orders = [item for item in data if isinstance(item, dict)]
                elif isinstance(data, dict):
                    if isinstance(data.get('list'), list):
                        orders = [item for item in data.get('list', []) if isinstance(item, dict)]
                    elif isinstance(data.get('orders'), list):
                        orders = [item for item in data.get('orders', []) if isinstance(item, dict)]
                    elif any(key in data for key in ['algoId', 'clientAlgoId', 'orderId']):
                        orders = [data]

                for order in orders:
                    status_text_raw = order.get('algoStatus') or order.get('status') or order.get('orderStatus') or ''
                    status_text = str(status_text_raw)
                    for key in ['algoId', 'clientAlgoId', 'orderId', 'clientOrderId', 'id']:
                        val = order.get(key)
                        if val:
                            status_lookup[str(val)] = status_text
                    all_orders.append(order)

                if not orders or len(orders) < page_size:
                    break

                page += 1
                time.sleep(0.05)

        return all_orders, status_lookup

    def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                # 对于同步ccxt，不需要close()方法
                self.exchange = None
            self.logger.info("[OK] 交易执行器资源已清理")

        except Exception as e:
            self.logger.error(f"[ERROR] 清理交易执行器失败: {e}")

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

            # 验证订单数量
            is_valid, error_msg, _ = self._validate_order_amount(symbol, amount)
            if not is_valid:
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
                env_prefix = f"[{self.config.ENV_NAME}]"
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

    def _fetch_ticker_with_rate_limit(self, symbol: str) -> dict:
        """
        带频率控制的单个价格获取方法
        Args:
            symbol: 交易对符号
        Returns:
            ticker 信息字典
        """
        import time

        # 控制API调用频率
        self._respect_rate_limit()
        ticker = self.exchange.fetch_ticker(symbol)
        self.last_api_call_time = time.time()

        return ticker
