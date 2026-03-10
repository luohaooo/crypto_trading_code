"""
find_in_out 策略主脚本
每分钟执行一次：
1) 监控已有空单的止盈/止损触发与动态追踪
2) 检查 15m K 线开空信号
3) 下单并记录保护单，循环对齐到每分钟 00 秒
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binance_implement.whole_strategy.trading_executor import TradingExecutor  # noqa: E402
from binance_implement.whole_strategy.optimized_data_processor import OptimizedDataProcessor  # noqa: E402
from binance_implement.only_in.config import get_config, TradingConfig  # noqa: E402
from dingding import send_dingtalk_message  # noqa: E402
from binance_implement.account_stat.asset_stat import get_usdt_futures_asset_total


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    """简单日志配置"""
    logger = logging.getLogger("find_in_out")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复 handler
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def condition(symbol_df: pd.DataFrame) -> bool:
    if symbol_df.empty or len(symbol_df) < 21: # 确保至少有21根数据（1根当前的 + 20根均线）
        return False

    # 简化提取流程
    symbol_only = symbol_df.droplevel("symbol")
    volumes = symbol_only["volume"].astype(float)
    open_p = symbol_only["open"].astype(float)
    close_p = symbol_only["close"].astype(float)

    # 1. 获取当前（最后一根）的数据
    current_volume = volumes.iloc[-1]
    current_open = open_p.iloc[-1]
    current_close = close_p.iloc[-1]

    # 2. 获取前20根（不含当前根）的成交量并计算均值
    # iloc[-21:-1] 表示从倒数第21根开始，到倒数第2根结束（刚好20根）
    previous_20_volumes = volumes.iloc[-21:-1]
    avg_volume = previous_20_volumes.mean()

    # 3. 判断逻辑
    volume_trigger = current_volume >= 20 * avg_volume # 你的注释写20倍，代码写6倍，这里按注释改
    bullish = current_close > current_open

    return bool(volume_trigger and bullish)


def _normalize_fapi_symbol(symbol: str) -> str:
    base = symbol.split(":")[0]
    return base.replace("/", "")


def _fetch_funding_interval_map(
    base_url: str,
    timeout: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    endpoint = f"{base_url}/fapi/v1/fundingInfo"
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.warning(f"[FILTER WARNING] 获取 fundingInfo 失败: {exc}")
        return {}

    if not isinstance(payload, list):
        logger.warning("[FILTER WARNING] fundingInfo 返回格式异常")
        return {}

    interval_map: Dict[str, int] = {}
    for item in payload:
        symbol = item.get("symbol")
        hours = item.get("fundingIntervalHours")
        if symbol and hours is not None:
            try:
                interval_map[symbol] = int(hours)
            except (TypeError, ValueError):
                continue

    return interval_map


def _fetch_mark_price(
    base_url: str,
    symbol: str,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    endpoint = f"{base_url}/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    response = requests.get(endpoint, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        for item in payload:
            if item.get("symbol") == symbol:
                return item
        return None
    return payload


def _get_funding_rate_and_interval(
    symbol: str,
    base_url: str,
    timeout: int,
    interval_map: Dict[str, int],
    logger: logging.Logger,
) -> Tuple[Optional[float], Optional[int]]:
    fapi_symbol = _normalize_fapi_symbol(symbol)
    try:
        mark_payload = _fetch_mark_price(base_url, fapi_symbol, timeout)
    except requests.RequestException as exc:
        logger.warning(f"[FILTER WARNING] 获取 {fapi_symbol} mark price 失败: {exc}")
        return None, None

    if not mark_payload:
        logger.warning(f"[FILTER WARNING] 未找到 {fapi_symbol} mark price")
        return None, None

    rate_str = mark_payload.get("lastFundingRate")
    try:
        funding_rate = float(rate_str) if rate_str is not None else None
    except (TypeError, ValueError):
        funding_rate = None

    funding_interval = interval_map.get(fapi_symbol, 8)
    return funding_rate, funding_interval


def _passes_funding_rate_filter(
    symbol: str,
    base_url: str,
    timeout: int,
    interval_map: Dict[str, int],
    logger: logging.Logger,
) -> bool:
    funding_rate, funding_interval = _get_funding_rate_and_interval(
        symbol, base_url, timeout, interval_map, logger
    )
    if funding_rate is None or not funding_interval:
        logger.warning(f"[FILTER WARNING] {symbol} 资金费率数据缺失，跳过过滤")
        return True

    rate_per_hour = funding_rate / funding_interval
    if rate_per_hour < -0.0025:
        logger.info(
            f"[FILTER] {symbol} fundingRate={funding_rate:.8f}, "
            f"interval={funding_interval}h, ratio={rate_per_hour:.8f} < -0.0025，排除"
        )
        return False

    logger.info(
        f"[FILTER] {symbol} fundingRate={funding_rate:.8f}, "
        f"interval={funding_interval}h, ratio={rate_per_hour:.8f} 通过"
    )
    return True


# def determine_next_stop_levels(
#     last_minute_candle: List[Any],
#     info: Dict[str, Any],
#     config: TradingConfig,
# ) -> Tuple[Optional[float], Optional[float], float]:
#     """
#     根据上一分钟 K 线更新止盈止损（仅空单）
#     - 止盈：跟随新低，保留最紧的触发价
#     - 止损：跟随上一分钟高点上方一定缓冲
#     """

#     high = float(last_minute_candle[2])
#     low = float(last_minute_candle[3])
#     close = float(last_minute_candle[4])


#     min_price = min(info.get("min_price", low), low)

#     current_tp = info.get("take_profit_price")
#     current_sl = info.get("stop_loss_price")

#     new_tp = min(current_tp, close * (1-config.TP_BUFFER))
#     new_sl = current_sl - config.INCR * (current_sl - min_price)

#     return new_tp, new_sl, min_price




class FindInOutStrategy:
    """find_in_out 策略执行器"""

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.executor = TradingExecutor(config, logger)
        self.data_processor = OptimizedDataProcessor(config, logger)
        # 所有待监控空单列表（每个元素是一个dict）
        self.stage_entries: List[Dict[str, Any]] = []

    def initialize(self):
        """初始化交易所和数据处理器"""
        self.logger.info("[SYSTEM] 初始化组件...")
        self.executor.initialize()
        self.data_processor.initialize()
        balance = self.executor.get_account_balance()
        if balance is None:
            raise RuntimeError("无法获取账户余额，停止启动")
        self.logger.info(f"[BALANCE] 当前 USDT 余额: {balance:.2f}")
        self.logger.info(f"[SYSTEM] 激活交易对数量: {len(self.executor.get_active_symbols())}")

    def _fetch_closed_minute_candle(self, symbol: str) -> Optional[List[Any]]:
        """获取上一根已收盘的1m K线"""
        try:
            candles = self.executor.exchange.fetch_ohlcv(symbol, timeframe="1m", limit=1)
            return candles[-1]  # 上一根已闭合 K 线
        except Exception as exc:
            self.logger.warning(f"[KLINE WARNING] 获取 {symbol} 1m K线失败: {exc}")
            return None

    def _place_custom_protective_orders(
        self, symbol: str, quantity: float, tp_price: Optional[float], sl_price: Optional[float]
    ) -> Dict[str, Any]:
        """按指定价格创建新的止盈/止损单"""
        results: Dict[str, Any] = {}
        try:
            precise_qty = float(self.executor.exchange.amount_to_precision(symbol, quantity))
        except Exception as exc:
            self.logger.error(f"[PROTECT ERROR] {symbol} 数量精度调整失败: {exc}")
            return results
        if precise_qty <= 0:
            self.logger.error(f"[PROTECT ERROR] {symbol} 数量无效，无法下保护单")
            return results

        working_type = getattr(self.config, "PROTECTIVE_WORKING_TYPE", "MARK_PRICE")

        if tp_price:
            try:
                precise_tp = float(self.executor.exchange.price_to_precision(symbol, tp_price))
                order = self.executor._create_algo_protective_order(
                    symbol=symbol,
                    order_type="TAKE_PROFIT_MARKET",
                    side="buy",
                    position_side="SHORT",
                    quantity=precise_qty,
                    trigger_price=precise_tp,
                    working_type=working_type,
                )
                if order:
                    results["take_profit_price"] = tp_price
                    results["take_profit_order_id"] = order.get("algo_id") or order.get("id")
                    results["take_profit_client_id"] = order.get("client_algo_id")
                    self.logger.info(f"[PROTECT] {symbol} 更新止盈 {tp_price}，订单止盈 {precise_tp}")
            except Exception as exc:
                self.logger.error(f"[PROTECT ERROR] {symbol} 更新止盈失败: {exc}")

        if sl_price:
            try:
                precise_sl = float(self.executor.exchange.price_to_precision(symbol, sl_price))
                order = self.executor._create_algo_protective_order(
                    symbol=symbol,
                    order_type="STOP_MARKET",
                    side="buy",
                    position_side="SHORT",
                    quantity=precise_qty,
                    trigger_price=precise_sl,
                    working_type=working_type,
                )
                if order:
                    results["stop_loss_price"] = sl_price
                    results["stop_loss_order_id"] = order.get("algo_id") or order.get("id")
                    results["stop_loss_client_id"] = order.get("client_algo_id")
                    self.logger.info(f"[PROTECT] {symbol} 更新止损 {sl_price}，订单止损 {precise_sl}")
            except Exception as exc:
                self.logger.error(f"[PROTECT ERROR] {symbol} 更新止损失败: {exc}")

        return results

    def _check_and_update_positions(self, now: Optional[datetime] = None):
        # """监控并更新已有空单"""
        now = now or datetime.now()
        is_funding_check = now.minute == 59
        if not is_funding_check and now.minute % 15 != 0:
            self.logger.info("[SCAN] 非15分钟节点，跳过执行平仓")
            return

        if is_funding_check:
            if not self.stage_entries:
                self.logger.info("[MONITOR] 当前无持仓需要监控")
                return

            funding_interval_map = _fetch_funding_interval_map(
                self.config.BASE_URL, 10, self.logger
            )

            entries_snapshot = list(self.stage_entries)
            for info in entries_snapshot:
                symbol = info.get("symbol")
                quantity = info.get("quantity")
                if not symbol or not quantity:
                    self.logger.warning("[MONITOR WARNING] 持仓信息不完整，移除")
                    self.stage_entries.remove(info)
                    continue

                if _passes_funding_rate_filter(
                    symbol,
                    self.config.BASE_URL,
                    10,
                    funding_interval_map,
                    self.logger,
                ):
                    continue

                positions_dict = {symbol: {"side": "short", "quantity": quantity}}
                close_success = self.executor.close_specific_positions_simple(positions_dict)
                if not close_success:
                    self.logger.info(f"[MONITOR] {symbol} 资金费率平仓失败，可能已触发止盈/止损")

                try:
                    self.executor._cancel_single_protective_order_simple(
                        symbol=symbol,
                        order_id=info.get("take_profit_order_id"),
                        client_order_id=info.get("take_profit_client_id"),
                        label="止盈",
                        position_side="short",
                    )
                    self.executor._cancel_single_protective_order_simple(
                        symbol=symbol,
                        order_id=info.get("stop_loss_order_id"),
                        client_order_id=info.get("stop_loss_client_id"),
                        label="止损",
                        position_side="short",
                    )
                except Exception as exc:
                    self.logger.warning(f"[MONITOR WARNING] 撤销 {symbol} 保护单失败: {exc}")

                exit_price: Optional[float] = None
                try:
                    ticker = self.executor._fetch_ticker_with_rate_limit(symbol)
                    exit_price = ticker.get("last") or ticker.get("close")
                except Exception as exc:
                    self.logger.warning(f"[MONITOR WARNING] 获取 {symbol} 最新价格失败: {exc}")

                if exit_price is None:
                    candle = self._fetch_closed_minute_candle(symbol)
                    if candle:
                        try:
                            exit_price = float(candle[4])
                        except (TypeError, ValueError):
                            exit_price = None

                if exit_price is None:
                    exit_price = (
                        info.get("take_profit_price")
                        or info.get("stop_loss_price")
                        or info.get("open_price")
                    )

                entry_price = info.get("open_price")
                pnl_pct = None
                pnl = None
                if entry_price and exit_price:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                    pnl = quantity * entry_price * pnl_pct / 100

                exit_price_display = f"{exit_price:.6f}" if exit_price is not None else "N/A"
                entry_price_display = f"{entry_price:.6f}" if entry_price is not None else "N/A"
                pnl_pct_display = f"{pnl_pct:.2f}%" if pnl_pct is not None else "N/A"
                pnl_display = f"{pnl:.2f} USDT" if pnl is not None else "N/A"
                close_note = "平仓失败，可能已触发止盈/止损" if not close_success else "平仓成功"

                message = (
                    f"[find_in_out] {symbol} 资金费率未通过，执行平仓\n"
                    f"开仓时间: {info.get('entry_time')}\n"
                    f"开仓价: {entry_price_display}\n"
                    f"平仓价: {exit_price_display}\n"
                    f"预计收益: {pnl_pct_display}\n"
                    f"预计盈亏: {pnl_display}\n"
                    f"备注: {close_note}\n"
                )
                send_dingtalk_message(message)
                self.logger.info(f"[MONITOR] {symbol} 资金费率未通过已处理")

                self.stage_entries.remove(info)

            return

        self.logger.info("[MONITOR] 当前 stage_entries 列表:")
        for idx, entry in enumerate(self.stage_entries, 1): 
            self.logger.info(f"[MONITOR] #{idx} {entry}")

        if not self.stage_entries:
            self.logger.info("[MONITOR] 当前无持仓需要监控")
            return

        entries_snapshot = list(self.stage_entries)
        for info in entries_snapshot:
            expected_exit_time = info.get("expected_exit_time")
            if not expected_exit_time:
                continue

            try:
                expected_dt = datetime.strptime(expected_exit_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                self.logger.warning(f"[MONITOR WARNING] 无法解析预计平仓时间: {expected_exit_time}")
                continue

            if expected_dt > now:
                continue

            symbol = info.get("symbol")
            quantity = info.get("quantity")
            if not symbol or not quantity:
                self.logger.warning("[MONITOR WARNING] 平仓信息不完整，跳过")
                self.stage_entries.remove(info)
                continue

            positions_dict = {symbol: {"side": "short", "quantity": quantity}}
            close_success = self.executor.close_specific_positions_simple(positions_dict)
            if not close_success:
                self.logger.info(f"[MONITOR] {symbol} 预计平仓失败，可能已触发止盈/止损")

            try:
                self.executor._cancel_single_protective_order_simple(
                    symbol=symbol,
                    order_id=info.get("take_profit_order_id"),
                    client_order_id=info.get("take_profit_client_id"),
                    label="止盈",
                    position_side="short",
                )
                self.executor._cancel_single_protective_order_simple(
                    symbol=symbol,
                    order_id=info.get("stop_loss_order_id"),
                    client_order_id=info.get("stop_loss_client_id"),
                    label="止损",
                    position_side="short",
                )
            except Exception as exc:
                self.logger.warning(f"[MONITOR WARNING] 撤销 {symbol} 保护单失败: {exc}")

            exit_price: Optional[float] = None
            try:
                ticker = self.executor._fetch_ticker_with_rate_limit(symbol)
                exit_price = ticker.get("last") or ticker.get("close")
            except Exception as exc:
                self.logger.warning(f"[MONITOR WARNING] 获取 {symbol} 最新价格失败: {exc}")

            if exit_price is None:
                candle = self._fetch_closed_minute_candle(symbol)
                if candle:
                    try:
                        exit_price = float(candle[4])
                    except (TypeError, ValueError):
                        exit_price = None

            if exit_price is None:
                exit_price = (
                    info.get("take_profit_price")
                    or info.get("stop_loss_price")
                    or info.get("open_price")
                )

            entry_price = info.get("open_price")
            pnl_pct = None
            pnl = None
            if entry_price and exit_price:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                pnl = quantity * entry_price * pnl_pct / 100

            exit_price_display = f"{exit_price:.6f}" if exit_price is not None else "N/A"
            entry_price_display = f"{entry_price:.6f}" if entry_price is not None else "N/A"
            pnl_pct_display = f"{pnl_pct:.2f}%" if pnl_pct is not None else "N/A"
            pnl_display = f"{pnl:.2f} USDT" if pnl is not None else "N/A"
            close_note = "平仓失败，可能已触发止盈/止损" if not close_success else "平仓成功"

            message = (
                f"[find_in_out] {symbol} 到期平仓\n"
                f"开仓时间: {info.get('entry_time')}\n"
                f"预计平仓时间: {expected_exit_time}\n"
                f"开仓价: {entry_price_display}\n"
                f"平仓价: {exit_price_display}\n"
                f"预计收益: {pnl_pct_display}\n"
                f"预计盈亏: {pnl_display}\n"
                f"备注: {close_note}\n"
            )
            send_dingtalk_message(message)
            self.logger.info(f"[MONITOR] {symbol} 到期平仓已处理")

            self.stage_entries.remove(info)

    def _scan_open_opportunities(self, now: Optional[datetime] = None):
        """扫描 15m K线并开空，仅每15分钟执行一次"""
        now = now or datetime.now(timezone.utc)
        # 判断单仓保证金
        if now.hour == 0 and now.minute == 0:
            balance_info = get_usdt_futures_asset_total(self.config.API_KEY, self.config.API_SECRET)
            usdt_balance = balance_info.get("wallet_balance", 0)
            self.logger.info(f"[BALANCE] 当前总资产: {usdt_balance:.2f}")
            send_dingtalk_message(f"[find_in_out] 当前总资产: {usdt_balance:.2f} USDT")
            self.config.SINGLE_MARGIN = usdt_balance * 0.98 / 40
            self.logger.info(f"[CONFIG] 单仓保证金设置为: {self.config.SINGLE_MARGIN} USDT")

        if now.minute % 15 != 0:
            self.logger.info("[SCAN] 非15分钟节点，跳过开仓扫描")
            return

        active_symbols = set(self.executor.get_active_symbols())
        if not active_symbols:
            self.logger.warning("[SCAN] 无活跃交易对可扫描")
            return

        # # 从文本加载白名单
        # whitelist_path = Path(__file__).with_name("profitable_symbols.txt")
        # whitelist = set()
        # if whitelist_path.exists():
        #     whitelist = {line.strip() for line in whitelist_path.read_text().splitlines() if line.strip()}

        # 加载黑名单
        blacklist_path = Path(__file__).with_name("neglected_symbols.txt")
        blacklist = set()
        if blacklist_path.exists():
            blacklist = {
                line.strip()
                for line in blacklist_path.read_text().splitlines()
                if line.strip()
            }

        # scan_symbols = list(active_symbols & whitelist) if whitelist else list(active_symbols)

        scan_symbols = list(active_symbols - blacklist)
        
        if not scan_symbols:
            self.logger.warning("[SCAN] 无可扫描交易对")
            return
        else:
            self.logger.info(f"[SCAN] 本轮扫描交易对数量: {len(scan_symbols)}")

        funding_interval_map = _fetch_funding_interval_map(
            self.config.BASE_URL, 10, self.logger
        )

        wait_for_execute: List[str] = []
        for symbol in scan_symbols:
            try:
                candles = self.executor.exchange.fetch_ohlcv(symbol, timeframe="15m", limit=25)
            except Exception as exc:
                self.logger.warning(f"[SCAN WARNING] 获取 {symbol} 15m K线失败: {exc}")
                continue

            if not candles:
                continue

            df = pd.DataFrame(
                candles,
                columns=["open_time", "open", "high", "low", "close", "volume"],
            )

            # 保留至少 20 根
            if df.empty:
                continue

            
            df["symbol"] = symbol
            df.set_index(["symbol", "open_time"], inplace=True)

            if condition(df):
                if not _passes_funding_rate_filter(
                    symbol,
                    self.config.BASE_URL,
                    10,
                    funding_interval_map,
                    self.logger,
                ):
                    continue
                wait_for_execute.append(symbol)
                print(symbol, df.tail(3))

        if not wait_for_execute:
            self.logger.info("[SCAN] 本轮无满足条件的标的")
            return

        # # 测试用
        # wait_for_execute = ['DOGS/USDT:USDT']

        # 去除当前已经有仓位的symbol
        existing_symbols = {
            entry.get("symbol")
            for entry in self.stage_entries
            if entry.get("symbol")
        }
        if existing_symbols:
            wait_for_execute = [
                symbol for symbol in wait_for_execute if symbol not in existing_symbols
            ]
            if not wait_for_execute:
                self.logger.info("[SCAN] 已有持仓覆盖所有标的，本轮跳过开仓")
                return


        # 批量开空
        balance = self.executor.get_account_balance()
        find_num = len(wait_for_execute)
        to_be_executed_list = []

        if balance < self.config.SINGLE_MARGIN:
            self.logger.warning("[OPEN WARNING] 余额不足，无法开仓")
            return
        elif balance < find_num * self.config.SINGLE_MARGIN:
            self.logger.warning("[OPEN WARNING] 余额不足，无法覆盖所有找到的symbol")
            num_to_be_executed = math.floor(balance / self.config.SINGLE_MARGIN)
            to_be_executed_list = wait_for_execute[:num_to_be_executed]
        else:
            to_be_executed_list = wait_for_execute
            self.logger.info("[OPEN] 余额能够完全覆盖所有找到的symbol")
        

        try:
            self.executor._set_margin_type_for_symbols(to_be_executed_list)
        except Exception as exc:
            self.logger.warning(f"[LEVERAGE WARNING] 批量设置保证金模式失败: {exc}")

        try:
            self.executor._set_leverage_for_symbols(to_be_executed_list)
        except Exception as exc:
            self.logger.warning(f"[LEVERAGE WARNING] 批量设置杠杆失败: {exc}")

        position_value = self.config.SINGLE_MARGIN * self.config.LEVERAGE
        success, opened = self.executor._open_short_positions(to_be_executed_list, position_value)
        if not success:
            self.logger.warning("[OPEN] 本轮开空全部失败")
            return

        for symbol, pos in opened.items():
            entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            expected_exit_time = (now + timedelta(minutes=self.config.EXPECTED_PERIOD-1)).strftime("%Y-%m-%d %H:%M:%S")

            stage_entry = {
                "symbol": symbol,
                "entry_time": entry_time,
                "expected_exit_time": expected_exit_time,
                "quantity": pos.get("quantity"),
                "open_price": pos.get("open_price"),
                "take_profit_price": pos.get("take_profit_price"),
                "stop_loss_price": pos.get("stop_loss_price"),
                "take_profit_order_id": pos.get("take_profit_order_id"),
                "stop_loss_order_id": pos.get("stop_loss_order_id"),
                "take_profit_client_id": pos.get("take_profit_client_id"),
                "stop_loss_client_id": pos.get("stop_loss_client_id"),
            }
            # 以配置的 TP/SL buffer 重置保护单
            entry_price = pos.get("open_price")
            tp_buffer = getattr(self.config, "TP_BUFFER", None)
            sl_buffer = getattr(self.config, "SL_BUFFER", None)
            custom_tp = entry_price * (1 - tp_buffer) if entry_price and tp_buffer else pos.get("take_profit_price")
            custom_sl = entry_price * (1 + sl_buffer) if entry_price and sl_buffer else pos.get("stop_loss_price")

            # 取消原有保护单后重新下单
            updates = self._place_custom_protective_orders(
                symbol=symbol,
                quantity=stage_entry["quantity"],
                tp_price=custom_tp,
                sl_price=custom_sl,
            )

            stage_entry.update(updates)

            self.stage_entries.append(stage_entry)
            send_dingtalk_message(
                f"[find_in_out] 开空 {symbol}\n"
                f"数量: {pos.get('quantity')}\n"
                f"开仓价: {pos.get('open_price')}\n"
                f"止盈: {stage_entry.get('take_profit_price'):.6f}\n"
                f"止损: {stage_entry.get('stop_loss_price'):.6f}"
            )


    def _sleep_to_next_minute(self, start_time: datetime):
        """等待到下一分钟的 00 秒"""
        next_minute = (start_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
        now = datetime.now()
        if now >= next_minute:
            return
        wait_seconds = (next_minute - now).total_seconds()
        self.logger.info(f"[WAIT] 等待 {wait_seconds:.1f} 秒进入下一轮")
        time.sleep(wait_seconds)

    async def start(self):
        """主循环：每分钟对齐执行"""
        self.initialize()
        self.logger.info("[SYSTEM] find_in_out 策略启动")

        # 判断单仓保证金
        balance_info = get_usdt_futures_asset_total(self.config.API_KEY, self.config.API_SECRET)
        usdt_balance = balance_info.get("wallet_balance", 0)
        self.logger.info(f"[BALANCE] 当前总资产: {usdt_balance:.2f}")
        send_dingtalk_message(f"[find_in_out] 当前总资产: {usdt_balance:.2f} USDT")
        self.config.SINGLE_MARGIN = usdt_balance * 0.98 / 40
        self.logger.info(f"[CONFIG] 单仓保证金设置为: {self.config.SINGLE_MARGIN} USDT")

        # 对齐到下一分钟
        now = datetime.now()
        first_tick = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        time.sleep((first_tick - now).total_seconds())

        while True:
            loop_start = datetime.now()
            try:
                self._check_and_update_positions(loop_start)
                self._scan_open_opportunities(loop_start)
            except Exception as exc:
                self.logger.error(f"[ERROR] 主循环异常: {exc}")
            self._sleep_to_next_minute(loop_start)

def main():
    config = get_config()
    logger = setup_logger(config.LOG_FILE, config.LOG_LEVEL)

    strategy = FindInOutStrategy(config, logger)
    try:
        asyncio.run(strategy.start())
    except KeyboardInterrupt:
        logger.info("[EXIT] 用户中断")
    except Exception as exc:
        logger.error(f"[EXIT ERROR] 程序异常: {exc}")


if __name__ == "__main__":
    main()
