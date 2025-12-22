"""
按指定时间戳计算因子值的辅助脚本。

使用 whole_strategy 的配置和 fs_factor_calculator 计算方式，但允许传入任意时间戳，
在该时间点之前的历史数据上计算因子值，便于回测或验证。
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd

strategy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, strategy_root)

from config import TradingConfig  # noqa: E402
from fs_factor_calculator import OptimizedFactorCalculator  # noqa: E402
from previous_data_processoe import OptimizedDataProcessor  # noqa: E402


class PrintHelper:
    """简单打印助手，提供 info/warning/error 接口以适配计算器。"""

    @staticmethod
    def info(msg: str) -> None:
        print(msg)

    @staticmethod
    def warning(msg: str) -> None:
        print(msg)

    @staticmethod
    def error(msg: str) -> None:
        print(msg)


class HistoricalDataProcessor:
    """
    只获取指定时间戳之前的1h数据，接口保持与 OptimizedFactorCalculator 兼容。
    """

    def __init__(
        self,
        config: TradingConfig,
        printer,
        target_timestamp: datetime,
        limit: int,
        symbols: Optional[List[str]] = None,
    ):
        self.config = config
        self.printer = printer
        self.target_timestamp = target_timestamp
        self.limit = limit
        self.symbols = symbols
        self.data_processor: Optional[OptimizedDataProcessor] = None

    def initialize(self) -> None:
        """初始化数据处理器。"""
        self.data_processor = OptimizedDataProcessor(self.config, self.printer)
        self.data_processor.initialize()
        self.printer.info("[SYSTEM] 已初始化数据处理器")

    def _fetch_symbol_1h(
        self, symbol: str, limit: int, target_ts: datetime
    ) -> Optional[pd.DataFrame]:
        """
        使用 OptimizedDataProcessor 获取1h数据，并裁剪到目标时间之前。
        """
        if self.data_processor is None:
            raise RuntimeError("data_processor 未初始化")

        # 估算需要的条数，并传入 since 限制，确保抓到目标时间之前的数据
        since_dt = target_ts - timedelta(hours=83)
        since_ms = int(since_dt.timestamp() * 1000)

        df = self.data_processor.get_symbol_multiframe_data(
            symbol, limit=83, since_ms=since_ms
        )
        if df is None or df.empty:
            self.printer.warning(f"[WARNING] {symbol} 无法获取到数据")
            return None

        # 确保 open_time 为 tz-aware（UTC），便于和 target_ts 比较
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize(timezone.utc)
        else:
            df["open_time"] = df["open_time"].dt.tz_convert(timezone.utc)

        # 过滤目标时间之前的数据
        df = df[df["open_time"] < target_ts]

        if len(df) < limit:
            self.printer.warning(
                f"[DATA] {symbol} 数据不足: 需要 {limit} 条, 实际 {len(df)} 条"
            )
            return None

        return df.iloc[-limit:].reset_index(drop=True)

    def get_active_symbols(self) -> List[str]:
        """
        返回需要计算的交易对；默认遍历所有活跃 USDT 永续，并补充 BTC/USDT:USDT 作为基准。
        """
        if self.symbols:
            symbols = list(dict.fromkeys(self.symbols))
        else:
            assert self.data_processor is not None
            symbols = self.data_processor.get_active_symbols()

        if "BTC/USDT:USDT" not in symbols:
            symbols.append("BTC/USDT:USDT")
        return symbols

    def get_all_symbols_factors(
        self, timeframes: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        提供给 OptimizedFactorCalculator 使用的接口。
        """
        _ = timeframes  # 未使用，但保持签名一致
        limit = limit or self.limit
        symbols = self.get_active_symbols()

        all_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol == "USDC/USDT:USDT":
                continue
            data = self._fetch_symbol_1h(symbol, limit, self.target_timestamp)
            if data is not None:
                all_data[symbol] = data

        if not all_data:
            self.printer.error("[ERROR] 未获取到任何有效数据")
            return None

        return all_data

    def cleanup(self) -> None:
        if self.data_processor:
            try:
                self.data_processor.cleanup()
            except AttributeError as exc:
                self.printer.warning(f"[WARNING] 清理数据处理器时忽略错误: {exc}")
            finally:
                self.printer.info("[SYSTEM] 已关闭数据处理器")


def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """
    支持秒/毫秒时间戳或 ISO 格式字符串，统一转换为 UTC。
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        ts_int = int(str(value).strip())
        if ts_int > 10**12:
            return datetime.fromtimestamp(ts_int / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(ts_int, tz=timezone.utc)

    ts_dt = pd.to_datetime(value, utc=True)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
    return ts_dt.to_pydatetime()


def calculate_factors_at_timestamp(
    timestamp: Union[str, int, float, datetime],
    symbols: Optional[List[str]] = None,
    limit: int = 83,
) -> Optional[pd.Series]:
    """
    在指定时间戳计算给定交易对的因子值。

    Args:
        timestamp: 秒/毫秒时间戳或 ISO 时间字符串。
        symbols: 指定交易对列表；若为空则自动遍历所有活跃 USDT 永续（并补充 BTC/USDT:USDT）。
        limit: 需要的 1h K线数量，默认 83。

    Returns:
        因子 Series，index 为 symbol。
    """
    target_ts = parse_timestamp(timestamp)
    config = TradingConfig()
    printer = PrintHelper()
    printer.info(f"[SYSTEM] 目标时间: {target_ts.isoformat()}")

    data_processor = HistoricalDataProcessor(
        config=config,
        printer=printer,
        target_timestamp=target_ts,
        limit=limit,
        symbols=symbols,
    )
    data_processor.initialize()

    factor_calculator = OptimizedFactorCalculator(config, printer)
    factor_calculator.limit = limit
    factor_calculator.initialize()

    try:
        factors = factor_calculator.calculate_factors(data_processor)
        if factors is None or factors.empty:
            printer.error("[ERROR] 因子计算失败")
            return None


        printer.info(
            f"[OK] 因子计算完成，有效交易对 {len(factors)} 个，范围 [{factors.min():.6f}, {factors.max():.6f}]"
        )
        return factors
    finally:
        data_processor.cleanup()
        factor_calculator.cleanup()


if __name__ == "__main__":
    # 示例：在此处直接修改参数以运行（symbols 为空则遍历全部活跃 USDT 永续）
    demo_symbols: Optional[List[str]] = None
    demo_timestamp = "2025-11-11T01:00:00Z"
    result = calculate_factors_at_timestamp(
        timestamp=demo_timestamp,
        symbols=demo_symbols,
        limit=83,
    )

    pd.set_option("display.max_rows", None)
    if result is not None:
        print(result.sort_values(ascending=False))
