"""
实用函数：通过 Binance USDT-M Websocket 获取指定 symbol 的上一分钟 K 线。
用法示例：
    from fetch_last_minute import get_last_minute_kline
    data = asyncio.run(get_last_minute_kline('btcusdt'))
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import websockets

WS_BASE = "wss://fstream.binance.com/ws"


async def get_last_minute_kline(symbol: str) -> Optional[Dict[str, Any]]:
    """
    获取指定 symbol 的上一分钟 K 线（USDT-M 期货 1m）。

    Args:
        symbol: 交易对，使用小写，如 'btcusdt'

    Returns:
        dict 或 None，其中包含 open/high/low/close/volume 等字段。
    """
    stream = f"{symbol.lower()}@kline_1m"
    ws_url = f"{WS_BASE}/{stream}"

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)
        k = data.get("k") or data.get("data", {}).get("k") or data

        if not k:
            return None

        return {
            "symbol": k.get("s"),
            "open_time": datetime.fromtimestamp(k.get("t", 0) / 1000),
            "close_time": datetime.fromtimestamp(k.get("T", 0) / 1000),
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "is_final": bool(k.get("x")),
        }


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    res = asyncio.get_event_loop().run_until_complete(get_last_minute_kline("btcusdt"))
    print(res)
