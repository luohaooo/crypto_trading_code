"""独立检查 only_in 策略保护单状态的脚本。

默认查询一组预置的止盈/止损订单，并打印每个订单当前在 Binance Algo
订单列表中的状态。运行前需要设置环境变量:

    BINANCE_API_KEY
    BINANCE_API_SECRET
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binance_implement.only_in.config import get_config


DEFAULT_ENTRIES: List[Dict[str, Any]] = [
    {
        "symbol": "TA/USDT:USDT",
        "take_profit_order_id": "4000000871698161",
        "stop_loss_order_id": "4000000871698162",
        "take_profit_client_id": "prot_tp_3472504423",
        "stop_loss_client_id": "prot_sl_3472504449",
    },
    {
        "symbol": "1000XEC/USDT:USDT",
        "take_profit_order_id": "4000000871761629",
        "stop_loss_order_id": "4000000871761631",
        "take_profit_client_id": "prot_tp_3473405075",
        "stop_loss_client_id": "prot_sl_3473405101",
    },
    {
        "symbol": "BSV/USDT:USDT",
        "take_profit_order_id": "4000000871761695",
        "stop_loss_order_id": "4000000871761699",
        "take_profit_client_id": "prot_tp_3473406093",
        "stop_loss_client_id": "prot_sl_3473406114",
    },
    {
        "symbol": "ALPINE/USDT:USDT",
        "take_profit_order_id": "4000000857967230",
        "stop_loss_order_id": "4000000857967239",
        "take_profit_client_id": "prot_tp_3313204287",
        "stop_loss_client_id": "prot_sl_3313204314",
    },
    {
        "symbol": "C/USDT:USDT",
        "take_profit_order_id": "4000000871890724",
        "stop_loss_order_id": "4000000871890736",
        "take_profit_client_id": "prot_tp_3475205172",
        "stop_loss_client_id": "prot_sl_3475205198",
    },
    {
        "symbol": "ASR/USDT:USDT",
        "take_profit_order_id": "4000000857967340",
        "stop_loss_order_id": "4000000857967343",
        "take_profit_client_id": "prot_tp_3313205498",
        "stop_loss_client_id": "prot_sl_3313205524",
    },
    {
        "symbol": "WLD/USDT:USDT",
        "take_profit_order_id": "4000000858401564",
        "stop_loss_order_id": "4000000858401569",
        "take_profit_client_id": "prot_tp_3318604512",
        "stop_loss_client_id": "prot_sl_3318604538",
    },
    {
        "symbol": "GRT/USDT:USDT",
        "take_profit_order_id": "4000000858474831",
        "stop_loss_order_id": "4000000858474832",
        "take_profit_client_id": "prot_tp_3319504423",
        "stop_loss_client_id": "prot_sl_3319504448",
    },
    {
        "symbol": "PUFFER/USDT:USDT",
        "take_profit_order_id": "4000000859439506",
        "stop_loss_order_id": "4000000859439513",
        "take_profit_client_id": "prot_tp_3328504355",
        "stop_loss_client_id": "prot_sl_3328504396",
    },
    {
        "symbol": "TRUMP/USDT:USDT",
        "take_profit_order_id": "4000000860871283",
        "stop_loss_order_id": "4000000860871284",
        "take_profit_client_id": "prot_tp_3343804345",
        "stop_loss_client_id": "prot_sl_3343804373",
    },
    {
        "symbol": "MELANIA/USDT:USDT",
        "take_profit_order_id": "4000000860871463",
        "stop_loss_order_id": "4000000860871470",
        "take_profit_client_id": "prot_tp_3343805399",
        "stop_loss_client_id": "prot_sl_3343805425",
    },
    {
        "symbol": "COW/USDT:USDT",
        "take_profit_order_id": "4000000861689439",
        "stop_loss_order_id": "4000000861689440",
        "take_profit_client_id": "prot_tp_3354604426",
        "stop_loss_client_id": "prot_sl_3354604455",
    },
    {
        "symbol": "F/USDT:USDT",
        "take_profit_order_id": "4000000862212644",
        "stop_loss_order_id": "4000000862212649",
        "take_profit_client_id": "prot_tp_3360905867",
        "stop_loss_client_id": "prot_sl_3360905893",
    },
    {
        "symbol": "ANIME/USDT:USDT",
        "take_profit_order_id": "4000000862212834",
        "stop_loss_order_id": "4000000862212836",
        "take_profit_client_id": "prot_tp_3360907483",
        "stop_loss_client_id": "prot_sl_3360907507",
    },
    {
        "symbol": "TAIKO/USDT:USDT",
        "take_profit_order_id": "4000000862355142",
        "stop_loss_order_id": "4000000862355143",
        "take_profit_client_id": "prot_tp_3361805762",
        "stop_loss_client_id": "prot_sl_3361805784",
    },
    {
        "symbol": "BCH/USDT:USDT",
        "take_profit_order_id": "4000000862470989",
        "stop_loss_order_id": "4000000862470990",
        "take_profit_client_id": "prot_tp_3362704261",
        "stop_loss_client_id": "prot_sl_3362704287",
    },
    {
        "symbol": "KAT/USDT:USDT",
        "take_profit_order_id": "4000000862879547",
        "stop_loss_order_id": "4000000862879551",
        "take_profit_client_id": "prot_tp_3367205650",
        "stop_loss_client_id": "prot_sl_3367205678",
    },
    {
        "symbol": "APR/USDT:USDT",
        "take_profit_order_id": "4000000863864271",
        "stop_loss_order_id": "4000000863864273",
        "take_profit_client_id": "prot_tp_3379804819",
        "stop_loss_client_id": "prot_sl_3379804853",
    },
    {
        "symbol": "TAG/USDT:USDT",
        "take_profit_order_id": "4000000864296956",
        "stop_loss_order_id": "4000000864296962",
        "take_profit_client_id": "prot_tp_3385204376",
        "stop_loss_client_id": "prot_sl_3385204404",
    },
    {
        "symbol": "ETHW/USDT:USDT",
        "take_profit_order_id": "4000000867466859",
        "stop_loss_order_id": "4000000867466860",
        "take_profit_client_id": "prot_tp_3418504491",
        "stop_loss_client_id": "prot_sl_3418504518",
    },
    {
        "symbol": "PHA/USDT:USDT",
        "take_profit_order_id": "4000000867816849",
        "stop_loss_order_id": "4000000867816857",
        "take_profit_client_id": "prot_tp_3422104294",
        "stop_loss_client_id": "prot_sl_3422104319",
    },
    {
        "symbol": "CYBER/USDT:USDT",
        "take_profit_order_id": "4000000869858420",
        "stop_loss_order_id": "4000000869858428",
        "take_profit_client_id": "prot_tp_3447307173",
        "stop_loss_client_id": "prot_sl_3447307197",
    },
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("check_only_in_protective_order_status")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def resolve_status(
    status_lookup: Dict[str, str],
    order_id: Optional[str],
    client_id: Optional[str],
) -> str:
    if order_id and str(order_id) in status_lookup:
        return str(status_lookup[str(order_id)])
    if client_id and str(client_id) in status_lookup:
        return str(status_lookup[str(client_id)])
    return "NOT_FOUND"


def to_symbol_id(symbol: str) -> str:
    normalized = symbol.replace("/", "")
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized


def fetch_current_algo_orders(
    base_url: str,
    api_key: str,
    api_secret: str,
    symbols: List[str],
    logger: logging.Logger,
) -> Dict[str, str]:
    url = f"{base_url}/fapi/v1/allAlgoOrders"
    headers = {"X-MBX-APIKEY": api_key}
    status_lookup: Dict[str, str] = {}
    page_size = 50

    for symbol in symbols:
        symbol_id = to_symbol_id(symbol)
        page = 1
        while True:
            timestamp = int(time.time() * 1000)
            payload = {
                "recvWindow": 5000,
                "timestamp": timestamp,
                "page": page,
                "pageSize": page_size,
                "symbol": symbol_id,
            }
            query = urlencode(payload)
            signature = hmac.new(
                api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params = dict(payload)
            params["signature"] = signature

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            orders: List[Dict[str, Any]] = []
            if isinstance(data, list):
                orders = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                if isinstance(data.get("list"), list):
                    orders = [item for item in data["list"] if isinstance(item, dict)]
                elif isinstance(data.get("orders"), list):
                    orders = [item for item in data["orders"] if isinstance(item, dict)]
                elif any(key in data for key in ["algoId", "clientAlgoId", "orderId"]):
                    orders = [data]

            for order in orders:
                status_text = str(
                    order.get("algoStatus")
                    or order.get("status")
                    or order.get("orderStatus")
                    or ""
                )
                for key in ["algoId", "clientAlgoId", "orderId", "clientOrderId", "id"]:
                    value = order.get(key)
                    if value:
                        status_lookup[str(value)] = status_text

            logger.info(
                "[CHECK] symbol=%s page=%s fetched=%s",
                symbol,
                page,
                len(orders),
            )
            if not orders or len(orders) < page_size:
                break

            page += 1
            time.sleep(0.05)

    return status_lookup


def main() -> int:
    logger = setup_logger()
    config = get_config()
    symbols = [entry["symbol"] for entry in DEFAULT_ENTRIES]
    try:
        status_lookup = fetch_current_algo_orders(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            symbols=symbols,
            logger=logger,
        )
    except Exception as exc:
        logger.error("无法获取 Algo 订单状态: %s", exc)
        return 1

    summary: Dict[str, int] = {}
    for entry in DEFAULT_ENTRIES:
        symbol = entry["symbol"]
        tp_status = resolve_status(
            status_lookup,
            entry.get("take_profit_order_id"),
            entry.get("take_profit_client_id"),
        )
        sl_status = resolve_status(
            status_lookup,
            entry.get("stop_loss_order_id"),
            entry.get("stop_loss_client_id"),
        )
        summary[tp_status] = summary.get(tp_status, 0) + 1
        summary[sl_status] = summary.get(sl_status, 0) + 1

        print(
            f"{symbol}\n"
            f"  TP {entry['take_profit_order_id']} ({entry['take_profit_client_id']}): {tp_status}\n"
            f"  SL {entry['stop_loss_order_id']} ({entry['stop_loss_client_id']}): {sl_status}"
        )

    print("\nSummary:")
    for status in sorted(summary):
        print(f"  {status}: {summary[status]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
