#!/usr/bin/env python3
"""
Fetch funding info from Binance USDⓈ-Margined Futures REST API.

Reference:
https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price
https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-Info
"""

import argparse
from typing import Any, Dict, List, Optional, Union

import requests


FundingResponse = Union[Dict[str, Any], List[Dict[str, Any]]]


def fetch_mark_price(
    base_url: str,
    symbol: Optional[str],
    timeout: int,
) -> FundingResponse:
    """
    Fetch mark price info (includes current funding rate) from USDⓈ-Margined Futures.

    Args:
        base_url: API base URL.
        symbol: Optional symbol like BTCUSD_PERP.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.
    """
    endpoint = f"{base_url}/fapi/v1/premiumIndex"
    params: Dict[str, str] = {}
    if symbol:
        params["symbol"] = symbol
    response = requests.get(endpoint, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_funding_rate_info(
    base_url: str,
    timeout: int,
) -> FundingResponse:
    """
    Fetch funding rate info for symbols with adjusted funding intervals/caps.
    """
    endpoint = f"{base_url}/fapi/v1/fundingInfo"
    response = requests.get(endpoint, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture funding info.")
    parser.add_argument(
        "--symbol",
        default="ACEUSDT",
        help="Symbol like BTCUSDT; omit to fetch all symbols.",
    )
    parser.add_argument(
        "--base-url",
        default="https://fapi.binance.com",
        help="Binance USDⓈ-Margined Futures base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    try:
        mark_payload = fetch_mark_price(args.base_url, args.symbol, args.timeout)
        info_payload = fetch_funding_rate_info(args.base_url, args.timeout)
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        return 1

    if isinstance(mark_payload, list):
        print(f"[OK] Retrieved {len(mark_payload)} mark price records.")
        if not mark_payload:
            print("[ERROR] Empty mark price records.")
            return 1
        if args.symbol:
            current = next(
                (item for item in mark_payload if item.get("symbol") == args.symbol),
                None,
            )
            if current is None:
                print("[ERROR] Symbol not found in mark price records.")
                return 1
        else:
            current = mark_payload[0]
    else:
        current = mark_payload

    funding_interval = "unknown"
    if isinstance(info_payload, list):
        for item in info_payload:
            if item.get("symbol") == args.symbol:
                funding_interval = item.get("fundingIntervalHours", "unknown")
                break

    print(
        {
            "symbol": current.get("symbol"),
            "fundingTime": current.get("nextFundingTime"),
            "fundingRate": current.get("lastFundingRate"),
            "markPrice": current.get("markPrice"),
            "fundingIntervalHours": funding_interval,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
