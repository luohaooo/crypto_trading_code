"""
账户资产统计工具

提供查询币安U本位永续合约账户资产（余额 + 当前订单保证金）的方法。
"""

from __future__ import annotations

import os
from typing import Dict, List

import ccxt


def _build_exchange(
    api_key: str,
    api_secret: str,
    *,
    use_testnet: bool,
) -> ccxt.Exchange:
    """创建期货环境下的Binance交易所实例。"""
    exchange_config = {
        "apiKey": api_key,
        "secret": api_secret,
        "timeout": 10000,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "defaultMarket": "future",
        },
    }

    exchange_ids: List[str] = ["binanceusdm", "binance"]
    if use_testnet:
        exchange_ids = ["binance"]  # 仅binance支持期货测试网

    last_error: Exception | None = None
    for exchange_id in exchange_ids:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            continue
        try:
            exchange = exchange_class(exchange_config)
            if use_testnet:
                exchange.set_sandbox_mode(True)
                exchange.urls["api"]["public"] = (
                    "https://testnet.binancefuture.com/fapi/v1"
                )
                exchange.urls["api"]["private"] = (
                    "https://testnet.binancefuture.com/fapi/v1"
                )
            return exchange
        except Exception as err:
            last_error = err

    if last_error:
        raise last_error
    raise RuntimeError("无法初始化币安交易所客户端")


def get_usdt_futures_asset_total(
    api_key: str,
    api_secret: str,
    *,
    use_testnet: bool = False,
) -> Dict[str, float]:
    """
    获取币安U本位合约账户资产信息。

    Args:
        api_key: 币安账户 API Key。
        api_secret: 币安账户 API Secret。
        use_testnet: 是否使用测试网。

    Returns:
        包含钱包余额、订单保证金和资产总额的字典。
    """
    resolved_key = os.getenv(api_key, api_key)
    resolved_secret = os.getenv(api_secret, api_secret)

    if not resolved_key or not resolved_secret:
        raise ValueError("API Key 和 Secret 不能为空")

    exchange = _build_exchange(
        api_key=resolved_key,
        api_secret=resolved_secret,
        use_testnet=use_testnet,
    )

    try:
        balance = exchange.fetch_balance()
    except Exception as exc:  # pragma: no cover - 直接抛给调用者
        raise RuntimeError("无法从币安获取账户信息") from exc

    usdt_balance = balance.get("USDT", {})
    wallet_balance = float(usdt_balance.get("total", 0.0))
    open_order_margin = float(usdt_balance.get("used", 0.0))

    return {
        "wallet_balance": wallet_balance,
        # "open_order_margin": open_order_margin,
        # "total_assets": wallet_balance + open_order_margin,
    }

def get_all_accounts_total_balance(
    api_key_list: List[str],
    api_secret_list: List[str],
) -> float:
    """
    获取多个币安U本位合约账户的总资产余额。

    Args:
        api_key_list: 币安账户 API Key 列表。
        api_secret_list: 币安账户 API Secret 列表。
    Returns:
        所有账户的总余额。
    """
    total_balance = 0.0
    for api_key, api_secret in zip(api_key_list, api_secret_list):
        balance_info = get_usdt_futures_asset_total(api_key, api_secret)
        total_balance += balance_info['wallet_balance']
    return total_balance


if __name__ == "__main__":
    # 示例用法
    import os
    api_key_list = [
        # 'BINANCE_API_KEY_UESTC',
        # 'BINANCE_API_KEY_CD',
        # 'BINANCE_API_KEY_HK',
        # 'BINANCE_API_KEY_SH',
        'BINANCE_API_KEY_PUBLIC'
    ]
    api_secret_list = [
        # 'BINANCE_API_SECRET_UESTC',
        # 'BINANCE_API_SECRET_CD',
        # 'BINANCE_API_SECRET_HK',
        # 'BINANCE_API_SECRET_SH',
        'BINANCE_API_SECRET_PUBLIC'
    ]

    total_balance = 0.0
    for key, secret in zip(api_key_list, api_secret_list):
        try:
            balance = get_usdt_futures_asset_total(key, secret)
            total_balance += balance['wallet_balance']
            print(f"API: {key}, 余额: {balance['wallet_balance']:.2f} USDT")
        except Exception as e:
            print(f"获取账户资产失败，API Key 环境变量: {key}，错误: {e}")

    print(f"总余额: {total_balance:.2f} USDT")
