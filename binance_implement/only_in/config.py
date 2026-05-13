"""配置文件 - find_in_out 策略"""

import os
from typing import Any, Dict


class TradingConfig:
    """交易策略配置"""

    def __init__(self):
        # API 配置（仅实盘）
        self.API_KEY = os.environ["BINANCE_API_KEY_PUBLIC"]
        self.API_SECRET = os.environ["BINANCE_API_SECRET_PUBLIC"]
        self.BASE_URL = "https://fapi.binance.com"
        self.ENV_NAME = "LIVE"

        # 日志配置
        self.LOG_LEVEL = "INFO"
        base_dir = os.path.dirname(__file__)
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.LOG_FILE = os.path.join(log_dir, "craz_0315.log")
        self.PAUSE_OPEN_ORDERS_FLAG = os.path.join(base_dir, "pause_open_orders.flag")

        # 交易参数
        self.LEVERAGE = 5

        self.SINGLE_MARGIN = None
        self.ENABLE_MARGIN_TYPE_SETTING = True
        self.MARGIN_TYPE_RETRY_COUNT = 3
        self.MARGIN_TYPE = 'CROSSED'

        self.PROTECTIVE_WORKING_TYPE = "MARK_PRICE"

    def get_ccxt_config(self) -> Dict[str, Any]:
        """获取 CCXT 交易所配置"""
        return {
            "apiKey": self.API_KEY,
            "secret": self.API_SECRET,
            "timeout": 10000,
            "enableRateLimit": False,
            "options": {"defaultType": "future"},
        }


def get_config() -> TradingConfig:
    """获取配置实例"""
    return TradingConfig()
