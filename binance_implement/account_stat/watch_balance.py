"""
账户余额监控工具

每 10 分钟查询所有账户的总余额并进行记录。
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from asset_stat import get_all_accounts_total_balance


DEFAULT_API_KEY_ENV_NAMES: List[str] = [
    "BINANCE_API_KEY_UESTC",
    "BINANCE_API_KEY_CD",
    "BINANCE_API_KEY_HK",
    "BINANCE_API_KEY_SH",
]

DEFAULT_API_SECRET_ENV_NAMES: List[str] = [
    "BINANCE_API_SECRET_UESTC",
    "BINANCE_API_SECRET_CD",
    "BINANCE_API_SECRET_HK",
    "BINANCE_API_SECRET_SH",
]


def _parse_env_list(value: str | None) -> List[str]:
    """将逗号分隔的字符串解析为列表。"""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class BalanceWatcherConfig:
    """余额监控配置。"""

    api_key_env_names: List[str]
    api_secret_env_names: List[str]
    csv_path: Path
    interval_minutes: int = 1

    @property
    def interval_seconds(self) -> int:
        return self.interval_minutes * 60

    @classmethod
    def from_environment(cls) -> "BalanceWatcherConfig":
        """从环境变量创建配置。"""
        api_keys = _parse_env_list(os.getenv("BALANCE_WATCH_API_KEYS"))
        api_secrets = _parse_env_list(os.getenv("BALANCE_WATCH_API_SECRETS"))

        if not api_keys:
            api_keys = DEFAULT_API_KEY_ENV_NAMES
        if not api_secrets:
            api_secrets = DEFAULT_API_SECRET_ENV_NAMES

        if len(api_keys) != len(api_secrets):
            raise ValueError("API Key 与 Secret 数量不一致")

        output_dir = Path(__file__).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_filename = "mars.csv"
        interval_minutes = 1

        return cls(
            api_key_env_names=api_keys,
            api_secret_env_names=api_secrets,
            csv_path=output_dir / csv_filename,
            interval_minutes=interval_minutes,
        )


def append_balance_record(csv_path: Path, timestamp: datetime, balance: float) -> None:
    """将余额记录附加到 CSV 文件中。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "total_balance"])
        writer.writerow([timestamp.isoformat(), f"{balance:.8f}"])


def monitor_balances(config: BalanceWatcherConfig) -> None:
    """按照配置的时间间隔监控并记录账户余额。"""
    print(f"[SYSTEM] 启动账户余额监控，间隔: {config.interval_minutes} 分钟")

    try:
        while True:
            timestamp = datetime.now()
            try:
                total_balance = get_all_accounts_total_balance(
                    config.api_key_env_names, config.api_secret_env_names
                )
            except Exception as exc:  # pragma: no cover - 网络错误直接告警
                print(f"[ERROR] {timestamp.isoformat()} 获取账户余额失败: {exc}")
            else:
                append_balance_record(config.csv_path, timestamp, total_balance)
                print(
                    f"[OK] {timestamp.isoformat(timespec='seconds')} 记录余额 {total_balance:.2f} USDT -> {config.csv_path}"
                )
            time.sleep(config.interval_seconds)
    except KeyboardInterrupt:
        print("[SYSTEM] 余额监控已停止")


if __name__ == "__main__":
    watcher_config = BalanceWatcherConfig.from_environment()
    monitor_balances(watcher_config)
