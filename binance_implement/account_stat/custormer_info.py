"""
客户余额通知工具。

每天零点读取 mars_normalized.csv 最后一行净值，计算 bearwifi 客户余额后发送钉钉消息。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

try:
    from dingding import send_dingtalk_message
except ImportError:  # pragma: no cover - 支持作为 package 模块导入
    from .dingding import send_dingtalk_message


DEFAULT_NAV_BASE = 1.10640614
DEFAULT_PRINCIPAL = 5000.0
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "mars_normalized.csv"


@dataclass
class Customer:
    """根据账户净值文件计算并发送客户余额。"""

    name: str
    csv_path: Path = DEFAULT_CSV_PATH
    nav_base: float = DEFAULT_NAV_BASE
    principal: float = DEFAULT_PRINCIPAL

    def read_latest_nav(self) -> Tuple[Optional[str], float]:
        """读取 CSV 最后一行的时间戳和净值。"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

        latest_row = None
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    latest_row = row

        if latest_row is None:
            raise ValueError(f"CSV 文件为空: {self.csv_path}")

        try:
            latest_nav = float(latest_row[-1])
        except ValueError as exc:
            raise ValueError(f"CSV 最后一行净值无法解析: {latest_row}") from exc

        timestamp = latest_row[0] if len(latest_row) >= 2 else None
        return timestamp, latest_nav

    def calculate_balance(self, latest_nav: float) -> float:
        """按客户份额计算余额。"""
        return latest_nav / self.nav_base * self.principal

    def build_balance_message(self) -> str:
        """生成钉钉通知内容。"""
        timestamp, latest_nav = self.read_latest_nav()
        balance = self.calculate_balance(latest_nav)
        record_time = timestamp or "unknown"
        notify_time = datetime.now().isoformat(timespec="seconds")

        return (
            f"[OK] 客户余额通知\n"
            f"客户: {self.name}\n"
            f"余额: {balance:.2f}"
        )

    def send_balance_message(self) -> None:
        """发送客户余额钉钉通知。"""
        send_dingtalk_message(self.build_balance_message())


def seconds_until_next_midnight(now: Optional[datetime] = None) -> float:
    """计算距离下一个零点的秒数。"""
    current = now or datetime.now()
    next_midnight = (current + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return max((next_midnight - current).total_seconds(), 0.0)


def run_daily_midnight_notification(customer: Customer) -> None:
    """每天零点发送客户余额通知。"""
    print(f"[SYSTEM] 启动客户余额通知，客户: {customer.name}")
    customer.send_balance_message()

    try:
        while True:
            sleep_seconds = seconds_until_next_midnight()
            print(f"[SYSTEM] 距离下次零点通知还有 {sleep_seconds:.0f} 秒")
            time.sleep(sleep_seconds)

            try:
                customer.send_balance_message()
            except Exception as exc:  # pragma: no cover - 运行时错误继续下一轮
                print(f"[ERROR] 发送客户余额通知失败: {exc}")
            else:
                print(f"[OK] 已发送 {customer.name} 客户余额通知")
    except KeyboardInterrupt:
        print("[SYSTEM] 客户余额通知已停止")


if __name__ == "__main__":
    bearwifi = Customer(name="bearwifi")
    run_daily_midnight_notification(bearwifi)
