"""
日志工具模块
提供统一的日志记录功能

功能：
- 控制台和文件日志输出
- 不同级别的日志记录
- 时间戳和格式化
"""

import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(log_file: str, log_level: str = 'INFO') -> logging.Logger:
    """
    设置日志记录器

    Args:
        log_file: 日志文件名
        log_level: 日志级别

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger('AutoTrading')
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"[LOG] 日志文件: {log_file}")
    except Exception as e:
        logger.warning(f"[WARNING] 无法创建日志文件 {log_file}: {e}")

    # 防止重复日志
    logger.propagate = False

    return logger


class TradingLogger:
    """交易专用日志记录器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def trade_start(self, cycle: int):
        """记录交易周期开始"""
        self.logger.info(f"[START] ========== 第 {cycle} 轮交易开始 ==========")

    def trade_end(self, cycle: int, success: bool, duration: float):
        """记录交易周期结束"""
        status = "成功" if success else "失败"
        self.logger.info(f"[END] ========== 第 {cycle} 轮交易{status} (耗时: {duration:.1f}秒) ==========")

    def position_action(self, action: str, symbol: str, details: str = ""):
        """记录仓位操作"""
        actions_map = {
            'open_long': '[LONG] 开多',
            'open_short': '[SHORT] 开空',
            'close': '[CLOSE] 平仓'
        }
        action_str = actions_map.get(action, action)
        self.logger.info(f"{action_str} {symbol} {details}")

    def factor_info(self, total_symbols: int, valid_factors: int, factor_range: tuple):
        """记录因子信息"""
        self.logger.info(f"[FACTOR] 因子计算: {valid_factors}/{total_symbols} 个有效, 范围: [{factor_range[0]:.6f}, {factor_range[1]:.6f}]")

    def balance_info(self, balance: float, change: Optional[float] = None):
        """记录余额信息"""
        if change is not None:
            change_str = f" ({change:+.2f})" if change != 0 else ""
            self.logger.info(f"[BALANCE] 账户余额: {balance:.2f} USDT{change_str}")
        else:
            self.logger.info(f"[BALANCE] 账户余额: {balance:.2f} USDT")

    def error(self, message: str):
        """记录错误"""
        self.logger.error(f"[ERROR] {message}")

    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(f"[WARNING] {message}")

    def success(self, message: str):
        """记录成功"""
        self.logger.info(f"[OK] {message}")

    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)