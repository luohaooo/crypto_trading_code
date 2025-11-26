"""
监控工具模块
提供交易系统的性能监控和统计功能

功能：
- 交易周期监控
- 成功率统计
- 性能指标记录
- 系统状态监控
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CycleStats:
    """交易周期统计"""
    cycle_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    balance_before: float = 0.0
    balance_after: float = 0.0
    positions_opened: int = 0
    positions_closed: int = 0
    error_message: Optional[str] = None


class TradingMonitor:
    """交易监控器"""

    def __init__(self, logger):
        """
        初始化监控器

        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.start_time = datetime.now()
        self.cycle_stats: List[CycleStats] = []
        self.success_count = 0
        self.failure_count = 0
        self.total_cycles = 0
        self.cycle_times: List[float] = []

    def record_success(self):
        """记录成功的交易周期"""
        self.success_count += 1
        self.total_cycles += 1

    def record_failure(self):
        """记录失败的交易周期"""
        self.failure_count += 1
        self.total_cycles += 1

    def record_cycle_time(self, duration: float):
        """记录周期执行时间"""
        self.cycle_times.append(duration)

        # 只保留最近100次的记录
        if len(self.cycle_times) > 100:
            self.cycle_times = self.cycle_times[-100:]

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_cycles == 0:
            return 0.0
        return self.success_count / self.total_cycles * 100

    def get_average_cycle_time(self) -> float:
        """获取平均周期时间"""
        if not self.cycle_times:
            return 0.0
        return sum(self.cycle_times) / len(self.cycle_times)

    def get_runtime(self) -> timedelta:
        """获取总运行时间"""
        return datetime.now() - self.start_time

    def print_stats(self):
        """打印当前统计信息"""
        runtime = self.get_runtime()
        success_rate = self.get_success_rate()
        avg_cycle_time = self.get_average_cycle_time()

        self.logger.info("=" * 50)
        self.logger.info("[STATS] 交易系统统计")
        self.logger.info("=" * 50)
        self.logger.info(f"[TIME] 运行时间: {runtime}")
        self.logger.info(f"[CYCLES] 总周期数: {self.total_cycles}")
        self.logger.info(f"[SUCCESS] 成功周期: {self.success_count}")
        self.logger.info(f"[FAILED] 失败周期: {self.failure_count}")
        self.logger.info(f"[RATE] 成功率: {success_rate:.1f}%")
        self.logger.info(f"[TIME] 平均周期时间: {avg_cycle_time:.1f}秒")
        self.logger.info("=" * 50)

    def print_final_stats(self):
        """打印最终统计信息"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[SUMMARY] 交易系统运行总结")
        self.logger.info("=" * 60)

        runtime = self.get_runtime()
        success_rate = self.get_success_rate()
        avg_cycle_time = self.get_average_cycle_time()

        self.logger.info(f"[START] 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[END] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[RUNTIME] 总运行时间: {runtime}")
        self.logger.info(f"[TOTAL] 总执行周期: {self.total_cycles}")
        self.logger.info(f"[SUCCESS] 成功周期: {self.success_count}")
        self.logger.info(f"[FAILED] 失败周期: {self.failure_count}")
        self.logger.info(f"[OVERALL] 整体成功率: {success_rate:.1f}%")

        if self.cycle_times:
            self.logger.info(f"[TIME] 平均周期时间: {avg_cycle_time:.1f}秒")
            self.logger.info(f"[FASTEST] 最快周期: {min(self.cycle_times):.1f}秒")
            self.logger.info(f"[SLOWEST] 最慢周期: {max(self.cycle_times):.1f}秒")

        # 性能评估
        if success_rate >= 90:
            performance = "[EXCELLENT] 优秀"
        elif success_rate >= 70:
            performance = "[GOOD] 良好"
        elif success_rate >= 50:
            performance = "[AVERAGE] 一般"
        else:
            performance = "[NEEDS_IMPROVEMENT] 需要改进"

        self.logger.info(f"[PERFORMANCE] 系统性能: {performance}")
        self.logger.info("=" * 60)

    def get_monitoring_data(self) -> Dict:
        """获取监控数据"""
        runtime = self.get_runtime()
        success_rate = self.get_success_rate()
        avg_cycle_time = self.get_average_cycle_time()

        return {
            'start_time': self.start_time.isoformat(),
            'runtime_seconds': runtime.total_seconds(),
            'total_cycles': self.total_cycles,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'average_cycle_time': avg_cycle_time,
            'min_cycle_time': min(self.cycle_times) if self.cycle_times else 0,
            'max_cycle_time': max(self.cycle_times) if self.cycle_times else 0
        }