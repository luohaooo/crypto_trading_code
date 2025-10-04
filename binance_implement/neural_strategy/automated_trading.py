"""
自动化交易策略主脚本
基于神经网络OHLC因子的量化交易系统

功能：
- 支持每日定时执行 (默认20:00) 或固定间隔执行模式
- 支持testnet和实盘环境
- 基于神经网络预测的因子选币
- 自动平仓、开仓和风险控制
- 钉钉通知集成
"""

import sys
import os
import time
import asyncio
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import warnings

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# 导入配置和工具
from config import get_config, TradingConfig
from trading_executor import TradingExecutor
from optimized_data_processor import OptimizedDataProcessor
from optimized_factor_calculator import OptimizedFactorCalculator
from trading_utils.logger import setup_logger
from trading_utils.monitor import TradingMonitor

class AutomatedTradingSystem:
    """自动化交易系统主类"""

    def __init__(self, use_testnet: bool = True):
        """
        初始化交易系统

        Args:
            use_testnet: True为testnet，False为实盘
        """
        self.use_testnet = use_testnet
        self.config = get_config(use_testnet)

        # 初始化组件
        self.logger = setup_logger(self.config.LOG_FILE, self.config.LOG_LEVEL)
        self.monitor = TradingMonitor(self.logger)
        self.executor = TradingExecutor(self.config, self.logger)
        self.data_processor = OptimizedDataProcessor(self.config, self.logger)
        self.factor_calculator = OptimizedFactorCalculator(self.config, self.logger)

        # 交易状态
        self.cycle_count = 0
        self.last_execution_time = None
        self.current_positions = {}
        self.current_balance = 0.0

        self.is_16h_open = False  # 标记是否刚执行完16小时的开仓

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        os._exit(0)

    async def start(self):
        """启动自动化交易系统"""
        self.logger.info("="*60)
        self.logger.info("[SYSTEM] 启动自动化交易系统")
        self.logger.info("="*60)

        # 打印配置信息
        self.config.print_config()

        # 验证配置
        if not self.config.validate_config():
            self.logger.error("[ERROR] 配置验证失败，无法启动")
            return False

        # 初始化组件
        try:
            await self._initialize_components()
        except Exception as e:
            self.logger.error(f"[ERROR] 组件初始化失败: {e}")
            return False

        # 启动主循环
        if self.config.EXECUTION_MODE == 'daily':
            next_execution = self._get_next_execution_time()
            self.logger.info(f"[OK] 系统初始化完成，每日 {self.config.EXECUTION_HOUR:02d}:00 执行交易")
            self.logger.info(f"[SCHEDULE] 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
        elif self.config.EXECUTION_MODE == '16h':
            next_execution = self._get_next_16h_execution_time()
            self.logger.info(f"[OK] 系统初始化完成，16小时周期执行 (0点、8点、16点)")
            self.logger.info(f"[SCHEDULE] 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            self.logger.info(f"[OK] 系统初始化完成，每 {self.config.REBALANCE_INTERVAL//60} 分钟执行交易")

        try:
            await self._main_trading_loop()
        except Exception as e:
            self.logger.error(f"[ERROR] 交易循环异常: {e}")

        return True

    async def _initialize_components(self):
        """初始化各个组件"""
        self.logger.info("[INIT] 初始化交易组件...")

        # 初始化交易执行器 (现在是同步方法)
        self.executor.initialize()

        # 初始化数据处理器
        self.data_processor.initialize()

        # 初始化因子计算器
        self.factor_calculator.initialize()

        # 验证连接 (现在是同步方法)
        balance = self.executor.get_account_balance()
        if balance is None:
            raise Exception("无法获取账户余额，请检查API配置")

        self.current_balance = balance
        self.logger.info(f"[BALANCE] 当前账户余额: {balance:.2f} USDT")

        # 发送余额通知
        self.executor.send_balance_notification()

    async def _main_trading_loop(self):
        """主交易循环"""
        self.logger.info("[LOOP] 开始主交易循环")

        # 如果是每日执行模式，先等待到执行时间
        if self.config.EXECUTION_MODE == 'daily':
            next_execution = self._get_next_execution_time()
            now = datetime.now()

            # 如果下次执行时间不是现在，先等待
            if (next_execution - now).total_seconds() > 60:  # 超过1分钟才等待
                self.logger.info(f"[DAILY] 等待首次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
                await self._wait_for_daily_execution()
        elif self.config.EXECUTION_MODE == '16h':
            next_execution = self._get_next_16h_execution_time()
            now = datetime.now()

            # 如果下次执行时间不是现在，先等待
            if (next_execution - now).total_seconds() > 60:  # 超过1分钟才等待
                self.logger.info(f"[16H] 等待首次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
                await self._wait_for_16h_execution()

        while True:
            try:
                cycle_start_time = datetime.now()
                self.cycle_count += 1

                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"[CYCLE] 第 {self.cycle_count} 轮交易周期")
                self.logger.info(f"[TIME] 开始时间: {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*50}")

                # 执行一轮完整的交易流程
                success = await self._execute_trading_cycle()

                if success:
                    self.logger.info("[OK] 交易周期执行成功")
                    self.monitor.record_success()
                else:
                    self.logger.warning("[WARNING] 交易周期执行失败")
                    self.monitor.record_failure()
                
                self.is_16h_open = True

                # 记录执行时间
                cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                self.last_execution_time = cycle_start_time
                self.monitor.record_cycle_time(cycle_duration)

                self.logger.info(f"[TIME] 周期耗时: {cycle_duration:.1f}秒")

                # 等待下一个周期
                await self._wait_for_next_cycle()

            except Exception as e:
                self.logger.error(f"[ERROR] 交易周期异常: {e}")
                self.monitor.record_failure()

                # 异常后等待一段时间再继续
                self.logger.info("[WAIT] 异常后等待30秒...")
                await asyncio.sleep(30)

    async def _execute_trading_cycle(self) -> bool:
        """
        执行一轮完整的交易周期

        Returns:
            bool: 执行是否成功
        """
        try:
            # 1. 平仓字典中存储的仓位
            self.logger.info("[STEP 1] 平仓字典中存储的仓位...")
            if self.current_positions:
                self.logger.info(f"[POSITIONS] 当前持有仓位: {self.current_positions}")
                close_success = self.executor.close_specific_positions(self.current_positions)
                if not close_success:
                    self.logger.error("[ERROR] 平仓失败")
                    return False
                # 清空仓位字典
                self.current_positions.clear()
                self.logger.info("[INFO] 仓位字典已清空")
            else:
                self.logger.info("[INFO] 仓位字典为空，无需平仓")

            # 2. 跳过验证步骤
            self.logger.info("[STEP 2] 跳过验证步骤...")

            # 3. 获取当前保证金余额 (现在是同步方法)
            self.logger.info("[STEP 3] 获取当前合约保证金...")
            balance = self.executor.get_account_balance()
            if balance is None:
                self.logger.error("[ERROR] 无法获取账户余额")
                return False

            self.current_balance = balance
            self.logger.info(f"[BALANCE] 当前保证金余额: {balance:.2f} USDT")

            # 发送交易前余额通知
            self.executor.send_balance_notification()

            # 4. 数据处理和因子计算
            self.logger.info("[STEP 4] 提取OHLC数据并计算因子...")
            factors = self._calculate_factors()
            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 因子计算失败或无有效因子")
                return False

            self.logger.info(f"[FACTORS] 计算得到 {len(factors)} 个有效因子")

            # 5. 选币策略
            self.logger.info("[STEP 5] 根据因子值选择交易标的...")
            long_symbols, short_symbols = self._select_trading_symbols(factors)

            if len(long_symbols) == 0 and len(short_symbols) == 0:
                self.logger.warning("[WARNING] 没有符合条件的交易标的")
                return True  # 不算失败，只是这轮不交易

            self.logger.info(f"[SELECTION] 选择交易标的:")
            self.logger.info(f"   做多: {long_symbols}")
            self.logger.info(f"   做空: {short_symbols}")

            # 6. 开仓交易 (现在是同步方法)
            self.logger.info("[STEP 6] 执行开仓交易...")
            position_success, opened_positions = self.executor.open_positions(
                long_symbols, short_symbols, balance
            )

            if not position_success:
                self.logger.error("[ERROR] 开仓失败")
                return False

            # 将新开仓位添加到字典中
            if opened_positions:
                self.current_positions.update(opened_positions)
                self.logger.info(f"[POSITIONS] 新仓位已添加到字典: {opened_positions}")
                self.logger.info(f"[POSITIONS] 当前仓位字典: {self.current_positions}")

            # 7. 验证开仓完成 (暂时简化处理)
            self.logger.info("[STEP 7] 开仓交易完成")
            self.logger.info("[INFO] 跳过详细仓位验证，交易周期完成")

            return True

        except Exception as e:
            self.logger.error(f"[ERROR] 交易周期执行异常: {e}")
            return False

    def _calculate_factors(self) -> Optional[pd.Series]:
        """计算交易因子"""
        try:
            # 使用优化的因子计算器直接计算因子
            factors = self.factor_calculator.calculate_factors(self.data_processor)

            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 因子计算失败")
                return None

            return factors

        except Exception as e:
            self.logger.error(f"[ERROR] 因子计算异常: {e}")
            return None

    def _select_trading_symbols(self, factors: pd.Series) -> tuple[List[str], List[str]]:
        """
        根据因子值选择交易标的

        Args:
            factors: 因子值序列

        Returns:
            tuple: (做多标的列表, 做空标的列表)
        """
        if len(factors) == 0:
            return [], []

        # 排序因子值
        sorted_factors = factors.sort_values(ascending=False)

        # 选择前N名做多，后N名做空
        top_n_long = min(self.config.TOP_N_LONG, len(sorted_factors))
        top_n_short = min(self.config.TOP_N_SHORT, len(sorted_factors))

        long_symbols = sorted_factors.head(top_n_long).index.tolist()
        short_symbols = sorted_factors.tail(top_n_short).index.tolist()

        # 记录因子信息
        self.logger.info(f"[STATS] 因子统计:")
        self.logger.info(f"   总计: {len(factors)} 个标的")
        self.logger.info(f"   因子值范围: {factors.min():.6f} ~ {factors.max():.6f}")
        self.logger.info(f"   选择做多: {len(long_symbols)} 个")
        self.logger.info(f"   选择做空: {len(short_symbols)} 个")

        # 打印详细的做多标的及其因子值
        if len(long_symbols) > 0:
            self.logger.info("[LONG SYMBOLS] 做多标的详情:")
            for symbol in long_symbols:
                factor_value = sorted_factors[symbol]
                self.logger.info(f"   {symbol}: {factor_value:.6f}")

        # 打印详细的做空标的及其因子值
        if len(short_symbols) > 0:
            self.logger.info("[SHORT SYMBOLS] 做空标的详情:")
            for symbol in short_symbols:
                factor_value = sorted_factors[symbol]
                self.logger.info(f"   {symbol}: {factor_value:.6f}")

        return long_symbols, short_symbols

    def _get_next_execution_time(self) -> datetime:
        """获取下次执行时间"""
        now = datetime.now()
        target_hour = self.config.EXECUTION_HOUR
        today_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        if now >= today_target:
            return today_target + timedelta(days=1)
        else:
            return today_target

    def _get_next_16h_execution_time(self) -> datetime:
        """获取下次16小时模式执行时间 (0点、8点、16点)"""
        now = datetime.now()

        # 16小时模式的执行时间点
        execution_hours = [0, 8, 16]

        # 获取当前日期，时分秒设为0
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.is_16h_open:
            base = now.replace(minute=0, second=0, microsecond=0)
            return base + timedelta(hours=16)
        else:
            # 计算今天所有的执行时间点
            today_targets = []
            for hour in execution_hours:
                target = today.replace(hour=hour)
                today_targets.append(target)

            # 寻找下一个执行时间
            for target in today_targets:
                if now < target:
                    return target

            # 如果今天的执行时间都已过，返回明天的第一个执行时间 (0点)
            tomorrow_first = today + timedelta(days=1)
            return tomorrow_first.replace(hour=0)

    async def _wait_for_next_cycle(self):
        """等待下一个交易周期"""
        if self.config.EXECUTION_MODE == 'daily':
            await self._wait_for_daily_execution()
        elif self.config.EXECUTION_MODE == '16h':
            await self._wait_for_16h_execution()
        else:
            await self._wait_for_interval_execution()

    async def _wait_for_daily_execution(self):
        """等待每日执行时间 (20:00)"""
        now = datetime.now()
        target_hour = self.config.EXECUTION_HOUR

        # 计算下次执行时间
        today_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        if now >= today_target:
            # 如果今天的执行时间已过，等到明天的执行时间
            next_execution = today_target + timedelta(days=1)
        else:
            # 如果今天的执行时间未到，等到今天的执行时间
            next_execution = today_target

        wait_seconds = (next_execution - now).total_seconds()

        self.logger.info(f"[DAILY] 每日执行模式 - 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[WAIT] 等待 {wait_seconds:.0f} 秒 ({wait_seconds/3600:.1f} 小时)")

        # 分批等待，每小时打印一次状态
        while wait_seconds > 0:
            if wait_seconds > 3600:
                # 等待1小时
                await asyncio.sleep(3600)
                wait_seconds -= 3600
                remaining_hours = wait_seconds / 3600
                self.logger.info(f"[WAIT] 剩余等待时间: {remaining_hours:.1f} 小时")
            else:
                # 最后的等待时间
                self.logger.info(f"[WAIT] 最后等待: {wait_seconds:.0f} 秒")
                await asyncio.sleep(wait_seconds)
                break

    async def _wait_for_16h_execution(self):
        """等待16小时模式执行时间 (0点、8点、16点)"""
        now = datetime.now()

        # 计算下次执行时间
        next_execution = self._get_next_16h_execution_time()
        wait_seconds = (next_execution - now).total_seconds()

        self.logger.info(f"[16H] 16小时执行模式 - 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[WAIT] 等待 {wait_seconds:.0f} 秒 ({wait_seconds/3600:.1f} 小时)")

        # 分批等待，每小时打印一次状态
        while wait_seconds > 0:
            if wait_seconds > 3600:
                # 等待1小时
                await asyncio.sleep(3600)
                wait_seconds -= 3600
                remaining_hours = wait_seconds / 3600
                self.logger.info(f"[WAIT] 剩余等待时间: {remaining_hours:.1f} 小时")
            else:
                # 最后的等待时间
                self.logger.info(f"[WAIT] 最后等待: {wait_seconds:.0f} 秒")
                await asyncio.sleep(wait_seconds)
                break

    async def _wait_for_interval_execution(self):
        """等待固定间隔执行"""
        wait_seconds = self.config.REBALANCE_INTERVAL

        self.logger.info(f"[INTERVAL] 间隔执行模式 - 等待 {wait_seconds} 秒 ({wait_seconds//60} 分钟)")
        await asyncio.sleep(wait_seconds)


def main():
    """主函数"""
    print("[SYSTEM] 自动化交易系统")
    print("=" * 40)

    # 环境选择
    while True:
        env_choice = input("选择交易环境 (1: Testnet, 2: 实盘): ").strip()
        if env_choice == '1':
            use_testnet = True
            env_name = "Testnet"
            break
        elif env_choice == '2':
            use_testnet = False
            env_name = "实盘"
            # 实盘确认
            confirm = input("[WARNING] 确认使用实盘环境？这将使用真实资金 (yes/no): ").strip().lower()
            if confirm == 'yes':
                break
            else:
                print("已取消实盘操作")
                continue
        else:
            print("[ERROR] 无效选择，请输入 1 或 2")
            continue

    print(f"[OK] 选择环境: {env_name}")
    print("[SYSTEM] 启动交易系统...")
    print()

    # 创建和启动交易系统
    trading_system = AutomatedTradingSystem(use_testnet=use_testnet)

    try:
        # 运行主程序
        asyncio.run(trading_system.start())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户中断")
    except Exception as e:
        print(f"\n[ERROR] 系统异常: {e}")
    finally:
        print("[EXIT] 程序结束")


if __name__ == "__main__":
    main()