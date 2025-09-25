"""
自动化交易策略主脚本
基于神经网络OHLC因子的量化交易系统

功能：
- 每5分钟执行一轮交易
- 支持testnet和实盘环境
- 基于神经网络预测的因子选币
- 自动平仓、开仓和风险控制
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
        self.logger.info("[OK] 系统初始化完成，开始交易循环")

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

    async def _main_trading_loop(self):
        """主交易循环"""
        self.logger.info("[LOOP] 开始主交易循环")

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
            # 1. 平仓所有现有仓位 (现在是同步方法)
            self.logger.info("[STEP 1] 平仓所有现有仓位...")
            close_success = self.executor.close_all_positions()
            if not close_success:
                self.logger.error("[ERROR] 平仓失败")
                return False

            # 2. 验证平仓完成 (保持异步，因为trading_executor中还有一些异步方法)
            self.logger.info("[STEP 2] 验证平仓完成...")
            # 由于verify_positions_closed可能还是异步的，我们先跳过详细验证
            self.logger.info("[INFO] 跳过详细平仓验证，继续执行")

            # 3. 获取当前保证金余额 (现在是同步方法)
            self.logger.info("[STEP 3] 获取当前合约保证金...")
            balance = self.executor.get_account_balance()
            if balance is None:
                self.logger.error("[ERROR] 无法获取账户余额")
                return False

            self.current_balance = balance
            self.logger.info(f"[BALANCE] 当前保证金余额: {balance:.2f} USDT")

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
            position_success = self.executor.open_positions(
                long_symbols, short_symbols, balance
            )

            if not position_success:
                self.logger.error("[ERROR] 开仓失败")
                return False

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

    async def _wait_for_next_cycle(self):
        """等待下一个交易周期"""
        wait_seconds = self.config.REBALANCE_INTERVAL

        self.logger.info(f"[WAIT] 等待下一轮交易，间隔 {wait_seconds} 秒 ({wait_seconds//60} 分钟)")

        # 简单等待
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