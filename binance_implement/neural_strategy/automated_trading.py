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

        # 因子表现分析
        self.symbol_info = None  # 存储上一周期的symbol价格和因子信息
        self.ic_history = []     # IC历史记录
        self.rankic_history = []  # RankIC历史记录
        self.max_history_length = 20  # 保留最近20期的历史记录

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

            # 2.5. 因子表现分析 (基于上一周期因子预测当前收益)
            self.logger.info("[STEP 2.5] 分析上一轮因子表现 (IC/RankIC)...")
            if self.symbol_info is not None:
                # 创建上一轮的因子序列用于IC分析
                previous_factors = pd.Series(
                    {symbol: data['factor'] for symbol, data in self.symbol_info['data'].items()},
                    name='previous_factors'
                )
                self._calculate_factor_performance(previous_factors)
            else:
                self.logger.info("[FACTOR PERFORMANCE] 首次运行，无历史数据进行IC分析")

            # 3. 数据处理和因子计算
            self.logger.info("[STEP 3] 提取OHLC数据并计算因子...")
            factors = self._calculate_factors()
            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 因子计算失败或无有效因子")
                return False

            self.logger.info(f"[FACTORS] 计算得到 {len(factors)} 个有效因子")

            # 4. 选币策略
            self.logger.info("[STEP 4] 根据因子值选择交易标的...")
            long_symbols, short_symbols = self._select_trading_symbols(factors)

            if len(long_symbols) == 0 and len(short_symbols) == 0:
                self.logger.warning("[WARNING] 没有符合条件的交易标的")
                return True  # 不算失败，只是这轮不交易

            self.logger.info(f"[SELECTION] 选择交易标的:")
            self.logger.info(f"   做多: {long_symbols}")
            self.logger.info(f"   做空: {short_symbols}")

            # 发送交易标的选择通知
            selection_message = f"交易标的选择完成:\n"

            # 添加做多标的及其因子值
            if long_symbols:
                selection_message += f"做多 ({len(long_symbols)}个):\n"
                for symbol in long_symbols:
                    factor_value = factors[symbol]
                    selection_message += f"  {symbol}: {factor_value:.6f}\n"
            else:
                selection_message += "做多: 无\n"

            # 添加做空标的及其因子值
            if short_symbols:
                selection_message += f"做空 ({len(short_symbols)}个):\n"
                for symbol in short_symbols:
                    factor_value = factors[symbol]
                    selection_message += f"  {symbol}: {factor_value:.6f}\n"
            else:
                selection_message += "做空: 无\n"

            # 添加因子统计信息
            selection_message += f"\n因子统计:\n"
            selection_message += f"  总计: {len(factors)} 个标的\n"
            selection_message += f"  范围: {factors.min():.6f} ~ {factors.max():.6f}"

            self.executor._send_dingding_notification(selection_message)

            # 5. 获取当前保证金余额 (现在是同步方法)
            self.logger.info("[STEP 5] 获取当前合约保证金...")
            balance = self.executor.get_account_balance()
            if balance is None:
                self.logger.error("[ERROR] 无法获取账户余额")
                return False

            self.current_balance = balance
            self.logger.info(f"[BALANCE] 当前保证金余额: {balance:.2f} USDT")

            # 发送交易前余额通知
            self.executor.send_balance_notification()

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

            # 8. 记录symbol信息用于下次IC分析
            self.logger.info("[STEP 8] 记录当前交易对价格和因子信息...")
            self._record_symbol_info(factors)

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

        # 打印因子值排名前20的交易对和对应的因子值
        top_20_count = min(20, len(sorted_factors))
        self.logger.info(f"[TOP 20] 因子值排名前{top_20_count}的交易对:")
        for i in range(top_20_count):
            symbol = sorted_factors.index[i]
            factor_value = sorted_factors.iloc[i]
            self.logger.info(f"   第{i+1}名: {symbol}: {factor_value:.6f}")

        # 打印因子值排名后20的交易对和对应的因子值
        bottom_20_count = min(20, len(sorted_factors))
        self.logger.info(f"[BOTTOM 20] 因子值排名后{bottom_20_count}的交易对:")
        for i in range(bottom_20_count):
            idx = len(sorted_factors) - bottom_20_count + i
            symbol = sorted_factors.index[idx]
            factor_value = sorted_factors.iloc[idx]
            rank = len(sorted_factors) - bottom_20_count + i + 1
            self.logger.info(f"   第{rank}名: {symbol}: {factor_value:.6f}")

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

    def _calculate_factor_performance(self, current_factors: pd.Series) -> None:
        """
        计算因子表现 - IC和RankIC
        Args:
            current_factors: 当前周期的因子值序列
        """
        try:
            # 检查是否有上一周期的数据
            if self.symbol_info is None:
                self.logger.info("[FACTOR PERFORMANCE] 首次运行，无历史数据进行IC分析")
                return

            # 获取当前所有symbol的价格
            current_prices = {}
            failed_symbols = []

            for symbol in current_factors.index:
                try:
                    ticker = self.executor.exchange.fetch_ticker(symbol)
                    if ticker and 'last' in ticker and ticker['last']:
                        current_prices[symbol] = ticker['last']
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    self.logger.warning(f"[PRICE FETCH] 获取 {symbol} 价格失败: {e}")
                    failed_symbols.append(symbol)

            if failed_symbols:
                self.logger.warning(f"[PRICE FETCH] {len(failed_symbols)} 个交易对价格获取失败: {failed_symbols[:5]}...")

            # 计算收益率和IC
            valid_symbols = []
            returns = []
            factor_values = []

            for symbol in current_factors.index:
                if (symbol in current_prices and
                    symbol in self.symbol_info['data']):

                    current_price = current_prices[symbol]
                    previous_price = self.symbol_info['data'][symbol]['price']
                    previous_factor = self.symbol_info['data'][symbol]['factor']

                    # 计算收益率
                    if previous_price > 0:
                        return_rate = (current_price - previous_price) / previous_price
                        returns.append(return_rate)
                        factor_values.append(previous_factor)
                        valid_symbols.append(symbol)

            # 检查是否有足够的数据进行分析
            if len(returns) < 10:
                self.logger.warning(f"[FACTOR PERFORMANCE] 有效数据不足 ({len(returns)} < 10)，跳过IC分析")
                return

            # 转换为pandas Series进行相关性计算
            returns_series = pd.Series(returns, index=valid_symbols)
            factors_series = pd.Series(factor_values, index=valid_symbols)

            # 计算IC (Information Coefficient)
            ic = returns_series.corr(factors_series)

            # 计算RankIC (Rank Information Coefficient)
            returns_rank = returns_series.rank()
            factors_rank = factors_series.rank()
            rank_ic = returns_rank.corr(factors_rank)

            # 记录到历史
            if not pd.isna(ic):
                self.ic_history.append(ic)
                if len(self.ic_history) > self.max_history_length:
                    self.ic_history.pop(0)

            if not pd.isna(rank_ic):
                self.rankic_history.append(rank_ic)
                if len(self.rankic_history) > self.max_history_length:
                    self.rankic_history.pop(0)

            # 计算历史统计
            ic_mean = pd.Series(self.ic_history).mean() if self.ic_history else 0
            ic_std = pd.Series(self.ic_history).std() if len(self.ic_history) > 1 else 0
            rankic_mean = pd.Series(self.rankic_history).mean() if self.rankic_history else 0
            rankic_std = pd.Series(self.rankic_history).std() if len(self.rankic_history) > 1 else 0

            # 记录详细日志
            self.logger.info("="*50)
            self.logger.info("[FACTOR PERFORMANCE] IC分析结果:")
            self.logger.info(f"   参与分析的交易对数量: {len(valid_symbols)}")
            self.logger.info(f"   时间跨度: {self.symbol_info['timestamp'].strftime('%Y-%m-%d %H:%M')} -> {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            self.logger.info(f"   因子IC: {ic:.4f} {'(正相关)' if ic > 0 else '(负相关)' if ic < 0 else '(无相关)'}")
            self.logger.info(f"   因子RankIC: {rank_ic:.4f} {'(正相关)' if rank_ic > 0 else '(负相关)' if rank_ic < 0 else '(无相关)'}")

            if len(self.ic_history) > 1:
                self.logger.info(f"   历史IC均值: {ic_mean:.4f} (最近{len(self.ic_history)}期)")
                self.logger.info(f"   历史IC标准差: {ic_std:.4f}")
                self.logger.info(f"   历史RankIC均值: {rankic_mean:.4f} (最近{len(self.rankic_history)}期)")
                self.logger.info(f"   历史RankIC标准差: {rankic_std:.4f}")

            # 统计收益率分布
            positive_returns = sum(1 for r in returns if r > 0)
            negative_returns = sum(1 for r in returns if r < 0)
            self.logger.info(f"   收益率分布: {positive_returns}个正收益, {negative_returns}个负收益")
            self.logger.info(f"   收益率范围: {min(returns):.4f} ~ {max(returns):.4f}")
            self.logger.info("="*50)

            # 发送DingTalk通知
            self._send_factor_performance_notification(
                ic, rank_ic, len(valid_symbols), ic_mean, ic_std,
                len(self.ic_history), positive_returns, negative_returns
            )

        except Exception as e:
            self.logger.error(f"[ERROR] 因子表现分析异常: {e}")

    def _send_factor_performance_notification(self, ic: float, rank_ic: float,
                                            sample_size: int, ic_mean: float, ic_std: float,
                                            history_length: int, positive_returns: int,
                                            negative_returns: int) -> None:
        """发送因子表现分析的DingTalk通知"""
        try:
            # 判断IC表现
            ic_status = "✅" if ic > 0.02 else "⚠️" if ic > -0.02 else "❌"
            rankic_status = "✅" if rank_ic > 0.02 else "⚠️" if rank_ic > -0.02 else "❌"

            message = f"🔍 因子表现分析:\n"
            message += f"IC: {ic:.4f} {ic_status}\n"
            message += f"RankIC: {rank_ic:.4f} {rankic_status}\n"
            message += f"分析样本: {sample_size}个交易对\n"
            message += f"收益分布: {positive_returns}↗️ {negative_returns}↘️\n"

            if history_length > 1:
                message += f"历史表现: 均值{ic_mean:.4f}±{ic_std:.4f} ({history_length}期)\n"

            # 添加表现评价
            if ic > 0.05:
                message += "📈 因子表现优秀"
            elif ic > 0.02:
                message += "📊 因子表现良好"
            elif ic > -0.02:
                message += "📉 因子表现一般"
            else:
                message += "⚠️ 因子表现较差"

            self.executor._send_dingding_notification(message)

        except Exception as e:
            self.logger.error(f"[ERROR] 发送因子表现通知失败: {e}")

    def _record_symbol_info(self, factors: pd.Series) -> None:
        """
        记录当前周期的symbol信息（价格和因子值）
        Args:
            factors: 当前因子值序列
        """
        try:
            current_data = {}
            failed_symbols = []

            for symbol in factors.index:
                try:
                    ticker = self.executor.exchange.fetch_ticker(symbol)
                    if ticker and 'last' in ticker and ticker['last']:
                        current_data[symbol] = {
                            'price': ticker['last'],
                            'factor': factors[symbol]
                        }
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    self.logger.warning(f"[RECORD] 获取 {symbol} 价格失败: {e}")
                    failed_symbols.append(symbol)

            # 更新symbol_info
            self.symbol_info = {
                'timestamp': datetime.now(),
                'data': current_data
            }

            self.logger.info(f"[RECORD] 记录了 {len(current_data)} 个交易对的价格和因子信息")
            if failed_symbols:
                self.logger.warning(f"[RECORD] {len(failed_symbols)} 个交易对记录失败: {failed_symbols[:5]}...")

        except Exception as e:
            self.logger.error(f"[ERROR] 记录symbol信息异常: {e}")

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
        # execution_hours = [0, 8, 16]
        execution_hours = [4, 12, 20]

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