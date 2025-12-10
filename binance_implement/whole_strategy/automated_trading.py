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
from typing import Optional, Dict, List, Any
import pandas as pd
import warnings

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# 导入配置和工具
from config import get_config, TradingConfig
from trading_executor import TradingExecutor
from optimized_data_processor import OptimizedDataProcessor
# from optimized_factor_calculator import OptimizedFactorCalculator
from fs_factor_calculator import OptimizedFactorCalculator
from trading_utils.logger import setup_logger
from trading_utils.monitor import TradingMonitor

class AutomatedTradingSystem:
    """自动化交易系统主类"""

    def __init__(self, use_testnet: bool = True, start_hour_16h: Optional[int] = None):
        """
        初始化交易系统

        Args:
            use_testnet: True为testnet，False为实盘
            start_hour_16h: 16小时执行模式起点小时 (0-23)
        """
        self.use_testnet = use_testnet
        if start_hour_16h is None:
            start_hour_16h = 0
        if not 0 <= start_hour_16h < 24:
            raise ValueError("start_hour_16h 必须在 0-23 之间")
        self.start_hour_16h = start_hour_16h

        self.config = get_config(use_testnet)

        # 初始化组件
        self.logger = setup_logger(self.config.LOG_FILE, self.config.LOG_LEVEL)
        self.monitor = TradingMonitor(self.logger)
        self.executor = TradingExecutor(self.config, self.logger)
        self.data_processor = OptimizedDataProcessor(self.config, self.logger)
        self.factor_calculator = OptimizedFactorCalculator(self.config, self.logger)

        # 交易状态
        self.cycle_count = 0
        self.current_balance = 0.0
        self.initial_cycle_balance = None
        self.cycle_states: List[Dict[str, Any]] = []
        self._initialize_cycle_states()

        # 因子表现分析
        self.symbol_info = None  # 存储上一周期的symbol价格和因子信息
        self.ic_history = []     # IC历史记录
        self.rankic_history = []  # RankIC历史记录
        self.max_history_length = 20  # 保留最近20期的历史记录

        # API频率控制
        self.last_api_call_time = 0
        self.api_call_interval = 0.1  # 最小间隔100ms
        self.batch_call_cache = {}  # 批量调用缓存
        self.cache_expire_time = 30  # 缓存过期30秒

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_cycle_states(self):
        """初始化cycle状态"""
        self.cycle_states = []
        now = datetime.now()

        if self.config.EXECUTION_MODE == '16h':
            base_hour = (self.start_hour_16h + getattr(self.config, 'CYCLE_START_OFFSET', 0)) % 24
            base_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)

            for idx in range(self.config.CYCLE_COUNT):
                next_time = base_time + timedelta(hours=idx * self.config.CYCLE_SPACING_HOURS)
                while next_time < now:
                    next_time += timedelta(hours=self.config.CYCLE_PERIOD_HOURS)

                self.cycle_states.append({
                    'id': idx,
                    'next_execution': next_time,
                    'last_execution': None,
                    'positions': {},
                    'has_traded': False,
                })
        else:
            self.cycle_states.append({
                'id': 0,
                'next_execution': None,
                'last_execution': None,
                'positions': {},
                'has_traded': False,
            })

    def _format_cycle_label(self, cycle_state: Optional[Dict[str, Any]]) -> str:
        """格式化cycle标签"""
        if cycle_state and self.config.EXECUTION_MODE == '16h':
            return f"Cycle {cycle_state['id']}/{self.config.CYCLE_COUNT}"
        return "Cycle 0"

    def _get_next_cycle_state(self) -> Dict[str, Any]:
        """获取下一个需要执行的cycle"""
        if not self.cycle_states:
            raise RuntimeError("未初始化cycle状态")

        def sort_key(state: Dict[str, Any]):
            next_time = state.get('next_execution')
            if next_time is None:
                return datetime.max
            return next_time

        return min(self.cycle_states, key=sort_key)

    def _sync_positions_with_exchange(self):
        """同步仓位状态，移除已由止盈止损平仓的仓位"""
        for cycle_state in self.cycle_states:
            positions = cycle_state.get('positions', {})
            if not positions:
                continue

            results = self.executor.refresh_positions_status(positions)
            for item in results:
                symbol = item.get('symbol')
                reason = item.get('reason', '未知原因')
                if symbol in positions:
                    position_info = positions.pop(symbol)
                    self.logger.info(
                        f"[PROTECT] {self._format_cycle_label(cycle_state)} {symbol} "
                        f"仓位已由保护单或外部操作平仓 ({reason})，自动移除"
                    )
                    message = (
                        f"🛡️ 仓位监控\n"
                        f"Cycle: {self._format_cycle_label(cycle_state)}\n"
                        f"交易对: {symbol}\n"
                        f"方向: {position_info.get('side', '未知')}\n"
                        f"原因: {reason}"
                    )
                    self.executor._send_dingding_notification(message)

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
            next_cycle = self._get_next_cycle_state()
            next_execution = next_cycle['next_execution']
            cycle_label = self._format_cycle_label(next_cycle)
            self.logger.info(f"[OK] 系统初始化完成，16小时周期执行 (起始 {self.start_hour_16h:02d}:00)")
            if next_execution:
                self.logger.info(
                    f"[SCHEDULE] {cycle_label} 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}"
                )
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

        if self.config.EXECUTION_MODE == '16h' and self.config.CYCLE_COUNT > 0:
            self.initial_cycle_balance = balance / self.config.CYCLE_COUNT
            self.logger.info(
                f"[BALANCE] 16小时模式首次每个cycle可用资金: {self.initial_cycle_balance:.2f} USDT"
            )

    def _load_neglect_symbols(self) -> List[str]:
        """
        读取需要忽略的交易对列表
        Returns:
            忽略的symbol列表
        """
        file_path = os.path.join(os.path.dirname(__file__), "neglect_symbols.txt")
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                symbols = [
                    line.strip()
                    for line in file
                    if line.strip() and not line.strip().startswith("#")
                ]
            return symbols
        except Exception as exc:
            self.logger.error(f"[ERROR] 读取 neglect_symbols.txt 失败: {exc}")
            return []

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
        while True:
            try:
                cycle_state = None
                if self.config.EXECUTION_MODE == '16h':
                    cycle_state = await self._prepare_cycle_execution()
                else:
                    cycle_state = self.cycle_states[0]

                cycle_start_time = datetime.now()
                self.cycle_count += 1

                cycle_label = self._format_cycle_label(cycle_state)
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"[CYCLE] 第 {self.cycle_count} 轮交易周期 ({cycle_label})")
                self.logger.info(f"[TIME] 开始时间: {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*50}")

                # 执行一轮完整的交易流程
                success = await self._execute_trading_cycle(cycle_state)

                if success:
                    self.logger.info("[OK] 交易周期执行成功")
                    self.monitor.record_success()
                else:
                    self.logger.warning("[WARNING] 交易周期执行失败")
                    self.monitor.record_failure()

                # 记录执行时间
                cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                self.monitor.record_cycle_time(cycle_duration)

                self.logger.info(f"[TIME] 周期耗时: {cycle_duration:.1f}秒")

                if self.config.EXECUTION_MODE == '16h' and cycle_state is not None:
                    cycle_state['last_execution'] = cycle_start_time
                    cycle_state['next_execution'] = cycle_state['next_execution'] + timedelta(
                        hours=self.config.CYCLE_PERIOD_HOURS
                    )

                # 等待下一个周期
                if self.config.EXECUTION_MODE != '16h':
                    await self._wait_for_next_cycle()

            except Exception as e:
                self.logger.error(f"[ERROR] 交易周期异常: {e}")
                self.monitor.record_failure()

                # 异常后等待一段时间再继续
                self.logger.info("[WAIT] 异常后等待30秒...")
                await asyncio.sleep(30)

    async def _execute_trading_cycle(self, cycle_state: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行一轮完整的交易周期

        Returns:
            bool: 执行是否成功
        """
        try:
            if cycle_state is None:
                cycle_state = self.cycle_states[0]

            positions_holder = cycle_state.get('positions', {})
            cycle_label = self._format_cycle_label(cycle_state)

            # 周期开始前同步仓位状态，剔除已由保护单平仓的仓位
            self._sync_positions_with_exchange()

            # 1. 平仓字典中存储的仓位
            self.logger.info(f"[STEP 1] {cycle_label} 平仓字典中存储的仓位...")
            if positions_holder:
                self.logger.info(f"[POSITIONS] 当前持有仓位: {positions_holder}")
                close_success = self.executor.close_specific_positions(positions_holder)
                if not close_success:
                    self.logger.error("[ERROR] 平仓失败")
                    return False
                # 清空仓位字典
                positions_holder.clear()
                self.logger.info(f"[INFO] {cycle_label} 仓位字典已清空")
            else:
                self.logger.info("[INFO] 仓位字典为空，无需平仓")

            # 2. 跳过验证步骤
            self.logger.info("[STEP 2] 跳过验证步骤...")

            # 3. 数据处理和因子计算
            self.logger.info("[STEP 3] 提取OHLC数据并计算因子...")
            factors = self._calculate_factors()
            if factors is None or len(factors) == 0:
                self.logger.error("[ERROR] 因子计算失败或无有效因子")
                return False

            self.logger.info(f"[FACTORS] 计算得到 {len(factors)} 个有效因子")

            # 4. 选币策略
            self.logger.info("[STEP 4] 根据因子值选择交易标的...")
            long_symbols, short_symbols = self._select_trading_symbols(factors, cycle_state)

            if len(long_symbols) == 0 and len(short_symbols) == 0:
                self.logger.warning("[WARNING] 没有符合条件的交易标的")
                return True  # 不算失败，只是这轮不交易

            self.logger.info(f"[SELECTION] 选择交易标的:")
            self.logger.info(f"   做多: {long_symbols}")
            self.logger.info(f"   做空: {short_symbols}")

            # 发送交易标的选择通知
            selection_message = f"{cycle_label} 交易标的选择完成:\n"

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

            # 5.5 计算本轮可用资金
            balance_to_use = balance
            if self.config.EXECUTION_MODE == '16h':
                if not cycle_state.get('has_traded', False):
                    if self.initial_cycle_balance is None and self.config.CYCLE_COUNT > 0:
                        self.initial_cycle_balance = balance / self.config.CYCLE_COUNT
                    balance_to_use = self.initial_cycle_balance or balance
                    cycle_state['has_traded'] = True
                    self.logger.info(
                        f"[BALANCE] {cycle_label} 首次交易，使用预分配资金 {balance_to_use:.2f} USDT"
                    )
                else:
                    self.logger.info(
                        f"[BALANCE] {cycle_label} 使用当前可用余额 {balance_to_use:.2f} USDT 执行开仓"
                    )

            # 6. 开仓交易 (现在是同步方法)
            self.logger.info("[STEP 6] 执行开仓交易...")
            position_success, opened_positions = self.executor.open_positions(
                long_symbols, short_symbols, balance_to_use
            )
  
            if not position_success:
                self.logger.error("[ERROR] 开仓失败")
                return False

            # 将新开仓位添加到字典中
            if opened_positions:
                positions_holder.update(opened_positions)
                self.logger.info(f"[POSITIONS] 新仓位已添加到字典: {opened_positions}")
                self.logger.info(f"[POSITIONS] 当前仓位字典: {positions_holder}")

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

    def _select_trading_symbols(
        self, factors: pd.Series, current_cycle_state: Optional[Dict[str, Any]] = None
    ) -> tuple[List[str], List[str]]:
        """
        根据因子值选择交易标的

        Args:
            factors: 因子值序列
            current_cycle_state: 当前执行周期的状态，用于排除自身仓位

        Returns:
            tuple: (做多标的列表, 做空标的列表)
        """
        if len(factors) == 0:
            return [], []

        # 过滤需要忽略的symbol
        neglect_symbols = self._load_neglect_symbols()
        if neglect_symbols:
            filtered_symbols = [symbol for symbol in neglect_symbols if symbol in factors.index]
            if filtered_symbols:
                factors = factors.drop(filtered_symbols, errors="ignore")
                self.logger.info(f"[FILTER] 忽略 {len(filtered_symbols)} 个交易对: {filtered_symbols}")

        if len(factors) == 0:
            self.logger.warning("[WARNING] 忽略配置生效后没有可用的交易对")
            return [], []

        # 排序因子值
        sorted_factors = factors.sort_values(ascending=False)

        # 统计其它周期中每个symbol出现次数，用于限制重复持仓
        symbol_usage: Dict[str, int] = {}
        for state in self.cycle_states:
            if current_cycle_state is not None and state is current_cycle_state:
                continue
            positions = state.get('positions', {})
            for symbol in positions.keys():
                symbol_usage[symbol] = symbol_usage.get(symbol, 0) + 1

        max_symbol_occurrence = 4  # 其它周期内同一symbol最多出现4次

        def pick_symbols(candidates: List[str], target_count: int) -> List[str]:
            """按照候选顺序挑选，确保其它周期未超过限制"""
            if target_count <= 0:
                return []

            selected: List[str] = []
            for symbol in candidates:
                if len(selected) >= target_count:
                    break

                current_usage = symbol_usage.get(symbol, 0)
                if current_usage >= max_symbol_occurrence:
                    self.logger.info(
                        f"[FILTER] {symbol} 已在其它周期出现 {current_usage} 次，达到上限 {max_symbol_occurrence}，跳过"
                    )
                    continue

                selected.append(symbol)
                symbol_usage[symbol] = current_usage + 1

            return selected

        # 选择前N名做多，后N名做空
        top_n_long = min(self.config.TOP_N_LONG, len(sorted_factors))
        top_n_short = min(self.config.TOP_N_SHORT, len(sorted_factors))

        long_candidates = sorted_factors.index.tolist()
        short_candidates = list(reversed(long_candidates))

        long_symbols = pick_symbols(long_candidates, top_n_long)
        short_symbols = pick_symbols(short_candidates, top_n_short)

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

    def _record_symbol_info(self, factors: pd.Series) -> None:
        """
        记录当前周期的symbol信息（价格和因子值）
        Args:
            factors: 当前因子值序列
        """
        try:
            # 批量获取价格（使用频率控制）
            symbols_list = factors.index.tolist()
            prices = self._fetch_prices_with_rate_limit(symbols_list)

            # 构建 current_data
            current_data = {}
            failed_symbols = []

            for symbol in symbols_list:
                if symbol in prices:
                    current_data[symbol] = {
                        'price': prices[symbol],
                        'factor': factors[symbol]
                    }
                else:
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

    def _fetch_prices_with_rate_limit(self, symbols: List[str]) -> Dict[str, float]:
        """
        带频率控制的价格获取方法
        Args:
            symbols: 需要获取价格的交易对列表
        Returns:
            {symbol: price} 字典
        """
        current_time = time.time()
        cache_key = ','.join(sorted(symbols))

        # 检查缓存
        if (cache_key in self.batch_call_cache and
            current_time - self.batch_call_cache[cache_key]['timestamp'] < self.cache_expire_time):
            self.logger.debug(f"[CACHE] 使用缓存价格数据")
            return self.batch_call_cache[cache_key]['data']

        prices = {}
        failed_symbols = []

        try:
            # 首先尝试批量获取
            self.logger.debug(f"[API] 批量获取 {len(symbols)} 个交易对价格")
            tickers = self.executor.exchange.fetch_tickers(symbols)

            for symbol in symbols:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    if ticker and 'last' in ticker and ticker['last']:
                        prices[symbol] = ticker['last']
                    else:
                        failed_symbols.append(symbol)
                else:
                    failed_symbols.append(symbol)

            # 缓存成功的结果
            if prices:
                self.batch_call_cache[cache_key] = {
                    'timestamp': current_time,
                    'data': prices.copy()
                }

        except Exception as e:
            self.logger.warning(f"[API] 批量获取失败，回退到单个获取: {e}")
            # 批量获取失败，使用单个获取并控制频率
            for symbol in symbols:
                try:
                    # 控制API调用频率
                    elapsed = time.time() - self.last_api_call_time
                    if elapsed < self.api_call_interval:
                        time.sleep(self.api_call_interval - elapsed)

                    ticker = self.executor.exchange.fetch_ticker(symbol)
                    self.last_api_call_time = time.time()

                    if ticker and 'last' in ticker and ticker['last']:
                        prices[symbol] = ticker['last']
                    else:
                        failed_symbols.append(symbol)

                except Exception as e:
                    self.logger.warning(f"[API] 获取 {symbol} 价格失败: {e}")
                    failed_symbols.append(symbol)

        if failed_symbols:
            self.logger.warning(f"[API] {len(failed_symbols)} 个交易对价格获取失败: {failed_symbols[:5]}...")

        self.logger.info(f"[API] 成功获取 {len(prices)} 个交易对价格")
        return prices

    def _get_next_execution_time(self) -> datetime:
        """获取下次执行时间"""
        now = datetime.now()
        target_hour = self.config.EXECUTION_HOUR
        today_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        if now >= today_target:
            return today_target + timedelta(days=1)
        else:
            return today_target

    async def _prepare_cycle_execution(self) -> Dict[str, Any]:
        """等待并获取下一个需要执行的cycle"""
        cycle_state = self._get_next_cycle_state()
        await self._wait_for_cycle_execution(cycle_state)
        return cycle_state

    async def _wait_for_next_cycle(self):
        """等待下一个交易周期"""
        if self.config.EXECUTION_MODE == 'daily':
            await self._wait_for_daily_execution()
        elif self.config.EXECUTION_MODE == '16h':
            return
        else:
            await self._wait_for_interval_execution()

    # async def _wait_for_daily_execution(self):
    #     """等待每日执行时间 (20:00)"""
    #     now = datetime.now()
    #     target_hour = self.config.EXECUTION_HOUR

    #     # 计算下次执行时间
    #     today_target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

    #     if now >= today_target:
    #         # 如果今天的执行时间已过，等到明天的执行时间
    #         next_execution = today_target + timedelta(days=1)
    #     else:
    #         # 如果今天的执行时间未到，等到今天的执行时间
    #         next_execution = today_target

    #     wait_seconds = (next_execution - now).total_seconds()

    #     self.logger.info(f"[DAILY] 每日执行模式 - 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}")
    #     self.logger.info(f"[WAIT] 等待 {wait_seconds:.0f} 秒 ({wait_seconds/3600:.1f} 小时)")

    #     # 分批等待，每小时打印一次状态
    #     while wait_seconds > 0:
    #         if wait_seconds > 3600:
    #             # 等待1小时
    #             await self._sleep_and_sync(3600)
    #             wait_seconds -= 3600
    #             remaining_hours = wait_seconds / 3600
    #             self.logger.info(f"[WAIT] 剩余等待时间: {remaining_hours:.1f} 小时")
    #         else:
    #             # 最后的等待时间
    #             self.logger.info(f"[WAIT] 最后等待: {wait_seconds:.0f} 秒")
    #             await self._sleep_and_sync(wait_seconds, chunk=min(300, wait_seconds))
    #             break

    async def _wait_for_cycle_execution(self, cycle_state: Dict[str, Any]):
        """等待指定cycle的执行时间"""
        next_execution = cycle_state.get('next_execution')
        if not next_execution:
            return

        cycle_label = self._format_cycle_label(cycle_state)
        now = datetime.now()
        wait_seconds = max(0, (next_execution - now).total_seconds())

        self.logger.info(
            f"[16H] {cycle_label} 下次执行时间: {next_execution.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(f"[WAIT] 等待 {wait_seconds:.0f} 秒 ({wait_seconds/3600:.1f} 小时)")

        counter = 0

        while wait_seconds > 0:
            counter += 1
            if wait_seconds > 600:
                await asyncio.sleep(300)
                self._sync_positions_with_exchange()
                now = datetime.now()
                wait_seconds = max(0, (next_execution - now).total_seconds())
                remaining_mins = wait_seconds / 60
                if counter % 10 == 0:
                    self.logger.info(f"[WAIT] {cycle_label} 剩余等待时间: {remaining_mins:.1f} 分钟")
            else:
                now = datetime.now()
                wait_seconds = max(0, (next_execution - now).total_seconds())
                self.logger.info(f"[WAIT] {cycle_label} 最后等待: {wait_seconds:.0f} 秒")

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                # self._sync_positions_with_exchange()
                break

    async def _wait_for_interval_execution(self):
        """等待固定间隔执行"""
        wait_seconds = self.config.REBALANCE_INTERVAL

        self.logger.info(f"[INTERVAL] 间隔执行模式 - 等待 {wait_seconds} 秒 ({wait_seconds//60} 分钟)")
        await self._sleep_and_sync(wait_seconds, chunk=min(60, wait_seconds))

    # async def _sleep_and_sync(self, total_seconds: float, chunk: float = 300):
    #     """在等待过程中定期同步仓位状态"""
    #     remaining = total_seconds
    #     if remaining <= 0:
    #         # self._sync_positions_with_exchange()
    #         return

    #     while remaining > 0:
    #         sleep_duration = min(chunk, remaining)
    #         # self._sync_positions_with_exchange()
    #         await asyncio.sleep(sleep_duration)
    #         remaining -= sleep_duration
    #     # 最后再同步一次，捕捉等待结束后的状态
    #     self._sync_positions_with_exchange()

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

    # 16小时模式起始时间设置
    while True:
        hour_input = input("设置16小时模式起始小时 (0-23，直接回车默认为0): ").strip()
        if not hour_input:
            start_hour_16h = 0
            break
        try:
            start_hour_16h = int(hour_input)
            if 0 <= start_hour_16h < 24:
                break
            print("[ERROR] 起始小时必须在 0-23 范围内")
        except ValueError:
            print("[ERROR] 请输入有效的整数小时")

    print(f"[OK] 选择环境: {env_name}")
    print("[SYSTEM] 启动交易系统...")
    print()

    # 创建和启动交易系统
    trading_system = AutomatedTradingSystem(use_testnet=use_testnet, start_hour_16h=start_hour_16h)

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
