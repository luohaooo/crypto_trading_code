"""
配置文件 - 自动化交易策略
支持 testnet 和实盘环境切换
"""

import os
from typing import Dict, Any

class TradingConfig:
    """交易策略配置类"""

    def __init__(self, use_testnet: bool = True):
        """
        初始化配置

        Args:
            use_testnet: 是否使用testnet (True: testnet, False: 实盘)
        """
        self.use_testnet = use_testnet

        # 基础交易配置
        self.EXECUTION_MODE = 'interval'     # 执行模式: 'interval' 或 'daily'
        self.EXECUTION_HOUR = 16           # 每日执行时间 (24小时制)
        self.REBALANCE_INTERVAL = 60      # 5分钟 = 300秒 (仅在interval模式下使用)
        self.TOP_N_LONG = 2            # 做多币种数量
        self.TOP_N_SHORT = 2             # 做空币种数量
        self.LEVERAGE = 1                 # 杠杆倍数
        self.LOOKBACK_HOURS = 80          # 历史数据回看小时数

        # 神经网络模型配置
        self.TIMEFRAMES = ['1h', '2h', '4h']  # 多时间框架
        self.LOOKBACK_PERIODS = 20    # OHLC回看周期数

        # 订单执行配置 - 简化为市价单模式
        self.USE_MARKET_ORDERS_ONLY = True  # 仅使用市价单，提高执行速度

        # 模型路径配置
        self.MODEL_PATH = os.path.join(
            os.path.dirname(__file__), '..', '..', 'figure_model', 'model_saved',
            't_2025-06_2025-06_mse_256_1d.pt'
        )

        # API配置
        self._setup_api_config()

        # 日志配置
        self.LOG_LEVEL = 'INFO'
        env_suffix = 'testnet' if use_testnet else 'live'

        # 创建日志目录路径
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        self.LOG_FILE = os.path.join(log_dir, f'trading_{env_suffix}_{self._get_date_str()}.log')

    def _setup_api_config(self):
        """设置API配置"""
        if self.use_testnet:
            # Testnet 配置
            self.API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', 'YOUR_TESTNET_API_KEY')
            self.API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', 'YOUR_TESTNET_API_SECRET')
            self.BASE_URL = 'https://testnet.binancefuture.com'
            self.ENV_NAME = 'TESTNET'
        else:
            # 实盘配置
            self.API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_LIVE_API_KEY')
            self.API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_LIVE_API_SECRET')
            self.BASE_URL = 'https://fapi.binance.com'
            self.ENV_NAME = 'LIVE'

    def _get_date_str(self) -> str:
        """获取当前日期字符串"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d')

    def get_ccxt_config(self) -> Dict[str, Any]:
        """获取CCXT交易所配置"""
        config = {
            'apiKey': self.API_KEY,
            'secret': self.API_SECRET,
            'timeout': 10000,  # 降低到10秒
            'enableRateLimit': False,  # 禁用速率限制以获得最快速度
            'options': {
                'defaultType': 'future',  # 期货交易
            }
        }

        if self.use_testnet:
            config['sandbox'] = True
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }

        return config

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        # 检查API密钥
        if not self.API_KEY or self.API_KEY.startswith('YOUR_'):
            print(f"错误: 请设置有效的API密钥 ({'testnet' if self.use_testnet else '实盘'})")
            return False

        if not self.API_SECRET or self.API_SECRET.startswith('YOUR_'):
            print(f"错误: 请设置有效的API密钥 ({'testnet' if self.use_testnet else '实盘'})")
            return False

        # 检查参数范围
        if self.TOP_N_LONG <= 0 or self.TOP_N_SHORT <= 0:
            print("错误: 做多/做空数量必须大于0")
            return False

        if self.LEVERAGE <= 0 or self.LEVERAGE > 10:
            print("错误: 杠杆倍数必须在1-10之间")
            return False

        if self.EXECUTION_MODE == 'interval' and self.REBALANCE_INTERVAL < 60:
            print("错误: 重新平衡间隔不能少于60秒")
            return False

        if self.EXECUTION_MODE == 'daily' and (self.EXECUTION_HOUR < 0 or self.EXECUTION_HOUR >= 24):
            print("错误: 每日执行时间必须在0-23之间")
            return False

        return True

    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print(f"交易策略配置 - {self.ENV_NAME}")
        print("=" * 50)
        print(f"环境类型: {'Testnet' if self.use_testnet else '实盘'}")
        print(f"执行模式: {self.EXECUTION_MODE}")
        if self.EXECUTION_MODE == 'daily':
            print(f"每日执行时间: {self.EXECUTION_HOUR:02d}:00")
        else:
            print(f"重新平衡间隔: {self.REBALANCE_INTERVAL}秒 ({self.REBALANCE_INTERVAL//60}分钟)")
        print(f"做多标的数量: {self.TOP_N_LONG}")
        print(f"做空标的数量: {self.TOP_N_SHORT}")
        print(f"杠杆倍数: {self.LEVERAGE}x")
        print(f"历史数据回看: {self.LOOKBACK_HOURS}小时")
        print(f"时间框架: {', '.join(self.TIMEFRAMES)}")
        print(f"OHLC回看周期: {self.LOOKBACK_PERIODS}")
        print(f"日志文件: {self.LOG_FILE}")
        print("=" * 50)


# 环境配置示例
def get_config(use_testnet: bool = True) -> TradingConfig:
    """
    获取配置实例

    Args:
        use_testnet: True为testnet，False为实盘

    Returns:
        TradingConfig: 配置实例
    """
    return TradingConfig(use_testnet=use_testnet)


# 环境变量设置指南
SETUP_GUIDE = """
环境变量设置指南

1. Testnet (测试环境):
   export BINANCE_TESTNET_API_KEY="your_testnet_api_key"
   export BINANCE_TESTNET_API_SECRET="your_testnet_secret"

2. 实盘环境:
   export BINANCE_API_KEY="your_live_api_key"
   export BINANCE_API_SECRET="your_live_secret"

3. 获取API密钥:
   - Testnet: https://testnet.binancefuture.com/
   - 实盘: https://www.binance.com/ (API管理)

安全提醒:
   - 请先在testnet环境充分测试
   - 实盘API请设置IP白名单
   - 建议API权限仅开启期货交易
   - 定期更换API密钥
"""

if __name__ == "__main__":
    print(SETUP_GUIDE)

    # 测试配置
    print("\nTestnet 配置:")
    testnet_config = get_config(use_testnet=True)
    testnet_config.print_config()
    print(f"配置有效性: {'有效' if testnet_config.validate_config() else '无效'}")

    print("\n实盘配置:")
    live_config = get_config(use_testnet=False)
    live_config.print_config()
    print(f"配置有效性: {'有效' if live_config.validate_config() else '无效'}")