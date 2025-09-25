"""
测试环境配置和API连接
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import TradingConfig, SETUP_GUIDE
from exchange_connector import BinanceConnector


def test_configuration():
    """测试配置模块"""
    print("🔧 测试配置模块...")
    print("=" * 50)

    # 测试testnet配置
    print("1. Testnet配置测试:")
    testnet_config = TradingConfig(use_testnet=True)
    testnet_config.print_config()
    print(f"配置验证: {'✅ 通过' if testnet_config.validate_config() else '❌ 失败'}")

    print("\n" + "=" * 50)

    # 测试实盘配置
    print("2. 实盘配置测试:")
    live_config = TradingConfig(use_testnet=False)
    live_config.print_config()
    print(f"配置验证: {'✅ 通过' if live_config.validate_config() else '❌ 失败'}")

    return testnet_config, live_config


def test_exchange_connection():
    """测试交易所连接"""
    print("\n🔗 测试交易所连接...")
    print("=" * 50)

    # 使用testnet配置进行测试
    config = TradingConfig(use_testnet=True)

    if not config.validate_config():
        print("❌ 配置无效，跳过连接测试")
        print("\n📋 请按照以下指南设置环境变量:")
        print(SETUP_GUIDE)
        return None

    # 创建连接器
    connector = BinanceConnector(config)

    # 测试连接
    print("正在连接到Binance Testnet...")
    if connector.connect():
        print("✅ 连接成功!")

        try:
            # 测试账户信息
            print("\n📊 账户信息:")
            account_info = connector.get_account_info()
            print(f"   总权益: {account_info['total_equity']:.4f} USDT")
            print(f"   可用余额: {account_info['available_balance']:.4f} USDT")
            print(f"   保证金余额: {account_info['margin_balance']:.4f} USDT")
            print(f"   持仓数量: {account_info['position_count']}")
            print(f"   账户类型: {account_info['account_type']}")
            print(f"   可交易: {'✅' if account_info['can_trade'] else '❌'}")

            # 测试获取交易对
            print("\n📈 获取交易对:")
            symbols = connector.get_futures_symbols(min_volume_24h=500000)
            print(f"   找到 {len(symbols)} 个活跃期货交易对")
            if symbols:
                print(f"   前10个: {symbols[:10]}")

            # 测试获取历史数据
            if symbols:
                test_symbol = symbols[0]
                print(f"\n📊 测试获取 {test_symbol} 历史数据:")
                df = connector.get_ohlcv_data(test_symbol, '1m', limit=60)
                if not df.empty:
                    print(f"   成功获取 {len(df)} 条1分钟数据")
                    print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
                    print(f"   最新价格: {df['close'].iloc[-1]:.4f}")
                    print(f"   24小时涨跌: {((df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100):.2f}%")
                else:
                    print("   ❌ 未获取到数据")

            # 健康检查
            print("\n🔍 连接健康检查:")
            health = connector.check_connection_health()
            print(f"   连接状态: {'✅' if health['is_connected'] else '❌'}")
            print(f"   交易所状态: {health['exchange_status']}")
            if 'time_diff' in health:
                print(f"   时间差: {health['time_diff']:.0f}ms")

        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")

        finally:
            connector.close()

    else:
        print("❌ 连接失败")
        print("\n可能的原因:")
        print("1. API密钥配置错误")
        print("2. 网络连接问题")
        print("3. Binance服务暂时不可用")

    return connector


def main():
    """主测试函数"""
    print("🚀 自动化交易策略 - 环境配置和API连接测试")
    print("=" * 60)

    try:
        # 测试配置
        testnet_config, live_config = test_configuration()

        # 测试连接（仅testnet）
        test_exchange_connection()

        print("\n" + "=" * 60)
        print("🎉 测试完成!")
        print("\n💡 下一步:")
        print("1. 确保API密钥配置正确")
        print("2. 在testnet环境充分测试策略")
        print("3. 验证账户权限和余额")
        print("4. 实施仓位管理和风险控制")

    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()