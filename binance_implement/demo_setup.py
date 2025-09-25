"""
简化的API连接演示
展示如何配置和使用Binance API
"""

import os
from config import TradingConfig, SETUP_GUIDE

def demo_api_setup():
    """演示API设置流程"""
    print("🚀 Binance API连接配置演示")
    print("=" * 50)

    print("📋 API密钥设置指南:")
    print(SETUP_GUIDE)

    print("\n🔧 当前配置状态:")

    # Testnet配置
    print("\n1. Testnet环境:")
    testnet_config = TradingConfig(use_testnet=True)

    print(f"   API Key: {'✅ 已设置' if not testnet_config.API_KEY.startswith('YOUR_') else '❌ 未设置'}")
    print(f"   API Secret: {'✅ 已设置' if not testnet_config.API_SECRET.startswith('YOUR_') else '❌ 未设置'}")
    print(f"   环境: {testnet_config.ENV_NAME}")
    print(f"   沙盒模式: {'✅ 启用' if testnet_config.use_testnet else '❌ 关闭'}")

    # 实盘配置
    print("\n2. 实盘环境:")
    live_config = TradingConfig(use_testnet=False)

    print(f"   API Key: {'✅ 已设置' if not live_config.API_KEY.startswith('YOUR_') else '❌ 未设置'}")
    print(f"   API Secret: {'✅ 已设置' if not live_config.API_SECRET.startswith('YOUR_') else '❌ 未设置'}")
    print(f"   环境: {live_config.ENV_NAME}")
    print(f"   沙盒模式: {'✅ 启用' if live_config.use_testnet else '❌ 关闭'}")

    print("\n💡 下一步操作:")
    if testnet_config.validate_config():
        print("✅ Testnet配置完整，可以开始测试交易策略")
    else:
        print("⚠️ 请先设置Testnet API密钥:")
        print("   export BINANCE_TESTNET_API_KEY='your_testnet_api_key'")
        print("   export BINANCE_TESTNET_API_SECRET='your_testnet_secret'")

    if live_config.validate_config():
        print("✅ 实盘配置完整，但建议先在testnet测试")
    else:
        print("⚠️ 实盘API密钥未设置（建议先在testnet测试）")

    print("\n🎯 推荐流程:")
    print("1. 注册Binance Testnet账户: https://testnet.binancefuture.com/")
    print("2. 生成API密钥（开启期货交易权限）")
    print("3. 设置环境变量")
    print("4. 在testnet环境测试策略")
    print("5. 验证策略有效性后再考虑实盘")

def demo_trading_config():
    """演示交易配置"""
    print("\n📊 交易策略配置演示")
    print("=" * 50)

    config = TradingConfig(use_testnet=True)
    config.print_config()

    print("🎛️ 配置说明:")
    print(f"• 重新平衡间隔: 每{config.REBALANCE_INTERVAL//60}分钟重新计算仓位")
    print(f"• 做多/做空数量: 根据神经网络因子排序选择前{config.TOP_N_LONG}名做多，后{config.TOP_N_SHORT}名做空")
    print(f"• 杠杆倍数: {config.LEVERAGE}倍（保守策略）")
    print(f"• 历史数据: 使用前{config.LOOKBACK_HOURS}小时的OHLC数据")
    print(f"• 时间框架: {', '.join(config.TIMEFRAMES)} 多时间周期分析")
    print(f"• 风险控制: 单仓位最大{config.MAX_POSITION_PCT*100}%，最小保证金{config.MIN_MARGIN_BALANCE}U")

def demo_ccxt_config():
    """演示CCXT配置"""
    print("\n⚙️ CCXT配置演示")
    print("=" * 50)

    config = TradingConfig(use_testnet=True)
    ccxt_config = config.get_ccxt_config()

    print("CCXT配置参数:")
    for key, value in ccxt_config.items():
        if key in ['apiKey', 'secret']:
            # 隐藏敏感信息
            display_value = value[:8] + "..." if value and not value.startswith('YOUR_') else "未设置"
            print(f"  {key}: {display_value}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    try:
        demo_api_setup()
        demo_trading_config()
        demo_ccxt_config()

        print("\n" + "=" * 50)
        print("🎉 配置演示完成!")
        print("\n📝 总结:")
        print("• 环境配置模块已就绪")
        print("• 支持testnet和实盘切换")
        print("• 包含完整的风险控制参数")
        print("• 提供详细的配置验证")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()