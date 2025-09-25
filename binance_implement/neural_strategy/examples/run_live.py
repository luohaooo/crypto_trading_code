#!/usr/bin/env python3
"""
实盘环境运行示例
用于实际交易的自动化策略

⚠️ 警告: 这将使用真实资金进行交易！

使用方法:
python examples/run_live.py
"""

import sys
import os
import asyncio

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from automated_trading import AutomatedTradingSystem


def confirm_live_trading():
    """确认实盘交易"""
    print("🚨 实盘交易风险提示 🚨")
    print("=" * 50)
    print("⚠️  您即将启动实盘自动化交易系统")
    print("💰 这将使用您的真实资金进行交易")
    print("📉 存在资金损失的风险")
    print("🔐 请确保已设置BINANCE_API_KEY和BINANCE_API_SECRET环境变量")
    print("🛡️  建议先在testnet环境充分测试")
    print()

    # 多重确认
    confirmations = [
        "我理解这是实盘交易，将使用真实资金",
        "我已经在testnet环境充分测试了策略",
        "我愿意承担可能的资金损失风险"
    ]

    for i, confirmation in enumerate(confirmations, 1):
        print(f"{i}. {confirmation}")
        response = input(f"   请输入 'yes' 确认: ").strip().lower()
        if response != 'yes':
            print("❌ 实盘交易已取消")
            return False
        print()

    # 最终确认
    print("🔴 最后确认: 启动实盘自动化交易？")
    final_confirm = input("请输入 'START_LIVE_TRADING' 来最终确认: ").strip()
    if final_confirm != 'START_LIVE_TRADING':
        print("❌ 实盘交易已取消")
        return False

    return True


async def main():
    """主函数"""
    print("🚀 实盘自动化交易系统")
    print("=" * 50)

    # 确认实盘交易
    if not confirm_live_trading():
        return

    print("✅ 确认完成，启动实盘交易系统...")
    print()

    # 创建实盘交易系统
    trading_system = AutomatedTradingSystem(use_testnet=False)

    try:
        # 启动系统
        success = await trading_system.start()
        if success:
            print("✅ 系统正常结束")
        else:
            print("❌ 系统异常结束")
    except KeyboardInterrupt:
        print("\n⚡ 用户中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ 系统异常: {e}")
    finally:
        print("👋 实盘交易系统已关闭")


if __name__ == "__main__":
    print("🤖 Binance 实盘自动化交易系统")
    print("Copyright (c) 2024")
    print()

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        sys.exit(1)

    # 运行主程序
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)