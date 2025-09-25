#!/usr/bin/env python3
"""
Testnet环境运行示例
用于测试和调试自动化交易策略

使用方法:
python examples/run_testnet.py
"""

import sys
import os
import asyncio

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from automated_trading import AutomatedTradingSystem


async def main():
    """主函数"""
    print("🧪 启动Testnet自动化交易系统")
    print("=" * 50)
    print("⚠️  这是测试环境，使用虚拟资金")
    print("🔧 请确保已设置BINANCE_TESTNET_API_KEY和BINANCE_TESTNET_API_SECRET环境变量")
    print()

    # 创建testnet交易系统
    trading_system = AutomatedTradingSystem(use_testnet=True)

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
        print("👋 Testnet交易系统已关闭")


if __name__ == "__main__":
    print("🤖 Binance Testnet 自动化交易系统")
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