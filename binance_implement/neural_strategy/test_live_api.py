#!/usr/bin/env python3
"""
实盘API测试脚本
用于测试Binance实盘API的合约交易功能，诊断-4061错误

使用方法:
python test_live_api.py

注意:
- 请确保已设置正确的环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET
- 该脚本将使用少量资金进行测试
- 建议先在Testnet环境测试确认无误后再使用
"""

import os
import sys
import ccxt
import time
from datetime import datetime
from typing import Optional, Dict, Any

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from config import get_config


class LiveAPITester:
    """实盘API测试器"""

    def __init__(self):
        """初始化测试器"""
        self.config = get_config(use_testnet=False)  # 使用实盘配置
        self.exchange = None
        self.test_symbol = "SFP/USDT:USDT"  # 使用流动性最好的交易对
        self.test_amount_usdt = 20.0   # 测试金额，使用5 USDT（Binance最小要求）

    def print_step(self, step_num: int, description: str):
        """打印测试步骤"""
        print(f"\n{'='*60}")
        print(f"步骤 {step_num}: {description}")
        print(f"{'='*60}")

    def print_result(self, success: bool, message: str):
        """打印测试结果"""
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status}: {message}")

    def step1_test_connection(self) -> bool:
        """步骤1: 测试API连接"""
        self.print_step(1, "测试API连接")

        try:
            # 创建交易所实例
            self.exchange = ccxt.binance(self.config.get_ccxt_config())
            print(f"API Key: {self.config.API_KEY[:8]}...")
            print(f"环境: {self.config.ENV_NAME}")

            # 测试连接
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('total', 0)

            self.print_result(True, f"API连接成功，USDT余额: {usdt_balance:.2f}")

            if usdt_balance < self.test_amount_usdt:
                print(f"⚠️ 警告: 账户余额 {usdt_balance:.2f} USDT 少于测试金额 {self.test_amount_usdt} USDT")
                response = input("是否继续测试? (y/N): ")
                if response.lower() != 'y':
                    return False

            return True

        except Exception as e:
            self.print_result(False, f"API连接失败: {e}")
            return False

    def step2_check_position_mode(self) -> bool:
        """步骤2: 检查持仓模式"""
        self.print_step(2, "检查账户持仓模式")

        try:
            # 尝试获取持仓模式信息
            print("正在检查持仓模式...")

            # 方法1: 尝试获取账户信息
            try:
                account_info = self.exchange.fapiPrivateGetAccount()
                multi_assets_margin = account_info.get('multiAssetsMargin', False)
                print(f"多资产保证金模式: {multi_assets_margin}")
            except Exception as e:
                print(f"无法获取账户详细信息: {e}")

            # 方法2: 尝试获取持仓模式
            try:
                position_mode = self.exchange.fapiPrivateGetPositionSideDual()
                dual_side = position_mode.get('dualSidePosition', False)
                print(f"双向持仓模式: {dual_side}")

                if not dual_side:
                    print("⚠️ 当前为单向持仓模式")
                    print("建议: 可能需要切换到双向持仓模式以支持同时做多做空")
                else:
                    print("✅ 当前为双向持仓模式")

            except Exception as e:
                print(f"无法获取持仓模式: {e}")

            self.print_result(True, "持仓模式检查完成")
            return True

        except Exception as e:
            self.print_result(False, f"检查持仓模式失败: {e}")
            return False

    def step3_check_market_data(self) -> bool:
        """步骤3: 检查市场数据"""
        self.print_step(3, "检查市场数据和交易对信息")

        try:
            # 加载市场信息
            markets = self.exchange.load_markets()

            if self.test_symbol not in markets:
                self.print_result(False, f"交易对 {self.test_symbol} 不存在")
                return False

            market = markets[self.test_symbol]
            print(f"交易对: {self.test_symbol}")
            print(f"类型: {market.get('type', 'unknown')}")
            print(f"状态: {'活跃' if market.get('active', False) else '不活跃'}")
            print(f"基础币种: {market.get('base', 'unknown')}")
            print(f"计价币种: {market.get('quote', 'unknown')}")

            # 获取当前价格
            ticker = self.exchange.fetch_ticker(self.test_symbol)
            current_price = ticker['last']
            print(f"当前价格: {current_price}")

            # 计算测试数量
            test_quantity = self.test_amount_usdt / current_price
            print(f"测试数量: {test_quantity:.6f} {market.get('base', '')}")

            # 检查最小交易量
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if min_amount and test_quantity < min_amount:
                print(f"⚠️ 警告: 计算的交易量 {test_quantity:.6f} 小于最小交易量 {min_amount}")
                # 调整为最小交易量
                test_quantity = min_amount
                actual_usdt = test_quantity * current_price
                print(f"调整后数量: {test_quantity:.6f}")
                print(f"调整后金额: {actual_usdt:.2f} USDT")
                self.test_amount_usdt = actual_usdt

            self.print_result(True, "市场数据检查完成")
            return True

        except Exception as e:
            self.print_result(False, f"检查市场数据失败: {e}")
            return False

    def step4_test_order_creation(self) -> bool:
        """步骤4: 测试订单创建"""
        self.print_step(4, "测试订单创建")

        try:
            # 获取当前价格和计算数量
            ticker = self.exchange.fetch_ticker(self.test_symbol)
            current_price = ticker['last']
            test_quantity = self.test_amount_usdt / current_price

            # 获取精度信息
            market = self.exchange.markets[self.test_symbol]
            amount_precision = market['precision']['amount']

            # 处理精度
            if isinstance(amount_precision, float):
                decimal_places = len(str(amount_precision).split('.')[-1]) if '.' in str(amount_precision) else 0
                test_quantity = round(test_quantity, decimal_places)
                if test_quantity < amount_precision:
                    test_quantity = amount_precision

            print(f"准备创建订单:")
            print(f"  交易对: {self.test_symbol}")
            print(f"  方向: 买入 (做多)")
            print(f"  数量: {test_quantity}")
            print(f"  价格: {current_price} (市价)")
            print(f"  预估价值: {test_quantity * current_price:.2f} USDT")

            # 确认创建订单
            print("\n⚠️  这将创建一个真实的合约订单!")
            response = input("确认创建订单? (y/N): ")
            if response.lower() != 'y':
                print("用户取消订单创建")
                return False

            print("\n正在创建订单...")

            # 尝试不同的参数组合
            test_cases = [
                {"name": "指定positionSide=LONG", "params": {"positionSide": "LONG"}},
                {"name": "指定positionSide=BOTH", "params": {"positionSide": "BOTH"}},
            ]

            for i, test_case in enumerate(test_cases, 1):
                print(f"\n测试 {i}: {test_case['name']}")
                try:
                    order = self.exchange.create_market_order(
                        symbol=self.test_symbol,
                        side='buy',
                        amount=test_quantity,
                        params=test_case['params']
                    )

                    print(f"✅ 订单创建成功!")
                    print(f"  订单ID: {order.get('id', 'Unknown')}")
                    print(f"  状态: {order.get('status', 'Unknown')}")
                    print(f"  成交量: {order.get('filled', 0)}")

                    # 如果订单成功，立即平仓
                    if order.get('status') == 'closed' or order.get('filled', 0) > 0:
                        print(f"\n立即平仓...")

                        # 构建平仓参数，保持与开仓相同的positionSide
                        close_params = {}
                        if "positionSide" in test_case['params']:
                            close_params["positionSide"] = test_case['params']["positionSide"]

                        # 尝试平仓，首先尝试带reduceOnly，如果失败则不带该参数重试
                        close_success = False

                        # 第一次尝试：带reduceOnly参数
                        try:
                            close_params_with_reduce = close_params.copy()
                            close_params_with_reduce["reduceOnly"] = True

                            close_order = self.exchange.create_market_order(
                                symbol=self.test_symbol,
                                side='sell',
                                amount=order.get('filled', test_quantity),
                                params=close_params_with_reduce
                            )
                            print(f"✅ 平仓成功 (with reduceOnly): {close_order.get('id', 'Unknown')}")
                            close_success = True

                        except Exception as close_e:
                            error_str = str(close_e)
                            if "-1106" in error_str:
                                print(f"⚠️ reduceOnly参数不被接受，尝试不带该参数...")

                                # 第二次尝试：不带reduceOnly参数
                                try:
                                    close_order = self.exchange.create_market_order(
                                        symbol=self.test_symbol,
                                        side='sell',
                                        amount=order.get('filled', test_quantity),
                                        params=close_params
                                    )
                                    print(f"✅ 平仓成功 (without reduceOnly): {close_order.get('id', 'Unknown')}")
                                    close_success = True

                                except Exception as close_e2:
                                    print(f"❌ 平仓仍然失败: {close_e2}")
                            else:
                                print(f"❌ 平仓失败: {close_e}")

                        if not close_success:
                            print("⚠️ 平仓失败，但开仓测试已成功")

                    self.print_result(True, f"订单测试成功: {test_case['name']}")
                    return True

                except Exception as e:
                    print(f"❌ 订单创建失败: {e}")

                    # 分析错误
                    error_str = str(e)
                    if "-4061" in error_str:
                        print("这是-4061错误: 持仓方向不匹配")
                        if "positionSide" not in test_case['params']:
                            print("建议: 可能需要指定positionSide参数")
                    elif "-1111" in error_str:
                        print("这是精度错误: 数量精度不正确")
                    elif "-2019" in error_str:
                        print("这是余额不足错误")

                    continue

            self.print_result(False, "所有订单创建测试都失败了")
            return False

        except Exception as e:
            self.print_result(False, f"订单测试失败: {e}")
            return False

    def step5_try_position_mode_change(self) -> bool:
        """步骤5: 尝试更改持仓模式"""
        self.print_step(5, "尝试设置双向持仓模式")

        try:
            print("⚠️  这将尝试更改您的持仓模式设置")
            response = input("是否尝试设置为双向持仓模式? (y/N): ")
            if response.lower() != 'y':
                print("跳过持仓模式更改")
                return True

            # 尝试设置双向持仓模式
            try:
                result = self.exchange.fapiPrivatePostPositionSideDual({
                    'dualSidePosition': 'true'
                })
                print(f"✅ 双向持仓模式设置成功: {result}")

                # 再次测试订单创建
                print("\n现在重新测试订单创建...")
                time.sleep(1)  # 等待设置生效
                return self.step4_test_order_creation()

            except Exception as e:
                print(f"❌ 设置双向持仓模式失败: {e}")
                return False

        except Exception as e:
            self.print_result(False, f"持仓模式更改测试失败: {e}")
            return False

    def run_test(self):
        """运行完整测试"""
        print("🚀 开始实盘API测试")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试交易对: {self.test_symbol}")
        print(f"测试金额: {self.test_amount_usdt} USDT (Binance最小要求)")

        # 执行测试步骤
        if not self.step1_test_connection():
            return False

        if not self.step2_check_position_mode():
            return False

        if not self.step3_check_market_data():
            return False

        if not self.step4_test_order_creation():
            # 如果订单创建失败，尝试更改持仓模式
            if not self.step5_try_position_mode_change():
                return False

        print(f"\n{'='*60}")
        print("🎉 测试完成!")
        print(f"{'='*60}")
        return True


def main():
    """主函数"""
    print("实盘API测试脚本")
    print("=" * 40)

    # 安全确认
    print("⚠️  警告: 这将使用真实的API密钥进行实盘交易测试!")
    print("   - 请确保您已经充分理解风险")
    print("   - 建议使用少量资金进行测试")
    print("   - 确保您的API密钥权限设置正确")
    print()

    response = input("确认继续实盘测试? (yes/NO): ")
    if response.lower() != 'yes':
        print("测试已取消")
        return

    # 创建测试器并运行
    tester = LiveAPITester()

    try:
        success = tester.run_test()
        if success:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 测试过程中出现问题，请检查错误信息")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中发生未预期错误: {e}")
    finally:
        print("\n测试结束")


if __name__ == "__main__":
    main()