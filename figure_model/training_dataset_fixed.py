#!/usr/bin/env python3
"""
修复版本的训练脚本 - 解决损失函数计算不一致问题
"""

import torch
import torch.nn as nn
from tqdm import tqdm

def train_loop(dataloader, net, loss_fn, optimizer):
    """训练循环 - 修复版本"""
    running_loss = 0.0
    current = 0
    net.train()

    with tqdm(dataloader) as t:
        for batch, (X, y) in enumerate(t):
            X = X.to(device)
            y = y.to(device)

            # 前向传播
            y_pred = net(X)

            # 确保标签类型正确（回归任务用float）
            if loss_fn.__class__.__name__ in ['L1Loss', 'MSELoss']:
                y = y.float()
            else:
                y = y.long()

            loss = loss_fn(y_pred.squeeze(), y)  # 确保维度匹配

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 正确的损失计算
            batch_loss = loss.item()
            running_loss = (len(X) * batch_loss + running_loss * current) / (len(X) + current)
            current += len(X)

            t.set_postfix({'running_loss': running_loss})

    return running_loss

def val_loop(dataloader, net, loss_fn):
    """验证循环 - 修复版本"""
    running_loss = 0.0
    current = 0
    net.eval()

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)

                # 前向传播
                y_pred = net(X)

                # 确保标签类型正确（回归任务用float）
                if loss_fn.__class__.__name__ in ['L1Loss', 'MSELoss']:
                    y = y.float()
                else:
                    y = y.long()

                loss = loss_fn(y_pred.squeeze(), y)  # 确保维度匹配

                # 修复：正确的损失计算（与训练一致）
                batch_loss = loss.item()
                running_loss = (len(X) * batch_loss + running_loss * current) / (len(X) + current)
                current += len(X)

                t.set_postfix({'running_loss': running_loss})

    return running_loss

def train_model_fixed(
    net,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    epochs=100,
    early_stopping_epoch=5,
    start_epoch=0,
    save_dir="./model_checkpoint",
    device='cuda'
):
    """修复版本的训练函数"""
    import os
    import datetime

    # 记录训练和验证损失
    train_losses = []
    val_losses = []

    # 记录最佳验证损失
    min_val_loss = float("inf")
    last_min_ind = -1

    # 模型存储目录
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, start_time)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 开始训练，保存目录: {save_dir}")
    print(f"📊 损失函数: {loss_fn.__class__.__name__}")
    print(f"🎯 优化器: {optimizer.__class__.__name__}")
    print(f"📱 设备: {device}")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        print(f"\n===== Epoch {epoch} =====")

        # 训练
        train_loss = train_loop(train_dataloader, net, loss_fn, optimizer)
        train_losses.append(train_loss)

        # 验证
        val_loss = val_loop(val_dataloader, net, loss_fn)
        val_losses.append(val_loss)

        # 打印对比信息
        print(f"📈 训练损失: {train_loss:.6f}")
        print(f"📉 验证损失: {val_loss:.6f}")
        print(f"📊 训练/验证比值: {train_loss/val_loss:.3f}")

        # 保存当前模型快照
        ckpt_name = f"baseline_epoch_{epoch}_train_{train_loss:.5f}_val_{val_loss:.5f}.pt"
        ckpt_path = os.path.join(save_dir, ckpt_name)

        # 保存模型状态
        save_dict = {
            'epoch': epoch,
            'model_state_dict': net.state_dict() if not isinstance(net, nn.DataParallel) else net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        torch.save(save_dict, ckpt_path)
        print(f"✔ Saved checkpoint: {ckpt_path}")

        # 早停逻辑
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            last_min_ind = epoch
            best_ckpt_path = ckpt_path
            print(f"🌟 新的最佳模型! 验证损失: {min_val_loss:.6f}")
        elif epoch - last_min_ind >= early_stopping_epoch:
            print(f"⏹ Early stopping at epoch {epoch} (no improvement for {early_stopping_epoch} epochs)")
            break

    print("\n🎉 Training done!")
    print(f"Best epoch: {last_min_ind}, best val_loss: {min_val_loss:.6f}")

    # 损失曲线分析
    if len(train_losses) > 1:
        print(f"\n📈 训练曲线分析:")
        print(f"   初始训练损失: {train_losses[0]:.6f}")
        print(f"   最终训练损失: {train_losses[-1]:.6f}")
        print(f"   训练损失下降: {(train_losses[0]-train_losses[-1])/train_losses[0]*100:.1f}%")

        print(f"📉 验证曲线分析:")
        print(f"   初始验证损失: {val_losses[0]:.6f}")
        print(f"   最终验证损失: {val_losses[-1]:.6f}")
        print(f"   验证损失下降: {(val_losses[0]-val_losses[-1])/val_losses[0]*100:.1f}%")

    return best_ckpt_path, train_losses, val_losses

# 使用示例
if __name__ == "__main__":
    # 假设已有数据加载器和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("修复版本的训练脚本已准备就绪!")
    print("主要修复:")
    print("1. 统一训练和验证的损失计算方式")
    print("2. 修复标签类型转换问题")
    print("3. 确保输出维度匹配")
    print("4. 添加详细的损失分析")